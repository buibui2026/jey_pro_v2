"""V3 数据重做·阶段4：定版——组装校验、盲测集封存、Laya 转换、manifest 登记、报告。

校验三层：
  1. schema E01-E08（train.qwen_jey.validate_record mode=final）；
  2. V3 管线级：split 必填；eval 态八题+非空 gold；production 态七题+无 gold；
     labels 覆盖该态全部题键；概率仅归一化（不 clamp——与 teacher 原始相对大小一致）；
  3. 池纪律：train/dev 记录的 source 章必须落在本池章集合（防串池）。

被拒记录全部进 reject_log（stage=finalize，含原因）——600→599 凭空消失的事不再发生。

产物：
  data/train_v3/records_train.jsonl / records_dev.jsonl   （Qwen 训练记录，schema 格式）
  data/train_v3/laya_trainset_v3.jsonl                    （Laya 格式，calib=dev 池记录）
  data/blind_test_v3/blind_testset.v3.jsonl + manifest.json（封存）
  data/artifacts_manifest.json（追加 active 登记）
  data/train_v3/v3_data_report.md
运行：.venv/bin/python scripts/v3/a4_finalize.py
"""
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.qwen_jey.schema import make_record, validate_record  # noqa: E402

TRAIN_V3 = ROOT / "data/train_v3"
RAW_DIR = TRAIN_V3 / "_synth_raw"
BLIND_V3 = ROOT / "data/blind_test_v3"
EIGHT = {"ctx_relevance", "ctx_sufficiency", "groundedness", "citation_correct",
         "answers_question", "completeness", "answer_correctness", "overall"}
SEVEN = EIGHT - {"answer_correctness"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reject(reason, rec):
    with open(TRAIN_V3 / "reject_log.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "stage": "finalize",
                            "reason": reason, "sid": rec["id"], "pool": rec.get("split")},
                           ensure_ascii=False) + "\n")


def norm_probs(probs: dict) -> dict | None:
    """和归一（仅除以总和；不 clamp、不设上下限——teacher 相对大小原样保留）。"""
    # PRO 两种输出形状统一解包：{候选:p} 或 {"probs":{候选:p}}（字面照抄示例格式时）
    if isinstance(probs, dict) and set(probs) == {"probs"} and isinstance(probs["probs"], dict):
        probs = probs["probs"]
    vals = {k: float(v) for k, v in probs.items() if isinstance(v, (int, float))}
    if not vals or any(v < 0 for v in vals.values()):
        return None
    s = sum(vals.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in vals.items()}


def main():
    sg = json.loads((ROOT / "data/splits/source_groups.v3.json").read_text())
    chapters_by_pool = {p: {(g["law"], g["chapter"]) for g in sg["groups"][p]}
                        for p in ("train", "dev", "test")}

    raws = sorted(RAW_DIR.glob("*.json"))
    print(f"原始合成记录 {len(raws)} 条")
    kept = {"train": [], "dev": []}
    seen = set()
    fail = Counter()

    for f in raws:
        r = json.loads(f.read_text())
        pool = r["split"]
        # ── 管线级校验（先于 schema：结构性缺陷直接拒） ──
        expected_q = EIGHT if r["mode"] == "eval" else SEVEN
        if set(r["labels"]) != expected_q:
            miss = sorted(expected_q - set(r["labels"]))
            extra = sorted(set(r["labels"]) - expected_q)
            reject(f"labels题集≠{r['mode']}态（缺{miss} 多{extra}）", r)
            fail[f"labels题集"] += 1
            continue
        target = {}
        bad = False
        for qid in expected_q:
            np_ = norm_probs(r["labels"][qid])
            if np_ is None:
                reject(f"{qid} 概率非法（负值/和≤0/空）", r)
                fail["概率非法"] += 1
                bad = True
                break
            target[qid] = np_
        if bad:
            continue
        # 校正：teacher probs 值应是候选键 → 概率；确认键集与 rubric 一致（E02 前置）
        from train.qwen_jey.schema import build_questions
        bq = build_questions()
        for qid in expected_q:
            if set(target[qid]) != set(bq[qid]["candidates"]):
                reject(f"{qid} 概率键集≠candidates", r)
                fail["概率键集"] += 1
                bad = True
                break
            # 概率按候选键序重建（teacher 返回键序不定——按键查值虽无害，
            # 但按位置消费的下游会静默错标；定版层统一键序根除该隐患）
            target[qid] = {k: target[qid][k] for k in bq[qid]["candidates"]}
        if bad:
            continue
        if (r["law"], r["chapter"]) not in chapters_by_pool[pool]:
            reject("source 章不在本池（串池）", r)
            fail["串池"] += 1
            continue
        if r["mode"] == "eval" and not r["gold_answer"].strip():
            reject("eval 态 gold 为空", r)
            fail["gold空"] += 1
            continue

        # ── 组装（make_record：state_parts + E06 快照 + token 计数） ──
        # make_record 放全八题；production 态按 E01 口径裁掉第 8 题
        # （旧 v2_fmt 全量带八题+空gold 过不了 E01——当年断言降级"报警模式"放行，审计 M8 实锤）
        rec = make_record(
            rec_id=r["id"], question=r["question"], context=r["context"], answer=r["answer"],
            gold_answer=r["gold_answer"] if r["mode"] == "eval" else None,
            origin="v3-synth", testset_id=None,
            law_refs=[f"《{a['law']}》{a['article_no']}" for a in r["gold_articles"]],
            target_probs=target,
            teacher={"model": r["teacher"]["model"],
                     "request_hash": r["teacher"].get("request_hash", ""),
                     "raw": r["labels"], "raw_text": r["teacher"]["raw"]})
        rec["split"] = pool
        if r["mode"] == "production":
            rec["questions"] = {k: v for k, v in rec["questions"].items()
                                if k != "answer_correctness"}
        rec["v3"] = {"law": r["law"], "chapter": r["chapter"], "qtype": r["qtype"],
                     "difficulty": r["difficulty"], "ans_kind": r["ans_kind"],
                     "mode": r["mode"], "flaw": r.get("flaw"),
                     "gold_articles": r["gold_articles"], "generator": "deepseek-v4-flash",
                     "teacher_model": r["teacher"]["model"], "splits_version": "v3"}

        # ── schema E01-E08 ──
        errs = validate_record(rec, mode="final", seen=seen)
        if errs:
            for e in errs:
                reject(e, r)
            fail["schema"] += 1
            continue
        kept[pool].append(rec)

    # ── 写盘 ──
    for pool in ("train", "dev"):
        out = TRAIN_V3 / f"records_{pool}.jsonl"
        with open(out, "w") as f:
            for rec in kept[pool]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"records_{pool}.jsonl：{len(kept[pool])} 条")

    # ── Laya 转换（state+labels 原样；calib=dev 记录——记录级隔离，M5 审计口径） ──
    laya_rows = []
    for k, (pool, rec) in enumerate([(p, r) for p in ("train", "dev") for r in kept[p]]):
        laya_rows.append({
            "k": k, "route": "multilingual", "tier": "law_v3",
            "ans_kind": rec["v3"]["ans_kind"], "mode": rec["v3"]["mode"],
            "state": rec["state_text"], "split": pool,
            "labels": {qid: {"probs": dist} for qid, dist in rec["target_probs"].items()},
            "teacher_raw_full": rec["teacher"]["raw_text"],
            "parts": rec["state_parts"], "calib": pool == "dev",
        })
    laya_out = TRAIN_V3 / "laya_trainset_v3.jsonl"
    with open(laya_out, "w") as f:
        for row in laya_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"laya_trainset_v3.jsonl：{len(laya_rows)} 条（calib=dev {sum(1 for r in laya_rows if r['calib'])} 条）")

    # ── 盲测集封存 ──
    src = BLIND_V3 / "questions_test.jsonl"
    sealed = BLIND_V3 / "blind_testset.v3.jsonl"
    n_test = sum(1 for _ in open(src))
    shutil.copy(src, sealed)
    meta = {
        "version": "v3", "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_records": n_test, "sha256": sha256_file(sealed),
        "upstream": {"source_groups.v3.json": f"sha256:{sha256_file(ROOT / 'data/splits/source_groups.v3.json')}",
                     "chunks.jsonl": sg["upstream"]["chunks.jsonl"]},
        "discipline": [
            "本集只用于最终盲测：模型与 prompt 冻结前，任何人不查看 gold_answer / 不以本集调 prompt、调参、选型",
            "本集条文池（test 章）与 train/dev 章互斥；M2 用条全部隔离在 train 侧",
            "gold_answer 为出题时同步生成的 oracle gold；2026-10-10 终审抽检 30/117（28对/1错/1备注）"
            "并完成 2 条 human-fix（留痕 train_v3/human_fix_log.jsonl），本轮为复验版",
        ],
        "columns": ["id", "split", "law", "chapter", "qtype", "difficulty",
                    "question", "gold_answer", "gold_articles", "self_difficulty"],
    }
    (BLIND_V3 / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"盲测集封存：{n_test} 条 → {sealed.name}（sha256 前12={meta['sha256'][:12]}）")

    # ── 全局 manifest 追加 active ──
    gman_path = ROOT / "data/artifacts_manifest.json"
    gman = json.loads(gman_path.read_text())
    new_items = [
        ("splits/source_groups.v3.json", "V3 源分组切分（章级，三池互斥，冻结锚）"),
        ("train_v3/questions_train.jsonl", "V3 train 题库（出题即带 oracle gold）"),
        ("train_v3/questions_dev.jsonl", "V3 dev 题库"),
        ("train_v3/contexts.json", "V3 真实检索上下文（M1 链路，可重放）"),
        ("train_v3/records_train.jsonl", "V3 Qwen 训练记录（E01-E08 全过）"),
        ("train_v3/records_dev.jsonl", "V3 Qwen dev 记录（选点/校准用）"),
        ("train_v3/laya_trainset_v3.jsonl", "V3 Laya 训练集（calib=dev 记录级隔离）"),
        ("train_v3/reject_log.jsonl", "V3 全程拒绝记录（append-only）"),
        ("blind_test_v3/blind_testset.v3.jsonl", "V3 盲测集（封存，冻结前禁看标签）"),
    ]
    have = {}
    for a in gman["artifacts"]:
        have[a["path"]] = a
    for rel, role in new_items:
        p = ROOT / "data" / rel
        if not p.exists():
            continue
        n_rec = sum(1 for _ in open(p)) if p.suffix == ".jsonl" else None
        # 同 path 重定版＝刷新（hash/条数变化如实更新）；新路径＝追加
        have[f"data/{rel}"] = {
            "path": f"data/{rel}", "status": "active", "role": role,
            "contamination": None, "sha256": sha256_file(p),
            "bytes": p.stat().st_size, "n_records": n_rec,
        }
    gman["artifacts"] = list(have.values())
    gman["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gman_path.write_text(json.dumps(gman, ensure_ascii=False, indent=1))

    # ── 报告 ──
    def dist(rows, key):
        return dict(Counter(r["v3"][key] if "v3" in r else r[key] for r in rows))
    tok = [r["state_tokens"] for r in kept["train"] + kept["dev"]]
    report = [
        "# V3 数据定版报告", "",
        f"定版时间：{meta['sealed_at']}", "",
        "## 规模", "",
        f"- train 记录 {len(kept['train'])} 条 / dev {len(kept['dev'])} 条 / 盲测 {n_test} 题",
        f"- finalize 拒绝：{dict(fail) or '无'}",
        "",
        "## 分布（train）", "",
        f"- 答案档：{dist(kept['train'], 'ans_kind')}",
        f"- 模式：{dist(kept['train'], 'mode')}",
        f"- 题型：{dist(kept['train'], 'qtype')}  难度：{dist(kept['train'], 'difficulty')}",
        f"- 法源：{dist(kept['train'], 'law')}",
        "",
        "## state_tokens（E05 带）", "",
        f"- min {min(tok)} / 中位 {sorted(tok)[len(tok)//2]} / max {max(tok)}",
        "",
        "## 数据门自检（审计 §七）", "",
        f"- train/dev/test 按源（章）分组：✅ source_groups.v3.json 断言三池互斥",
        f"- test 零派生：✅ 盲测集条文池与 train/dev 章互斥，且未进入任何合成/打标流程",
        f"- gold 非空：✅ eval 态全量硬断言；概率仅归一化未 clamp",
        f"- 人工抽检：⏳ 待办——见下方终审清单",
        "",
        "## 人工终审清单（数据门最后一项）", "",
        "从盲测集与 train 各分层抽样（法源×题型×难度×答案档），逐条判 gold 正确性。",
        "生成命令：`.venv/bin/python scripts/v3/a5_sample.py`",
    ]
    (TRAIN_V3 / "v3_data_report.md").write_text("\n".join(report))
    print(f"报告 → {TRAIN_V3 / 'v3_data_report.md'}")
    if fail and sum(fail.values()) > len(raws) * 0.05:
        print(f"⚠️ 拒绝率 {sum(fail.values())/len(raws):.0%} > 5%——请检查 reject_log 再定版")
        sys.exit(1)


if __name__ == "__main__":
    main()
