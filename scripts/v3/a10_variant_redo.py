"""V2·P0-3 变体重设计：trunc 档 149(train)+~15(dev) 条按 gold 结构分流三形态（2026-10-11 有效性检查修复）。

原缺陷："取第一句"在结论先行结构下得到"简洁的正确答案"——trunc 档与 gold 档无差异
（answer_correctness 97% 判一致、completeness 1.69 vs 1.72），中间带监督缺失。

三形态（形态=judge 要学会判的答案样式）：
  partial_keypoints  要点缺失：多要点 gold（≥3 句或含"；"）截取前 ⌈n/2⌉ 句——真丢一半要点【answer 变→重打标】
  partial_offconcl   结论偏离：短 gold 改派 flawed"部分正确但核心结论偏离"【answer 新生成→重打标】
  concl_no_basis     结论无依据：保留"仅结论句"原形态，角色明确为 citation 维度信号【answer 不变→零 API】

分流：多要点→keypoints；短 gold 按 id sha 稳定交替 offconcl/no_basis。
留痕：variant_redo_log.jsonl；raw 覆写后由 a4 重定版。
运行：.venv/bin/python -u scripts/v3/a10_variant_redo.py
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))
from t3_common import call_flash, call_pro  # noqa: E402

RAW = ROOT / "data/train_v3/_synth_raw"
FLAW_OFFCONCL = "答案部分正确但核心结论偏离问题（答了一半跑题了）"


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[。；;!？])", text) if s.strip()]


def parse_labels(raw):
    """两种 PRO 输出形态统一解包（照抄 v1 a3.parse_labels 语义）。"""
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return {}
    items = parsed
    if isinstance(parsed.get("qid"), dict) and "probs" in parsed["qid"]:
        items = parsed["qid"]
        if "probs" in items:
            return {k: {"probs": v} for k, v in items["probs"].items()}
    out = {}
    for k, v in items.items():
        if isinstance(v, dict) and "probs" in v:
            out[k] = v["probs"]
    return out


def main():
    from rag.judge.judge import build_state
    from train.qwen_jey.schema import build_questions
    q8 = build_questions()
    seven = {k: v for k, v in q8.items() if k != "answer_correctness"}

    raws = sorted(RAW.glob("*.json"))
    trunc = []
    for f in raws:
        r = json.loads(f.read_text())
        if r.get("ans_kind") == "trunc":
            trunc.append((f, r))
    print(f"trunc 档记录：{len(trunc)} 条（train+dev）")

    log = open(ROOT / "data/train_v3/variant_redo_log.jsonl", "a")
    n = {"partial_keypoints": 0, "partial_offconcl": 0, "concl_no_basis": 0}
    t0 = time.time()
    for i, (f, r) in enumerate(trunc):
        sents = split_sentences(r["gold_answer"])
        multi = len(sents) >= 3 or "；" in r["gold_answer"]
        if multi:
            new_kind = "partial_keypoints"
            new_answer = "".join(sents[:max(1, (len(sents) + 1) // 2)])
        else:
            h = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16)
            new_kind = "partial_offconcl" if h % 2 == 0 else "concl_no_basis"
            if new_kind == "concl_no_basis":
                new_answer = "。".join(r["gold_answer"].split("。")[:1]) + "。"
            else:
                prompt = (f"问题：{r['question']}\n请写一段约 80 字的【有缺陷】回答，缺陷要求：{FLAW_OFFCONCL}。\n只输出回答正文。")
                new_answer = call_flash(prompt, 300).strip()

        if new_kind == "concl_no_basis" and new_answer == r["answer"]:
            # 零 API：state 不变，仅更新形态标注（原标签/teacher 原样保留）
            r["ans_kind"] = new_kind
            f.write_text(json.dumps(r, ensure_ascii=False))
            log.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "id": r["id"],
                                  "old": "trunc", "new": new_kind,
                                  "action": "relabel_only"}, ensure_ascii=False) + "\n")
            n[new_kind] += 1
            continue

        # state 变 → 重打标（mode 沿用原记录：eval 八题带 gold / production 七题）
        if r["mode"] == "eval":
            state = build_state(r["question"], r["context"], new_answer, r["gold_answer"])
            qs_used = q8
        else:
            state = build_state(r["question"], r["context"], new_answer, None)
            qs_used = seven
        p = (state + "\n\n你是评测裁判。对下列各题给出选项概率分布与一句依据，输出 JSON。"
             "硬性要求：每题概率之和为 1。")
        for qid, q in qs_used.items():
            crit = list(q["candidates"].keys())
            p += f"\n- {qid}：{q['instructions']}，选项：{'/'.join(crit)}"
        p += '\n输出格式：{"qid": {"probs": {...}, "reason": "一句话"}, ...}'
        raw_t = call_pro(p, 1400)
        labels = parse_labels(raw_t)

        r["ans_kind"] = new_kind
        r["answer"] = new_answer
        r["labels"] = labels
        r["teacher"] = {"model": "deepseek-v4-pro", "raw": raw_t}
        if new_kind == "partial_offconcl":
            r["flaw"] = "结论偏离·部分正确"
        f.write_text(json.dumps(r, ensure_ascii=False))
        log.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "id": r["id"],
                              "old": "trunc", "new": new_kind, "action": "regen+relabel",
                              "answer_sha": hashlib.sha256(new_answer.encode()).hexdigest()[:12]},
                  ensure_ascii=False) + "\n")
        n[new_kind] += 1
        if (i + 1) % 10 == 0 or i + 1 == len(trunc):
            print(f"  [{i+1}/{len(trunc)}] {dict(n)} {time.time()-t0:.0f}s", flush=True)
    log.close()
    print(f"\n变体重设计完成：{dict(n)} → 下一步 a4 重定版")


if __name__ == "__main__":
    main()
