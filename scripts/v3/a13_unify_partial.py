"""V2·P0-5b 形态收敛：无效子形态回收为已证有效的 offconcl + 补 40 条 irrelevant。

复检证据（2026-10-11）：
  offconcl            ✅ 50% 中间档（部分一致 9 + 不一致 9 / 36）、completeness 0.68 梯度明显
  keypoints 判一致部分 ❌ 54/78 仍判一致（截半对结论集中的 gold 伤害不足）
  no_basis 判正确部分  ❌ 29/35 citation 判正确（teacher 对"结论对但无引用"宽容）

动作：
  1. keypoints 且 correctness argmax=一致 → 转 offconcl（重生成+重打标）
  2. no_basis  且 citation argmax=正确   → 转 offconcl（同上）
     （各自保留真实有效的少量样本，维持形态多样性）
  3. 补 40 条 pure-irrelevant（噪声全换），拉无关档到 ≥8%
运行：.venv/bin/python -u scripts/v3/a13_unify_partial.py
"""
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))
from t3_common import call_flash, call_pro  # noqa: E402
from a10_variant_redo import parse_labels  # noqa: E402
from a12_retry_rejected import labels_valid  # noqa: E402

RAW = ROOT / "data/train_v3/_synth_raw"
FLAW = "答案部分正确但核心结论偏离问题（答了一半跑题了）"
SEED = 43


def relabel(r, q8, seven, build_state):
    if r["mode"] == "eval":
        state = build_state(r["question"], r["context"], r["answer"], r["gold_answer"])
        qs_used = q8
    else:
        state = build_state(r["question"], r["context"], r["answer"], None)
        qs_used = seven
    p = (state + "\n\n你是评测裁判。对下列各题给出选项概率概率分布与一句依据，输出 JSON。"
         "硬性要求：每题概率之和为 1。"
         "\n注意：每题必须给出【每个选项】的概率（0~1 小数、总和为1），禁止只输出最终选项名或档位数字。")
    for qid, q in qs_used.items():
        crit = list(q["candidates"].keys())
        p += f"\n- {qid}：{q['instructions']}，选项：{'/'.join(crit)}"
    p += '\n输出格式：{"qid": {"probs": {...}, "reason": "一句话"}, ...}'
    raw_t = call_pro(p, 1400)
    labels = parse_labels(raw_t)
    return labels, raw_t


def main():
    from rag.judge.judge import build_state
    from train.qwen_jey.schema import build_questions
    q8 = build_questions()
    seven = {k: v for k, v in q8.items() if k != "answer_correctness"}

    log = open(ROOT / "data/train_v3/variant_redo_log.jsonl", "a")
    # ── 1+2. 无效子形态回收 → offconcl ──
    recs = [json.loads(l) for l in open(ROOT / "data/train_v3/records_train.jsonl")]
    convert = []
    for r in recs:
        kind = r["v3"]["ans_kind"]
        if kind == "partial_keypoints" and "answer_correctness" in r["target_probs"] \
           and max(r["target_probs"]["answer_correctness"],
                   key=r["target_probs"]["answer_correctness"].get) == "一致":
            convert.append(r["id"])
        elif kind == "concl_no_basis" \
             and max(r["target_probs"]["citation_correct"],
                     key=r["target_probs"]["citation_correct"].get) == "正确":
            convert.append(r["id"])
    print(f"回收转 offconcl：{len(convert)} 条")
    n_ok = 0
    for i, rid in enumerate(convert):
        f = RAW / f"{rid}.json"
        r = json.loads(f.read_text())
        prompt = (f"问题：{r['question']}\n请写一段约 80 字的【有缺陷】回答，缺陷要求：{FLAW}。\n只输出回答正文。")
        r["answer"] = call_flash(prompt, 300).strip()
        r["ans_kind"] = "partial_offconcl"
        r["flaw"] = "结论偏离·部分正确"
        labels, raw_t = relabel(r, q8, seven, build_state)
        if labels_valid(labels, q8 if r["mode"] == "eval" else seven, q8):
            r["labels"] = labels
            r["teacher"] = {"model": "deepseek-v4-pro", "raw": raw_t}
            n_ok += 1
        f.write_text(json.dumps(r, ensure_ascii=False))
        log.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "id": rid,
                              "old": "invalid_subform", "new": "partial_offconcl",
                              "action": "regen+relabel"}, ensure_ascii=False) + "\n")
        if (i + 1) % 15 == 0:
            print(f"  转换 [{i+1}/{len(convert)}] 有效 {n_ok}", flush=True)

    # ── 3. 补 40 条 pure-irrelevant ──
    chunks = [json.loads(l) for l in open(ROOT / "data/rag_store/chunks.jsonl")]
    sg = json.loads((ROOT / "data/splits/source_groups.v3.json").read_text())
    banned = {(b["law"], b["article_num"]) for b in sg["banned_articles"]}
    pool_by_law = defaultdict(list)
    for c in chunks:
        if c["article_sub"]:
            continue
        for p, gs in sg["groups"].items():
            if p != "train":
                continue
            for g in gs:
                if g["law"] == c["law"] and c["article_num"] in g["articles"] \
                   and (c["law"], c["article_num"]) not in banned:
                    pool_by_law[c["law"]].append(c)
    qs = [json.loads(l) for l in open(ROOT / "data/train_v3/questions_train.jsonl")]
    rng = random.Random(SEED)
    used_q = {r["state_parts"]["question"] for r in recs if r["v3"]["ans_kind"].startswith("gold_") and r["v3"]["ans_kind"].endswith("_ctx")}
    by_law = defaultdict(list)
    for q in qs:
        if q["question"] not in used_q:
            by_law[q["law"]].append(q)
    picked = []
    laws = sorted(by_law)
    while len(picked) < 40:
        for law in laws:
            if by_law[law]:
                picked.append(by_law[law].pop(rng.randrange(len(by_law[law]))))
                if len(picked) >= 40:
                    break
    n_new = 0
    for i, q in enumerate(picked):
        gold_keys = {(a["law"], a["article_no"]) for a in q["gold_articles"]}
        noise = [c for c in pool_by_law[q["law"]] if (c["law"], c["article_no"]) not in gold_keys]
        rng.shuffle(noise)
        ctx = noise[:6]
        if len(ctx) < 4:
            continue
        mode = "production" if i % 5 == 4 else "eval"
        ctx_rec = [{"law": c["law"], "article_no": c["article_no"],
                    "article_num": c.get("article_num"), "text": c["text"]} for c in ctx]
        r = {"id": f"v3n-{q['law']}-{600+i:04d}", "split": "train", "law": q["law"],
             "chapter": q["chapter"], "qtype": q["qtype"], "difficulty": q["difficulty"],
             "ans_kind": "gold_irrelevant_ctx", "mode": mode, "flaw": None,
             "question": q["question"], "gold_answer": q["gold_answer"],
             "gold_articles": q["gold_articles"], "context": ctx_rec,
             "answer": q["gold_answer"].strip()}
        labels, raw_t = relabel(r, q8, seven, build_state)
        r["labels"] = labels
        r["teacher"] = {"model": "deepseek-v4-pro", "raw": raw_t}
        (RAW / f"{r['id']}.json").write_text(json.dumps(r, ensure_ascii=False))
        n_new += 1
    log.close()
    print(f"\n完成：回收 {n_ok}/{len(convert)} 有效转换 + 新增 irrelevant {n_new} 条 → a4 重定版")


if __name__ == "__main__":
    main()
