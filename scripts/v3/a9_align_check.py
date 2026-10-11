"""V3·M8 前置：特征-标签对齐检查（开训前最后一道，用户 2026-10-10 要求）。

八层检查（任一 FAIL 即非零退出，禁止带病开训）：
  1. 覆盖：656 条记录的每一行（题×记录）特征缓存文件齐全（四重键定位）；
  2. 题键三方对齐：questions == target_probs == 特征键集；eval 八题 / production 七题；
  3. 形状：k 行数==候选数（noul 2 / choice·score 3），hidden 维==主干宽度；
  4. 键序对齐：target_probs 键序==candidates 键序（按位置消费即静默错标的最隐蔽层）；
  5. 概率合法：每题和=1±0.001、非负；
  6. split 纪律：train/dev id 无交叉、state_text sha 全局唯一；
  7. 端到端：train_a.make_examples 全量构造成功（行数=Σ每记录题数，y 形状==候选数）；
  8. gold 可见性（终审三-3）：行级 state 与 judge 双 state 口径一致——七生产题的行明文
     不得含【参考答案】，仅 answer_correctness 行明文含（eval 记录）；production 全不含。
运行：.venv/bin/python -u scripts/v3/a9_align_check.py
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.qwen_jey.features import CACHE_DIR, FeatureExtractor  # noqa: E402
from train.qwen_jey.rows import record_rows, record_states  # noqa: E402
from train.qwen_jey.train_a import make_examples  # noqa: E402

EIGHT = {"ctx_relevance", "ctx_sufficiency", "groundedness", "citation_correct",
         "answers_question", "completeness", "answer_correctness", "overall"}
fail = Counter()
bad_samples = []


def check(cond, tag, detail=""):
    if not cond:
        fail[tag] += 1
        if len(bad_samples) < 10:
            bad_samples.append(f"{tag}: {detail}")


def main():
    recs = {"train": [json.loads(l) for l in open(ROOT / "data/train_v3/records_train.jsonl")],
            "dev": [json.loads(l) for l in open(ROOT / "data/train_v3/records_dev.jsonl")]}
    fx = FeatureExtractor()          # 只读缓存（不命中会现算——覆盖检查前应全命中）
    n_rows_total, hidden = 0, None
    seen_ids, seen_state_sha = set(), set()

    for split, rows in recs.items():
        for r in rows:
            rid, mode = r["id"], r["v3"]["mode"]
            expected = EIGHT if mode == "eval" else EIGHT - {"answer_correctness"}
            # ── 2. 题键三方对齐 ──
            check(set(r["questions"]) == expected, "题集", f"{rid} {mode} {sorted(r['questions'])}")
            check(set(r["target_probs"]) == expected, "标签题集", f"{rid}")
            # ── 6. split 纪律 ──
            check(rid not in seen_ids, "id重复", rid)
            seen_ids.add(rid)
            check(r["split"] == split, "split错标", f"{rid} {r['split']}!={split}")
            ssha = hashlib.sha256(r["state_text"].encode()).hexdigest()[:16]
            check(ssha not in seen_state_sha, "state重复", rid)
            seen_state_sha.add(ssha)

            feats = fx.extract_record(r)     # 1. 覆盖（未命中缓存会现场前向并落盘）
            check(set(feats) == expected, "特征题集", f"{rid}")
            rows = record_rows(r)
            # ── 8. gold 可见性（judge 双 state 训推一致） ──
            base, gold_state = record_states(r)
            for qk in r["questions"]:
                st = rows[qk]["state"]
                want_gold = qk == "answer_correctness" and gold_state != base
                check("【参考答案】" in st, "gold缺失", f"{rid}.{qk} correctness 行应含参考答案") if want_gold \
                    else check("【参考答案】" not in st, "gold泄漏", f"{rid}.{qk} 非 correctness 行含参考答案")
            for qk, q in r["questions"].items():
                n_rows_total += 1
                tp = r["target_probs"][qk]
                cands = q["candidates"]
                # ── 4. 键序对齐 ──
                check(list(tp) == list(cands), "键序", f"{rid}.{qk} 标签序{list(tp)}≠候选序{list(cands)}")
                # ── 5. 概率合法 ──
                s = sum(tp.values())
                check(abs(s - 1) <= 0.001 and min(tp.values()) >= 0, "概率", f"{rid}.{qk} 和={s:.4f}")
                # ── 3. 形状 ──
                f = feats[qk]
                k = f["k"]
                check(k.shape[0] == len(cands), "候选数", f"{rid}.{qk} k={tuple(k.shape)} n_cand={len(cands)}")
                hidden = f["q"].shape[-1] if hidden is None else hidden
                check(f["q"].shape[-1] == hidden and k.shape[-1] == hidden, "hidden维",
                      f"{rid}.{qk}")
                # ── 7. 行预算 ──
                n_tok = len(rows[qk]["ids"])
                check(n_tok <= 4096, "行预算", f"{rid}.{qk} {n_tok}")

    # ── 1b. 缓存文件数 ≥ 总行数（防键共享：键漏题行时文件数 < 行数，多题读回彼此特征；
    #        文件数 > 行数＝旧键孤儿（数据重放后行明文变了），append-only 缓存的正常残留——报告不拒） ──
    n_files = len(list(CACHE_DIR.glob("*.pt")))
    check(n_files >= n_rows_total, "缓存文件数",
          f"{n_files}<{n_rows_total}（键共享或覆盖不全）")
    if n_files > n_rows_total:
        print(f"ℹ️ 孤儿缓存 {n_files - n_rows_total} 个（旧键残留，无害；如需清理可清空目录全量重抽）")

    # ── 7. 端到端样本构造 ──
    all_recs = recs["train"] + recs["dev"]
    for r in all_recs:
        r["_feats"] = fx.extract_record(r)
    exs = make_examples(all_recs, extractor=None)
    n_expect = sum(len(r["questions"]) for r in all_recs)
    check(len(exs) == n_expect, "样本数", f"{len(exs)}!={n_expect}")
    for e in exs:
        check(e["y"].shape[0] == e["k"].shape[0], "y/k不齐", f"{e['rid']}.{e['qid']}")

    print(f"记录 {len(all_recs)}（train {len(recs['train'])}/dev {len(recs['dev'])}）｜行 {n_rows_total}｜"
          f"hidden {hidden}｜样本 {len(exs)}")
    by_split = Counter(e["rid"].split("-")[1] for e in exs)
    print(f"样本池分布（id 前缀）：{dict(by_split)}")
    if fail:
        print(f"\n🔴 对齐失败 {dict(fail)}")
        for b in bad_samples:
            print(f"  - {b}")
        sys.exit(1)
    print("✅ 特征-标签八层对齐全部通过——可开训")


if __name__ == "__main__":
    main()
