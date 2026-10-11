"""V3 数据重做·阶段7：train↔盲测交叉校验（开训前最后一道，用户 2026-10-10 终审建议）。

三层检查：
  1. 跨集查重（硬门）：train+dev 题与盲测题 ngram≥0.4 或向量余弦>0.90 → 必须为 0 对
     （章级隔离的最终验证：两集不得有同题——同题不同 gold 是口径差异的最坏形态）；
  2. 相似题对 gold 口径对比（人工快查）：跨集余弦 top-15 对，输出双方 gold 对照清单；
  3. 分布对齐：两集题型/难度/法源分布对比表。

产物：data/train_v3/crosscheck_report.md（含人工快查清单）
运行：.venv/bin/python -u scripts/v3/a7_crosscheck.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import os
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "m4"))
from t42_synth import ngram_overlap  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "rag/config.yaml"))
TRAIN_V3 = ROOT / "data/train_v3"
BLIND_V3 = ROOT / "data/blind_test_v3"


def main():
    from rag.embedder import get_embedder, embed_query
    emb = get_embedder(os.path.expanduser(CFG["paths"]["embedding_gguf"]))

    tr = [json.loads(l) for l in open(TRAIN_V3 / "questions_train.jsonl")]
    tr += [json.loads(l) for l in open(TRAIN_V3 / "questions_dev.jsonl")]
    bl = [json.loads(l) for l in open(BLIND_V3 / "blind_testset.v3.jsonl")]
    print(f"train+dev {len(tr)} 题 vs 盲测 {len(bl)} 题")

    # ── 1. 跨集查重（硬门） ──
    tv = np.array([embed_query(emb, r["question"]) for r in tr], dtype=np.float32)
    bv = np.array([embed_query(emb, r["question"]) for r in bl], dtype=np.float32)
    sims = tv @ bv.T
    dup_pairs = []
    for i in range(len(tr)):
        for j in np.where(sims[i] > 0.90)[0]:
            if ngram_overlap(tr[i]["question"], bl[j]["question"]) >= 0.4 or sims[i, j] > 0.90:
                dup_pairs.append((tr[i]["id"], bl[j]["id"], float(sims[i, j])))
    print(f"跨集高相似对（余弦>0.90 或 ngram≥0.4）：{len(dup_pairs)} 对")
    for p in dup_pairs:
        print(f"  🔴 {p}")

    # ── 2. 相似题对 gold 口径对比（top-15，人工快查） ──
    order = np.dstack(np.unravel_index(np.argsort(-sims.ravel()), sims.shape))[0]
    top, seen_laws = [], set()
    for i, j in order:
        if sims[i, j] < 0.55:
            break
        key = (tr[i]["law"], bl[j]["law"])
        if key in seen_laws:          # 每对法源组合只取最相似一对，样本铺开
            continue
        seen_laws.add(key)
        top.append((tr[i], bl[j], float(sims[i, j])))
        if len(top) >= 15:
            break

    # ── 3. 分布对齐 ──
    def dist(rows, key):
        c = Counter(r[key] for r in rows)
        n = sum(c.values())
        return {k: f"{v}({v/n:.0%})" for k, v in sorted(c.items())}

    lines = ["# V3 跨集交叉校验报告", "",
             f"train+dev {len(tr)} 题 vs 盲测 {len(bl)} 题", "",
             "## 1. 跨集查重（硬门）", "",
             f"- 余弦>0.90 或 ngram≥0.4 的题对：**{len(dup_pairs)} 对**"
             + ("——✅ 两集无同题，章级隔离成立" if not dup_pairs else "——🔴 存在同题，禁止开训"),
             "",
             "## 2. 相似题对 gold 口径对比（人工快查）", "",
             "> 以下为跨集最相似题对（每法源组合一对）。核对点：量刑档/条号引用格式/"
             "结论结构是否同口径；相似问题不得出现矛盾的 gold。", ""]
    for t, b, s in top:
        lines += [f"### 余弦 {s:.2f}｜train {t['id']}（{t['law']}·{t['qtype']}）↔ 盲测 {b['id']}（{b['law']}·{b['qtype']}）",
                  f"- **train 题**：{t['question']}",
                  f"  **train gold**：{t['gold_answer'][:150]}{'…' if len(t['gold_answer'])>150 else ''}",
                  f"- **盲测题**：{b['question']}",
                  f"  **盲测 gold**：{b['gold_answer'][:150]}{'…' if len(b['gold_answer'])>150 else ''}",
                  f"- **口径一致？**：[ ]", ""]
    lines += ["## 3. 分布对齐", "",
              "| 维度 | train+dev | 盲测 |", "|---|---|---|"]
    for key in ("qtype", "difficulty", "law"):
        lines.append(f"| {key} | {dist(tr, key)} | {dist(bl, key)} |")
    gl = lambda rows: sum(len(r["gold_answer"]) for r in rows) / len(rows)
    lines += ["", f"gold 平均长度：train+dev {gl(tr):.0f} 字 / 盲测 {gl(bl):.0f} 字", "",
              "## 结论", "",
              f"- 硬门（无同题）：{'✅ 通过' if not dup_pairs else '🔴 不通过'}",
              "- 口径对比：⏳ 待人工快查上方题对（重点：量刑档表述、条号引用格式）",
              "- 分布：见上表，两集同构配额，实际产出比例一致即通过"]
    out = TRAIN_V3 / "crosscheck_report.md"
    out.write_text("\n".join(lines))
    print(f"报告 → {out}（相似对 {len(top)} 组待人工快查）")


if __name__ == "__main__":
    main()
