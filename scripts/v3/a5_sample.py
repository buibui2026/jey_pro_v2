"""V3 数据重做·阶段5：人工终审抽样清单（数据门最后一项：gold 非空且人工抽检）。

分层：法源 × 题型 × 答案档（train 记录）；盲测集按 法源 × 题型。
产出：data/train_v3/human_review_*.md——每条含问题/oracle gold/金条文原文，
人工只需对 gold 判 [对/错/存疑] 并记录，终审结果回填 a4 报告后数据门才算闭合。
运行：.venv/bin/python scripts/v3/a5_sample.py [--n-blind 30] [--n-train 30]
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_V3 = ROOT / "data/train_v3"
BLIND_V3 = ROOT / "data/blind_test_v3"
SEED = 42


def render(title, rows, headers):
    out = [f"# {title}", "",
           "> 逐条判定 gold：把行末 `[ ]` 改为 `[对]` / `[错]` / `[存疑]`；"
           "错/存疑请在备注列写一句原因。", "",
           "> 口径说明：**oracle gold**＝出题时锁定的标准答案（全量非空）；**answer**＝该记录实际"
           "进入 state 的答案（production 态不拼【参考答案】段、只有七题——这是设计，不是缺陷）。", ""]
    for i, r in enumerate(rows, 1):
        out.append(f"## {i}. [{r['tag']}] {r['id']}  [{r['law']}·{r.get('qtype','')}{r.get('ans_kind','')}]")
        out.append("")
        out.append(f"**问题**：{r['question']}")
        out.append("")
        out.append(f"**oracle gold**：{r['gold_answer']}")
        if r.get("answer"):
            out.append("")
            out.append(f"**answer（入 state，{r.get('ans_kind','')} 档）**：{r['answer']}")
        out.append("")
        out.append("**金条文**：")
        for c in r["ctx"]:
            out.append(f"- 《{c['law']}》{c['article_no']}：{c['text'][:160]}{'…' if len(c['text'])>160 else ''}")
        out.append("")
        out.append(f"**判定**：[ ]    **备注**：")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blind", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42, help="抽样种子（复验轮换用 43 等）")
    ap.add_argument("--include", default="", help="强制纳入 train 清单的 id（逗号分隔，如 human-fix 复验项）")
    ap.add_argument("--include-blind", default="", help="强制纳入盲测清单的 id（逗号分隔）")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    chunks = {f"{c['law']}|{c['article_no']}": c["text"]
              for c in (json.loads(l) for l in open(ROOT / "data/rag_store/chunks.jsonl"))}
    oracle = {}
    for f in (TRAIN_V3 / "questions_train.jsonl", TRAIN_V3 / "questions_dev.jsonl"):
        if f.exists():
            oracle.update({json.loads(l)["id"]: json.loads(l) for l in open(f)})

    inc_blind = [x for x in args.include_blind.split(",") if x]
    inc_train = [x for x in args.include.split(",") if x]

    blind = [json.loads(l) for l in open(BLIND_V3 / "blind_testset.v3.jsonl")]
    by = defaultdict(list)
    for r in blind:
        if r["id"] in inc_blind:
            continue
        by[(r["law"], r["qtype"])].append(r)
    picked = [{"tag": "盲测", "id": r["id"], "law": r["law"],
               "qtype": r["qtype"], "question": r["question"],
               "gold_answer": r["gold_answer"],
               "ctx": [{"law": a["law"], "article_no": a["article_no"],
                        "text": ""} for a in r["gold_articles"]]}
              for r in blind if r["id"] in inc_blind]
    keys = sorted(by)
    while len(picked) < min(args.n_blind, len(blind)):
        progressed = False
        for k in keys:                      # 轮转分层，保证每格先出一条
            if by[k]:
                r = by[k].pop(rng.randrange(len(by[k])))
                picked.append({"tag": "盲测", "id": r["id"], "law": r["law"],
                               "qtype": r["qtype"], "question": r["question"],
                               "gold_answer": r["gold_answer"],
                               "ctx": [{"law": a["law"], "article_no": a["article_no"],
                                        "text": ""} for a in r["gold_articles"]]})
                progressed = True
                if len(picked) >= args.n_blind:
                    break
        if not progressed:
            break
    # 金条文原文回填
    chunks = {f"{c['law']}|{c['article_no']}": c["text"]
              for c in (json.loads(l) for l in open(ROOT / "data/rag_store/chunks.jsonl"))}
    for p in picked:
        for c in p["ctx"]:
            c["text"] = chunks.get(f"{c['law']}|{c['article_no']}", "（未找到）")
    (TRAIN_V3 / "human_review_blind.md").write_text(
        render(f"盲测集终审抽样（{len(picked)}/{len(blind)} 条）", picked, None))

    train = [json.loads(l) for l in open(TRAIN_V3 / "records_train.jsonl")]
    by = defaultdict(list)
    for r in train:
        if r["id"] in inc_train:
            continue
        by[(r["v3"]["law"], r["v3"]["qtype"], r["v3"]["ans_kind"])].append(r)
    tmap = {r["id"]: r for r in train}
    picked2 = [{"tag": "train", "id": r["id"], "law": r["v3"]["law"],
                "qtype": r["v3"]["qtype"], "ans_kind": r["v3"]["ans_kind"],
                "mode": r["v3"]["mode"],
                "question": r["state_parts"]["question"],
                "gold_answer": (oracle.get(r["id"], {}).get("gold_answer")
                                or r["state_parts"].get("gold_answer") or "（缺失）") + (
                               f"（{r['v3']['mode']} 态：state 不拼参考答案段、七题）"
                               if r["v3"]["mode"] == "production" else ""),
                "answer": r["state_parts"]["answer"][:120],
                "ctx": r["state_parts"]["context"]}
               for r in (tmap[i] for i in inc_train if i in tmap)]
    keys = sorted(by)
    while len(picked2) < min(args.n_train, len(train)):
        progressed = False
        for k in keys:
            if by[k]:
                r = by[k].pop(rng.randrange(len(by[k])))
                picked2.append({"tag": "train", "id": r["id"], "law": r["v3"]["law"],
                                "qtype": r["v3"]["qtype"], "ans_kind": r["v3"]["ans_kind"],
                                "mode": r["v3"]["mode"],
                                "question": r["state_parts"]["question"],
                                "gold_answer": (oracle.get(r["id"], {}).get("gold_answer")
                                                or r["state_parts"].get("gold_answer") or "（缺失）") + (
                                               f"（{r['v3']['mode']} 态：state 不拼参考答案段、七题）"
                                               if r["v3"]["mode"] == "production" else ""),
                                "answer": r["state_parts"]["answer"][:120],
                                "ctx": r["state_parts"]["context"]})
                progressed = True
                if len(picked2) >= args.n_train:
                    break
        if not progressed:
            break
    (TRAIN_V3 / "human_review_train.md").write_text(
        render(f"train 记录终审抽样（{len(picked2)}/{len(train)} 条）", picked2, None))
    print(f"终审清单：盲测 {len(picked)} 条 → human_review_blind.md；"
          f"train {len(picked2)} 条 → human_review_train.md")


if __name__ == "__main__":
    main()
