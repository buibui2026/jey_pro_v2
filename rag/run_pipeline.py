#!/usr/bin/env python3
"""T1.5 链路串联：一条命令跑通 问题→检索→重排→生成→judge→指标。

用法：
  .venv/bin/python rag/run_pipeline.py --question "打架斗殴会被拘留多少天？"
  .venv/bin/python rag/run_pipeline.py --file data/testset/smoke_questions.jsonl
  .venv/bin/python rag/run_pipeline.py --file ... --stage judge --run-id <已有run>   # 分段/续跑
分段产物落 data/runs/{run_id}/{i}_*.json——分析闭环"隔离"手段的工程落点（T1.5 方案 §一）。
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

CFG = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
RUNS = Path(CFG["paths"]["runs"])


def est_tokens(text: str) -> int:
    return int(len(text) * CFG["retrieval"]["chars_per_token"]) + 1


def retrieve(question: str) -> dict:
    """粗排 top_n → 重排 → token 预算优先取 K（满 3200 即止，K∈[1, top_k_max]）。"""
    from rag.embedder import embed_query, get_embedder
    from rag.reranker import get_reranker, rerank
    from rag.vector_store import load_store, search

    chunks, mat = load_store()
    ll_e = get_embedder(CFG["paths"]["embedding_gguf"].replace("~", str(Path.home())))
    ll_r = get_reranker(CFG["paths"]["reranker_gguf"].replace("~", str(Path.home())))

    cand_idx = search(embed_query(ll_e, question), mat, CFG["retrieval"]["top_n"])
    cands = [chunks[i] for i in cand_idx]
    scores = rerank(ll_r, question, [f"{c['law']} {c['article_no']}：{c['text']}" for c in cands])
    ranked = sorted(zip(scores, cand_idx, cands), key=lambda x: -x[0])

    budget, selected = CFG["retrieval"]["context_token_budget"], []
    for s, idx, c in ranked[: CFG["retrieval"]["top_k_max"]]:
        t = est_tokens(f"{c['law']} {c['article_no']}：{c['text']}")
        if selected and est_tokens(_join(selected)) + t > budget:
            break
        selected.append({"score": round(s, 4), "law": c["law"], "article_no": c["article_no"],
                         "text": c["text"]})
    return {"question": question, "candidates_scored": len(ranked), "context": selected}


def _join(sel: list[dict]) -> str:
    return "".join(f"{c['law']}{c['article_no']}{c['text']}" for c in sel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question")
    ap.add_argument("--file")
    ap.add_argument("--stage", default="all", choices=["retrieve", "generate", "judge", "all"])
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    records = ([{"id": "q0", "question": args.question}] if args.question
               else [json.loads(l) for l in open(args.file)])
    run_dir = RUNS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        i, base = rec["id"], run_dir / rec["id"]
        t0 = time.time()
        if args.stage in ("retrieve", "all"):
            # 坏例注入：手写 context 直通，不经检索/重排（T1.6 方案 §一）
            r = ({"question": rec["question"], "candidates_scored": 0, "context": rec["fixed_context"]}
                 if rec.get("fixed_context") else retrieve(rec["question"]))
            r.update({"id": i, **{k: rec[k] for k in ("gold_answer", "law_hint") if k in rec}})
            base.with_suffix(".retrieval.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        if args.stage in ("generate", "all") and rec.get("fixed_answer") is None:
            ret = json.loads(base.with_suffix(".retrieval.json").read_text())
            from rag.generator import generate
            base.with_suffix(".answer.json").write_text(json.dumps(
                {"id": i, "answer": generate(rec["question"], ret["context"])}, ensure_ascii=False))
        if args.stage in ("judge", "all"):
            ret = json.loads(base.with_suffix(".retrieval.json").read_text())
            ans = (rec.get("fixed_answer") if rec.get("fixed_answer") is not None
                   else json.loads(base.with_suffix(".answer.json").read_text())["answer"])
            from rag.judge.judge import run as judge_run
            base.with_suffix(".judge.json").write_text(json.dumps(
                {"id": i, **judge_run(rec["question"], ret["context"], ans, rec.get("gold_answer"))},
                ensure_ascii=False))
        print(f"[{i}] {args.stage} done in {time.time() - t0:.1f}s -> {run_dir}")

    if args.stage in ("judge", "all"):
        from rag.metrics import summarize
        summarize(run_dir)


if __name__ == "__main__":
    main()
