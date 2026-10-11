"""V3·M8 前置：V3 全量特征抽取（656 条 → data/train_v3/feat_cache/，断点安全）。

缓存四重键（backbone sha + tokenizer sha + dtype + 行明文 sha）幂等——重跑零成本。
运行：.venv/bin/python -u scripts/v3/a8_extract.py [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.qwen_jey.features import CACHE_DIR, FeatureExtractor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = []
    for f in ("records_train.jsonl", "records_dev.jsonl"):
        recs += [json.loads(l) for l in open(ROOT / "data/train_v3" / f)]
    if args.limit:
        recs = recs[:args.limit]
    n_rows = sum(len(r["questions"]) for r in recs)
    print(f"待抽取：{len(recs)} 条记录 ≈ {n_rows} 行（eval 八题/production 七题）")

    fx = FeatureExtractor()
    t0 = time.time()
    for i, r in enumerate(recs, 1):
        fx.extract_record(r)
        if i % 20 == 0 or i == len(recs):
            rate = i / (time.time() - t0)
            eta = (len(recs) - i) / rate if rate > 0 else 0
            print(f"[{i}/{len(recs)}] {time.time()-t0:.0f}s | {rate:.2f}条/s | ETA {eta/60:.0f}min",
                  flush=True)
    n_cached = len(list(CACHE_DIR.glob("*.pt")))
    print(f"特征就绪：{len(recs)} 条，缓存 {n_cached} 个行文件，耗时 {(time.time()-t0)/60:.0f} 分钟")


if __name__ == "__main__":
    main()
