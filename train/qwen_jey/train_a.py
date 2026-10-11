"""T8.1 阶段 A 训练循环：冻结主干（特征已缓存）＋训头 262,400。

规格出处＝M6 T6.4 §二：L＝CE(teacher 软标签)＋0.5·Brier；AdamW(0.9,0.95) wd0.01 lr1e-3；
cosine＋warmup5%；批 16 行；≤10 轮 patience2（val KL）；选点 val KL 主＋ECE 辅；seed 固定重跑逐位一致。

用法：
    .venv/bin/python -m train.qwen_jey.train_a --data data/train/qwen_jey_dev_samples.jsonl   # 冒烟
    .venv/bin/python -m train.qwen_jey.train_a --data data/train/qwen_jey_trainset.jsonl --batch 1 --overfit

════════════════════ 实测输出（2026-10-08 冒烟：dev C 的 8 行，epochs=30） ════════════════════
# [Aep0] loss=1.1341 val: agree=100.0% KL=0.4438 brier=0.3161   ← 初始 loss≈ln3+Brier 项，预测锚命中
# [Aep14] loss=0.6319 val: agree=100.0% KL=0.0115 brier=0.0062
# [Aep29] loss=0.6118 val: agree=100.0% KL=0.0001 brier=0.0000  ← 8 行小集故意过拟合，容量验证 ✓
# == 训练集 == agree=100.0% KL=0.0073
# 同 seed 双跑：ep29 与汇总逐字符一致（确定性 ✓）
# S4：训出的头对无标签 A/B 行前向，概率和=1、候选数对、判断合理（全部有据 0.82~0.84）
══════════════════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

from .features import FeatureExtractor
from .head import DecisionHead
from .schema import build_questions


def load_dataset(path: str, *, batch_filter: int | None = None, limit: int | None = None,
                 split: str | None = None) -> list[dict]:
    recs = [json.loads(l) for l in open(path)]
    if batch_filter:
        recs = [r for r in recs if r.get("batch") == batch_filter]
    if split:
        recs = [r for r in recs if r.get("split", "train") == split]
    if limit:
        recs = recs[:limit]
    return recs


def make_examples(recs: list[dict], extractor: FeatureExtractor | None,
                  cache_only: bool = False) -> list[dict]:
    """记录 → 样本列表（每题一行）：{q:[H], k:[N,H], y:[N], qid, rid}。"""
    out = []
    for r in recs:
        feats = extractor.extract_record(r) if extractor else r.get("_feats")
        for qk, q in r["questions"].items():
            tp = (r.get("target_probs") or {}).get(qk)
            if not tp or feats is None:
                continue
            keys = list(q["candidates"])
            y = torch.tensor([tp[k] for k in keys])
            f = feats[qk]
            out.append({"q": f["q"], "k": f["k"], "y": y, "qid": qk, "rid": r["id"]})
    return out


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    logp = torch.log_softmax(logits, dim=-1)
    ce = -(y * logp).sum(-1).mean()
    p = torch.softmax(logits, dim=-1)
    brier = ((p - y) ** 2).sum(-1).mean()
    return ce + 0.5 * brier, {"ce": ce.item(), "brier": brier.item()}


def evaluate(examples: list[dict], head: DecisionHead) -> dict:
    """标准 10 桶 ECE（审计 M8 🔴：ece_proxy 的'错误样本置信度和/样本数'不是 ECE，2026-10-10 换正式实现）。"""
    head.eval()
    agree = kl_sum = brier_sum = n = 0
    bins = [[0.0, 0, 0] for _ in range(10)]   # [置信度累计, 样本数, 正确数]
    with torch.no_grad():
        for ex in examples:
            logits = head(ex["q"].unsqueeze(0), ex["k"].unsqueeze(0))[0]
            p = torch.softmax(logits, -1)
            y = ex["y"]
            agree += (p.argmax() == y.argmax()).item()
            kl = (y * (torch.log(y + 1e-9) - torch.log(p + 1e-9))).sum().item()
            kl_sum += kl
            brier_sum += ((p - y) ** 2).sum().item()
            b = min(9, int(p.max().item() * 10))
            bins[b][0] += p.max().item(); bins[b][1] += 1
            bins[b][2] += int(p.argmax() == y.argmax())
            n += 1
    ece = sum((cnt / n) * abs(acc / cnt - conf / cnt) for conf, cnt, acc in bins if cnt)
    return {"agree": agree / n, "kl": kl_sum / n, "brier": brier_sum / n, "ece": ece, "n": n}


def run(recs: list[dict], *, epochs: int = 10, seed: int = 0, overfit: bool = False,
        extractor: FeatureExtractor | None = None, tag: str = "") -> dict:
    """split 纪律（审计 M8 🔴 修复 2026-10-10）：val 只取 dev/val split 的记录；
    数据里没有就报错中止——禁止拿训练样本前 10% 代用（v2 时代 599 条无 split 正是这么泄漏的）。"""
    torch.manual_seed(seed)
    random.Random(seed).shuffle(recs)
    if overfit:
        train, val = recs, []
    else:
        val = [r for r in recs if r.get("split") in ("dev", "val")]
        train = [r for r in recs if r.get("split") not in ("dev", "val")]
        if not val:
            raise ValueError("数据中无 split=dev/val 的记录——早停/选点需要独立 dev（V3: records_dev.jsonl），"
                             "拒绝用训练样本代用（训练泄漏）")
        tr_ids = {r["id"] for r in train}
        va_ids = {r["id"] for r in val}
        assert not (tr_ids & va_ids), "train/dev 记录 id 交叉"
    tr_ex = make_examples(train, extractor)
    va_ex = make_examples(val, extractor) if val else tr_ex

    head = DecisionHead()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    steps = max(1, math.ceil(len(tr_ex) / 16)) * epochs
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.05, total_iters=max(1, int(steps * 0.05)))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [sched, torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)],
        milestones=[max(1, int(steps * 0.05))])

    best = {"kl": float("inf"), "state": None, "epoch": -1}
    patience = 0
    for ep in range(epochs):
        head.train()
        random.Random(seed * 1000 + ep).shuffle(tr_ex)
        losses = []
        for i in range(0, len(tr_ex), 16):
            chunk = tr_ex[i:i + 16]
            # 候选数因题型而异（noul=2/choice·score=3）——逐行前向、批内均值（头仅 26 万参数，毫秒级）
            loss = sum(loss_fn(head(c["q"].unsqueeze(0), c["k"].unsqueeze(0))[0], c["y"])[0]
                       for c in chunk) / len(chunk)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        m = evaluate(va_ex, head)
        marker = ""
        if m["kl"] < best["kl"]:
            best = {"kl": m["kl"], "state": {k2: v2.clone() for k2, v2 in head.state_dict().items()},
                    "epoch": ep}
            patience = 0
        else:
            patience += 1
            marker = " ↑valKL连涨"
        print(f"[{tag}ep{ep}] loss={sum(losses)/len(losses):.4f} val: agree={m['agree']:.1%} "
              f"KL={m['kl']:.4f} brier={m['brier']:.4f}{marker}", flush=True)
        if patience >= 2 and not overfit:
            print(f"[{tag}] 早停（val KL patience 2）", flush=True)
            break
    if best["state"]:
        head.load_state_dict(best["state"])
    final = evaluate(va_ex, head)
    tr_final = evaluate(tr_ex, head)
    return {"head": head, "val": final, "train": tr_final, "best_epoch": best["epoch"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=None, help="只训某批（过拟合门用 --batch 1）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--overfit", action="store_true", help="过拟合门：全量训练集上拟合")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-extract", action="store_true", help="跳过特征抽取（全部缓存命中时）")
    args = ap.parse_args()

    recs = load_dataset(args.data, batch_filter=args.batch, limit=args.limit)
    print(f"记录 {len(recs)} 条")
    # --no-extract：cache_only（不加载主干）；注意须先跑过 a8 全量抽取，未命中会明确报错
    extractor = None if args.no_extract else FeatureExtractor()
    t0 = time.time()
    if extractor is None:
        extractor = FeatureExtractor(cache_only=True)
    for i, r in enumerate(recs):
        r["_feats"] = extractor.extract_record(r)
        if (i + 1) % 100 == 0:
            print(f"  特征加载 {i+1}/{len(recs)}（{time.time()-t0:.0f}s）", flush=True)
    res = run(recs, epochs=args.epochs, seed=args.seed, overfit=args.overfit,
              extractor=None, tag="A")
    print(f"\n== 训练集 == agree={res['train']['agree']:.1%} KL={res['train']['kl']:.4f}")
    print(f"== 验证（独立 dev）== agree={res['val']['agree']:.1%} KL={res['val']['kl']:.4f} "
          f"brier={res['val']['brier']:.4f} ece={res['val']['ece']:.4f}")
    Path("data/train_v3").mkdir(parents=True, exist_ok=True)
    torch.save(res["head"].state_dict(), f"data/train_v3/head_{'overfit' if args.overfit else 'smoke'}.pt")
    print("头权重已存 data/train_v3/")


if __name__ == "__main__":
    main()
