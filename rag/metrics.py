#!/usr/bin/env python3
"""T1.5 指标汇总：02 方案 §四 的实现。

口径：score 题用期望分（Σ level×p，不用 argmax——保留不确定性）；
judge-gold 一致率仅在记录带金标时输出（M2 之前缺省，不硬造）；
归因联读：answer_correctness 低 × ctx_relevance 低 → 检索；低 × 高 → 生成。
"""

import json
from pathlib import Path

CHOICE_BAD = {"无据", "引用但内容不符", "不一致或相反", "无关"}   # 各 choice 题的坏侧候选
CTX_BAD = {"无关", "部分相关"}


def _expected(probs: dict[str, float]) -> float:
    return sum(int(k) * v for k, v in probs.items())


def summarize(run_dir: Path) -> dict:
    records = [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.judge.json"))]
    if not records:
        return {}
    agg: dict[str, dict] = {}
    attribution = {"检索": 0, "生成": 0, "无判据": 0}
    for r in records:
        for key, a in r["answers"].items():
            # laya 的 noul 型答案无 probabilities 键（只有标量 noul）——展开为 true/false 二值
            probs = a.get("probabilities") or {"true": a["noul"], "false": 1 - a["noul"]}
            slot = agg.setdefault(key, {"n": 0, "expected": 0.0, "prob_sum": {k: 0.0 for k in probs}})
            slot["n"] += 1
            slot["expected"] += _expected(probs) if a["type"] == "score" else 0.0
            for k, v in probs.items():
                slot["prob_sum"][k] += v
        # 归因联读（02 §二 规则）——需要两题同时存在
        ac, cr = r["answers"].get("answer_correctness"), r["answers"].get("ctx_relevance")
        if ac and cr:
            ac_bad = max(ac["probabilities"], key=ac["probabilities"].get) in CHOICE_BAD
            cr_bad = max(cr["probabilities"], key=cr["probabilities"].get) in CTX_BAD
            attribution["检索" if (ac_bad and cr_bad) else "生成" if ac_bad else "无判据"] += 1

    out = {"n_records": len(records), "attribution": attribution, "questions": {}}
    for key, s in agg.items():
        q = {"n": s["n"], "mean_prob": {k: round(v / s["n"], 4) for k, v in s["prob_sum"].items()}}
        if s["expected"]:
            q["mean_expected_score"] = round(s["expected"] / s["n"], 3)
        out["questions"][key] = q
    (run_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out
