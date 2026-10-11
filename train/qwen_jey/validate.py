"""T7.1 校验器 CLI：jsonl 全量校验 → 错误清单（退出码＝错误数）＋质控报告。

用法（T7.1 方案 §八）：
    .venv/bin/python -m train.qwen_jey.validate data/train/qwen_jey_dev_samples.jsonl --mode draft
产物：data/runs/{时间戳}_t71_validate.md（长度分布/题型计数/错误清单）。

════════════════════ 实测输出（2026-10-07，dev_samples＝A长1074/B短424/C完整1879） ════════════════════
# --mode draft（退出码=1：仅 B 的预期拒绝——424 开发样例）
# 记录 3 条｜模式 draft｜错误 1
# [行2] QJ-E05: qj-t001 state_tokens=424（期望 1000~3800；3800 上限给题行留余量保证整行 ≤4096）
# state_tokens: min=424 中位=1074 max=1879
# 长度直方(500/桶): {0: 1, 1000: 1, 1500: 1}
# 题型计数: {'choice': 12, 'noul': 6, 'score': 6}
# ↑ 解读：A(1074)/C(1879) 过长度带；B 是方案 §三的真实负例，正确被拒
# --mode final（退出码=5：前体 A 缺软标签被 E01+E08 拦、B 三项、C 完整记录零错误全过）
# ↑ 解读：draft/final 两档行为符合方案 §一——前体进 final 必被拦，定版记录全绿
═════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from .schema import validate_record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--mode", choices=("draft", "final"), default="draft")
    args = parser.parse_args(argv)

    records, errors, seen = [], [], set()
    for lineno, line in enumerate(open(args.jsonl), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        records.append(rec)
        rec_errors = validate_record(rec, mode=args.mode, seen=seen)
        errors += [f"[行{lineno}] {e}" for e in rec_errors]

    toks = sorted(r.get("state_tokens", 0) for r in records)
    qtypes = Counter(q["type"] for r in records for q in r.get("questions", {}).values())
    hist = Counter((t // 500) * 500 for t in toks)

    print(f"记录 {len(records)} 条｜模式 {args.mode}｜错误 {len(errors)}")
    for e in errors:
        print(e)
    print(f"state_tokens: min={toks[0] if toks else '-'} 中位={toks[len(toks)//2] if toks else '-'} "
          f"max={toks[-1] if toks else '-'}")
    print(f"长度直方(500/桶): {dict(sorted(hist.items()))}")
    print(f"题型计数: {dict(qtypes)}")

    run = Path("data/runs") / f"{time.strftime('%Y%m%d_%H%M%S')}_t71_validate.md"
    run.parent.mkdir(exist_ok=True)
    with open(run, "w") as f:
        f.write(f"# T7.1 校验报告：{args.jsonl.name}（mode={args.mode}）\n\n"
                f"- 记录 {len(records)} 条｜错误 {len(errors)}\n"
                f"- state_tokens: min={toks[0] if toks else '-'} / 中位={toks[len(toks)//2] if toks else '-'} "
                f"/ max={toks[-1] if toks else '-'}\n"
                f"- 长度直方(500/桶): {dict(sorted(hist.items()))}\n- 题型计数: {dict(qtypes)}\n\n## 错误清单\n"
                + ("\n".join(f"- {e}" for e in errors) if errors else "- 无"))
    print(f"报告: {run}")
    return len(errors)  # 退出码＝错误数：0 才过 schema 门


if __name__ == "__main__":
    sys.exit(main())
