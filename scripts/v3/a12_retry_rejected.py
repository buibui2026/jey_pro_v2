"""V2·P0-5a 被拒记录重试：对 a4 finalize 拒绝的 raw（teacher 硬标签退化）带 emphasize 重打标。

a10/a11 单次打标无 emphasize 重试（v1 a3 有）——退化输出（如 completeness: 2）被 a4 拒。
本脚本：解析最近一轮 finalize 拒绝 → 重建 state → emphasize 重打标 → 通过则覆写 raw → 重跑 a4。
运行：.venv/bin/python -u scripts/v3/a12_retry_rejected.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))
from t3_common import call_pro  # noqa: E402
from a10_variant_redo import parse_labels  # noqa: E402

RAW = ROOT / "data/train_v3/_synth_raw"


def labels_valid(labels, qs_used, bq):
    for qid in qs_used:
        d = labels.get(qid)
        if isinstance(d, dict) and set(d) == {"probs"} and isinstance(d["probs"], dict):
            d = d["probs"]
        if not isinstance(d, dict):
            return False
        if any(not isinstance(v, (int, float)) for v in d.values()):
            return False
        if set(d) != set(bq[qid]["candidates"]) or sum(d.values()) <= 0:
            return False
    return True


def label(state, qs_used, emphasize):
    p = (state + "\n\n你是评测裁判。对下列各题给出选项概率分布与一句依据，输出 JSON。"
         "硬性要求：每题概率之和为 1。")
    if emphasize:
        p += "\n注意：每题必须给出【每个选项】的概率（0~1 小数、总和为1），禁止只输出最终选项名或档位数字。"
    for qid, q in qs_used.items():
        crit = list(q["candidates"].keys())
        p += f"\n- {qid}：{q['instructions']}，选项：{'/'.join(crit)}"
    p += '\n输出格式：{"qid": {"probs": {...}, "reason": "一句话"}, ...}'
    return call_pro(p, 1400)


def main():
    from rag.judge.judge import build_state
    from train.qwen_jey.schema import build_questions
    q8 = build_questions()
    seven = {k: v for k, v in q8.items() if k != "answer_correctness"}

    # 最近一轮 a4 的 finalize 拒绝（按时间取尾段）
    rejected = []
    for l in open(ROOT / "data/train_v3/reject_log.jsonl"):
        r = json.loads(l)
        if r.get("stage") == "finalize":
            rejected.append(r["sid"])
    rejected = list(dict.fromkeys(rejected))[-40:]   # 最近一批
    print(f"待重试：{len(rejected)} 条")

    ok = fail = 0
    for sid in rejected:
        f = RAW / f"{sid}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        qs_used = q8 if r["mode"] == "eval" else seven
        if r["mode"] == "eval":
            state = build_state(r["question"], r["context"], r["answer"], r["gold_answer"])
        else:
            state = build_state(r["question"], r["context"], r["answer"], None)
        raw_t = label(state, qs_used, emphasize=True)
        labels = parse_labels(raw_t)
        if labels_valid(labels, qs_used, q8):
            r["labels"] = labels
            r["teacher"] = {"model": "deepseek-v4-pro", "raw": raw_t}
            f.write_text(json.dumps(r, ensure_ascii=False))
            ok += 1
        else:
            fail += 1
        print(f"  {sid}: {'✅' if labels_valid(labels, qs_used, q8) else '❌ 仍退化'}", flush=True)
    print(f"\n重试完成：救回 {ok}，仍失败 {fail}（失败的保持被拒状态，不进定版）")


if __name__ == "__main__":
    main()
