"""V2·P0-4 检索劣化补批：~60 条噪声 context 样本（修复 ctx_relevance=无关 仅 1.5% 的信号稀薄）。

形态设计（教 judge 判检索质量）：
  irrelevant  全噪声：6 条 context 全部换成本题金条文以外章节的随机条文
              → teacher 应判 ctx_relevance=无关 / ctx_sufficiency=否（answer 用 oracle gold，
                "答案对但检索无关"——检索维度的专属信号，不混入答案质量信号）
  partial     部分相关：保留 2 条金条文 + 4 条噪声
              → ctx_relevance=部分相关 中间档信号

题源：复用已终审的 train 题（分层抽样，同题不同 context 不撞 E07 指纹——指纹含法条号集）。
answer：oracle gold（本批目标是检索维度信号，答案质量信号保持恒定）。
teacher：与管线同 prompt。留痕 raw + a11_log。
运行：.venv/bin/python -u scripts/v3/a11_noisy_batch.py
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
from t3_common import call_pro  # noqa: E402
from a10_variant_redo import parse_labels  # noqa: E402

RAW = ROOT / "data/train_v3/_synth_raw"
SEED = 42


def main():
    from rag.judge.judge import build_state
    from train.qwen_jey.schema import build_questions
    q8 = build_questions()
    seven = {k: v for k, v in q8.items() if k != "answer_correctness"}

    chunks = [json.loads(l) for l in open(ROOT / "data/rag_store/chunks.jsonl")]
    sg = json.loads((ROOT / "data/splits/source_groups.v3.json").read_text())
    banned = {(b["law"], b["article_num"]) for b in sg["banned_articles"]}
    # train 池法条（噪声来源：同法任意非金条文、非 banned）
    pool_by_law = defaultdict(list)
    for c in chunks:
        if c["article_sub"]:
            continue
        for p, gs in sg["groups"].items():
            hit = next((g for g in gs if g["law"] == c["law"]
                        and c["article_num"] in g["articles"]), None)
            if hit:
                # 只从 train 池章取（不碰 dev/test 章）
                if p == "train" and (c["law"], c["article_num"]) not in banned:
                    pool_by_law[c["law"]].append(c)
                break

    qs = [json.loads(l) for l in open(ROOT / "data/train_v3/questions_train.jsonl")]
    rng = random.Random(SEED)
    by_law = defaultdict(list)
    for q in qs:
        by_law[q["law"]].append(q)
    picked = []
    laws = sorted(by_law)
    while len(picked) < 60:
        for law in laws:
            if by_law[law]:
                picked.append(by_law[law].pop(rng.randrange(len(by_law[law]))))
                if len(picked) >= 60:
                    break

    log = open(ROOT / "data/train_v3/a11_log.jsonl", "a")
    n = {"irrelevant": 0, "partial": 0}
    t0 = time.time()
    for i, q in enumerate(picked):
        kind = "irrelevant" if i % 2 == 0 else "partial"
        gold_keys = {(a["law"], a["article_no"]) for a in q["gold_articles"]}
        noise_pool = [c for c in pool_by_law[q["law"]] if (c["law"], c["article_no"]) not in gold_keys]
        rng.shuffle(noise_pool)
        if kind == "irrelevant":
            ctx = noise_pool[:6]
            # 保证噪声确实不含金条文
            ctx = [c for c in ctx if (c["law"], c["article_no"]) not in gold_keys][:6]
            if len(ctx) < 4:
                continue
        else:
            gold_ctx = [c for c in chunks
                        if (c["law"], c["article_no"]) in gold_keys and not c["article_sub"]][:2]
            ctx = gold_ctx + [c for c in noise_pool if (c["law"], c["article_no"]) not in gold_keys][:4]
        mode = "production" if i % 5 == 4 else "eval"
        oracle = q["gold_answer"].strip()
        ctx_rec = [{"law": c["law"], "article_no": c["article_no"],
                    "article_num": c.get("article_num"), "text": c["text"]} for c in ctx]

        if mode == "eval":
            state = build_state(q["question"], ctx_rec, oracle, oracle)
            qs_used = q8
        else:
            state = build_state(q["question"], ctx_rec, oracle, None)
            qs_used = seven
        p = (state + "\n\n你是评测裁判。对下列各题给出选项概率分布与一句依据，输出 JSON。"
             "硬性要求：每题概率之和为 1。")
        for qid, qq in qs_used.items():
            crit = list(qq["candidates"].keys())
            p += f"\n- {qid}：{qq['instructions']}，选项：{'/'.join(crit)}"
        p += '\n输出格式：{"qid": {"probs": {...}, "reason": "一句话"}, ...}'
        raw_t = call_pro(p, 1400)
        labels = parse_labels(raw_t)

        rid = f"v3n-{q['law']}-{i:04d}"
        rec = {
            "id": rid, "split": "train", "law": q["law"], "chapter": q["chapter"],
            "qtype": q["qtype"], "difficulty": q["difficulty"],
            "ans_kind": f"gold_{kind}_ctx", "mode": mode, "flaw": None,
            "question": q["question"], "gold_answer": oracle,
            "gold_articles": q["gold_articles"], "context": ctx_rec, "answer": oracle,
            "labels": labels, "teacher": {"model": "deepseek-v4-pro", "raw": raw_t},
        }
        (RAW / f"{rid}.json").write_text(json.dumps(rec, ensure_ascii=False))
        log.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "id": rid, "kind": kind,
                              "mode": mode, "n_ctx": len(ctx_rec),
                              "gold_in_ctx": kind == "partial"},
                             ensure_ascii=False) + "\n")
        n[kind] += 1
        if (i + 1) % 10 == 0 or i + 1 == len(picked):
            print(f"  [{i+1}/{len(picked)}] {dict(n)} {time.time()-t0:.0f}s", flush=True)
    log.close()
    print(f"\n检索劣化补批完成：{dict(n)} → a4 重定版")


if __name__ == "__main__":
    main()
