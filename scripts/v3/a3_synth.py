"""V3 数据重做·阶段3：检索 → 答案变体 → teacher 打标（train+dev 池；test 盲测集不进本流程）。

对旧 v2 管线的修正（审计🔴-3 / M4 / M7）：
  1. oracle gold 唯一来源＝出题时的 gold_answer，永不为空；gold 档直接用、trunc 档
     程序截断（零 API、可复现），不再事后调 API 补答案（v2 的 480 条无 gold 悲剧根源）；
  2. eval 态（八题）state 必拼非空 gold，production 态（七题）state 不拼 gold——
     硬断言，违反即中止；
  3. teacher prompt 不做概率形状强制（v2 强制最高≤0.90/次高≥0.03——人工概率模板），
     只保留"概率之和为1"；
  4. 每条记录的 teacher 原始输出内嵌（E08 可回放），中间产物按 id 落盘，可重入。

产物：data/train_v3/_synth_raw/{id}.json（每题一件，含检索/变体/标签全量上下文）
     data/train_v3/contexts.json（检索结果，可重放）
运行：set -a && source .env && set +a && .venv/bin/python -u scripts/v3/a3_synth.py [--limit N]
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "m4"))

CFG = yaml.safe_load(open(ROOT / "rag/config.yaml"))
CACHE = ROOT / "data/eval_cache/llm"
TRAIN_V3 = ROOT / "data/train_v3"
RAW_DIR = TRAIN_V3 / "_synth_raw"
for line in open(ROOT / ".env"):
    if line.startswith("DEEPSEEK_API_KEY"):
        os.environ.setdefault("DEEPSEEK_API_KEY", line.split("=", 1)[1].strip())

FLASH, PRO = "deepseek-v4-flash", "deepseek-v4-pro"
SEED = 42
FLAWS = [
    ("编造一个法条中不存在的论断（看似合理但法条原文没说）", "编造论断·难"),
    ("编造一个明显不存在的条文引用（法名或条号完全虚构）", "编造论断·易"),
    ("引用错误的条文号（引同法相邻条，条号改错一两位）", "引错条号·难"),
    ("引用完全无关的法条的条文号", "引错条号·易"),
    ("答案部分正确但核心结论偏离问题（答了一半跑题了）", "答非所问·难"),
    ("答案完全离题（讨论与问题无关的法律问题）", "答非所问·易"),
]


def call(model, messages, tag, max_tokens=1200):
    payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
               "messages": messages, "thinking": {"type": "disabled"}, "tag": tag}
    h = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{h}.json"
    if f.exists():
        return json.loads(f.read_text())["resp"], h
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("缺 DEEPSEEK_API_KEY")
    client = OpenAI(api_key=key, base_url=CFG["generator"]["base_url"], timeout=90, max_retries=1)
    body = {k: v for k, v in payload.items() if k not in ("thinking", "tag")}
    for attempt in range(3):
        try:
            r = client.chat.completions.create(**body, extra_body={"thinking": payload["thinking"]})
            c = r.choices[0].message.content
            f.write_text(json.dumps({"req": payload, "resp": c}, ensure_ascii=False))
            return c, h
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 + attempt * 5)


def retrieve(question, top_k=6):
    """真实 M1 检索链路（embed 20 → rerank → top6）。"""
    from rag.embedder import get_embedder, embed_query
    from rag.reranker import get_reranker, rerank
    from rag.vector_store import load_store, search
    e = get_embedder(os.path.expanduser(CFG["paths"]["embedding_gguf"]))
    r = get_reranker(os.path.expanduser(CFG["paths"]["reranker_gguf"]))
    chunks, mat = load_store()
    idx = search(embed_query(e, question), mat, 20)
    cand = [chunks[i] for i in idx]
    scores = rerank(r, question, [c["text"] for c in cand])
    order = sorted(range(len(cand)), key=lambda i: -scores[i])
    return [cand[i] for i in order[:top_k]]


def gen_flawed(question, flaw_desc):
    prompt = (f"问题：{question}\n"
              f"请写一段约 80 字的【有缺陷】回答，缺陷要求：{flaw_desc}。\n只输出回答正文。")
    text, h = call(FLASH, [{"role": "user", "content": prompt}], "v3-a3-flawed", 300)
    return text.strip(), h


def teacher_label(state, questions_def, emphasize=False):
    """PRO 打标：只约束和为 1，不强制概率形状（审计 M4 正确做法）。
    emphasize=True 为退化输出（硬标签而非分布）后的重试形态——不同 prompt 才能绕开缓存。"""
    p = (state + "\n\n你是评测裁判。对下列各题给出选项概率分布与一句依据，输出 JSON。"
         "硬性要求：每题概率之和为 1。")
    if emphasize:
        p += "\n注意：每题必须给出【每个选项】的概率（0~1 小数、总和为1），禁止只输出最终选项名或档位数字。"
    for qid, q in questions_def.items():
        crit = list(q["candidates"].keys())
        p += f"\n- {qid}：{q['instructions']}，选项：{'/'.join(crit)}"
    p += '\n输出格式：{"qid": {"probs": {...}, "reason": "一句话"}, ...}'
    tag = "v3-a3-label-pro-retry" if emphasize else "v3-a3-label-pro"
    return call(PRO, [{"role": "user", "content": p}], tag, 1400)


def labels_valid(labels, qs_used, bq) -> bool:
    """每题都是数值分布、键集=candidates——硬标签/字符串答案视为退化输出。"""
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


def parse_labels(raw):
    """宽松解析 PRO 的 JSON（含 qid 包裹/直接 probs 两种形态）。"""
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return {}
    out = {}
    items = parsed
    if isinstance(parsed.get("qid"), dict) and "probs" in parsed["qid"]:
        items = parsed["qid"]
        if "probs" in items:
            return {k: {"probs": v} for k, v in items["probs"].items()}
    for k, v in items.items():
        if isinstance(v, dict) and isinstance(v.get("probs"), dict):
            out[k] = v["probs"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每池上限（冒烟）")
    args = ap.parse_args()

    from rag.judge.judge import build_state
    from train.qwen_jey.schema import build_questions
    q8 = build_questions()
    seven = {k: v for k, v in q8.items() if k != "answer_correctness"}

    qs_all = []
    for pool in ("train", "dev"):
        f = TRAIN_V3 / f"questions_{pool}.jsonl"
        rows = [json.loads(l) for l in open(f)]
        if args.limit:
            rows = rows[:args.limit]
        qs_all += [(pool, r) for r in rows]
    print(f"待合成：{len(qs_all)} 题（train+dev）")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── 检索（缓存于 contexts.json，可重放） ──
    ctx_file = TRAIN_V3 / "contexts.json"
    contexts = json.loads(ctx_file.read_text()) if ctx_file.exists() else {}
    t0 = time.time()
    miss = [r for _, r in qs_all if r["question"] not in contexts]
    for i, r in enumerate(miss):
        ctx = retrieve(r["question"])
        contexts[r["question"]] = [{"law": c["law"], "article_no": c["article_no"],
                                    "article_num": c.get("article_num"),
                                    "text": c["text"]} for c in ctx]
        if (i + 1) % 10 == 0 or i + 1 == len(miss):
            ctx_file.write_text(json.dumps(contexts, ensure_ascii=False))
            print(f"  检索 [{i+1}/{len(miss)}] {time.time()-t0:.0f}s", flush=True)
    ctx_file.write_text(json.dumps(contexts, ensure_ascii=False))

    # ── 逐题：变体 + state + teacher（4 线程并行；检索已缓存） ──
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from train.qwen_jey.schema import build_questions as _bq
    bq8 = _bq()

    def synth_one(i, pool, q):
        if (RAW_DIR / f"{q['id']}.json").exists():
            return "skip"
        ctx = contexts.get(q["question"], [])
        if not ctx:
            return "noctx"
        # 变体分配（确定性）：gold 50% / trunc 25% / flawed 25%；production 20%
        kind = ["gold", "gold", "trunc", "flawed"][i % 4]
        mode = "production" if i % 5 == 4 else "eval"
        oracle = q["gold_answer"].strip()
        assert oracle, f"{q['id']} oracle gold 为空——出题数据损坏，中止"

        if kind == "gold":
            answer = oracle
        elif kind == "trunc":
            answer = "。".join(oracle.split("。")[:1]) + "。"
        else:
            flaw = FLAWS[i % len(FLAWS)]
            answer, _ = gen_flawed(q["question"], flaw[0])

        # state：eval 拼【非空】gold（八题）；production 不拼（七题）——审计口径硬断言
        if mode == "eval":
            state = build_state(q["question"], ctx, answer, oracle)
            assert "【参考答案】" in state
            qs_used = q8
        else:
            state = build_state(q["question"], ctx, answer, None)
            assert "【参考答案】" not in state
            qs_used = seven

        raw, req_hash = teacher_label(state, qs_used)
        labels = parse_labels(raw)
        if not labels_valid(labels, qs_used, bq8):
            # 退化输出（硬标签/字符串）→ 强调重试一次；仍退化则照存，由 a4 拒绝留痕
            raw2, req_hash2 = teacher_label(state, qs_used, emphasize=True)
            labels2 = parse_labels(raw2)
            if labels_valid(labels2, qs_used, bq8):
                raw, req_hash, labels = raw2, req_hash2, labels2

        rec = {
            "id": q["id"], "split": pool, "law": q["law"], "chapter": q["chapter"],
            "qtype": q["qtype"], "difficulty": q["difficulty"],
            "ans_kind": kind, "mode": mode,
            "flaw": FLAWS[i % len(FLAWS)][1] if kind == "flawed" else None,
            "question": q["question"], "gold_answer": oracle,
            "gold_articles": q["gold_articles"],
            "context": ctx, "answer": answer,
            "labels": labels, "teacher": {"model": PRO, "request_hash": f"sha256:{req_hash}", "raw": raw},
        }
        (RAW_DIR / f"{q['id']}.json").write_text(json.dumps(rec, ensure_ascii=False))
        return "ok"

    todo = [(i, pool, q) for i, (pool, q) in enumerate(qs_all)]
    todo = [t for t in todo if not (RAW_DIR / f"{t[2]['id']}.json").exists()]
    n_done, n_skip, n_noctx = len(qs_all) - len(todo), 0, 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(synth_one, i, pool, q): (i, q["id"]) for i, pool, q in todo}
        for fut in as_completed(futs):
            r = fut.result()
            n_done += 1
            if r == "noctx":
                n_noctx += 1
                _, qid = futs[fut]
                with open(TRAIN_V3 / "reject_log.jsonl", "a") as f:
                    f.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "stage": "retrieve",
                                        "reason": "检索空", "sid": qid}, ensure_ascii=False) + "\n")
            if n_done % 10 == 0 or n_done == len(qs_all):
                print(f"  合成 [{n_done}/{len(qs_all)}] {time.time()-t0:.0f}s", flush=True)

    print(f"\n合成完成：{len(qs_all)} 条（新做 {len(todo)}，跳过 {len(qs_all)-len(todo)}，检索空 {n_noctx}）→ {RAW_DIR}")


if __name__ == "__main__":
    main()
