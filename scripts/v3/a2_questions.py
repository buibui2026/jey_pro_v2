"""V3 数据重做·阶段2：三池出题（出题即带 oracle gold——审计 P0-3/M7 正确做法）。

与旧管线的四点差异（每一处都对应审计🔴）：
  1. 只从本池法条出题（split 先行，v2 是先出题后随机切）；
  2. gold_answer 出题时同步生成、永不为空；gold_articles 程序注入（模型不自报条号）；
  3. 防撞双检：与 M2 234 题 ngram<0.4 且向量余弦<0.90（v2 只做内部去重）；
  4. C 型题给 2~3 条同池法条（v2 给单条，与"跨条文比较"定义冲突）。

产物：questions_{train,dev}.jsonl（data/train_v3/）、questions_test.jsonl（data/blind_test_v3/）
     reject_log.jsonl（append-only，全部拒绝可追因）
配额：按各池各法【可用条文】占比分配 + 下限；题型 A/B/C=40/40/20，难度 1/3。
可重入：按槽位 id 跳过已完成槽位。
运行：set -a && source .env && set +a && .venv/bin/python -u scripts/v3/a2_questions.py \
        [--limit N]（三池各上限 N 题，冒烟用）
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "m4"))
from t42_synth import ngram_overlap  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "rag/config.yaml"))
CACHE = ROOT / "data/eval_cache/llm"
TRAIN_V3 = ROOT / "data/train_v3"
BLIND_V3 = ROOT / "data/blind_test_v3"
for line in open(ROOT / ".env"):
    if line.startswith("DEEPSEEK_API_KEY"):
        os.environ.setdefault("DEEPSEEK_API_KEY", line.split("=", 1)[1].strip())

GEN_MODEL = CFG["generator"]["model"]
SEED = 42
QTYPES = ["A", "A", "B", "B", "C"]          # 40/40/20
DIFFS = ["easy", "mid", "hard"]
QUOTA_TOTAL = {"train": 600, "dev": 60, "test": 120}
QUOTA_MIN_PER_LAW = {"train": 20, "dev": 3, "test": 6}
PREFIX = {"train": "v3t", "dev": "v3d", "test": "v3b"}

PROMPT = {
    ("A", "easy"): "只依据给定条文，出一道【直查直答】的中文法律问题：答案就是条文里某一个明确要点。",
    ("A", "mid"): "只依据给定条文，出一道【需要理解适用条件】的中文法律问题：答案涉及条文内多个要点或条件判断。",
    ("A", "hard"): "只依据给定条文，出一道【易混淆】的中文法律问题：问题中涉及条文内相近概念（如不同情节对应的处罚档），需仔细区分。",
    ("B", "easy"): "只依据给定条文，构造一个简单生活情境（人物+行为），问该情境下的法律处理。情境要素单一。",
    ("B", "mid"): "只依据给定条文，构造一个含两到三个情节要素的情境（人物+行为+情节加重或减轻），问法律处理。",
    ("B", "hard"): "只依据给定条文，构造一个含干扰要素的情境（看似适用实则不适用、或情节跨档），问法律处理。",
    ("C", "easy"): "只依据给定的两三条条文，出一道【直接比较】题：比较两个条文对同一类行为的规定差异。",
    ("C", "mid"): "只依据给定的两三条条文，出一道【衔接适用】题：问某行为在这几个条文之间如何衔接或选择适用。",
    ("C", "hard"): "只依据给定的两三条条文，出一道【辨析】题：给出易混淆的一对情形，问各适用哪个条文、为什么。",
}


def call_llm(messages, temperature=0.7, max_tokens=700):
    payload = {"model": GEN_MODEL, "temperature": temperature, "max_tokens": max_tokens,
               "messages": messages, "thinking": {"type": "disabled"}, "tag": "v3-a2-question"}
    h = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{h}.json"
    if f.exists():
        return json.loads(f.read_text())["resp"]
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("缺 DEEPSEEK_API_KEY")
    client = OpenAI(api_key=key, base_url=CFG["generator"]["base_url"], timeout=60, max_retries=1)
    body = {k: v for k, v in payload.items() if k not in ("thinking", "tag")}
    resp = client.chat.completions.create(**body, extra_body={"thinking": payload["thinking"]})
    content = resp.choices[0].message.content
    f.write_text(json.dumps({"req": payload, "resp": content}, ensure_ascii=False))
    return content


class RejectLog:
    def __init__(self):
        self.f = open(TRAIN_V3 / "reject_log.jsonl", "a")

    def add(self, **kw):
        kw["ts"] = time.strftime("%H:%M:%S")
        self.f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.f.flush()


def build_quota(usable_by_law, pool):
    """按可用条文占比分配 + 每法下限，对齐总量。"""
    total_usable = sum(usable_by_law.values())
    quota = {law: max(min_q, round(QUOTA_TOTAL[pool] * n / total_usable))
             for law, n in usable_by_law.items()
             for min_q in [QUOTA_MIN_PER_LAW[pool]]}
    quota = dict(sorted(quota.items()))
    # 按差额调整最大法，对齐总量
    delta = QUOTA_TOTAL[pool] - sum(quota.values())
    biggest = max(quota, key=lambda l: quota[l])
    quota[biggest] += delta
    return quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每池出题上限（冒烟）")
    args = ap.parse_args()
    rng = random.Random(SEED)
    TRAIN_V3.mkdir(parents=True, exist_ok=True)
    BLIND_V3.mkdir(parents=True, exist_ok=True)

    chunks_all = [json.loads(l) for l in open(ROOT / "data/rag_store/chunks.jsonl")]
    chunks = [c for c in chunks_all if not c["article_sub"]]
    sg = json.loads((ROOT / "data/splits/source_groups.v3.json").read_text())
    banned = {(b["law"], b["article_num"]) for b in sg["banned_articles"]}
    # 池 → 法 → 可用条文 chunk 列表（池互斥由切分断言保证；banned 条文剔除）
    chapter_of = {}
    pool_arts = {p: defaultdict(set) for p in ("train", "dev", "test")}
    for p, gs in sg["groups"].items():
        for g in gs:
            for a in g["articles"]:
                chapter_of[(g["law"], a)] = g["chapter"]
                if (g["law"], a) not in banned:
                    pool_arts[p][g["law"]].add(a)
    pool_law_chunks = {p: defaultdict(list) for p in ("train", "dev", "test")}
    for c in chunks:
        for p in ("train", "dev", "test"):
            if c["article_num"] in pool_arts[p][c["law"]]:
                pool_law_chunks[p][c["law"]].append(c)
                break

    # M2 防撞基准：题向量 + 题文本
    from rag.embedder import get_embedder, embed_query
    emb = get_embedder(os.path.expanduser(CFG["paths"]["embedding_gguf"]))
    m2 = [json.loads(l) for l in open(ROOT / "data/testset/testset_final.jsonl")]
    m2_vec_file = TRAIN_V3 / "_m2_qembeds.npy"
    if m2_vec_file.exists():
        m2_vecs = np.load(m2_vec_file).astype(np.float32)
    else:
        m2_vecs = np.array([embed_query(emb, r["question"]) for r in m2], dtype=np.float32)
        np.save(m2_vec_file, m2_vecs)
    m2_qs = [r["question"] for r in m2]

    embeds = np.load(ROOT / "data/rag_store/embeds.npy").astype(np.float32)
    idx_of = {f"{c['law']}|{c['article_num']}": i for i, c in enumerate(chunks_all)}

    reject = RejectLog()
    TRAIN_V3.mkdir(parents=True, exist_ok=True)
    BLIND_V3.mkdir(parents=True, exist_ok=True)
    out_files = {"train": TRAIN_V3 / "questions_train.jsonl",
                 "dev": TRAIN_V3 / "questions_dev.jsonl",
                 "test": BLIND_V3 / "questions_test.jsonl"}

    summary = {}
    for pool in ("train", "dev", "test"):
        usable = {l: len(v) for l, v in pool_law_chunks[pool].items()}
        quota = build_quota(usable, pool)
        if args.limit:
            scale = args.limit / QUOTA_TOTAL[pool]
            quota = {l: max(1, round(n * scale)) for l, n in quota.items()}
        # 槽位展开：法内 (qtype, difficulty) 轮换
        slots = []
        for law, n in sorted(quota.items()):
            pool_chunks = pool_law_chunks[pool][law]
            for i in range(n):
                slots.append({"law": law, "qtype": QTYPES[i % len(QTYPES)],
                              "difficulty": DIFFS[i % len(DIFFS)],
                              "sid": f"{PREFIX[pool]}-{law}-{i:04d}",
                              "pool_chunks": pool_chunks})
        # 已完成槽位（可重入）
        done = set()
        if out_files[pool].exists():
            for line in open(out_files[pool]):
                done.add(json.loads(line)["sid"])
        todo = [s for s in slots if s["sid"] not in done]
        existing = []
        if out_files[pool].exists():
            existing = [json.loads(l) for l in open(out_files[pool])]

        accepted, acc_vecs, t0 = [], [], time.time()
        fail = Counter()
        for n, slot in enumerate(todo, 1):
            law, qt, diff = slot["law"], slot["qtype"], slot["difficulty"]
            pool_chunks = slot["pool_chunks"]
            def pick_gold():
                # 选条：A/B 单条轮换；C 锚条+池内相似 top2（同法同池、非 banned）
                if qt in "AB":
                    return [pool_chunks[rng.randrange(len(pool_chunks))]]
                anchor = pool_chunks[rng.randrange(len(pool_chunks))]
                ok_arts = pool_arts[pool][law] - {anchor["article_num"]}
                sims = embeds @ embeds[idx_of[f"{anchor['law']}|{anchor['article_num']}"]]
                cand = [chunks_all[i] for i in np.argsort(-sims)
                        if chunks_all[i]["law"] == law
                        and not chunks_all[i].get("article_sub")
                        and chunks_all[i]["article_num"] in ok_arts][:2]
                return [anchor] + cand

            def build_prompt(gold_articles):
                ctx = "\n\n".join(f"《{c['law']}》{c['article_no']}：{c['text']}" for c in gold_articles)
                return (f"{PROMPT[(qt, diff)]}\n\n【条文】\n{ctx}\n\n"
                        "【输出要求】只输出一个 JSON 对象（不要 markdown 代码块），字段：\n"
                        '{"question": "问题，10~60字，问句中不得照抄条文原句",\n'
                        ' "gold_answer": "标准答案：结论＋《法名》第X条依据＋一句适用说明，120字以内",\n'
                        ' "self_difficulty": "本题实际难度自评：easy/mid/hard"}\n'
                        f"目标难度：{diff}。问题必须能用且仅用给定条文回答。")

            gold = pick_gold()
            prompt = build_prompt(gold)

            rec = None
            for attempt in range(3):
                if attempt == 1:   # 首轮失败多为"该条文难出题"——换一条条文再试
                    gold = pick_gold()
                    prompt = build_prompt(gold)
                raw = call_llm([{"role": "user", "content": prompt}])
                try:
                    cand = json.loads(re.search(r"\{.*\}", raw, re.S).group())
                except Exception:
                    fail["JSON解析"] += 1
                    reject.add(pool=pool, stage="parse", reason="JSON解析失败",
                               law=law, qtype=qt, difficulty=diff, sid=slot["sid"])
                    continue
                q = cand.get("question", "")
                ga = cand.get("gold_answer", "")
                # QC1 schema
                if not (8 <= len(q) <= 80):
                    fail["schema:问题长度"] += 1
                    reject.add(pool=pool, stage="qc", reason=f"问题长度{len(q)}", law=law,
                               qtype=qt, difficulty=diff, sid=slot["sid"])
                    continue
                if not ga or len(ga) > 230:
                    fail["schema:答案长度"] += 1
                    reject.add(pool=pool, stage="qc", reason=f"gold长度{len(ga)}", law=law,
                               qtype=qt, difficulty=diff, sid=slot["sid"])
                    continue
                if cand.get("self_difficulty") not in DIFFS:
                    fail["schema:难度自评"] += 1
                    reject.add(pool=pool, stage="qc", reason="难度自评非法", law=law,
                               qtype=qt, difficulty=diff, sid=slot["sid"])
                    continue
                # QC2 防泄漏：问题 vs 金条文 15 字连续片段
                leak = any(t[i:i + 15] in q for c in gold for t in [c["text"]]
                           for i in range(max(1, len(t) - 15)))
                if leak:
                    fail["泄漏"] += 1
                    reject.add(pool=pool, stage="qc", reason="含金条文≥15字片段", law=law,
                               qtype=qt, difficulty=diff, sid=slot["sid"])
                    continue
                # QC3 防撞 M2：test/dev 硬拒（盲测语义隔离）；train 仅记录
                # （M2 用条全部在 train 池，同条文出题向量天然相近；M2 已退役，
                #   train 撞 M2 不构成污染——污染方向是"M2 进评测"，不是反向）
                if any(ngram_overlap(q, m) >= 0.4 for m in m2_qs):
                    if pool in ("dev", "test"):
                        fail["撞M2:ngram"] += 1
                        reject.add(pool=pool, stage="qc", reason="与M2题ngram≥0.4", law=law,
                                   qtype=qt, difficulty=diff, sid=slot["sid"])
                        continue
                    reject.add(pool=pool, stage="qc-warn", reason="train题与M2题ngram≥0.4（放行）",
                               law=law, qtype=qt, difficulty=diff, sid=slot["sid"])
                qvec = np.array(embed_query(emb, q), dtype=np.float32)
                if m2_vecs.shape[0] and float(np.max(m2_vecs @ qvec)) > 0.90:
                    if pool in ("dev", "test"):
                        fail["撞M2:向量"] += 1
                        reject.add(pool=pool, stage="qc", reason="与M2题余弦>0.90", law=law,
                                   qtype=qt, difficulty=diff, sid=slot["sid"])
                        continue
                    reject.add(pool=pool, stage="qc-warn", reason="train题与M2题余弦>0.90（放行）",
                               law=law, qtype=qt, difficulty=diff, sid=slot["sid"])
                # QC4 组内去重
                if any(ngram_overlap(q, p["question"]) >= 0.4 for p in accepted + existing):
                    fail["组内重复:ngram"] += 1
                    reject.add(pool=pool, stage="qc", reason="组内ngram≥0.4", law=law,
                               qtype=qt, difficulty=diff, sid=slot["sid"])
                    continue
                if any(float(v @ qvec) > 0.90 for v in acc_vecs):
                    fail["组内重复:向量"] += 1
                    reject.add(pool=pool, stage="qc", reason="组内余弦>0.90", law=law,
                               qtype=qt, difficulty=diff, sid=slot["sid"])
                    continue
                # QC5 难度对齐（最后一次接受自评）
                if cand["self_difficulty"] != diff and attempt < 2:
                    fail["难度重生成"] += 1
                    reject.add(pool=pool, stage="qc", reason=f"自评{cand['self_difficulty']}≠{diff}",
                               law=law, qtype=qt, difficulty=diff, sid=slot["sid"])
                    continue
                rec = {"sid": slot["sid"], "id": slot["sid"], "split": pool,
                       "law": law, "chapter": chapter_of[(law, gold[0]["article_num"])],
                       "qtype": qt, "difficulty": diff, "question": q,
                       "gold_answer": ga,
                       "gold_articles": [{"law": c["law"], "article_no": c["article_no"],
                                          "article_num": c["article_num"]} for c in gold],
                       "self_difficulty": cand["self_difficulty"]}
                accepted.append(rec)
                acc_vecs.append(qvec)
                break
            if not rec:
                fail["槽位放弃"] += 1
                reject.add(pool=pool, stage="slot", reason="3次尝试全失败，槽位放弃", law=law,
                           qtype=qt, difficulty=diff, sid=slot["sid"])
            if n % 25 == 0 or n == len(todo):
                print(f"[{pool}] {n}/{len(todo)} 通过{len(accepted)} 失败{dict(fail)} "
                      f"{time.time()-t0:.0f}s", flush=True)

        # 写盘（追加，保序：先旧后新）
        with open(out_files[pool], "a") as f:
            for r in accepted:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[pool] = {"quota": quota, "todo": len(todo), "accepted": len(accepted),
                         "fail": dict(fail)}

    print("\n=== 出题汇总 ===")
    for p, s in summary.items():
        print(f"{p:5s} 本次出题 {s['accepted']}/{s['todo']}  失败 {s['fail']}")
        print(f"      配额 {s['quota']}")


if __name__ == "__main__":
    main()
