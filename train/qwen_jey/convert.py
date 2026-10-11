"""T7.1 转换器：M2 测试集记录（＋chunk 上下文＋答案）→ qwen-jey 记录。

字段映射＝T7.1 方案 §四（逐字段）：context/answer 属 T7.2 域（真实检索＋teacher 生成），
本转换器只承接外部供数——开发样例允许手工指定（方案 §四"开发样例"条款）。
M4 管线产物接入（T7.2）走同一入口：state/questions/target_probs 字段名公因数对齐。
"""

from __future__ import annotations

import json
import re

from .schema import make_record


def testset_to_record(ts: dict, context: list[dict], answer: str,
                      *, gold_in_state: bool = True, **kwargs) -> dict:
    """M2 testset 一条 → 记录（前体：无 target_probs/teacher）。

    ts 字段消费：id→source.testset_id、question→state_parts.question、
    gold_answer→state_parts.gold_answer（gold_in_state=False 走生产式样例）、
    gold_articles/law/qtype/difficulty→source 元数据（对账用）。
    """
    return make_record(
        rec_id=kwargs.pop("rec_id", f"qj-{ts['id']}"),
        question=ts["question"],
        context=context,
        answer=answer,
        gold_answer=ts.get("gold_answer") if gold_in_state else None,
        origin="M2-testset",
        testset_id=ts["id"],
        law_refs=[f"《{a['law']}》{a['article_no']}" for a in ts.get("gold_articles", [])],
        **kwargs,
    )


def load_chunks(article_keys: list[tuple[str, int]]) -> list[dict]:
    """从 chunk 库按 (law, article_num) 取法条原文；顺序保持入参序（调用方控制 K 与排序）。"""
    want = set(article_keys)
    out = {}
    with open("data/rag_store/chunks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            k = (r["law"], r["article_num"])
            if k in want and k not in out:
                out[k] = r
    missing = want - set(out)
    if missing:
        raise KeyError(f"chunk 库缺条: {sorted(missing)}")
    return [out[k] for k in article_keys]


# 声明项 C（T7.2 方案 §三）：8K 档服务 M3 扩展，qwen-jey 训练窗 4096，不进本集
LEN_TIER_SKIP = {"8k", "8K"}


def _parse_context(ctx_text: str) -> list[dict]:
    """parts.context 是拼好的法条段文本（行首《法》第N条：原文）→ chunk 列表。
    只在行首《边界切——条文内部含换行（分款），按 \\n 暴力切会截断条文。"""
    chunks = []
    for blk in re.split(r"\n(?=《)", ctx_text.strip()):
        m = re.match(r"《(.+?)》(第.+?条)：(.*)", blk, re.S)
        if m:
            chunks.append({"law": m.group(1), "article_no": m.group(2), "text": m.group(3)})
    if not chunks:
        raise ValueError("parts.context 解析失败：无法拆出法条行")
    return chunks


def _remap_probs(probs: dict, q_type: str, score_order: list[str]) -> dict:
    """teacher_raw_full 键 → schema 键：noul true/false→否/是；score 描述→零基；choice 原样。
    和≠1±0.05 内轻归一（teacher 两位小数常见 0.99/1.01；原样留 teacher.raw 供审计）。"""
    if q_type == "noul":
        probs = {"true": probs.get("true", 0), "false": probs.get("false", 0)}
        mapping = {"true": "是", "false": "否"}
    elif q_type == "score":
        desc2idx = {d: str(i) for i, d in enumerate(score_order)}
        mapping = desc2idx
    else:
        mapping = {}
    out = {mapping.get(k, k): float(v) for k, v in probs.items() if mapping.get(k, k) is not None}
    s = sum(out.values())
    if not out or abs(s - 1) > 0.05 or min(out.values()) < 0:
        raise ValueError(f"teacher 分布畸形: sum={s}")
    return {k: v / s for k, v in out.items()} if abs(s - 1) > 0.001 else out


def draft_to_record(draft: dict, *, production: bool | None = None) -> dict | None:
    """M4 定版 draft（真实格式）→ qwen-jey schema 记录。

    真实字段：k/route/tier/ans_kind/mode/state/intent/labels/teacher_raw(_full)/parts/qc/final。
    蒸馏目标＝teacher_raw_full（真实 teacher 分布；labels 经 jitter 不作目标——干跑 D1）。
    题面＝canonical rubric（记录不携带变体文本，变体在 teacher prompt 侧）。
    """
    from .schema import build_questions, rubric

    if draft.get("tier") in LEN_TIER_SKIP:
        return None
    is_prod = (draft.get("mode") == "production") if production is None else production

    parts = draft["parts"]
    sp = {"question": parts["question"], "context": _parse_context(parts["context"]),
          "answer": parts["answer"]}
    if not is_prod and parts.get("gold_answer"):
        sp["gold_answer"] = parts["gold_answer"]

    raw = json.loads(draft["teacher_raw_full"])
    q8, score_orders = build_questions(), {k: list(v["criteria"]) for k, v in rubric().items()
                                           if v["type"] == "score"}
    tp = {}
    for qk, q in q8.items():
        if is_prod and qk == "answer_correctness":
            continue
        tp[qk] = _remap_probs(raw.get(qk, {}).get("probs", {}), q["type"], score_orders.get(qk, []))

    rec = make_record(
        rec_id=f"qj-b{draft.get('k', '?')}", question=sp["question"], context=sp["context"],
        answer=sp["answer"], gold_answer=sp.get("gold_answer"),
        origin=draft.get("route", "M4-pipeline"), law_refs=[],
        target_probs=tp,
        teacher={"model": "deepseek-v4-flash", "method": "temp0-1call", "temperature": 0.0,
                 "raw": raw},  # E08 内嵌审计：原始 teacher 输出随身携带
    )
    rec["questions"] = json.loads(json.dumps(q8, ensure_ascii=False))
    if is_prod:
        rec["questions"].pop("answer_correctness")
    rec["source"].update({"len_tier": draft.get("tier"), "ans_kind": draft.get("ans_kind"),
                          "draft_index": draft.get("k"), "intent": draft.get("intent")})
    if not is_prod and draft.get("state"):
        # draft 的 state 无【状态】前缀（M4 存四段本体）；比对我们重建的去前缀内层
        inner = rec["state_text"][len("【状态】\n"):]
        rec["_draft_state_match"] = draft["state"].rstrip("\n") == inner.rstrip("\n")
    return rec
