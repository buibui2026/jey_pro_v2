"""T7.1 schema：记录结构＋8 条硬校验（E01-E08）＋题目注入＋token 计数。

校验口径＝M6 T6.4 §1.2 与 T7.1 方案 §五；tokenizer 契约＝T7.1 方案 §六
（Qwen3.5、add_special_tokens=False、state_tokens 只计 state 段）。
draft/final 两档：final 要求 target_probs/teacher 必备（E02/E03/E08 全跑）；
draft（前体）没有软标签——E02/E03/E08 只在字段存在时执行，缺字段不报错。
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
RUBRIC_PATH = ROOT / "rag" / "judge" / "rubric.yaml"
TOKENIZER_PATH = yaml.safe_load(open(ROOT / "rag/config.yaml"))["paths"]["qwen_tokenizer"]  # 2026-10-10 消除硬编码（审计 M8🔴）；Base/工件同源不变
EVAL_CACHE = ROOT / "data" / "eval_cache" / "llm"

# noul 两题的候选描述：rubric.yaml 对 noul 不出 criteria（02 方案设计），
# 行为化文案来自 T7.1 方案 §三/§四定稿文本——集中一处，不许散落
NOUL_CANDIDATES = {
    "ctx_sufficiency": {"否": "检索法条缺少回答所需的关键依据", "是": "检索法条已含回答所需依据"},
    "answers_question": {"否": "答案未正面回应所问", "是": "答案正面回应了所问"},
}


@lru_cache(maxsize=1)
def rubric() -> dict:
    return yaml.safe_load(open(RUBRIC_PATH))


@lru_cache(maxsize=1)
def build_questions() -> dict:
    """rubric.yaml 八题 → schema questions 格式（零改写注入）。

    choice：candidates＝rubric criteria 原文；noul：固定"否／是"对（上方文案）；
    score：键＝零基档位号 "0".."n-1"（期望分 Σ i·p_i 与键号直接对应，E04 口径）。
    """
    questions = {}
    for key, q in rubric().items():
        if q["type"] == "choice":
            candidates = dict(q["criteria"])
        elif q["type"] == "noul":
            candidates = NOUL_CANDIDATES[key]
        else:
            candidates = {str(i): text for i, text in enumerate(q["criteria"])}
        questions[key] = {"type": q["type"], "instructions": q["instructions"], "candidates": candidates}
    return questions


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def count_tokens(text: str) -> int:
    """§六契约：add_special_tokens=False。"""
    return len(_tokenizer().encode(text, add_special_tokens=False))


def state_tokens(state_text: str) -> int:
    """state_tokens 只计 state 段（＝整个 state_text；题行在 row 层，不在此计）。"""
    return count_tokens(state_text)


def validate_record(rec: dict, *, mode: str, seen: set | None = None) -> list[str]:
    """返回错误清单（空＝过）。错误行格式：QJ-E0x: {id}.{题key} {事实}（{期望}）。"""
    errors: list[str] = []
    rid = rec.get("id", "?")
    canonical = build_questions()
    qs = rec.get("questions", {})

    # E01 题集合法：键 ⊆ rubric 八题；final（训练定版）按模式定必备题数
    # （T7.2 方案 §四 mode 分支：评测态含金标→八题；生产态无金标→七题，第 8 题不得在场）
    bad_keys = set(qs) - set(canonical)
    if bad_keys:
        errors.append(f"QJ-E01: {rid} 非法题键 {sorted(bad_keys)}（期望 ⊆ rubric 八题）")
    if mode == "final":
        has_gold = "gold_answer" in rec.get("state_parts", {})
        required = set(canonical) if has_gold else set(canonical) - {"answer_correctness"}
        missing = required - set(qs)
        if missing:
            errors.append(f"QJ-E01: {rid} {'评测态' if has_gold else '生产态'}缺题 {sorted(missing)}"
                          f"（期望{'八题全带' if has_gold else '生产态七题'}）")
        if not has_gold and "answer_correctness" in qs:
            errors.append(f"QJ-E01: {rid} 生产态含 answer_correctness（期望七题，第 8 题仅评测态）")

    # E02 键集相等（target_probs 存在时执行；final 要求必须存在）
    tp = rec.get("target_probs")
    if tp is None and mode == "final":
        errors.append(f"QJ-E01: {rid} final 模式缺 target_probs（期望 teacher 软标签已填）")
    if tp is not None:
        for key, q in qs.items():
            if key in tp and set(tp[key]) != set(q["candidates"]):
                errors.append(
                    f"QJ-E02: {rid}.{key} 键集不匹配 {sorted(set(tp[key]) ^ set(q['candidates']))}"
                    f"（期望 target_probs 与 candidates 键集严格相等）")

    # E03 概率合法
    if tp is not None:
        for key, dist in (tp or {}).items():
            total, minimum = sum(dist.values()), min(dist.values(), default=0)
            if abs(total - 1) > 0.001 or minimum < 0:
                errors.append(f"QJ-E03: {rid}.{key} 概率和={total:.4f} 最小值={minimum}"
                              f"（期望 和=1±0.001 且全部 ≥0）")

    # E04 题型键形
    for key, q in qs.items():
        cands = q["candidates"]
        if q["type"] == "noul" and set(cands) != {"否", "是"}:
            errors.append(f"QJ-E04: {rid}.{key} noul 键 {sorted(cands)}（期望 否／是）")
        if q["type"] == "score" and list(cands) != [str(i) for i in range(len(cands))]:
            errors.append(f"QJ-E04: {rid}.{key} score 键 {list(cands)}（期望 零基连续 0..n-1）")

    # E05 长度带（2026-10-07 依裁决A校准：下限 1000→200——实测 M4 交付中位 ~1K、min 240，
    # 原下限基于"3~4K 对齐"假设；混合分布方案下短档合法，上限 3800（行预算 4096 减题行）不变）
    st = rec.get("state_text")
    n_tok = state_tokens(st) if st else -1
    if not (200 <= n_tok <= 3800):
        errors.append(f"QJ-E05: {rid} state_tokens={n_tok}（期望 200~3800；"
                      f"3800 上限给题行留余量保证整行 ≤4096）")

    # E06 快照一致：state_parts 重建必须逐字符相等（防模板漂移）
    if st is not None:
        from .state_builder import build_state_text
        if build_state_text(rec["state_parts"]) != st:
            errors.append(f"QJ-E06: {rid} state_text 与 state_parts 重建不一致"
                          f"（期望 同一构建器逐字符相等；模板改动须升 builder_version）")

    # E07 判重：question＋context 法条号集 哈希全局唯一
    if seen is not None:
        parts = rec.get("state_parts", {})
        refs = sorted(f"《{c['law']}》{c['article_no']}" for c in parts.get("context", []))
        fp = hashlib.sha1((parts.get("question", "") + "|" + ",".join(refs)).encode()).hexdigest()[:16]
        if fp in seen:
            errors.append(f"QJ-E07: {rid} 与前条记录重复（question＋法条号集指纹 {fp} 相同）")
        seen.add(fp)

    # E08 可回放：teacher.raw 内嵌（原始 teacher 输出随身携带＝自足审计，2026-10-07 口径），
    # 或 sha256 链接 eval_cache 命中
    teacher = rec.get("teacher")
    if teacher is not None:
        raw = teacher.get("raw")
        h = teacher.get("request_hash", "")
        if raw is not None:
            if not isinstance(raw, dict) or not raw:
                errors.append(f"QJ-E08: {rid} teacher.raw 非法（期望可解析的原始分布）")
        elif h.startswith("sha256:") and (EVAL_CACHE / f"{h[7:]}.json").exists():
            pass
        else:
            errors.append(f"QJ-E08: {rid} 无内嵌 raw 且缓存未命中 {h[:24] if h else '（空）'}"
                          f"（期望 teacher_raw_full 内嵌或 eval_cache 可回放）")
    if teacher is None and mode == "final":
        errors.append(f"QJ-E08: {rid} final 模式缺 teacher（期望原始 teacher 输出随软标签同存）")

    return errors


def make_record(*, rec_id: str, question: str, context: list[dict], answer: str,
                gold_answer: str | None, origin: str = "M2-testset",
                testset_id: str | None = None, law_refs: list[str] | None = None,
                target_probs: dict | None = None, teacher: dict | None = None) -> dict:
    """转换器的组装件：字段映射＝T7.1 方案 §四。"""
    from .state_builder import build_state_text
    state_parts = {"question": question, "context": context, "answer": answer}
    if gold_answer is not None:
        state_parts["gold_answer"] = gold_answer
    rec = {
        "id": rec_id,
        "builder_version": "v2",
        "source": {"origin": origin, "testset_id": testset_id, "law_refs": law_refs or []},
        "state_parts": state_parts,
        "state_text": build_state_text(state_parts),
        "questions": json.loads(json.dumps(build_questions(), ensure_ascii=False)),  # 深拷贝防共享可变
    }
    rec["state_tokens"] = state_tokens(rec["state_text"])
    if target_probs is not None:
        rec["target_probs"] = target_probs
    if teacher is not None:
        rec["teacher"] = teacher
    return rec
