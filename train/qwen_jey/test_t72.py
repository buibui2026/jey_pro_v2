"""T7.2 单元测试：真实格式 draft → record 转换（键映射/轻归一/mode 分支/8K 过滤/交叉核对）。

运行：.venv/bin/python -m train.qwen_jey.test_t72（项目根目录）
fixture 按 M4 定版真实格式手造（k/route/tier/ans_kind/mode/state/intent/labels/
teacher_raw(_full)/parts/qc/final），t001 真数据；键形用真实键（noul true/false、score 全描述）。
"""

from __future__ import annotations

import json

from rag.judge.judge import build_state as laya_build_state
from .convert import draft_to_record
from .schema import build_questions, rubric, validate_record
from .test_t71 import ANSWER, LONG_CTX, SHORT_CTX, T001

Q8 = build_questions()
_RUB = rubric()
DESC_C = list(_RUB["completeness"]["criteria"])   # completeness 档位序
DESC_O = list(_RUB["overall"]["criteria"])        # overall 档位序（teacher 键形）


def raw_probs(spike_mode="normal"):
    """teacher_raw_full 的 probs（真实键形：choice=rubric 键、noul=true/false、score=全描述）。"""
    r = {"ctx_relevance": {"相关": 0.95, "部分相关": 0.04, "无关": 0.01},
         "ctx_sufficiency": {"true": 0.95, "false": 0.05},
         "groundedness": {"全部有据": 0.95, "部分有据": 0.04, "无据": 0.01},
         "citation_correct": {"正确": 0.95, "引用但内容不符": 0.04, "未引用": 0.01},
         "answers_question": {"true": 0.96, "false": 0.04},
         "completeness": {DESC_C[0]: 0.05, DESC_C[1]: 0.20, DESC_C[2]: 0.75},
         "answer_correctness": {"一致": 0.94, "部分一致": 0.05, "不一致或相反": 0.01},
         "overall": {DESC_O[0]: 0.02, DESC_O[1]: 0.23, DESC_O[2]: 0.75}}
    if spike_mode == "renorm":  # 和=0.99：teacher 两位小数的常见漂移
        r["groundedness"] = {"全部有据": 0.95, "部分有据": 0.04, "无据": 0.00}
    return r


def make_draft(production=False, tier="1k", ctx=None, probs=None):
    chunks = ctx or SHORT_CTX
    ctx_text = "\n".join(f"《{c['law']}》{c['article_no']}：{c['text']}" for c in chunks)
    parts = {"question": T001["question"], "context": ctx_text, "answer": ANSWER,
             "gold_answer": T001["gold_answer"]}
    state = laya_build_state(parts["question"], chunks, parts["answer"],
                             None if production else parts["gold_answer"])
    probs = probs if probs is not None else raw_probs()
    return {"k": 0, "route": "m2_fallout", "tier": tier, "ans_kind": "gold",
            "mode": "production" if production else "eval", "state": state,
            "intent": {"groundedness": "全部有据"},
            "labels": {"groundedness": {"probs": {"全部有据": 0.9184, "部分有据": 0.051, "无据": 0.0306}}},
            "teacher_raw": "截断占位", "teacher_raw_full": json.dumps(
                {k: {"probs": v, "reason": "…"} for k, v in probs.items()}, ensure_ascii=False),
            "parts": parts, "qc": "ok", "final": "R2"}


def test_key_mapping():
    rec = draft_to_record(make_draft())
    assert rec["target_probs"]["ctx_sufficiency"] == {"否": 0.05, "是": 0.95}
    assert list(rec["target_probs"]["overall"]) == ["0", "1", "2"]
    assert rec["target_probs"]["groundedness"]["全部有据"] == 0.95
    assert set(rec["target_probs"]["completeness"]) == {"0", "1", "2"}


def test_target_is_teacher_raw_not_labels():
    """蒸馏目标＝teacher_raw_full（labels 的 jitter 值 0.9184 不得混入）。"""
    rec = draft_to_record(make_draft())
    assert rec["target_probs"]["groundedness"]["全部有据"] == 0.95  # raw 值，非 labels 的 0.9184


def test_renorm():
    rec = draft_to_record(make_draft(probs=raw_probs("renorm")))
    assert abs(sum(rec["target_probs"]["groundedness"].values()) - 1) <= 1e-9


def test_eval_record_passes_final():
    rec = draft_to_record(make_draft(ctx=LONG_CTX))  # 长上下文过 E05
    assert rec["_draft_state_match"] is True
    assert validate_record(rec, mode="final") == []


def test_production_branch():
    rec = draft_to_record(make_draft(production=True, ctx=LONG_CTX))
    assert "answer_correctness" not in rec["questions"]
    assert "gold_answer" not in rec["state_parts"] and "【参考答案】" not in rec["state_text"]
    assert validate_record(rec, mode="final") == []


def test_8k_filter():
    assert draft_to_record(make_draft(tier="8k")) is None


def test_e01_mode_mismatch_caught():
    bad_prod = draft_to_record(make_draft(production=True, ctx=LONG_CTX))
    bad_prod["questions"]["answer_correctness"] = bad_prod["questions"]["groundedness"]
    assert any("E01" in e for e in validate_record(bad_prod, mode="final"))
    bad_eval = draft_to_record(make_draft(ctx=LONG_CTX))
    bad_eval["questions"].pop("answer_correctness")
    assert any("E01" in e for e in validate_record(bad_eval, mode="final"))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
    raise SystemExit(fails)
