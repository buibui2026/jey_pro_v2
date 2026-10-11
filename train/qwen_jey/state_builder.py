"""T7.1 state 构建器：state_parts → state_text（v2 模板，训推共用）。

模板原文＝T7.1 方案 §二（逐字符）：
    【状态】\\n【用户问题】{q}\\n【检索法条】\\n{《法》第N条：原文 ×K}\\n【生成的答案】{a}\\n【参考答案】{gold}
四段内层复用 rag.judge.judge.build_state（02 方案 §三的唯一实现）——三 judge 的 state 公因数，
不许在本包重写模板；test_t71.py 用逐字符等价断言钉死这一点。
"""

from __future__ import annotations

STATE_PREFIX = "【状态】\n"


def build_state_text(state_parts: dict) -> str:
    """state_parts{question,context[chunk...],answer,gold_answer?} → state_text。

    生产模式（无 gold_answer）自动省略【参考答案】段——02 方案 §三既定行为。
    """
    inner = _four_segment(
        state_parts["question"], state_parts["context"],
        state_parts["answer"], state_parts.get("gold_answer"),
    )
    return STATE_PREFIX + inner


def _four_segment(question: str, chunks: list[dict], answer: str, gold: str | None) -> str:
    # 与 rag.judge.build_state 保持逐字符同构；独立成函数是为让等价断言有明确靶子
    from rag.judge.judge import build_state
    return build_state(question, chunks, answer, gold)
