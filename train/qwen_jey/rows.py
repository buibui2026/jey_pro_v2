"""T8.1 行编码器：schema 记录 → 每题一行的 token 序列＋读取位（构造法，训推共用）。

模板＝T6.3 v2 定版（T7.1 方案 §二）：
    {state}\\n【问题】（{type}）{instructions}\\n【候选】\\n{key}：{desc}×n\\n【判断】
读取位由逐段 tokenize 拼接直接给出（不做事后文本切分）：
    key_pos[i]＝第 i 条候选行（含行尾换行）的末 token 位；query_pos＝【判断】段末位（行末）。

gold 可见性隔离（2026-10-10 终审问答三-3，与 judge P0-2 双 state 训推一致）：
每行只拼它被允许看的 state——七生产题用 base state（question/context/answer，无 gold）；
仅 answer_correctness 用含【参考答案】的 state。production 记录无 gold，全部行即 base。
记录级 state_text（E05/E06/E07/E08 锚）保持含 gold 的四段式不变——两层语义分开。
tokenizer 路径从 rag/config.yaml 读（2026-10-10 消除硬编码，审计 M8 🔴）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
TOKENIZER_PATH = yaml.safe_load(open(ROOT / "rag/config.yaml"))["paths"]["qwen_tokenizer"]


@lru_cache(maxsize=1)
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def row_tokens(state: str, question: dict) -> dict:
    """返回 {ids, query_pos, key_pos}；ids 为全行 token（add_special_tokens=False）。"""
    tok = _tok()

    def seg(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    ids = seg(state)
    ids += seg(f"\n【问题】（{question['type']}）{question['instructions']}\n【候选】\n")
    key_pos = []
    for key, desc in question["candidates"].items():
        ids += seg(f"{key}：{desc}\n")
        key_pos.append(len(ids) - 1)          # 候选行（含换行）末 token
    ids += seg("【判断】")
    query_pos = len(ids) - 1                   # 行末＝判断位
    return {"ids": ids, "query_pos": query_pos, "key_pos": key_pos}


def row_text(state: str, question: dict) -> str:
    """行的明文（特征缓存键用——比 token 列表更稳定的身份）。"""
    cands = "".join(f"{k}：{d}\n" for k, d in question["candidates"].items())
    return f"{state}\n【问题】（{question['type']}）{question['instructions']}\n【候选】\n{cands}【判断】"


def record_states(record: dict) -> tuple[str, str]:
    """（base_state, gold_state）：base 无 gold；gold_state 含【参考答案】（无 gold 时两者相等）。"""
    from .state_builder import build_state_text
    parts = record["state_parts"]
    base = build_state_text({"question": parts["question"], "context": parts["context"],
                             "answer": parts["answer"]})
    gold = build_state_text(parts)
    return base, gold


def record_rows(record: dict) -> dict[str, dict]:
    """一条记录 → {question_key: row}；每行携带自己的 state 变体与行明文。

    gold 可见性＝judge 推理口径（P0-2）：七题 base、answer_correctness gold。
    production 态天然少 answer_correctness 一行。
    """
    base, gold = record_states(record)
    rows = {}
    for qk, q in record["questions"].items():
        state = gold if (qk == "answer_correctness" and gold != base) else base
        # row_text＝完整行明文（state＋题行）——特征缓存键必须含题行：
        # 只用 state 会让同 state 的多题共享缓存、读回彼此的特征（2026-10-10 实测踩坑）
        rows[qk] = {"state": state, "row_text": row_text(state, q), **row_tokens(state, q)}
    return rows
