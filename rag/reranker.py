#!/usr/bin/env python3
"""T1.2 reranker 封装：Qwen3-Reranker-0.6B（yes/no 判别式，自实现打分）。

分数 = P(yes)/(P(yes)+P(no))——读首 token 的 top-logprobs 实现，不经 server 端点
（llama.cpp 生态对 Qwen3-Reranker 有已知打分 bug，打分必须自控——T1.2 方案 §三）。
决策门在对照实验（rag/bge_ref.py + T1.2 对照报告）：Spearman < 0.5 或分数退化即切 bge。
"""

from functools import lru_cache

from llama_cpp import Llama

PROMPT = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: 根据法律咨询问题，判断该法条作为参考依据的相关性\n"
    "<Query>: {query}\n"
    "<Document>: {doc}<|im_end|>\n"
    "<|im_start|>assistant\n"
    # 本 GGUF 转换版默认思考模式：不闭合 <think> 时首 token 是 <think>，yes/no 概率取不到、
    # 分数全 0（实测教训）。闭合思考段后首 token 即 yes/no，分数正常（相关0.997/无关0.000）
    "<think>\n\n</think>\n\n"
)


@lru_cache(maxsize=1)
def get_reranker(model_path: str) -> Llama:
    # logits_all=True 是 create_completion 返回 logprobs 的前提（llama-cpp-python 约束）
    return Llama(model_path=model_path, n_ctx=2048, logits_all=True, verbose=False)


def rerank(ll: Llama, query: str, docs: list[str]) -> list[float]:
    """逐对打分，返回与 docs 等长的分数列表（越高越相关）。"""
    return [score_once(ll, query, d) for d in docs]


def score_once(ll: Llama, query: str, doc: str) -> float:
    import math
    # 关键：prompt 必须先按 special=True 分词——直接传字符串时 <|im_start|> 等特殊
    # token 会被拆成普通字符，模型看到乱码，分数全 0（实测教训，决策门的假阳性来源）
    tokens = ll.tokenize(PROMPT.format(query=query, doc=doc).encode("utf-8"), add_bos=True, special=True)
    out = ll.create_completion(prompt=tokens, max_tokens=1, temperature=0.0, logprobs=20)
    top = out["choices"][0]["logprobs"]["top_logprobs"][0]  # {token: logprob}
    # yes/no 的常见形态统一归并（带前导空格与否）
    p_yes = sum(math.exp(lp) for t, lp in top.items() if t.strip().lower() == "yes")
    p_no = sum(math.exp(lp) for t, lp in top.items() if t.strip().lower() == "no")
    if p_yes + p_no == 0:
        return 0.0  # 两 token 均不在 top20 —— 退化信号，决策门的输入之一
    return p_yes / (p_yes + p_no)
