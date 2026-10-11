#!/usr/bin/env python3
"""T1.2 embedding 封装：Qwen3-Embedding-0.6B GGUF，llama-cpp-python 直载。

非对称编码（官方用法）：query 侧带 instruction 前缀，法条侧不带。
维度 1024，L2 归一化。决策依据见 T1.2 方案 §一（弃 LM Studio server）。
"""

import os
from functools import lru_cache

from llama_cpp import Llama

QUERY_PREFIX = "Instruct: 给定一个法律咨询问题，检索相关的法律条文\nQuery: "


@lru_cache(maxsize=1)
def get_embedder(model_path: str) -> Llama:
    # RAG_LLAMA_CPU=1 → 关 GPU 直跑 CPU：多进程同占 Metal 时 GPU 路径会瘫痪性卡顿
    # （实测 20 字 query 嵌入 17s~777s）；短文本 CPU 嵌入毫秒级，反而更快
    if os.environ.get("RAG_LLAMA_CPU"):
        return Llama(model_path=model_path, embedding=True, n_ctx=2048,
                     n_gpu_layers=0, n_threads=4, verbose=False)
    return Llama(model_path=model_path, embedding=True, n_ctx=2048, verbose=False)


def embed_query(ll: Llama, question: str) -> list[float]:
    return _norm(ll.create_embedding(input=[QUERY_PREFIX + question])["data"][0]["embedding"])


def embed_docs(ll: Llama, texts: list[str]) -> list[list[float]]:
    # 逐条嵌入：批量接口在长文本上偶发截断，逐条更稳（教训候选：批量与逐条结果需抽对账）
    return [_norm(ll.create_embedding(input=[t])["data"][0]["embedding"]) for t in texts]


def _norm(v: list[float]) -> list[float]:
    import math
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]
