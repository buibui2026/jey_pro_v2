#!/usr/bin/env python3
"""T1.2 向量库：numpy 暴力余弦（~1137×1024≈4.5MB，不引 faiss——方案 §四）。

CLI：
  build   全量嵌入 chunks.jsonl → embeds.npy（首次构建约 1~2 分钟）
  search  交互调试单问 top-N
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

CFG = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
STORE = Path(CFG["paths"]["rag_store"])


def load_store():
    chunks = [json.loads(l) for l in open(STORE / "chunks.jsonl")]
    mat = np.load(STORE / "embeds.npy").astype(np.float32)
    assert len(chunks) == mat.shape[0], "chunks 与向量数不一致——重跑 build"
    return chunks, mat


def search(qvec: list[float], mat: np.ndarray, top_n: int) -> list[int]:
    sims = mat @ np.array(qvec, dtype=np.float32)  # 双方已 L2 归一化，点积即余弦
    return np.argsort(-sims)[:top_n].tolist()


def build() -> None:
    from rag.embedder import embed_docs, get_embedder
    ll = get_embedder(CFG["paths"]["embedding_gguf"].replace("~", str(Path.home())))
    chunks = [json.loads(l) for l in open(STORE / "chunks.jsonl")]
    texts = [f"{c['law']} {c['article_no']}：{c['text']}" for c in chunks]
    mat = np.array(embed_docs(ll, texts), dtype=np.float32)
    np.save(STORE / "embeds.npy", mat)
    print(f"embedded {mat.shape} -> {STORE / 'embeds.npy'}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "search":
        from rag.embedder import embed_query, get_embedder
        ll = get_embedder(CFG["paths"]["embedding_gguf"].replace("~", str(Path.home())))
        chunks, mat = load_store()
        idx = search(embed_query(ll, sys.argv[2]), mat, CFG["retrieval"]["top_n"])
        for i in idx:
            print(f"{chunks[i]['law']} {chunks[i]['article_no']}: {chunks[i]['text'][:60]}")
