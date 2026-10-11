"""T8.1 特征抽取：逐行前向 → 读取位 hidden 缓存（预计算特征路线，T8.1 方案 §四）。

缓存防漂移四重键（manifest，审计 M8 🔴 修复 2026-10-10）：
  主干权重 sha16 ＋ tokenizer 路径 sha8 ＋ dtype 标记 ＋ 行明文 sha256——任一变即失效重算。
路径从 rag/config.yaml 读（原硬编码 /Users/.../model/backbone 一处不留）。
行预算 4096 硬门（超限报错不截断，V2-3）。
V3 起缓存目录＝data/train_v3/feat_cache/（与旧 v2 缓存物理隔离，旧键名结构不同亦不冲突）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml

from .rows import record_rows

ROOT = Path(__file__).resolve().parent.parent.parent
_CFG = yaml.safe_load(open(ROOT / "rag/config.yaml"))
BACKBONE = _CFG["paths"]["qwen_backbone"]
CACHE_DIR = ROOT / "data/train_v3/feat_cache"
MAX_ROW_TOKENS = 4096
DTYPE_TAG = "bf16"


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def backbone_fingerprint() -> str:
    return _sha_file(f"{BACKBONE}/model.safetensors")


def _tokenizer_tag() -> str:
    return hashlib.sha256(_CFG["paths"]["qwen_tokenizer"].encode()).hexdigest()[:8]


class FeatureExtractor:
    def __init__(self, device: str = "mps", cache_only: bool = False):
        """cache_only=True：不加载主干（跳过几 GB 显存），只读缓存；未命中即报错。"""
        self.device = device
        self.model = None
        if not cache_only:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(BACKBONE, dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").to(device).eval()
        self.fp = backbone_fingerprint()
        self.tok_tag = _tokenizer_tag()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text_sha: str) -> Path:
        return CACHE_DIR / f"{self.fp}_{self.tok_tag}_{DTYPE_TAG}_{text_sha}.pt"

    def extract_row(self, row: dict, text: str | None = None) -> dict:
        """单行 [1,L] 前向 → {q: [H], k: [N,H]}（bf16→float32 存档，训练数值稳）。

        text 缺省用 row['row_text']（完整行明文=state＋题行——键必须含题行，
        只用 state 段会让同 state 的多题共享缓存，2026-10-10 实测踩坑）。
        """
        if text is None:
            text = row["row_text"]
        if len(row["ids"]) > MAX_ROW_TOKENS:
            raise ValueError(f"行超预算：{len(row['ids'])} > {MAX_ROW_TOKENS}（报错不截断）")
        text_sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        f = self._cache_path(text_sha)
        if f.exists():
            return torch.load(f, weights_only=True)
        if self.model is None:
            raise RuntimeError(f"cache_only 模式缓存未命中：{f.name}——去掉 --no-extract 重新抽取")
        ids = torch.tensor([row["ids"]], device=self.device)
        with torch.inference_mode():
            h = self.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        out = {"q": h[row["query_pos"]].float().cpu(),
               "k": h[row["key_pos"]].float().cpu()}
        torch.save(out, f)
        return out

    def extract_record(self, record: dict) -> dict[str, dict]:
        rows = record_rows(record)
        feats = {}
        for qk, row in rows.items():
            feats[qk] = self.extract_row(row, row["row_text"])
        return feats
