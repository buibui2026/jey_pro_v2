#!/usr/bin/env python3
"""T1.3 生成模块：DeepSeek（OpenAI 兼容）＋请求/响应全量缓存。

可缓存性靠规范化：temperature=0、固定 system、无时间戳等随机源（T1.3 方案 §二）。
LLM_CACHE_ONLY=1 时只读缓存，未命中直接报错——不许静默穿透到网络。
key 只从环境变量 DEEPSEEK_API_KEY 读，任何文件不落盘。
"""

import hashlib
import json
import os
from pathlib import Path

import yaml

CFG = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
CACHE_DIR = Path(CFG["paths"]["eval_cache"]) / "llm"

# 与 judge 的三处硬约定（02 方案 §五 / T1.3 方案 §三）：
# ① 引用法条必须标注"第X条"（citation_correct 依赖）② 长度受限 ③ 只依据给定法条
SYSTEM = (
    "你是中国法律咨询助手。只依据提供的法条材料回答；引用时必须标注《法名》第X条；"
    "材料中没有依据时明确说明。答案控制在 500 字以内。"
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(f"《{c['law']}》{c['article_no']}：{c['text']}" for c in chunks)
    return f"【法条材料】\n{ctx}\n\n【咨询问题】\n{question}"


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def generate(question: str, chunks: list[dict]) -> dict:
    payload = {
        "model": CFG["generator"]["model"],
        "temperature": CFG["generator"]["temperature"],
        "max_tokens": CFG["generator"]["max_tokens"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    }
    # thinking 参数进 payload（从而进缓存键），调用时移到 extra_body——OpenAI SDK 不认裸 thinking
    if CFG["generator"].get("disable_thinking"):
        payload["thinking"] = {"type": "disabled"}
    h = _key(payload)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{h}.json"
    if cache_file.exists():
        # 命中与未命中必须返回同一形态（字符串）——曾因命中路径返回 dict 导致
        # 断网重放"不一致"假象（2026-10-06 实测教训）
        return json.loads(cache_file.read_text())["resp"]
    if os.environ.get("LLM_CACHE_ONLY"):
        raise RuntimeError(f"缓存未命中且 LLM_CACHE_ONLY=1：{question[:30]}（{h[:12]}）")

    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY 环境变量（key 不落盘，只走环境变量）")
    client = OpenAI(api_key=key, base_url=CFG["generator"]["base_url"])
    body = {k: v for k, v in payload.items() if k != "thinking"}
    resp = client.chat.completions.create(
        **body, extra_body={"thinking": payload["thinking"]} if "thinking" in payload else None)
    content = resp.choices[0].message.content
    cache_file.write_text(json.dumps(
        {"req": payload, "resp": content, "usage": resp.usage.model_dump()},
        ensure_ascii=False, indent=1))
    return content
