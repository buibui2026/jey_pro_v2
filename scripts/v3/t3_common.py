"""V3/V2 管线公共件：带缓存的 DeepSeek 调用（纪律照抄 v1 a3_synth.call：payload 进 key、命中返字符串）。"""
import hashlib
import json
import os
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "rag/config.yaml"))
CACHE = ROOT / "data/eval_cache/llm"
for line in open(ROOT / ".env"):
    if line.startswith("DEEPSEEK_API_KEY"):
        os.environ.setdefault("DEEPSEEK_API_KEY", line.split("=", 1)[1].strip())


def call(model, messages, tag, max_tokens=1400):
    payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
               "messages": messages, "thinking": {"type": "disabled"}, "tag": tag}
    h = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{h}.json"
    if f.exists():
        return json.loads(f.read_text())["resp"], h
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("缺 DEEPSEEK_API_KEY")
    client = OpenAI(api_key=key, base_url=CFG["generator"]["base_url"], timeout=90, max_retries=1)
    body = {k: v for k, v in payload.items() if k not in ("thinking", "tag")}
    for attempt in range(3):
        try:
            r = client.chat.completions.create(**body, extra_body={"thinking": payload["thinking"]})
            c = r.choices[0].message.content
            f.write_text(json.dumps({"req": payload, "resp": c}, ensure_ascii=False))
            return c, h
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 + attempt * 5)


def call_pro(prompt, max_tokens=1400):
    return call("deepseek-v4-pro", [{"role": "user", "content": prompt}],
                "v2-teacher-pro", max_tokens)[0]


def call_flash(prompt, max_tokens=400):
    return call("deepseek-v4-flash", [{"role": "user", "content": prompt}],
                "v2-gen-flash", max_tokens)[0]
