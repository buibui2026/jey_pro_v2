#!/usr/bin/env python3
"""T1.4 judge：JudgeRunner + LayaBackend（02 方案 §五 接口规格的实现）。

两个防线（T1.4 方案 §二）：
① max_len=8192 显式传——multilingual 默认 1024 静默截断是最隐蔽的坑；
② 每次预测后检查 usage.state_tokens_dropped == 0，非 0 报错——截断数据不许进指标。

输入隔离（审计 P0-2，2026-10-10 修复）：**每个维度只看它被允许看的证据**——
eval 模式构建两个 state：七个生产维度读 base state（question/context/answer，无 gold）；
仅 answer_correctness 额外读含【参考答案】的 state。production 模式不变（单 state 七题）。
代价：eval 模式两次前向——这是消除 gold 泄漏的必要成本（审计 §M1 正确做法）。
返回结构不变：{question_key: laya答案}——M3/M9 下游无感。
"""

import json
from functools import lru_cache
from pathlib import Path

import yaml

CFG = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))
RUBRIC = yaml.safe_load(open(Path(__file__).parent / "rubric.yaml"))


def build_state(question: str, chunks: list[dict], answer: str, gold: str | None = None) -> str:
    """02 方案 §三 四段式模板；生产模式（gold=None）省略【参考答案】段。"""
    ctx = "\n".join(f"《{c['law']}》{c['article_no']}：{c['text']}" for c in chunks)
    s = f"【用户问题】{question}\n【检索法条】\n{ctx}\n【生成的答案】{answer}"
    if gold:
        s += f"\n【参考答案】{gold}"
    return s


def _questions_for(keys: set[str]) -> dict:
    return {k: {kk: vv for kk, vv in v.items() if kk != "eval_only"}
            for k, v in RUBRIC.items() if k in keys}


@lru_cache(maxsize=1)
def _router():
    from laya import Router
    return Router(
        models={
            "english": CFG["paths"]["laya_root"],
            "multilingual": CFG["paths"]["laya_multilingual"],
        },
        device=CFG["judge"]["device"],
        default="english",
    )


def _predict(state: str, questions: dict) -> dict:
    result = _router().predict(state, questions, max_len=CFG["judge"]["max_len"])  # 防线①
    dropped = result.get("usage", {}).get("state_tokens_dropped", 0)  # 防线②
    if dropped:
        raise RuntimeError(f"state 被截断 {dropped} tokens——检查 context 预算后重跑，禁止带伤出数")
    return result


def run(question: str, chunks: list[dict], answer: str, gold: str | None = None) -> dict:
    """返回 {question_key: laya答案}，字段原样透传（换 judge 后端只换本文件，M9 接口保证）。

    输入隔离（P0-2）：七题读无 gold 的 base state；answer_correctness 只在 gold 提供时
    读含 gold 的 state——任何维度不再越权看到参考答案。
    """
    all_keys = set(RUBRIC)
    prod_keys = all_keys - {"answer_correctness"}
    base_state = build_state(question, chunks, answer)          # 永不含 gold

    result = _predict(base_state, _questions_for(prod_keys))
    answers = dict(result.get("answers", {}))

    if gold is not None:                                        # eval 模式：第 8 题单独喂
        gold_state = build_state(question, chunks, answer, gold)
        r2 = _predict(gold_state, _questions_for({"answer_correctness"}))
        answers.update(r2.get("answers", {}))
    return {"answers": answers, "usage": {**result.get("usage", {}),
                                          "n_forward": 2 if gold is not None else 1}}


if __name__ == "__main__":
    # 冒烟：一条手造记录（不经检索/生成——judge 单元可独立验证）
    demo = {
        "question": "打架斗殴会被拘留多少天？",
        "chunks": [{"law": "治安管理处罚法", "article_no": "第五十一条",
                    "text": "第五十一条　殴打他人的，或者故意伤害他人身体的，处五日以上十日以下拘留，并处五百元以上一千元以下罚款。"}],
        "answer": "依据《治安管理处罚法》第五十一条，殴打他人处五日以上十日以下拘留，并处罚款。",
        "gold": "依据第五十一条，处五日以上十日以下拘留，可并处罚款。",
    }
    print(json.dumps(run(**demo)["answers"], ensure_ascii=False, indent=1))
