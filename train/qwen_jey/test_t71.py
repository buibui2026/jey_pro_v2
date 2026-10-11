"""T7.1 单元测试：8 规则各配真实 pass/fail 用例（方案 §八清单）＋训推公因数钉死断言。

运行：.venv/bin/python -m train.qwen_jey.test_t71（项目根目录）
样例全部真实：t001 出自 M2 定版测试集，法条出自 chunk 库，E08 哈希出自 eval_cache 实存文件。

════════════════════ 实测输出（2026-10-07） ════════════════════
# PASS test_builder_parity        ← qwen-jey state 与 Laya 四段式逐字符同构（公因数钉死）
# PASS test_questions_canonical   ← rubric 零改写注入
# PASS test_e01~e08_pass_and_fail ← 8 规则各正反用例（E05 负例＝t001 开发样例 424 token）
# PASS test_final_mode_requires_soft_labels ← final 完整记录零错误
# ALL PASS（11/11）
# ↑ 解读：schema 门达成——判定标准"8 条全实现、样例 100% 通过/正确拒绝"
════════════════════════════════════════════════
"""

from __future__ import annotations

import json

from rag.judge.judge import build_state as laya_build_state
from .convert import load_chunks, testset_to_record
from .schema import build_questions, make_record, validate_record

T001 = next(json.loads(l) for l in open("data/testset/testset_final.jsonl")
           if json.loads(l)["id"] == "t001")
LONG_CTX = load_chunks([("刑法", n) for n in (260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270)] + [("刑法", 328)])
SHORT_CTX = load_chunks([("刑法", 264), ("刑法", 265), ("刑法", 328)])
ANSWER = ("最高可判无期徒刑。《刑法》第三百二十八条规定，盗掘确定为全国重点文物保护单位的古墓葬"
          "属于该条第（一）项情形，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。")
REAL_HASH = "sha256:2e82e884310e221241c3cdbf1ab8d86b10b840379f6386d442a44f9f00951ba0"  # eval_cache 实存文件


def good_target_probs():
    """八题各造一个合法分布（键集从 canonical 注入取——不允许手抄漂移）。"""
    tp = {}
    for key, q in build_questions().items():
        keys = list(q["candidates"])
        tp[key] = {k: (0.8 if i == 0 else round(0.2 / (len(keys) - 1), 4)) for i, k in enumerate(keys)}
    return tp


def long_record(**kw):  # 过 E05 的载体：K=12 真实法条
    return testset_to_record(T001, LONG_CTX, ANSWER, **kw)


def short_record(**kw):  # t001 开发样例：K=3＝424 token，E05 必拒（方案 §三预期行为）
    return testset_to_record(T001, SHORT_CTX, ANSWER, **kw)


def codes(rec, mode="draft"):
    return {e.split(":")[0] for e in validate_record(rec, mode=mode)}


def test_builder_parity():
    """公因数钉死：qwen-jey state 与 Laya judge 的四段式逐字符同构（只差【状态】前缀）。"""
    rec = short_record()
    inner = laya_build_state(T001["question"], SHORT_CTX, ANSWER, T001["gold_answer"])
    assert rec["state_text"] == "【状态】\n" + inner, "state 模板与 02 方案实现漂移"


def test_questions_canonical():
    """零改写注入：转换器产出的 questions 与 rubric canonical 完全相等。"""
    assert short_record()["questions"] == json.loads(json.dumps(build_questions()))


def test_e01_pass_and_fail():
    assert "QJ-E01" not in codes(short_record())
    bad = short_record()
    bad["questions"].pop("answer_correctness")  # final 定版缺题
    assert "QJ-E01" in codes(bad, mode="final")
    bad2 = short_record()
    bad2["questions"]["虚构题"] = bad2["questions"]["groundedness"]  # 非法题键
    assert "QJ-E01" in codes(bad2)


def test_e02_pass_and_fail():
    rec = long_record(target_probs=good_target_probs())
    assert "QJ-E02" not in codes(rec)
    bad = long_record(target_probs=good_target_probs())
    bad["target_probs"]["groundedness"].pop("无据")  # 少一个键
    assert "QJ-E02" in codes(bad)


def test_e03_pass_and_fail():
    rec = long_record(target_probs=good_target_probs())
    assert "QJ-E03" not in codes(rec)
    bad = long_record(target_probs=good_target_probs())
    bad["target_probs"]["overall"] = {"0": 0.4, "1": 0.4, "2": 0.142}  # 和=0.942
    assert "QJ-E03" in codes(bad)


def test_e04_pass_and_fail():
    assert "QJ-E04" not in codes(short_record())
    bad = short_record()
    q = bad["questions"]["completeness"]
    q["candidates"] = {"1": "只答了要点之一", "2": "覆盖主要要点", "3": "完整且说明了适用条件"}  # 非零基
    assert "QJ-E04" in codes(bad)


def test_e05_pass_and_fail():
    """带＝[200,3800]（2026-10-07 裁决A校准，原 [1000,3800]）：424 的 t001 样例现为带内正例；
    负例改用极短 state（单条微型法条，<200 token）。"""
    assert "QJ-E05" not in codes(short_record()), short_record()["state_tokens"]
    tiny = testset_to_record(
        T001, [{"law": "刑法", "article_no": "第一九九条",
                "text": "第一九九条　示例微型条文。"}],
        "无期徒刑。")  # 极短 state：约几十 token
    assert "QJ-E05" in codes(tiny), tiny["state_tokens"]
    assert "QJ-E05" not in codes(long_record())


def test_e06_pass_and_fail():
    assert "QJ-E06" not in codes(short_record())
    bad = short_record()
    bad["state_text"] = bad["state_text"].replace("【状态】", "【STATE】")  # 模板漂移
    assert "QJ-E06" in codes(bad)


def test_e07_pass_and_fail():
    seen = set()
    validate_record(short_record(), mode="draft", seen=seen)  # 首进建档
    dup = {e.split(":")[0] for e in validate_record(short_record(), mode="draft", seen=seen)}
    assert "QJ-E07" in dup  # 同题同法条集再进＝重复
    other = testset_to_record(T001, LONG_CTX, ANSWER)  # 同题但法条集不同＝不重复
    assert "QJ-E07" not in {e.split(":")[0]
                            for e in validate_record(other, mode="draft", seen=seen)}


def test_e08_pass_and_fail():
    ok = long_record(target_probs=good_target_probs(), teacher={"model": "deepseek-v4-flash",
                                                                "request_hash": REAL_HASH})
    assert "QJ-E08" not in codes(ok)
    bad = long_record(target_probs=good_target_probs(),
                      teacher={"model": "x", "request_hash": "sha256:" + "0" * 64})
    assert "QJ-E08" in codes(bad)


def test_final_mode_requires_soft_labels():
    assert {"QJ-E01", "QJ-E08"} & codes(long_record(), mode="final")  # 前体进 final 必被拦
    full = long_record(target_probs=good_target_probs(),
                       teacher={"model": "deepseek-v4-flash", "request_hash": REAL_HASH})
    assert validate_record(full, mode="final") == [], validate_record(full, mode="final")


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
