"""V3 数据重做·阶段6：人工终审 human-fix（2026-10-10 终审裁定落盘）。

终审结论（30+30 抽样）：盲测 1 错 1 备注建议、train 1 错 1 建议补、4 条"production
态 gold 空"经核实为清单展示误导（数据健康，无需修数据，改 a5 展示）。

本脚本只改源头 questions 文件并删对应 _synth_raw（触发 a3 重放）——下游 records/
盲测封存全部由重放再生，不手改。每处修正留痕 human_fix_log.jsonl（原文/新文/依据）。

运行：.venv/bin/python scripts/v3/a6_humanfix.py
之后：a3（重放 2 条）→ a4（重定版+重新封存）→ a5 --seed 43（复验清单）
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_V3 = ROOT / "data/train_v3"
BLIND_V3 = ROOT / "data/blind_test_v3"

# id → (新 gold, 追加金条文[(law, article_no, article_num)] 或 [], 终审依据)
FIXES = {
    # 盲测集：数罪当一罪加重情节——定性错误（终审#2）＋二轮终审删"或拘役"（第447条原文无拘役）
    "v3b-刑法-0002": (
        "标准答案：甲构成私放俘虏罪与故意杀人罪，应数罪并罚。私放普通俘虏依《刑法》第447条处五年以下"
        "有期徒刑；其协助俘虏杀害多名战友系独立犯罪行为，另构成故意杀人罪，依第232条论处，与私放"
        "俘虏罪并罚。",
        [],
        "二轮终审：第447条原文仅'处五年以下有期徒刑'，无拘役——删'或拘役'（一轮数罪并罚结论维持）"),
    # 盲测集：第11条两款主体不同（终审#23 备注建议，升级为 gold 修正）
    "v3b-未成年人保护法-0009": (
        "父母或其他监护人发现未成年人权益受侵害时，依《未成年人保护法》第二十条，应及时了解并采取保护"
        "措施，情况严重须立即报告；而第十一条第一款赋予任何组织和个人劝阻、制止、检举、控告的权利（一般"
        "主体无强制报告义务），该条第二款对国家机关、居（村）委会及密切接触未成年人的单位及其工作人员"
        "另设强制报告义务。",
        [],
        "终审备注：原表述'任何组织或个人无强制报告义务'易误读为第11条整体无强制报告；第2款另有特定主体强制报告义务"),
    # 盲测集二轮：玩具枪公交抢劫——gold 自相矛盾（终审#19）
    "v3b-刑法-0038": (
        "标准答案：应处十年以上有期徒刑、无期徒刑或者死刑，并处罚金或者没收财产。依《刑法》第263条第"
        "（二）项，在公共交通工具上抢劫属加重情节——公交车属公共交通工具；玩具枪不属于真枪，不适用第"
        "（七）项持枪抢劫，但不影响第（二）项的适用。",
        [],
        "二轮终审：gold 既承认公交车属公共交通工具又说不适用十年以上档，自相矛盾；第263条第(二)项明文'在公共交通工具上抢劫'处十年以上/无期/死刑"),
    # train：gold 引错条号（33/32→22）（终审#16）
    "v3t-反电信网络诈骗法-0019": (
        "标准答案：两措施在同一条内衔接：依《反电信网络诈骗法》第二十二条，互联网服务提供者对监测识别的"
        "涉诈异常账号应当重新核验，并根据国家有关规定采取限制功能、暂停服务等处置措施——先核验、后处置，"
        "处置以核验为前提。",
        [("反电信网络诈骗法", "第二十二条", 22)],
        "终审裁定：衔接规则明文在第22条（先核验后处置同条衔接）；第33条系身份认证公共服务、第32条系反制技术建设，均非处置依据"),
    # train：补"不处罚单位本身"（终审#8 存疑项，终审建议补全）
    "v3t-刑法-0024": (
        "标准答案：工程监理单位降低工程质量标准造成重大安全事故的，依《刑法》第137条，对直接责任人员处"
        "五年以下有期徒刑或拘役，并处罚金；后果特别严重的，处五年以上十年以下有期徒刑，并处罚金。该条"
        "不处罚工程监理单位本身，仅处罚直接责任人员——此即单位与个人的差异。",
        [],
        "终审裁定：第137条未规定对单位的处罚，'差异'应点明单位不受本条处罚，原答案不完整"),
    # train 二轮：gold 引第32条（反制技术）与申诉救济无关（终审#17）
    "v3t-反电信网络诈骗法-0002": (
        "标准答案：应依《反电信网络诈骗法》第三十一条第二款处理：用户对暂停措施有异议的可提出申诉，"
        "有关部门应建立健全申诉渠道、信用修复和救济制度——机构应及时受理核查，告知处置原因、救济渠道"
        "及需提交资料，核查通过的即解除暂停措施。",
        [("反电信网络诈骗法", "第三十一条", 31)],
        "二轮终审：第32条系反制技术研发条款，与告知原因/救济渠道/申诉无关；第31条第2款明文'对上述认定和措施有异议的，可以提出申诉……建立健全申诉渠道、信用修复和救济制度'（条号经法条库原文复核）"),
    # train 三轮：破坏计算机信息系统罪单位犯罪前提核实成立，gold 补全（终审存疑）
    "v3t-刑法-0030": (
        "标准答案：依《刑法》第二百八十六条第四款，单位犯破坏计算机信息系统罪的，对单位判处罚金，并对"
        "直接负责的主管人员和其他直接责任人员依照第一款规定处罚——即后果严重的处五年以下有期徒刑或"
        "拘役，后果特别严重的处五年以上有期徒刑。",
        [],
        "三轮终审存疑核实：第286条第4款明文规定单位犯罪（'单位犯前三款罪的，对单位判处罚金，并对其直接负责的主管人员…依照第一款的规定处罚'）——题目前提成立；gold 补'对单位判处罚金'使其完整"),
    # train 三轮：军事演习不属第451条战时，删"战时从重"（终审存疑）
    "v3t-刑法-0108": (
        "构成过失破坏军事通信罪，依《刑法》第三百六十九条第二款，造成严重后果的，处三年以下有期徒刑"
        "或者拘役。军事演习不属第四百五十一条规定的'战时'（宣布战争状态、受领作战任务、遭敌突然袭击），"
        "故不适用该条第三款的战时从重处罚。",
        [],
        "三轮终审存疑核实成立：第369条第3款'战时犯前两款罪的从重'，但第451条战时定义不含军事演习——删'战时从重处罚'并点明理由"),
}


def load_chunks():
    return {f"{c['law']}|{c['article_no']}": c
            for c in (json.loads(l) for l in open(ROOT / "data/rag_store/chunks.jsonl"))}


def main():
    chunks = load_chunks()
    fixlog = open(TRAIN_V3 / "human_fix_log.jsonl", "a")
    n_q, n_raw = 0, 0
    for path, pool in [(TRAIN_V3 / "questions_train.jsonl", "train"),
                       (BLIND_V3 / "questions_test.jsonl", "test")]:
        rows = [json.loads(l) for l in open(path)]
        changed = False
        for r in rows:
            if r["id"] not in FIXES:
                continue
            new_gold, extra_arts, basis = FIXES[r["id"]]
            have_arts = {(a["law"], a["article_no"]) for a in r["gold_articles"]}
            if r["gold_answer"] == new_gold and all(
                    (l, a) in have_arts for l, a, _ in extra_arts):
                continue  # 幂等：上一轮已修到位，跳过
            assert len(new_gold) <= 230, f"{r['id']} 修正文本超 230 字：{len(new_gold)}"
            for law, no, num in extra_arts:
                c = chunks[f"{law}|{no}"]
                r["gold_articles"].append({"law": law, "article_no": no, "article_num": num})
            fixlog.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "id": r["id"], "pool": pool,
                "old_gold": r["gold_answer"], "new_gold": new_gold,
                "extra_articles": [{"law": l, "article_no": a} for l, a, _ in extra_arts],
                "basis": basis, "reviewer": "用户终审 2026-10-10（30+30 抽样）",
            }, ensure_ascii=False) + "\n")
            r["gold_answer"] = new_gold
            r["human_fixed"] = True
            changed = True
            n_q += 1
            # train 记录触发重放：删 raw（a3 按 question 命中检索缓存，teacher 重打标）
            raw = TRAIN_V3 / "_synth_raw" / f"{r['id']}.json"
            if pool == "train" and raw.exists():
                raw.unlink()
                n_raw += 1
        if changed:
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fixlog.close()
    print(f"human-fix 完成：{n_q} 条 gold 修正（train 侧重放 {n_raw} 条 raw 已删）")
    print("后续：a3 → a4 → a5 --seed 43 --include <修复的train id>")


if __name__ == "__main__":
    main()
