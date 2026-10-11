"""V3 数据重做·阶段1：按语义源分组切分 train/dev/test 法条池（审计 P0-4 / M2 正确做法）。

切分单元＝章（chapter）：同一章的全部条文整家进同一 split——消除相邻条文泄漏，
且 C 型题（跨条文比较）可在组内取相似条文，不会跨组。
顺序纪律：先切分、后出题（旧 v2 是先合成 800 题再随机切，审计判定🔴）。

约束：
  1. train/dev/test 章集合互斥（程序断言）；
  2. test/dev 章优先选 M2 零用条的章（M2 已被旧训练链污染，新盲测集与其条文隔离）；
  3. 若某部法干净章不足，允许降级选 M2 用条最少的章，但该章内 M2 用过的条文
     进入 banned_articles——test/dev 出题时跳过（条文级豁免，登记在案）；
  4. 选择顺序确定性（无随机数）：m2_used 升序 → 主条文数降序 → 章名。

产物：data/splits/source_groups.v3.json（含 chunks.jsonl sha256 锚定上游）
运行：.venv/bin/python scripts/v3/a1_split.py
"""
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data/splits"
TARGET_FRAC = {"test": 0.15, "dev": 0.08}   # 每部法目标条文占比
MIN_ARTICLES = {"test": 12, "dev": 6}       # 每部法下限（反诈 50 条也要撑起 24/12 题）


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    chunks_path = ROOT / "data/rag_store/chunks.jsonl"
    chunks = [json.loads(l) for l in open(chunks_path)]
    m2 = [json.loads(l) for l in open(ROOT / "data/testset/testset_final.jsonl")]
    m2_used = {(a["law"], a["article_num"]) for r in m2 for a in r.get("gold_articles", [])}

    # 章聚合（只用主条文；article_sub 子款跟主条同章，天然同组）
    chapters = defaultdict(list)  # (law, chapter) -> [article_num...]
    for c in chunks:
        chapters[(c["law"], c["chapter"])].append(c["article_num"])

    # 每部法内选章
    groups = {"train": [], "dev": [], "test": []}
    banned = []  # (law, article_num)：test/dev 章内的 M2 用条（条文级豁免，禁出题）
    for law in sorted({k[0] for k in chapters}):
        chs = [(k, sorted(set(v))) for k, v in chapters.items() if k[0] == law]
        n_total = sum(len(a) for _, a in chs)
        stats = []
        for k, arts in chs:
            used = sum(1 for a in arts if (k[0], a) in m2_used)
            stats.append({"law": k[0], "chapter": k[1], "articles": arts,
                          "n": len(arts), "m2_used": used})
        # 确定性排序：M2 用条少优先 → 章大优先 → 章名
        stats.sort(key=lambda s: (s["m2_used"], -s["n"], s["chapter"]))

        taken = set()
        for split in ("test", "dev"):
            target = max(MIN_ARTICLES[split], round(n_total * TARGET_FRAC[split]))
            got, picked = 0, []
            for s in stats:
                if id(s) in taken:
                    continue
                picked.append(s)
                taken.add(id(s))
                got += s["n"]
                if got >= target:
                    break
                if len(picked) >= 3:   # 每部法每 split 至多 3 章，防碎片化
                    break
            for s in picked:
                s2 = {**{k: s[k] for k in ("law", "chapter", "n", "m2_used")},
                      "articles": s["articles"]}
                groups[split].append(s2)
                if s["m2_used"]:
                    banned += [(s["law"], a) for a in s["articles"]
                               if (s["law"], a) in m2_used]
        groups["train"] += [{**{k: s[k] for k in ("law", "chapter", "n", "m2_used")},
                             "articles": s["articles"]}
                            for s in stats if id(s) not in taken]

    # ── 断言：互斥 ──
    def art_set(gs):
        return {(c["law"], a) for c in gs for a in c["articles"]}
    tr, dv, te = art_set(groups["train"]), art_set(groups["dev"]), art_set(groups["test"])
    assert not (tr & dv) and not (tr & te) and not (dv & te), "split 法条交集非空！"
    # test/dev 的 M2 用条必须全部进 banned（出题豁免）
    m2_in_eval = {a for a in m2_used if a in (dv | te)}
    assert m2_in_eval == set(banned), "banned 集合与 test/dev 内 M2 用条不一致"

    out = {
        "version": "v3",
        "unit": "chapter",
        "policy": "先按源分组冻结，后出题；test/dev 章优先 M2 零用条；M2 用条在 test/dev 章内禁出题",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream": {"chunks.jsonl": f"sha256:{sha256_file(chunks_path)}"},
        "groups": groups,
        "banned_articles": [{"law": l, "article_num": a} for l, a in sorted(banned)],
        "article_counts": {"train": len(tr), "dev": len(dv), "test": len(te)},
        "m2_used_total": len(m2_used),
    }
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPLITS_DIR / "source_groups.v3.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    # ── 报告 ──
    print(f"切分完成 → {out_path.relative_to(ROOT)}")
    print(f"条文数：train {len(tr)} / dev {len(dv)} / test {len(te)}"
          f"（总 {len(tr | dv | te)}，M2 用过 {len(m2_used)} 条全部隔离在 train 侧）")
    print(f"banned（test/dev 章内 M2 用条，出题跳过）：{len(banned)} 条")
    for split in ("train", "dev", "test"):
        by_law = defaultdict(lambda: [0, 0])
        for c in groups[split]:
            by_law[c["law"]][0] += 1
            by_law[c["law"]][1] += c["n"]
        detail = "  ".join(f"{l}:{n}章/{a}条" for l, (n, a) in sorted(by_law.items()))
        print(f"  {split:5s} {len(groups[split])} 章 | {detail}")
    dirty = [c for c in groups["test"] + groups["dev"] if c["m2_used"]]
    if dirty:
        print(f"⚠️ test/dev 含 M2 用条的章（已条文豁免）：{[(c['law'], c['chapter'], c['m2_used']) for c in dirty]}")
    else:
        print("✅ test/dev 全部章 M2 零用条")


if __name__ == "__main__":
    main()
