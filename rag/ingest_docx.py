#!/usr/bin/env python3
"""T1.1 切分入库：5 部法律 docx → 法条级 chunk 库。

对应任务：01 依赖图 T1.1 / 方案 modules/M1_RAG链路/T1.1_切分入库方案.md
实验矩阵：单因素——切分按"第X条(之N)?"段首匹配，跨段合并，章标题记元数据
运行方式：.venv/bin/python rag/ingest_docx.py
产物：data/rag_store/chunks.jsonl ＋ modules/M1_RAG链路/results/T1.1_对账报告.md
边界：本任务到 chunks.jsonl 为止；embedding 向量库在 T1.2（方案 §3.5 已确认）
"""

import json
import re
import statistics
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# ── 全局契约区 ────────────────────────────────────────────────
LAW_DOCS = Path("data/law_docs")
OUT_DIR = Path("data/rag_store")
REPORT = Path("modules/M1_RAG链路/results/T1.1_对账报告.md")

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# 段首匹配"第X条(之N)?"——探测已验证（刑法 452+53=505 精确闭合）
ART_RE = re.compile(r"^第(?P<num>[零〇一二三四五六七八九十百千\d]+)条(?:之(?P<sub>[一二三四五六七八九十]+))?")
CHAP_RE = re.compile(r"^第[一二三四五六七八九十百]+章")

CN_DIG = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn2num(s: str) -> int:
    # 中文数字→整数，覆盖 1~9999（法条最大"第四百五十二条"量级）
    result, cur = 0, 0
    for ch in s:
        if ch in CN_DIG:
            cur = CN_DIG[ch]
        else:
            if cur == 0:
                cur = 1  # "十X"开头的十位省略一，如"第十五条"
            result += cur * CN_UNIT[ch]
            cur = 0
    return result + cur


def parse_docx(path: Path) -> list[str]:
    # 零依赖：docx 即 zip，拼 word/document.xml 的 w:p>w:t
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(f"{W_NS}p"):
        text = "".join(t.text or "" for t in p.iter(f"{W_NS}t")).strip()
        if text:
            paras.append(text)
    return paras


def split_articles(paras: list[str], law: str) -> tuple[list[dict], list[str]]:
    """游标扫描：条起开新 chunk、章起换元数据、其余并入当前条。"""
    chunks, chapters_seen = [], []
    cur = None
    chapter = ""
    for i, para in enumerate(paras):
        if CHAP_RE.match(para):
            if para not in chapters_seen:  # 目录回声去重：同名章标题只记一次
                chapters_seen.append(para)
            chapter = para
            continue
        m = ART_RE.match(para)
        if m:
            if cur:
                chunks.append(cur)
            num_raw = m.group("num")
            article_num = int(num_raw) if num_raw.isdigit() else cn2num(num_raw)
            cur = {
                "law": law,
                "article_no": para.split("　")[0] if "　" in para else m.group(0),
                "article_num": article_num,
                "article_sub": m.group("sub"),
                "chapter": chapter,
                "text": para,
                "n_chars": len(para),
                "_para_idx": i,
            }
        elif cur:
            cur["text"] += "\n" + para  # 跨段条文合并
            cur["n_chars"] += len(para)
    if cur:
        chunks.append(cur)
    return chunks, chapters_seen


def dedup(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    # 判重规则（方案 §3.4）：同编号保留正文条（更长者），处置全留痕
    by_key: dict[tuple, dict] = {}
    dropped = []
    for c in chunks:
        key = (c["law"], c["article_num"], c["article_sub"])
        if key not in by_key:
            by_key[key] = c
        else:
            keep, drop = (c, by_key[key]) if c["n_chars"] > by_key[key]["n_chars"] else (by_key[key], c)
            by_key[key] = keep
            dropped.append({"law": drop["law"], "article_no": drop["article_no"],
                            "kept_len": keep["n_chars"], "dropped_len": drop["n_chars"]})
    return list(by_key.values()), dropped


def p95(vals: list[int]) -> float:
    sv = sorted(vals)
    return sv[min(len(sv) - 1, int(0.95 * len(sv)))]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    all_chunks, per_law = [], []
    for path in sorted(LAW_DOCS.glob("*.docx")):
        law = path.stem.split("_")[0].replace("中华人民共和国", "")
        paras = parse_docx(path)
        raw, chapters = split_articles(paras, law)
        final, dropped = dedup(raw)

        nums = sorted({c["article_num"] for c in final})
        gaps = [n for n in range(1, (nums[-1] if nums else 0) + 1) if n not in set(nums)]
        body_chars = sum(len(p) for p in paras)
        chunk_chars = sum(c["n_chars"] for c in final)
        lengths = [c["n_chars"] for c in final]
        per_law.append({
            "law": law, "paras": len(paras), "raw": len(raw), "final": len(final),
            "max_num": nums[-1] if nums else 0, "sub_n": sum(1 for c in final if c["article_sub"]),
            "chapters": len(chapters), "gaps": gaps, "dropped": dropped,
            "coverage": chunk_chars / body_chars,
            "first": final[0] if final else None, "last": final[-1] if final else None,
            "sub_samples": [c for c in final if c["article_sub"]][:2],
            "lengths": lengths,
        })
        all_chunks.extend(final)

    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in sorted(all_chunks, key=lambda x: (x["law"], x["article_num"], x["article_sub"] or "")):
            c.pop("_para_idx", None)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ── 对账报告（自产文档：数字全部来自本次运行） ─────────────
    lines = ["# T1.1 对账报告", "", f"运行：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M} ｜ 产物：data/rag_store/chunks.jsonl", ""]
    lines += ["## 一、计数对账表", "", "| 法名 | 段落 | 探测条数 | 去重后 | 主编号最大 | 之N条 | 章数 | 编号缺口 | 字符覆盖率 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in per_law:
        lines.append(f"| {r['law']} | {r['paras']} | {r['raw']} | {r['final']} | {r['max_num']} | {r['sub_n']} | "
                     f"{r['chapters']} | {len(r['gaps'])}{'处:' + str(r['gaps'][:8]) if r['gaps'] else ''} | {r['coverage']:.1%} |")
    lines += ["", "**首末条核对**（人工一眼可核）：", ""]
    for r in per_law:
        if r["first"]:
            lines.append(f"- {r['law']} 首条：{r['first']['text'][:60]}…")
            lines.append(f"- {r['law']} 末条：{r['last']['text'][:60]}…")
    lines += ["", "## 二、判重处置表", ""]
    any_drop = [(r["law"], d) for r in per_law for d in r["dropped"]]
    if any_drop:
        lines += ["| 法名 | 编号 | 保留长度 | 丢弃长度 |", "|---|---|---|---|"]
        lines += [f"| {law} | {d['article_no']} | {d['kept_len']} | {d['dropped_len']} |" for law, d in any_drop]
    else:
        lines.append("无判重记录。")
    lines += ["", "## 三、长度分布（字）", "", "| 法名 | 均值 | 中位 | P95 | 最大 |", "|---|---|---|---|---|"]
    for r in per_law:
        ls = r["lengths"]
        lines.append(f"| {r['law']} | {statistics.mean(ls):.0f} | {statistics.median(ls):.0f} | {p95(ls)} | {max(ls)} |")
    total = [c["n_chars"] for c in all_chunks]
    lines += ["", f"**全库**：{len(all_chunks)} 条 ｜ 均值 {statistics.mean(total):.0f} ｜ P95 {p95(total)} ｜ 最大 {max(total)} 字", ""]
    lines += ["## 四、抽样核对（首/末/随机2 + 刑法之N×2，无拦腰截断检查）", ""]
    import random
    rng = random.Random(42)
    for r in per_law:
        pool = [c for c in all_chunks if c["law"] == r["law"]]
        picks = [pool[0], pool[-1]] + rng.sample(pool, min(2, len(pool)))
        if r["sub_samples"]:
            picks += r["sub_samples"][:2]
        for c in picks[:6]:
            lines.append(f"- **{c['law']} {c['article_no']}**（章：{c['chapter'] or '无'}，{c['n_chars']}字）：{c['text'][:80]}…")
    lines += ["", "## 五、预测对账", "",
              "| # | 预测 | 实测 | 判定 |", "|---|---|---|---|"]
    xfa = next(r for r in per_law if r["law"] == "刑法")
    msf = next(r for r in per_law if r["law"] == "民事诉讼法")
    preds = [
        ("P1", "刑法 505 chunks，无判重", f"{xfa['final']} 条，判重 {len(xfa['dropped'])} 处",
         "✓" if xfa["final"] == 505 and not xfa["dropped"] else "✗"),
        ("P2", "民诉法去重后 < 306（预测 284~296）", f"{msf['final']} 条",
         "✓" if 284 <= msf["final"] <= 296 else "✗"),
        ("P3", "全库 P95 < 600 字、最大 < 1500 字", f"P95 {p95(total)}、最大 {max(total)}",
         "✓" if p95(total) < 600 and max(total) < 1500 else "✗"),
        ("P4", "五部合计 ≈ 1110±40", f"{len(all_chunks)} 条",
         "✓" if 1070 <= len(all_chunks) <= 1150 else "✗"),
        ("P5", "各法覆盖率 ≥ 95%", "见 §一 表", "✓" if all(r["coverage"] >= 0.95 for r in per_law) else "✗"),
    ]
    lines += [f"| {p} | {d} | {a} | {v} |" for p, d, a, v in preds]
    bad = [p for p, d, a, v in preds if v == "✗"]
    if bad:
        lines += ["", f"**失手 {len(bad)} 条（{', '.join(bad)}）——原因分析与教训见下**", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"chunks: {len(all_chunks)} -> {OUT_DIR / 'chunks.jsonl'}")
    for r in per_law:
        print(f"  {r['law']}: raw={r['raw']} final={r['final']} gaps={len(r['gaps'])} cov={r['coverage']:.1%}")
    print(f"report -> {REPORT}")
    print(f"preds: {[p + v for p, d, a, v in preds]}")
# ═══════════════════ 实测输出（2026-10-06 运行） ═══════════════════
# chunks: 1137 -> data/rag_store/chunks.jsonl
#   刑法: raw=505 final=505 gaps=0 cov=97.8%
#   反电信网络诈骗法: raw=50 final=50 gaps=0 cov=98.0%
#   未成年人保护法: raw=132 final=132 gaps=0 cov=97.4%
#   民事诉讼法: raw=306 final=306 gaps=0 cov=96.5%
#   治安管理处罚法: raw=144 final=144 gaps=0 cov=98.3%
# preds: ['P1✓', 'P2✗', 'P3✓', 'P4✓', 'P5✓']
# ↑ P2 失手：民诉法 306 条是真实条数（2023 修正版官方即 306 条，涉外编扩充至第 305 条），
# ↑ "目录重复"假设不成立——外部核对与根因分析见对账报告 §六；其余四条全中
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    main()
