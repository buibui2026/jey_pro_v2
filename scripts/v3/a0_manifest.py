"""V3 数据重做·阶段0：全局 artifacts manifest（审计 P0-1）。

登记 data/ 下全部数据产物：status ∈ {active, retired_dev, retired, contaminated,
invalid, stale}。之后每批新数据由 a4_finalize.py 追加登记（active）。

状态语义（审计 §M0 正确做法）：
  active       当前有效，可进入训练/评测
  retired_dev  退役为调试/回归用，禁止再报告泛化指标（M2 全集即此档）
  retired      过程产物，已被后续版本取代
  contaminated 与评测集语义重叠，永久禁止进入任何训练（M2 派生链全部）
  invalid      生成缺陷（无 gold / 无 split / 随机切分），不可训练验收
  stale        冒烟期权重/中间物，不可引用为正式结果

运行：.venv/bin/python scripts/v3/a0_manifest.py
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# 固定盘点清单：path → (status, 用途, 污染说明)
INVENTORY = {
    # ── 法条库与索引（源头资产，active） ──
    "rag_store/chunks.jsonl": ("active", "法条库：5部法1137条chunk，V3全部合成的唯一源", None),
    "rag_store/embeds.npy": ("active", "法条向量索引（M1）", None),
    # ── M2 测试集（污染链头：被 M4/M7 当训练源） ──
    "testset/testset_final.jsonl": (
        "retired_dev", "M2 正式测试集234条",
        "被 M4/M7 用作训练语义源，审计🔴-1；只做调试回归，禁止泛化指标"),
    "testset/testset_draft.jsonl": ("retired", "M2 草稿集", "被 final 取代"),
    "testset/smoke10.jsonl": ("retired", "冒烟10题", None),
    "testset/smoke_questions.jsonl": ("retired", "冒烟题库", None),
    "testset/m3_t32_questions.jsonl": ("retired_dev", "M3 长度实验题（T3.2）", "仅实验用，不入训练"),
    "testset/m3_t33_questions.jsonl": ("retired_dev", "M3 长度实验题（T3.3）", "仅实验用，不入训练"),
    "testset/badcases.jsonl": ("retired_dev", "M1 坏例集", "仅调试"),
    # ── M4 Laya 数据（M2 改写派生） ──
    "train/laya_trainset.jsonl": (
        "contaminated", "M4 定版300条",
        "299/300 条为 M2 语义改写（审计🔴：语义隔离失败）"),
    "train/laya_trainset_draft.jsonl": ("contaminated", "M4 草稿", "同上"),
    "train/laya_trainset_pilot5.jsonl": ("contaminated", "M4 先导5条", "同上"),
    # ── M7 Qwen 数据 ──
    "train/qwen_jey_trainset.jsonl": (
        "contaminated", "Qwen 旧训练集495条",
        "继承 M2/M4 污染，且与 Laya 集228题相同（审计🔴）"),
    "train/qwen_jey_synth300.jsonl": ("contaminated", "M7 先导300条", "同上"),
    "train/qwen_jey_synth300.states.jsonl": ("contaminated", "先导states旁产物", "同上"),
    "train/qwen_jey_batch1.jsonl": ("contaminated", "M7 批1", "同上"),
    "train/qwen_jey_batch2.jsonl": ("contaminated", "M7 批2", "同上"),
    "train/qwen_jey_batch3.jsonl": ("contaminated", "M7 批3", "同上"),
    "train/qwen_jey_dev_samples.jsonl": ("contaminated", "M7 dev样本", "同上"),
    "train/qwen_jey_semantic_regression.jsonl": ("retired_dev", "语义回归断言集", "检测报警用，非训练"),
    # ── T7.2v2（题源换了但 gold/split 缺陷） ──
    "train/qwen_jey_trainset_v2.jsonl": (
        "invalid", "v2 原始600条",
        "480条无gold仍生成answer_correctness（审计🔴-3）"),
    "train/qwen_jey_trainset_v2_fmt.jsonl": (
        "invalid", "v2 格式化599条",
        "479条gold_answer为空、全量无split、600→599无reject记录、格式化脚本缺失"),
    "train/t72v2_questions.jsonl": (
        "invalid", "v2 题库800道",
        "先合成后随机切分，未按源分组（审计🔴）；C型题单条法条与定义冲突"),
    "train/t72v2_testset.jsonl": (
        "invalid", "v2 测试集136条（题元数据）",
        "无gold无context，且切分随机；不可作评测集"),
    "train/t72v2_contexts.json": ("invalid", "v2 检索上下文", "配套v2题库"),
    # ── 冒烟权重（非正式） ──
    "train/qwen_jey_head_smoke.pt": ("stale", "M8 冒烟head权重", "非正式训练产物"),
    "train/qwen_jey_head_overfit.pt": ("stale", "M8 过拟合自检权重", "非正式训练产物"),
    # ── 登记文档 ──
    "train/qwen_jey_门0三池清单.md": ("retired", "v2 三池清单", "口径已被审计推翻"),
    "train/qwen_jey_门0三池清单_v2.md": ("retired", "v2 三池清单修订", "同上"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_records(path: Path) -> int | None:
    if path.suffix not in {".jsonl", ".md", ".pt", ".npy", ".json"}:
        return None
    if path.suffix == ".jsonl":
        return sum(1 for _ in open(path))
    return None


def main() -> None:
    entries = []
    for rel, (status, role, contamination) in INVENTORY.items():
        p = DATA / rel
        if not p.exists():
            entries.append({"path": f"data/{rel}", "status": "missing"})
            continue
        entries.append({
            "path": f"data/{rel}",
            "status": status,
            "role": role,
            "contamination": contamination,
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
            "n_records": count_records(p),
        })
    manifest = {
        "version": "v3-rebuild",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": {
            "contaminated": "永久禁止进入任何训练或泛化评测",
            "retired_dev": "仅调试/回归，禁止报告泛化指标",
            "invalid": "生成缺陷，不可训练验收，仅留档",
            "active": "唯一可进入训练/评测的数据",
        },
        "artifacts": entries,
        "provenance": "scripts/v3/a0_manifest.py（2026-10-09 审计后全量重登记）",
    }
    out = DATA / "artifacts_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    n_by = {}
    for e in entries:
        n_by[e["status"]] = n_by.get(e["status"], 0) + 1
    print(f"artifacts_manifest.json：{len(entries)} 项")
    for k in sorted(n_by):
        print(f"  {k:14s} {n_by[k]}")


if __name__ == "__main__":
    main()
