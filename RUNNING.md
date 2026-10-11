# RUNNING：运行手册

## 环境

- 本机 Mac M4 / 16GB / MPS；`.venv/` 已从 v1 迁移（torch/transformers/openai/yaml/numpy 齐全）
- API key：`.env`（`DEEPSEEK_API_KEY`，不入库）
- 外部模型路径：`rag/config.yaml` 的 `paths.*`（唯一配置入口，无硬编码）

## 常用命令

```bash
# 数据完整性（P0 验收件）
.venv/bin/python -c "import hashlib; print(hashlib.sha256(open('data/blind_test_v3/blind_testset.v3.jsonl','rb').read()).hexdigest()[:12])"
# 预期 097d8814e97f

# 特征抽取（幂等四重键，新数据/缓存未命中必跑——先跑它再 --no-extract）
.venv/bin/python -u scripts/v3/a8_extract.py

# 特征-标签八层对齐（开训前必过）
.venv/bin/python -u scripts/v3/a9_align_check.py

# Qwen 训练（P2；val 自动取 split=dev，无 dev 即报错拒绝代用）
cat data/train_v3/records_train.jsonl data/train_v3/records_dev.jsonl > /tmp/all.jsonl
.venv/bin/python -m train.qwen_jey.train_a --data /tmp/all.jsonl --epochs 10
# 冒烟（2 epochs）加 --epochs 2；特征全缓存后可加 --no-extract（cache_only，未命中报错）

# judge 冒烟（双 state：eval 八题两前向 / production 七题单前向）
.venv/bin/python rag/judge/judge.py

# 推送 GitHub
bash scripts/git_push.sh
```

## 数据管线（scripts/v3/，append-only 日志全在 data/train_v3/）

| 脚本 | 作用 |
|---|---|
| a0-a2 | manifest 登记 / 章级切分 / 三池出题（出题即带 oracle gold） |
| a3 | 检索+变体+teacher 打标（主合成管线） |
| a4 | 定版：E01-E08+管线校验+盲测封存+Laya 转换+manifest 更新 |
| a5 | 人工终审抽样清单（--seed 换轮） |
| a6 | human-fix（源头重放） |
| a7 | 跨集交叉校验（train↔盲测 0 同题） |
| a8/a9 | 特征抽取 / 八层对齐检查 |
| a10/a13 | 变体重设计（trunc 三形态→offconcl 收敛，2026-10-11 有效性修复） |
| a11 | 检索劣化补批（噪声 context） |
| a12 | 被拒记录 emphasize 重试 |

**修复数据的标准路径**：改源头（questions/raw）→ 删受影响 _synth_raw → 重跑 a3/a10 → a4 重定版 → 重跑受影响检查（a9/a7/有效性四门）——禁止手改下游 jsonl。

## 扩产流程（冻结 596 已上调至 696，P2 若拟合不足）

1. 先诊断（数据质量/超参/架构归因），出诊断报告
2. 确需补数据：小批量 ≤300，走 a2/a3/a4 全管线 + a5 终审 + 有效性四门
3. 硬顶：补完即停，不自动扩产

## 日志文件（append-only，审计用）

- `data/train_v3/reject_log.jsonl`：全管线拒绝记录
- `data/train_v3/human_fix_log.jsonl`：人工终审修正（9 处，v1 遗产）
- `data/train_v3/variant_redo_log.jsonl`：变体重设计转换记录
- `data/train_v3/a11_log.jsonl`：检索劣化批次记录
