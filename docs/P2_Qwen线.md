# P2 Qwen 线：决策头训练与三 judge 对比（⬜ 待启动）

> 阶段 A（冻结主干+262K 决策头）为主线；LoRA 仅在阶段 A 不达标且数据量足够时触发。

## 任务

1. **阶段 A 正式训练**：train_a（v1 修复版：val=独立 dev、标准 10 桶 ECE、cache_only）
   - 数据：records_train 696 + records_dev 60（含 P0 修复后的 7 形态）
   - 特征：a8 全量抽取 + a9 八层对齐过（P0 收口件）
   - 选点：val KL 主 + ECE 辅，patience 2
2. **M6 三层验收**
   - [ ] 结构成立：候选换序一致率 ≥80%（架构门——不达标先改输入结构，不进 LoRA）
   - [ ] 蒸馏成立：dev KL ≤ teacher 自噪声对照（teacher 同题二次标注 KL）
   - [ ] 任务成立：盲测集人工真值上一致率 ≥ Laya 基线
3. **metrics.py 重写**（v1 欠账）：Recall@K/MRR/nDCG（检索）、citation 条号解析精确率、
   judge-人工一致率、标准分桶 ECE 曲线
4. **三 judge 对比收口**：未微调 Laya / 微调 Laya / qwen-jey——同输入、同 rubric、同 split、
   同温度纪律（各自 calibration 集拟合）+ **速度成本表**（吞吐/延迟/单样本成本）

## 验收门

- [ ] M6 三层全过
- [ ] metrics 重写并用于对比报告
- [ ] 三 judge 对比报告 + 速度成本表出
- [ ] 复盘回写 RUNNING.md

## 结果（完成后填）

（待执行）
