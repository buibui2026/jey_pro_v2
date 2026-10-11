# jey_pro_v2：法律 RAG Judge 双模型线

> v1（../jey_pro）审计后推倒重来的干净版本。验证性项目：重流程、轻规模、质量门机制化。
> 设计文档：[docs/spec/2026-10-11-jey-pro-v2-design.md](docs/spec/2026-10-11-jey-pro-v2-design.md)（九项裁决固化）

## 路线（三模块串行）

```
P0 数据资产迁移与收口 ──→ P1 Laya 法律适配+长度鲁棒性 ──→ P2 Qwen 决策头训练+三 judge 对比
     ✅ 完成 2026-10-11              ⬜ 待启动                        ⬜
```

| 模块 | 内容 | 状态 | 文档 |
|---|---|---|---|
| P0 | 数据迁移收口 + 有效性修复（变体重设计+检索劣化补批）+ 四维验收 | ✅ | [docs/P0_数据资产.md](docs/P0_数据资产.md) |
| P1 | Laya 微调（train_cuda 修三处 + 长样本 + 云端训练 + 基线评测） | ⬜ | [docs/P1_Laya线.md](docs/P1_Laya线.md) |
| P2 | Qwen 阶段 A 训练 + M6 三层验收 + metrics 重写 + 三 judge 对比 + 速度成本表 | ⬜ | [docs/P2_Qwen线.md](docs/P2_Qwen线.md) |

## 质量门清单（每阶段收口必过，v1 复盘教训机制化）

1. **校验一律拒绝式**，禁止报警模式——v1 亲手把三层断言降级为报警，放走了 production 带第八题等三批坏数据
2. **隔离靠机制**：章级源分组切分（`data/splits/source_groups.v3.json`）、盲测封存 hash（`097d8814e97f`，冻结前禁看标签）、train/dev/test 三池互斥
3. **修 X 必回归覆盖 X 的检查**——v1 两次"修完更危险"（judge 修复引入训推不一致、缓存键漏题行）
4. **每阶段收口前对抗性自审三问**：评测独立性？信息可见性？可复现（hash+命令+缓存）？
5. **产物绑定不可变证据**：manifest + sha256 + 生成命令；reject_log/human_fix_log/variant_redo_log append-only
6. **数据有效性先于结构正确**（2026-10-11 新增）：分布检查（形态可分性/维度覆盖/标签退化/极端档方向）过了才能训练——"数据不对其他都白做"

## 数据资产现状（P0 收口后）

| 数据集 | 规模 | 说明 |
|---|---|---|
| records_train | **696 条** | 7 形态：gold 298 / partial_offconcl 119 / flawed 149 / partial_keypoints 24 / concl_no_basis 6 / 检索劣化 100（irrelevant 70 + partial 30） |
| records_dev | 60 条 | 独立章池，选点/校准 |
| blind_testset | 117 题 | 封存 `097d8814e97f`，模型/prompt 冻结前禁看 |
| 有效性验收 | 四门全过 | offconcl 51% 中间档 / 无关 11% / 无退化 / gold 98%·flawed 82% 方向 |

## 运行手册

见 [RUNNING.md](RUNNING.md)——环境、常用命令、数据管线（a0-a13）、扩产流程。

## 外部资源（路径映射）

| 资源 | 位置 |
|---|---|
| Laya / Qwen3.5-0.8B backbone / tokenizer | `/Users/yangkun/Desktop/Projects/model`（rag/config.yaml 统一配置） |
| Embedding/Reranker GGUF | `~/.lmstudio/models/` |
| 答案生成/teacher | DeepSeek API（key 走 .env，不入库） |
| 训练算力 | 算力自由 gpufree.cn（Linux+CUDA，P1 使用） |
