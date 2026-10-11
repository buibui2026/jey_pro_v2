# jey_pro_v2 设计文档（spec）

> 日期：2026-10-11 ｜ 状态：待用户审批
> 来源：v1 全项目复盘（2026-10-11 头脑风暴）+ 三轮澄清（复用边界/长度目标/执行顺序）+ 骨架 A + 四项默认值 + 扩产口径
> 上游：v1 项目（`../onwer/jey_pro`，封存只读）、V3 数据资产（四轮人工终审过门）

---

## 0. 项目定义

**一句话**：法律 RAG judge 双模型线——Laya 微调基线（P1）与 Qwen3.5-0.8B 决策头自训（P2），站在已终审的 V3 数据资产上（P0 收口），轻治理、质量门全机制化。

**三大部件**（用户 2026-10-11 定义）：
1. Laya 微调适配法律 jev rag judge + 上下文扩展 → **长度鲁棒性口径**：微调后 1K~3.2K 真实工作区效果提升且 8K 压力测试不退化；不做任何位置编码改造（YaRN 等仅作方法参考）
2. Qwen3.5-0.8B 训练：自回归模型改输出决策任务头（冻结主干 + query/key 决策头，阶段 A；LoRA 阶段 B 条件触发）
3. 数据合成：复用 V3 管线（a0-a9），v2 内承担"迁移收口 + 长样本补充 + 条件性小批扩产"

**执行顺序**：P0 数据迁移收口 → P1 Laya 基线 → P2 Qwen 主线与三 judge 对比。

**关键裁决记录**：
| 裁决 | 结论 | 日期 |
|---|---|---|
| 复用边界 | 带验证过的核心：V3 数据全套+管线、rag/ 链路、qwen 训练件、治理模式；不迁 v1 文档/旧数据/旧脚本 | 10-11 |
| 上下文目标 | 长度鲁棒性（非外扩） | 10-11 |
| 执行顺序 | Laya 先、Qwen 后 | 10-11 |
| 治理骨架 | A：轻治理三模块（README+RUNNING+三份 P 文档） | 10-11 |
| eval/production 配比 | 维持 80/20（评测 judge 定位，eval 态是主场景） | 10-11 |
| retrieval_issue 覆盖 | P0 量化后再定，<10% 才补（~60 条走全管线） | 10-11 |
| 特征缓存 | 迁移续抽（键=行明文 hash，数据不变即有效；现 1662/5116） | 10-11 |
| 仓库形态 | 同级新目录 `jey_pro_v2/` + 独立 git 仓库；v1 封存只读 | 10-11 |
| 数据规模 | **冻结 596 先跑通全流程**；拟合不足先诊断原因，确需补数据小批量 ≤300，不设自动扩产 | 10-11 |

---

## 1. 目录结构

```
jey_pro_v2/
├── README.md          # 路线+状态表+质量门清单（唯一治理文档）
├── RUNNING.md         # 运行手册：环境/命令/迁移件清单/扩产流程
├── docs/
│   ├── P0_数据资产.md  # 方案+验收门+结果 三合一（每份 ≤200 行）
│   ├── P1_Laya线.md
│   ├── P2_Qwen线.md
│   └── spec/2026-10-11-jey-pro-v2-design.md（本文档副本）
├── rag/               # 复用 v1：embedder/reranker/vector_store/generator/judge（含双 state 修复）/run_pipeline/config.yaml
│                      # metrics.py 迁入但标记 P2 重写
├── train/
│   ├── qwen_jey/      # 复用 v1 全套：schema/rows/features/head/train_a/state_builder/convert/validate/test_*
│   └── laya/          # train_cuda.py 迁入，P1 修三处（见 §3）
├── scripts/
│   └── v3/            # 数据管线 a0-a9 原样迁入（收口/长样本/扩产共用工具）
└── data/
    ├── law_docs/               # 5 部法律 docx
    ├── rag_store/              # chunks + embeds
    ├── splits/                 # source_groups.v3.json
    ├── train_v3/               # questions/contexts/records/laya/reject_log/human_fix_log/feat_cache(1662半成品)
    ├── blind_test_v3/          # 盲测集（封存 097d8814e97f）+ manifest
    ├── artifacts_manifest.json
    └── eval_cache/             # API 缓存（可重放）
```

**迁移清单（从 v1 复制）**：
- 代码：`rag/`（除 t12_compare.py、bge_ref.py）、`train/qwen_jey/` 全包、`train/laya/`（train_cuda.py、training_config.md、upload_checklist.md、setup.sh、pack.sh；不带 finetuned/、trial_out/）、`scripts/v3/` 全部、`scripts/git_push.sh`、`.gitignore`
- 数据：`data/` 下 law_docs、rag_store、splits、train_v3（含 feat_cache 半成品与全部日志）、blind_test_v3、artifacts_manifest.json、eval_cache
- **不迁**：v1 `modules/`、`docs/`（叙事文档）、`data/testset/`（M2 retired）、`data/train/`（旧 contaminated/invalid）、`data/runs/`、`scripts/m3|m4|m7|m8*`、`synth/`

---

## 2. P0 数据资产迁移与收口（1-2 天）

> 2026-10-11 数据有效性检查（用户指令"数据不对其他都白做"）发现两个缺陷，修复任务并入 P0，
> 迁移之前先把数据修到有效。

**任务**：
1. 建 `jey_pro_v2/` + git init + 按迁移清单复制 + README/RUNNING/P0-P2 文档初版
2. 完整性验证：盲测集 sha256 须逐字等于 `097d8814e97f`（v1 封存值）；artifacts_manifest 复核；records E01-E08 抽验
3. **变体重设计（有效性缺陷 1 修复）**：trunc 档 149 条按 gold 结构分流为三形态——
   多要点题 → 要点缺失（程序化截取前半句数，~40 条）；短 gold 一半 → 结论偏离（改派
   flawed"答了一半跑题"，~55 条）；短 gold 另一半 → 结论无依据（保留原"仅结论句"形态，
   ~55 条，明确其角色为 citation 维度信号源）。全部重放 teacher 打标。
   理由：原"取第一句"在结论先行结构下得到"简洁的正确答案"（trunc 档 answer_correctness
   97% 判一致、completeness 1.69 vs gold 1.72——两档无差异），中间带监督缺失。
4. **检索劣化补批（有效性缺陷 2 修复）**：~60 条 context 部分替换为噪声/弱相关条文
   （同法他章），走全管线 + 终审，把 ctx_relevance=无关 从 1.5% 拉到 ~10%。
5. **有效性复检（P0 验收门）**：重跑四维检查——部分正确类（要点缺失+结论偏离）与 gold
   档可分（部分一致占比 ≥40%）；ctx_relevance=无关 ≥8%；无维度标签退化；gold/flawed
   极端档方向一致（沿用本次检查口径）。
6. 特征缓存续抽（trunc 重放后行明文变，重放部分需重抽）+ a9 八层对齐

**验收门**：
- [ ] 盲测 sha256 与 v1 一致（若变体重设计未触及盲测集则不变）
- [ ] 有效性四维检查全过（新增验收维度，v1 未有过）
- [ ] a9 八层全绿
- [ ] 修复走全管线+终审留痕（reject_log/human_fix_log 续用）
- [ ] 首次提交推 GitHub

---

## 3. P1 Laya 线（3-5 天）：法律适配 + 长度鲁棒性

**任务**：
1. **train_cuda.py 修三处**（v1 审计 M5 🔴）：
   - 按 calib 标记切分（用 laya_trainset_v3 的 calib=dev 60 条，弃"随机切 10%"）
   - 训练 max_len 覆盖目标长度带（≥3800，配合长样本数据）
   - 加未微调基线同输入对照 + 灾难遗忘检查（通用能力样本输出分布对比）
2. **长 context 补充数据**（V3 state_tokens 271~1137 中位 550，撑不起长度验证）：从 train 池法条构造 ~100 条长 context 训练样本（多条文拼接 3.2K~6K）+ ~30 条 dev 池压测样本；走 a3 合成 + a4 定版 + 终审（与 V3 同纪律）
3. 云端训练（算力自由 RTX 4090，沿用 v1 环境清单）：第一版全参（脚本现成、322M 成本低）；遗忘/过拟合则降 LoRA
4. 基线评测：未微调 Laya vs 微调 Laya 同输入同口径（盲测集、双 state judge、人工一致率抽样 50 条）

**验收门（长度鲁棒性口径）**：
- [ ] 工作区提升：judge-teacher 一致率较未微调 +5pp 以上（或 KL 降 20%+，二选一达成）
- [ ] 8K 压测：微调后 8K 档相对自身 4K 退化 ≤3pp 且不差于未微调 8K 档
- [ ] 灾难遗忘：通用样本分布漂移低于预设阈值（P1 文档定具体值）
- [ ] 人工一致率 ≥70%（50 条抽样）

---

## 4. P2 Qwen 线（5-7 天）：决策头训练与三 judge 对比

**任务**：
1. 阶段 A 正式训练：train_a（v1 修复版：val=独立 dev、标准 10 桶 ECE、cache_only）+ P0 续抽完成的特征；超参照 T8.1 方案（epochs≤10、val KL 选点、patience 2）
2. **M6 三层验收**：
   - 结构成立：候选换序一致率 ≥80%（架构门，不达标先改输入结构，不进 LoRA）
   - 蒸馏成立：dev KL ≤ teacher 自噪声对照（teacher 对同题二次标注的 KL）
   - 任务成立：盲测集人工真值上一致率 ≥ Laya 基线
3. **metrics.py 重写**（v1 欠账）：检索 Recall@K/MRR/nDCG、citation 条号解析精确率、judge-人工一致率、标准 ECE 曲线（分桶）
4. **三 judge 对比收口**（原 M9）：未微调 Laya / 微调 Laya / qwen-jey；同输入、同 rubric、同 split、同温度纪律（温度只在各自 calibration 集拟合）；**速度成本表**（吞吐/延迟/单样本成本对比——决策模型核心卖点，v1 从未测过）

**验收门**：
- [ ] M6 三层全过（LoRA 仅在阶段 A 不达标且数据量足够时触发）
- [ ] metrics 重写并用于对比报告
- [ ] 三 judge 对比报告 + 速度成本表出
- [ ] 复盘回写 RUNNING.md

---

## 5. 质量门清单（README 内置，v1 复盘教训机制化）

1. **校验一律拒绝式**，禁止报警模式（v1 三层断言降级教训）
2. **隔离靠机制**：章级源分组、封存 hash、盲测纪律（模型/prompt 冻结前禁看 gold 标签、禁调参）
3. **修 X 必回归覆盖 X 的检查**（v1 两次"修完更危险"教训：judge 修复引入训推不一致、缓存键漏题行）
4. **每阶段收口前对抗性自审三问**：独立性（评测与训练隔离）/ 隔离（信息可见性）/ 可复现（hash+命令+缓存）
5. **产物绑定不可变证据**：manifest + sha256 + 生成命令
6. **台账只记结论级事件**，过程靠 git log + reject_log + human_fix_log

---

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| 596 条拟合不足 | **先诊断**（数据质量/超参/架构归因），确需补数据小批量 ≤300 走全管线+终审；不设自动扩产 |
| Laya 全参灾难遗忘 | 降 LoRA 或混入通用数据 |
| 8K 压测退化超线 | 如实报告——长度鲁棒性是目标不是执念 |
| 云环境差异 | 复用 v1 upload_checklist |
| 扩产触发后成本失控 | 批量 ≤300 硬顶，触发前必须先出诊断报告 |

时间线：P0 1-2 天 / P1 3-5 天 / P2 5-7 天，合计约 2 周。

---

## 7. 自审记录（spec self-review）

- 无 placeholder/TBD；四项默认值与三轮澄清结论全部固化在 §0 裁决表
- 内部一致性：§3 长样本与 §2 数据冻结不冲突（长样本是 P1 新构造、走管线，不属于"扩产"）；§4 阶段 B 与 M6 架构门衔接
- 范围：三个 P 各自可独立验收，单一 spec 可支撑实施计划
- 歧义检查：扩产口径已由用户二次澄清（冻结 596 + 小批 ≤300）
