# V3 数据定版报告

定版时间：2026-10-11T11:31:26+00:00

## 规模

- train 记录 696 条 / dev 60 条 / 盲测 117 题
- finalize 拒绝：无

## 分布（train）

- 答案档：{'gold_irrelevant_ctx': 70, 'gold_partial_ctx': 30, 'gold': 298, 'partial_offconcl': 119, 'flawed': 149, 'partial_keypoints': 24, 'concl_no_basis': 6}
- 模式：{'eval': 557, 'production': 139}
- 题型：{'C': 135, 'B': 279, 'A': 282}  难度：{'hard': 235, 'easy': 228, 'mid': 233}
- 法源：{'刑法': 273, '反电信网络诈骗法': 41, '未成年人保护法': 80, '民事诉讼法': 206, '治安管理处罚法': 96}

## state_tokens（E05 带）

- min 271 / 中位 559 / max 1151

## 数据门自检（审计 §七）

- train/dev/test 按源（章）分组：✅ source_groups.v3.json 断言三池互斥
- test 零派生：✅ 盲测集条文池与 train/dev 章互斥，且未进入任何合成/打标流程
- gold 非空：✅ eval 态全量硬断言；概率仅归一化未 clamp
- 人工抽检：⏳ 待办——见下方终审清单

## 人工终审清单（数据门最后一项）

从盲测集与 train 各分层抽样（法源×题型×难度×答案档），逐条判 gold 正确性。
生成命令：`.venv/bin/python scripts/v3/a5_sample.py`