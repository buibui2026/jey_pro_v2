#!/bin/bash
# 打包上传清单：模型权重 + 训练数据 + 训练/评测脚本
set -e
cd "$(dirname "$0")/../.."

# 打包目录结构
rm -rf /tmp/laya_ft && mkdir -p /tmp/laya_ft/{model/multilingual/{encoder,tokenizer},data,scripts}

# ① 模型权重（647MB multilingual）
cp /Users/yangkun/Desktop/Projects/model/laya/multilingual/model.safetensors /tmp/laya_ft/model/multilingual/
cp /Users/yangkun/Desktop/Projects/model/laya/multilingual/rl_agent_config.json /tmp/laya_ft/model/multilingual/
cp /Users/yangkun/Desktop/Projects/model/laya/multilingual/encoder/config.json /tmp/laya_ft/model/multilingual/encoder/
cp /Users/yangkun/Desktop/Projects/model/laya/multilingual/tokenizer/tokenizer.json /tmp/laya_ft/model/multilingual/tokenizer/
cp /Users/yangkun/Desktop/Projects/model/laya/multilingual/tokenizer/tokenizer_config.json /tmp/laya_ft/model/multilingual/tokenizer/
cp /Users/yangkun/Desktop/Projects/model/laya/rl_common.py /tmp/laya_ft/model/

# ② 训练数据（~30MB）
cp data/train/laya_trainset.jsonl /tmp/laya_ft/data/
cp data/testset/testset_final.jsonl /tmp/laya_ft/data/  # 双模式评测用

# ③ 脚本
cp train/laya/setup.sh /tmp/laya_ft/
cp train/laya/train_cuda.py /tmp/laya_ft/
cp train/laya/eval_dual_mode.py /tmp/laya_ft/

# 打包
cd /tmp && tar czf laya_ft.tar.gz laya_ft/
echo "✅ 打包完成：/tmp/laya_ft.tar.gz ($(du -h laya_ft.tar.gz | cut -f1))"
echo "上传：scp /tmp/laya_ft.tar.gz root@<IP>:/root/"
echo "部署：ssh root@<IP> 'cd /root && tar xzf laya_ft.tar.gz && bash laya_ft/setup.sh'"
