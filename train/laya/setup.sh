#!/bin/bash
# T5.3 云端环境脚本：RTX 4090 / 24GB / CUDA 12.4.1 / PyTorch 2.3.0 / Python 3.10
# 验收：全新实例从零到训练启动 ≤30 分钟
# 用法：ssh 到实例后 bash setup.sh（假设已上传 laya_ft/ 整个目录到 /root/）

set -e
echo "=== ① GPU 检查 ==="
nvidia-smi || { echo "❌ nvidia-smi 失败"; exit 1; }

echo "=== ② 装依赖（PyTorch 2.3.0 已内置，只装缺的） ==="
pip install -q safetensors transformers==4.48.0 2>&1 | tail -1

echo "=== ③ Python 依赖检查 ==="
python3 -c "
import torch, transformers, safetensors
print(f'torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0)}')
print(f'transformers {transformers.__version__} | safetensors {safetensors.__version__}')
assert torch.cuda.is_available(), 'CUDA 不可用'
"

echo "=== ④ 检查数据与权重 ==="
for f in laya_ft/data/laya_trainset.jsonl laya_ft/model/multilingual/model.safetensors \
         laya_ft/model/multilingual/rl_agent_config.json \
         laya_ft/model/multilingual/encoder/config.json \
         laya_ft/model/multilingual/tokenizer/tokenizer.json; do
    [ -f "$f" ] && echo "  ✓ $f ($(du -h "$f" | cut -f1))" || { echo "  ❌ 缺 $f"; exit 1; }
done

echo "=== ⑤ 干跑验证（10 条 forward） ==="
python3 laya_ft/train_cuda.py --dry-run 2>&1 | tail -5

echo "=== ✅ 环境就绪，启动训练 ==="
echo "  bash setup.sh && python3 laya_ft/train_cuda.py --epochs 4"
echo "  或直接：nohup python3 laya_ft/train_cuda.py --epochs 4 > train.log 2>&1 &"
