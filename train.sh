#!/bin/bash

echo "🚀 Starting Florence-2 DocVQA Multi-Stage Fine-tuning..."
echo "================================================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查Python环境
echo "📋 Checking Python environment..."
python --version
echo "📋 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查必要的依赖
echo "📋 Checking required packages..."
python -c "
import torch
import transformers
import datasets
from PIL import Image
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein
import numpy as np
print('✅ All required packages are available')
"

# 创建输出目录
echo "📂 Creating output directories..."
mkdir -p logs
mkdir -p results
mkdir -p visualizations

# 启动训练
echo "🏋️ Starting training pipeline..."
python finetune.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📊 Results saved in:"
    echo "   - Logs: logs/"
    echo "   - Visualizations: visualization_*/"
    echo "   - Final results: final_results_all_stages.json"
else
    echo "❌ Training failed! Check the logs for details."
    exit 1
fi

echo "================================================================"
echo "🎉 Florence-2 DocVQA training pipeline completed!"
