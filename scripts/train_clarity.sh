#!/bin/bash

# 訓練 evidence_quality (clarity) 模型的腳本
# 針對極端不平衡進行穩定性優化
echo "===== Training Clarity (Evidence Quality) Model ====="

python -m src.train \
    --task clarity \
    --model-name-or-path "models/base/chinese-roberta-wwm-ext" \
    --train-path "data/train.json" \
    --valid-path "data/val.json" \
    --output-dir "checkpoints/clarity_model" \
    --epochs 10 \
    --batch-size 8 \
    --learning_rate 2e-5 \
    --max-length 256 \
    --early-stopping-patience 5 \
    --no-balance \
    --no-use-class-weights
