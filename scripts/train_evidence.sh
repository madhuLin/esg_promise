#!/bin/bash

# 訓練 evidence_status 模型的腳本
echo "===== Training Evidence Status Model ====="

python -m src.train \
    --task evidence \
    --model-name-or-path "models/base/chinese-roberta-wwm-ext" \
    --train-path "data/train.json" \
    --valid-path "data/val.json" \
    --output-dir "checkpoints/evidence_model" \
    --epochs 10 \
    --batch-size 8 \
    --learning_rate 2e-5 \
    --max-length 256 \
    --early-stopping-patience 3
