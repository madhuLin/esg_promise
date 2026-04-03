#!/bin/bash

# 訓練 verification_timeline 模型的腳本
echo "===== Training Verification Timeline Model ====="

python -m src.train \
    --task timeline \
    --model-name-or-path "models/base/chinese-roberta-wwm-ext" \
    --train-path "data/train.json" \
    --valid-path "data/val.json" \
    --output-dir "checkpoints/timeline_model" \
    --epochs 10 \
    --batch-size 8 \
    --learning_rate 2e-5 \
    --max-length 512 \
    --early_stopping_patience 3
