#!/bin/bash

# 訓練 promise_status 模型的腳本
echo "===== Training Promise Status Model ====="

python -m src.train \
    --task promise \
    --model_name_or_path "hfl/chinese-roberta-wwm-ext" \
    --train_path "data_examples/sample_data.json" \
    --valid_path "data_examples/sample_data.json" \
    --output_dir "checkpoints/promise_model" \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --max_length 256 \
    --early_stopping_patience 2
