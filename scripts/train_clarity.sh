#!/bin/bash

# 訓練 evidence_quality (clarity) 模型的腳本
echo "===== Training Clarity (Evidence Quality) Model ====="

python -m src.train \
    --task clarity \
    --model_name_or_path "hfl/chinese-roberta-wwm-ext" \
    --train_path "data_examples/sample_data.json" \
    --valid_path "data_examples/sample_data.json" \
    --output_dir "checkpoints/clarity_model" \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --max_length 512 \
    --early_stopping_patience 2 \
    --gradient_accumulation_steps 2
