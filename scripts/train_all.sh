#!/bin/bash

# 一鍵執行所有任務訓練的腳本
echo "🚀 Starting Full ESG Training Pipeline..."

# 1. Promise
bash scripts/train_promise.sh

# 2. Evidence
bash scripts/train_evidence.sh

# 3. Clarity (Clarity depends on evidence)
bash scripts/train_clarity.sh

# 4. Timeline
bash scripts/train_timeline.sh

echo ""
echo "✅ All training tasks completed!"
echo "📊 Current Best Scores Summary (checkpoints/all_results.json):"
if [ -f "checkpoints/all_results.json" ]; then
    cat checkpoints/all_results.json
else
    echo "Summary file not found."
fi
