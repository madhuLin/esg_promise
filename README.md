# ESG 承諾驗證繁體中文分類專案

本專案旨在解決「ESG 承諾驗證」多任務分類問題。專案基於 PyTorch 和 HuggingFace Transformers，提供了一個從訓練、評估到預測的完整解決方案。

## 專案特色

- **多任務架構**：支援四個獨立的分類任務（`promise_status`, `evidence_status`, `evidence_quality`, `verification_timeline`）。
- **模組化設計**：程式碼結構清晰，分為資料處理、模型、訓練、評估等模組，易於維護和擴充。
- **階層式預測**：推論階段會自動套用任務間的依賴關係，確保輸出邏輯一致。
- **高效能訓練**：支援混合精度（Mixed Precision）、梯度累積（Gradient Accumulation）和 Early Stopping。
- **詳細評估**：提供 Macro-F1、每個類別的 Precision/Recall/F1，並繪製混淆矩陣。
- **CLI 介面**：使用 `tyro` 提供簡潔易用的命令列介面。

## 環境安裝

1.  建議使用 Python 3.9+ 的虛擬環境。

2.  安裝相依套件：
    ```bash
    pip install -r requirements.txt
    ```

## 專案結構

```
esg_promise/
├── README.md
├── requirements.txt
├── configs/
│   └── label_maps.py     # 集中管理標籤定義
├── data_examples/        # 範例資料
│   └── sample_data.json
├── outputs/              # 儲存預測結果與評估報告
├── scripts/              # 訓練腳本
│   ├── train_promise.sh
│   ├── train_evidence.sh
│   ├── train_clarity.sh
│   └── train_timeline.sh
└── src/
    ├── data.py           # 資料集與資料載入器
    ├── evaluate.py       # 模型評估
    ├── metrics.py        # 指標計算與混淆矩陣
    ├── models.py         # 模型架構
    ├── predict.py        # 單任務預測
    ├── run_pipeline.py   # 執行完整預測流程（含階層邏輯）
    ├── train.py          # 訓練腳本主程式
    └── utils.py          # 工具函式
```

## 使用教學

### 1. 資料準備

將您的訓練、驗證、測試資料（JSON List 格式）放置在您選擇的目錄中。此處提供一個範例檔案 `data_examples/sample_data.json`。

### 2. 訓練模型

專案提供了針對四個任務的獨立訓練腳本。您可以直接執行：

- **訓練 `promise_status` 模型**
  ```bash
  bash scripts/train_promise.sh
  ```

- **訓練 `evidence_status` 模型**
  ```bash
  bash scripts/train_evidence.sh
  ```

- **訓練 `evidence_quality` (clarity) 模型**
  ```bash
  bash scripts/train_clarity.sh
  ```

- **訓練 `verification_timeline` 模型**
  ```bash
  bash scripts/train_timeline.sh
  ```

您也可以直接使用 `python -m src.train` 並傳入參數來自訂訓練：

```bash
python -m src.train \
    --task promise \
    --model_name_or_path "hfl/chinese-roberta-wwm-ext" \
    --train_path "data_examples/sample_data.json" \
    --valid_path "data_examples/sample_data.json" \
    --output_dir "checkpoints/promise_model" \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5
```

訓練完成後，最佳模型會儲存在 `--output_dir` 指定的路徑下。

### 3. 評估模型

使用 `src.evaluate` 來評估已訓練模型的表現。

```bash
python -m src.evaluate \
    --task promise \
    --model_path "checkpoints/promise_model/best_model" \
    --data_path "data_examples/sample_data.json" \
    --output_dir "outputs/promise_eval"
```

評估結果（包含 `classification_report.txt` 和 `confusion_matrix.png`）將會儲存在 `--output_dir` 中。

### 4. 產生預測

要對新資料進行預測，我們建議執行完整的預測流程，這會依序執行四個任務的預測，並套用階層式邏輯。

**完整預測流程**：

使用 `src.run_pipeline` 腳本，它會自動處理所有步驟。

```bash
python -m src.run_pipeline \
    --test_data_path "data_examples/sample_data.json" \
    --promise_model_path "checkpoints/promise_model/best_model" \
    --evidence_model_path "checkpoints/evidence_model/best_model" \
    --clarity_model_path "checkpoints/clarity_model/best_model" \
    --timeline_model_path "checkpoints/timeline_model/best_model" \
    --output_path "outputs/final_predictions.json"
```

這個腳本會：
1.  預測 `promise_status`。
2.  根據 `promise_status` 的結果，預測 `evidence_status` 和 `verification_timeline`。
3.  根據 `evidence_status` 的結果，預測 `evidence_quality`。
4.  將所有預測結果（包含符合邏輯的 `N/A`）整合並儲存到 `--output_path`。

**單任務預測** (進階)：

如果您只想執行單一任務的預測，可以使用 `src.predict`。

```bash
python -m src.predict \
    --task promise \
    --model_path "checkpoints/promise_model/best_model" \
    --input_path "data_examples/sample_data.json" \
    --output_path "outputs/promise_pred.json"
```

## 擴充指南

- **切換 Backbone 模型**：在 `train.py` 的 CLI 參數中，透過 `--model_name_or_path` 指定您想使用的 HuggingFace 模型名稱。
- **調整超參數**：`train.py` 的 CLI 參數支援調整學習率、批次大小、epoch 數、dropout 等。
- **多任務模型**：`src/models.py` 中已預留 `MultiTaskModel` 的結構，您可以基於此擴充實作共享編碼器的多任務學習。
- **自訂損失函數**：在 `train.py` 中，您可以輕鬆替換 `loss_fn`，例如改用 Focal Loss。
