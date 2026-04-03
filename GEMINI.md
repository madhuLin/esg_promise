# ESG Promise Verification Project Context

## 專案目標
自動化驗證企業 ESG 報告中的「承諾（Promise）」，包含四個子任務：
1. `promise_status` (承諾狀態 - 2 類: Yes, No)
2. `evidence_status` (證據狀態 - 3 類: Yes, No, N/A)
3. `evidence_quality` (證據品質 - 4 類: Clear, Not Clear, Misleading, N/A)
4. `verification_timeline` (驗證時間線 - 5 類: already, within_2_years, ..., N/A)

## 核心架構
- **模型**: Single-Task RoBERTa (基於 `hfl/chinese-roberta-wwm-ext`)。
- **輸入**: ESG 報告文本段落 (JSON List 格式, `max_length=256`)。
- **訓練策略**:
    - **階層式過濾**: 訓練 `evidence` 與 `timeline` 時僅保留 `promise=Yes` 的樣本；訓練 `clarity` 時僅保留 `evidence=Yes` 的樣本。
    - **類別權重**: 使用 `torch.bincount` 計算 `CrossEntropyLoss` 的權重以應對類別不均衡。
- **推理 Pipeline**: 實作 `run_pipeline.py` 進行帶約束的推理，若 `promise=No` 則後續欄位強制修正為 `N/A`。

## 目前進度
- [x] 完成四個單任務訓練腳本與 Macro-F1 驗證。
- [x] 實作階層式推理流程 (Hierarchical Pipeline)。
- [x] 下載本地端模型 `models/base/chinese-roberta-wwm-ext`。
- [x] 修復終端機日誌顯示與 `tkinter` 繪圖報錯問題。
- [x] 優化 `.gitignore` 排除大型權重與暫存檔。

## 常用指令
- 環境：`conda activate esg_promise`
- 訓練：`bash scripts/train_promise.sh` (或 `train_evidence.sh`, `train_clarity.sh`, `train_timeline.sh`)
- 推理：`python -m src.run_pipeline --test-data-path ... --output-path ...`
- 評估：`python -m src.evaluate --task promise --model_path ...`
