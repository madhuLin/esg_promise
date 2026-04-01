import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import torch
import tyro
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from configs.label_maps import LABEL_MAPS
from src.data import ESGDataset, format_input_text
from src.models import SingleTaskModel
from src.train import Task
from src.utils import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class PredictionArguments:
    """預測腳本的參數。"""
    task: Task = tyro.conf.arg(help="要預測的任務類型。")
    model_path: Path = tyro.conf.arg(help="已訓練模型的路徑。")
    input_path: Path = tyro.conf.arg(help="輸入的 JSON 檔案路徑。")
    output_path: Path = tyro.conf.arg(help="儲存預測結果的 JSON 檔案路徑。")

    batch_size: int = 32
    max_length: int = 512
    num_workers: int = 4

def main(args: PredictionArguments):
    """
    對給定資料集執行單一任務的預測，並將預測結果附加到原始資料上。
    """
    setup_logging()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"正在為任務 '{args.task}' 執行預測...")
    logger.info(f"模型路徑: {args.model_path}")
    logger.info(f"輸入資料: {args.input_path}")
    logger.info(f"輸出路徑: {args.output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_to_label = LABEL_MAPS["id_to_label"][args.task]
    num_labels = len(id_to_label)

    # 載入模型和 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = SingleTaskModel(
            model_name_or_path=str(args.model_path), 
            num_labels=num_labels
        ).to(device)
    except Exception as e:
        logger.error(f"從 {args.model_path} 載入模型或 tokenizer 失敗: {e}")
        return

    # 載入原始資料
    with open(args.input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 建立 Dataset 和 DataLoader (不含標籤)
    dataset = ESGDataset(
        data=raw_data,
        tokenizer=tokenizer,
        task=args.task,
        max_length=args.max_length,
        for_training=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 執行預測
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting {args.task}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast():
                logits = model(**batch)

            preds = torch.argmax(logits, dim=-1)
            all_predictions.extend(preds.cpu().numpy())

    # 將預測結果轉換為標籤字串
    predicted_labels = [id_to_label[pred_id] for pred_id in all_predictions]

    # 將預測結果附加到原始資料
    output_key = f"predicted_{args.task}"
    if len(predicted_labels) != len(raw_data):
        logger.error(f"預測結果數量 ({len(predicted_labels)}) 與輸入資料數量 ({len(raw_data)}) 不匹配。")
        return
        
    for i, sample in enumerate(raw_data):
        sample[output_key] = predicted_labels[i]

    # 儲存結果
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"預測完成，結果已儲存至 {args.output_path}")

if __name__ == "__main__":
    tyro.cli(main)
