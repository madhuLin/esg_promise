import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from configs.label_maps import TASK_TO_LABELS
from src.data import create_dataloader
from src.metrics import compute_metrics
from src.models import SingleTaskModel
from src.utils import set_seed, setup_logging
from src.train import Task

logger = logging.getLogger(__name__)

@dataclass
class EvaluationArguments:
    """評估腳本的參數。"""
    task: Task = tyro.conf.field(help="要評估的任務類型。")
    model_path: Path = tyro.conf.field(help="已訓練模型的路徑。")
    data_path: Path = tyro.conf.field(help="要評估的資料集 JSON 檔案路徑。")
    output_dir: Path = tyro.conf.field(help="儲存評估結果的目錄。")
    
    batch_size: int = tyro.conf.field(default=32, help="每個批次的樣本數。")
    max_length: int = tyro.conf.field(default=512, help="Tokenizer 的最大長度。")
    num_workers: int = tyro.conf.field(default=4, help="資料載入器的工作執行緒數量。")
    seed: int = tyro.conf.field(default=42, help="隨機種子。")

def main(args: EvaluationArguments):
    """主評估函式。"""
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    logger.info("評估參數:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用設備: {device}")

    # 載入 tokenizer 和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        num_labels = len(TASK_TO_LABELS[args.task])
        model = SingleTaskModel(
            model_name_or_path=str(args.model_path), 
            num_labels=num_labels
        ).to(device)
    except Exception as e:
        logger.error(f"從 {args.model_path} 載入模型或 tokenizer 失敗: {e}")
        return

    # 載入資料
    logger.info(f"正在從 {args.data_path} 載入資料...")
    eval_dataloader = create_dataloader(
        path=args.data_path,
        tokenizer=tokenizer,
        task=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        for_training=True, # 評估時也需要標籤
        num_workers=args.num_workers,
    )

    if eval_dataloader is None:
        logger.error("無法建立評估資料載入器，評估終止。")
        return

    # 執行評估
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            with autocast():
                logits = model(**batch)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 計算並儲存指標
    logger.info("評估完成，正在計算指標...")
    eval_metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        labels=TASK_TO_LABELS[args.task],
        output_dir=args.output_dir,
    )

    logger.info("--- 評估結果 ---")
    for key, value in eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"評估報告和混淆矩陣已儲存至: {args.output_dir}")

if __name__ == "__main__":
    tyro.cli(main)
