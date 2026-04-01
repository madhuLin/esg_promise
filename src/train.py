import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import tyro
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from configs.label_maps import LABEL_MAPS, TASK_TO_LABELS
from src.data import create_dataloader
from src.metrics import compute_metrics
from src.models import SingleTaskModel
from src.utils import set_seed, setup_logging

# 取得 logger
logger = logging.getLogger(__name__)

# 定義任務類型
Task = Literal["promise", "evidence", "clarity", "timeline"]

@dataclass
class TrainingArguments:
    """訓練腳本的參數。"""
    task: Task = tyro.conf.arg(help="要訓練的任務類型。")
    model_name_or_path: str = tyro.conf.arg(
        default="hfl/chinese-roberta-wwm-ext", help="預訓練模型的名稱或路徑。"
    )
    train_path: Path = tyro.conf.arg(help="訓練資料集的 JSON 檔案路徑。")
    valid_path: Path = tyro.conf.arg(help="驗證資料集的 JSON 檔案路徑。")
    output_dir: Path = tyro.conf.arg(help="儲存模型和日誌的目錄。")
    
    # 訓練超參數
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    dropout_prob: float = 0.1
    gradient_accumulation_steps: int = 1
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # 其他設定
    seed: int = 42
    use_amp: bool = True
    num_workers: int = 4

def main(args: TrainingArguments):
    """主訓練函式。"""
    # --- 1. 初始化 ---
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)
    
    logger.info("訓練參數:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用設備: {device}")

    # --- 2. 載入 Tokenizer 和資料 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    logger.info("正在建立訓練資料載入器...")
    train_dataloader = create_dataloader(
        path=args.train_path,
        tokenizer=tokenizer,
        task=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        for_training=True,
        num_workers=args.num_workers,
    )
    
    logger.info("正在建立驗證資料載入器...")
    valid_dataloader = create_dataloader(
        path=args.valid_path,
        tokenizer=tokenizer,
        task=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        for_training=True, # 驗證時也需要過濾和標籤
        num_workers=args.num_workers,
    )

    if train_dataloader is None or valid_dataloader is None:
        logger.error("無法建立資料載入器，訓練終止。")
        return

    # --- 3. 準備模型、優化器、損失函數 ---
    num_labels = len(TASK_TO_LABELS[args.task])
    model = SingleTaskModel(
        model_name_or_path=args.model_name_or_path,
        num_labels=num_labels,
        dropout_prob=args.dropout_prob,
    ).to(device)
    
    # 計算類別權重 (Class Weighting)
    labels_in_train = [item['labels'].item() for item in train_dataloader.dataset]
    class_counts = torch.bincount(torch.tensor(labels_in_train))
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum() * num_labels
    class_weights = class_weights.to(device)
    logger.info(f"計算出的類別權重: {class_weights}")
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=args.use_amp)

    # --- 4. 訓練迴圈 ---
    best_metric = -1.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # 訓練階段
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for i, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with autocast(enabled=args.use_amp):
                logits = model(**batch)
                loss = loss_fn(logits, labels)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": total_loss / (i + 1)})

        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"平均訓練損失: {avg_train_loss:.4f}")
        
        # 驗證階段
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                with autocast(enabled=args.use_amp):
                    logits = model(**batch)
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 計算指標
        eval_metrics = compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            labels=TASK_TO_LABELS[args.task],
            output_dir=args.output_dir / f"epoch_{epoch+1}",
        )
        logger.info(f"驗證指標 (Epoch {epoch+1}): Macro-F1 = {eval_metrics['macro_f1']:.4f}")
        
        # Early stopping 和模型儲存
        if eval_metrics["macro_f1"] > best_metric:
            best_metric = eval_metrics["macro_f1"]
            patience_counter = 0
            
            # 儲存最佳模型
            best_model_dir = args.output_dir / "best_model"
            best_model_dir.mkdir(exist_ok=True)
            model.model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            logger.info(f"找到新的最佳模型，Macro-F1 為 {best_metric:.4f}，已儲存至 {best_model_dir}")
        else:
            patience_counter += 1
            logger.info(f"模型表現未提升，Patience: {patience_counter}/{args.early_stopping_patience}")
            if patience_counter >= args.early_stopping_patience:
                logger.info("觸發 Early Stopping，訓練結束。")
                break

    end_time = time.time()
    logger.info(f"訓練完成，總耗時: {(end_time - start_time) / 60:.2f} 分鐘")
    logger.info(f"最佳 Macro-F1: {best_metric:.4f}")

if __name__ == "__main__":
    # 使用 tyro 解析命令行參數並執行 main 函式
    tyro.cli(main)
