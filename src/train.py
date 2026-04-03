import logging
import os
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import tyro
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from configs.label_maps import LABEL_MAPS, TASK_TO_LABELS
from src.data import create_dataloader
from src.metrics import compute_metrics
from src.models import SingleTaskModel
from src.utils import set_seed, setup_logging

# 抑制 tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

Task = Literal["promise", "evidence", "clarity", "timeline"]

@dataclass
class TrainingArguments:
    task: Task = field(metadata={"help": "要訓練的任務類型。"})
    train_path: Path = field(metadata={"help": "訓練資料集的 JSON 檔案路徑。"})
    valid_path: Path = field(metadata={"help": "驗證資料集的 JSON 檔案路徑。"})
    output_dir: Path = field(metadata={"help": "儲存模型和日誌的目錄。"})

    # 👉 有 default 的放後面
    model_name_or_path: str = field(
        default="hfl/chinese-roberta-wwm-ext",
        metadata={"help": "預訓練模型的名稱或路徑。"}
    )

    epochs: int = field(default=3, metadata={"help": "訓練 epoch 數。"})
    batch_size: int = field(default=16, metadata={"help": "每個批次的樣本數。"})
    learning_rate: float = field(default=2e-5, metadata={"help": "學習率。"})
    weight_decay: float = field(default=0.01, metadata={"help": "權重衰減。"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup 比例。"})
    max_length: int = field(default=512, metadata={"help": "Tokenizer 最大長度。"})
    dropout_prob: float = field(default=0.1, metadata={"help": "Dropout 機率。"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "梯度累積步數。"})
    early_stopping_patience: int = field(default=3, metadata={"help": "Early stopping 耐心值。"})
    seed: int = field(default=42, metadata={"help": "隨機種子。"})
    use_amp: bool = field(default=True, metadata={"help": "是否使用混合精度。"})
    num_workers: int = field(default=4, metadata={"help": "DataLoader workers。"})
    balance: bool = field(default=True, metadata={"help": "是否使用平衡採樣。"})
    use_class_weights: bool = field(default=True, metadata={"help": "是否使用類別權重。"})
    label_smoothing: float = field(default=0.1, metadata={"help": "Label smoothing 比例。"})

def main(args: TrainingArguments):
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    logger.info("訓練參數:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用設備: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataloader = create_dataloader(
        path=args.train_path,
        tokenizer=tokenizer,
        task=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        for_training=True,
        num_workers=args.num_workers,
        balance=args.balance,  # 傳入 balance 參數
        filter=True, # 訓練集維持過濾
    )

    valid_dataloader = create_dataloader(
        path=args.valid_path,
        tokenizer=tokenizer,
        task=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        for_training=True, # 恢復為 True 以載入標籤
        num_workers=args.num_workers,
        balance=False,
        filter=True, # 驗證集也要過濾，這樣分數才會「正常」
    )

    if train_dataloader is None or valid_dataloader is None:
        logger.error("資料載入器建立失敗")
        return

    logger.info(f"訓練樣本數: {len(train_dataloader.dataset)}")
    logger.info(f"驗證樣本數: {len(valid_dataloader.dataset)}")

    num_labels = len(TASK_TO_LABELS[args.task])
    model = SingleTaskModel(
        model_name_or_path=args.model_name_or_path,
        num_labels=num_labels,
        dropout_prob=args.dropout_prob,
    ).to(device)

    # 計算類別權重
    class_weights = None
    if args.use_class_weights:
        # 直接從 dataset.labels 獲取，排除 -1
        labels_in_train = [l for l in train_dataloader.dataset.labels if l != -1]
        if labels_in_train:
            class_counts = torch.bincount(torch.tensor(labels_in_train), minlength=num_labels)
            class_weights = torch.zeros(num_labels).to(device)
            nonzero_mask = class_counts > 0
            if nonzero_mask.any():
                class_weights[nonzero_mask] = 1. / (class_counts[nonzero_mask].float() + 1e-8).to(device)
                total_weight = class_weights.sum()
                if total_weight > 0:
                    class_weights = class_weights / total_weight * num_labels
            logger.info(f"使用類別權重: {class_weights.tolist()}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    scaler = GradScaler('cuda', enabled=args.use_amp)

    best_metric = -1.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch + 1} ---")

        model.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with autocast('cuda', enabled=args.use_amp):
                logits = model(**batch)
                loss = loss_fn(logits, labels)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps

        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")

                with autocast('cuda', enabled=args.use_amp):
                    logits = model(**batch)

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        eval_metrics = compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            labels=TASK_TO_LABELS[args.task],
            output_dir=args.output_dir / "temp", # 先存到暫存資料夾
        )

        logger.info(f"Macro-F1: {eval_metrics['macro_f1']:.4f}")

        if eval_metrics["macro_f1"] > best_metric:
            best_metric = eval_metrics["macro_f1"]
            patience_counter = 0

            best_model_dir = args.output_dir / "best_model"
            best_model_dir.mkdir(exist_ok=True)

            model.model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            # 將最佳評估報告移動到 best_model 資料夾
            import shutil
            shutil.copy(args.output_dir / "temp" / "classification_report.txt", best_model_dir / "classification_report.txt")
            shutil.copy(args.output_dir / "temp" / "confusion_matrix.png", best_model_dir / "confusion_matrix.png")

            # 記錄到全域結果檔案
            summary_path = args.output_dir.parent / "all_results.json"
            results = {}
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        results = json.load(f)
                except: pass
            
            # 儲存結構化數據
            results[args.task] = {
                "best_macro_f1": round(float(best_metric), 4),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch + 1,
                "metrics": eval_metrics["report_dict"] # 存入結構化字典
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 同時產生一個漂亮的 Markdown 摘要
            md_path = args.output_dir.parent / "summary.md"
            with open(md_path, "w", encoding="utf-8") as mf:
                mf.write("# 🚀 ESG Model Training Summary\n\n")
                mf.write("| Task | Best Macro-F1 | Best Epoch | Last Updated |\n")
                mf.write("| :--- | :---: | :---: | :--- |\n")
                for t, info in results.items():
                    mf.write(f"| **{t}** | `{info['best_macro_f1']:.4f}` | {info['epoch']} | {info['updated_at']} |\n")
                
                for t, info in results.items():
                    mf.write(f"\n## 📊 {t.capitalize()} Detailed Report\n")
                    mf.write("| Class | Precision | Recall | F1-Score | Support |\n")
                    mf.write("| :--- | :---: | :---: | :---: | :---: |\n")
                    for label, m in info["metrics"].items():
                        if isinstance(m, dict):
                            mf.write(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {int(m['support'])} |\n")

        else:
            patience_counter += 1

        # 清理暫存資料夾
        import shutil
        if (args.output_dir / "temp").exists():
            shutil.rmtree(args.output_dir / "temp")

        if patience_counter >= args.early_stopping_patience:
            logger.info("Early stopping")
            break

    logger.info(f"Best Macro-F1: {best_metric:.4f}")
    logger.info(f"Time: {(time.time() - start_time) / 60:.2f} min")

if __name__ == "__main__":
    args = tyro.cli(TrainingArguments)
    main(args)