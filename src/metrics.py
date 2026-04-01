import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from typing import List, Dict

logger = logging.getLogger(__name__)

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str],
    output_dir: Path,
) -> Dict[str, float]:
    """
    計算評估指標，包括 Macro-F1、分類報告和混淆矩陣。

    Args:
        y_true (List[int]): 真實標籤列表。
        y_pred (List[int]): 預測標籤列表。
        labels (List[str]): 標籤名稱列表。
        output_dir (Path): 儲存報告和圖表的目錄。

    Returns:
        Dict[str, float]: 包含 macro_f1 的字典。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 計算 Macro-F1
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # 產生分類報告
    report = classification_report(
        y_true, y_pred, target_names=labels, digits=4, output_dict=False
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=labels, digits=4, output_dict=True
    )
    
    logger.info(f"分類報告:\n{report}")
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 產生並儲存混淆矩陣
    try:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"繪製混淆矩陣時發生錯誤: {e}")

    results = {"macro_f1": macro_f1}
    # 添加每個類別的 f1-score
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict) and "f1-score" in metrics:
            results[f"{label}_f1"] = metrics["f1-score"]

    return results
