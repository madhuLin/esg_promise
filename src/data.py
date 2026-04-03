import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from configs.label_maps import LABEL_MAPS, TASK_TO_LABELS

logger = logging.getLogger(__name__)

Task = Literal["promise", "evidence", "clarity", "timeline"]

def load_data_from_json(path: Path) -> List[Dict[str, Any]]:
    """從 JSON 檔案載入資料列表。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"無法從 {path} 載入資料: {e}")
        raise

def format_input_text(sample: Dict[str, Any], task: Task) -> str:
    """
    根據任務類型，格式化輸入文字。

    Args:
        sample (Dict[str, Any]): 一筆資料樣本。
        task (Task): 任務類型。

    Returns:
        str: 格式化後的輸入文字。
    """
    if task == "promise":
        return sample["data"]
    elif task == "evidence":
        return sample["data"]
    elif task == "clarity":
        promise_str = sample.get("promise_string", "")
        evidence_str = sample.get("evidence_string", "")
        return f"[PROMISE] {promise_str} [EVIDENCE] {evidence_str}"
    elif task == "timeline":
        return sample.get("promise_string") or sample["data"]
    else:
        raise ValueError(f"未知的任務類型: {task}")

def filter_data_for_task(
    data: List[Dict[str, Any]], task: Task
) -> List[Dict[str, Any]]:
    """
    根據訓練時的特定規則過濾資料。

    Args:
        data (List[Dict[str, Any]]): 原始資料列表。
        task (Task): 任務類型。

    Returns:
        List[Dict[str, Any]]: 過濾後的資料列表。
    """
    if task == "promise":
        # promise 任務使用所有資料
        return data
    elif task == "evidence":
        # evidence 任務只在 promise_status == 'Yes' 的樣本上訓練
        return [s for s in data if s.get("promise_status") == "Yes"]
    elif task == "clarity":
        # clarity 任務只在 promise_status == 'Yes' 且 evidence_status == 'Yes' 的樣本上訓練
        return [
            s
            for s in data
            if s.get("promise_status") == "Yes" and s.get("evidence_status") == "Yes"
        ]
    elif task == "timeline":
        # timeline 任務只在 promise_status == 'Yes' 的樣本上訓練
        return [s for s in data if s.get("promise_status") == "Yes"]
    else:
        raise ValueError(f"未知的任務類型: {task}")

class ESGDataset(Dataset):
    """ESG 承諾驗證的 PyTorch 資料集。"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        task: Task,
        max_length: int = 512,
        for_training: bool = True,
        filter: bool = True, # 新增參數控制是否過濾
    ):
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        self.for_training = for_training
        
        # 根據參數決定是否過濾
        if filter:
            self.data = filter_data_for_task(data, task)
        else:
            self.data = data
            
        if not self.data:
            logger.warning(f"任務 '{task}' 在處理後沒有任何資料。")
            
        self.label_map = LABEL_MAPS["label_to_id"][task]

        self.texts = []
        self.labels = []

        task_key_map = {
            "promise": "promise_status",
            "evidence": "evidence_status",
            "clarity": "evidence_quality",
            "timeline": "verification_timeline",
        }
        label_key = task_key_map[task]

        for sample in self.data:
            self.texts.append(format_input_text(sample, task))
            # 無論是否為訓練，只要有標籤就載入，方便評估
            label = sample.get(label_key)
            if label in self.label_map:
                self.labels.append(self.label_map[label])
            else:
                # 推論時若無標籤則填入 -1
                self.labels.append(-1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].flatten()

        # 只要 labels 不是 -1 就回傳
        label = self.labels[idx]
        if label != -1:
            item["labels"] = torch.tensor(label, dtype=torch.long)
        
        return item

def create_dataloader(
    path: Path,
    tokenizer: AutoTokenizer,
    task: Task,
    batch_size: int,
    max_length: int,
    shuffle: bool = False,
    for_training: bool = True,
    num_workers: int = 4,
    balance: bool = False,
    filter: bool = True # 新增參數
) -> Optional[DataLoader]:
    """
    建立一個 DataLoader。
    """
    raw_data = load_data_from_json(path)
    dataset = ESGDataset(raw_data, tokenizer, task, max_length, for_training, filter=filter)

    if not dataset.data:
        logger.warning(f"無法為 {path} 建立 DataLoader，因為沒有可用資料。")
        return None

    sampler = None
    if for_training and balance:
        # 計算樣本權重以進行平衡採樣
        labels = torch.tensor(dataset.labels)
        if len(labels) > 0:
            num_labels = len(dataset.label_map)
            class_counts = torch.bincount(labels, minlength=num_labels)
            
            # 建立類別權重，處理數量為 0 的類別
            class_weights = torch.zeros(num_labels)
            nonzero_mask = class_counts > 0
            class_weights[nonzero_mask] = 1. / class_counts[nonzero_mask].float()
            
            sample_weights = class_weights[labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False  # 使用 sampler 時不能開啟 shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
