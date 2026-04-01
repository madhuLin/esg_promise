import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

class SingleTaskModel(nn.Module):
    """
    用於單一序列分類任務的基礎模型。
    """
    def __init__(self, model_name_or_path: str, num_labels: int, dropout_prob: float = 0.1):
        """
        Args:
            model_name_or_path (str): HuggingFace Hub 上的模型名稱或本地路徑。
            num_labels (int): 分類任務的標籤數量。
            dropout_prob (float): Dropout 的機率。
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            attention_probs_dropout_prob=dropout_prob,
            hidden_dropout_prob=dropout_prob,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None) -> torch.Tensor:
        """
        前向傳播。

        Args:
            input_ids (torch.Tensor): 輸入 token IDs。
            attention_mask (torch.Tensor): attention mask。
            token_type_ids (torch.Tensor, optional): token type IDs。

        Returns:
            torch.Tensor: 模型的 logits 輸出。
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits

class MultiTaskModel(nn.Module):
    """
    共享編碼器的多任務學習模型（進階，此處為擴充框架）。
    """
    def __init__(self, model_name_or_path: str, task_dims: dict, dropout_prob: float = 0.1):
        """
        Args:
            model_name_or_path (str): 共享的 backbone 模型名稱。
            task_dims (dict): 包含每個任務名稱和其標籤數量的字典。
                              例如: {'promise': 2, 'evidence': 2}
            dropout_prob (float): Dropout 機率。
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout_prob)
        
        # 為每個任務建立一個分類頭
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(self.backbone.config.hidden_size, num_labels)
            for task, num_labels in task_dims.items()
        })
        self.task_dims = task_dims

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task: str) -> torch.Tensor:
        """
        前向傳播。在多任務模型中，需要指定當前批次是哪個任務。

        Args:
            input_ids (torch.Tensor): 輸入 token IDs。
            attention_mask (torch.Tensor): attention mask。
            task (str): 當前要計算的任務名稱。

        Returns:
            torch.Tensor: 指定任務的 logits 輸出。
        """
        if task not in self.classifiers:
            raise ValueError(f"未知的任務: {task}。可用任務: {list(self.classifiers.keys())}")

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] token 的表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifiers[task](pooled_output)
        return logits
