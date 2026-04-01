from typing import Dict, List, Any

# 統一定義四個任務的標籤
# 這有助於確保標籤名稱的一致性
TASK_TO_LABELS: Dict[str, List[str]] = {
    "promise": ["No", "Yes"],
    "evidence": ["No", "Yes"],
    "clarity": ["Clear", "Not Clear", "Misleading"],
    "timeline": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"],
}

# N/A 標籤
NA_LABEL = "N/A"

def get_label_maps() -> Dict[str, Any]:
    """
    生成標籤到索引以及索引到標籤的映射。

    Returns:
        Dict[str, Any]: 包含 label_to_id 和 id_to_label 的字典。
    """
    label_to_id: Dict[str, Dict[str, int]] = {}
    id_to_label: Dict[str, Dict[int, str]] = {}

    for task, labels in TASK_TO_LABELS.items():
        label_to_id[task] = {label: i for i, label in enumerate(labels)}
        id_to_label[task] = {i: label for i, label in enumerate(labels)}

    return {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }

LABEL_MAPS = get_label_maps()

if __name__ == "__main__":
    # 執行此文件以檢查標籤映射是否正確
    import json
    print("--- Label to ID maps ---")
    print(json.dumps(LABEL_MAPS["label_to_id"], indent=2, ensure_ascii=False))
    print("\n--- ID to Label maps ---")
    print(json.dumps(LABEL_MAPS["id_to_label"], indent=2, ensure_ascii=False))
