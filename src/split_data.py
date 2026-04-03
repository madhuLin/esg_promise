import json
import random
from pathlib import Path

def split_data(input_path: str, train_path: str, val_path: str, ratio: float = 0.8):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 固定隨機種子，確保每次切分結果一致
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"資料切分完成！")
    print(f"總筆數: {len(data)}")
    print(f"訓練集 (Train): {len(train_data)} 筆 -> {train_path}")
    print(f"驗證集 (Val): {len(val_data)} 筆 -> {val_path}")

if __name__ == "__main__":
    split_data(
        "data/vpesg4k_train_1000_V1.json",
        "data/train.json",
        "data/val.json"
    )
