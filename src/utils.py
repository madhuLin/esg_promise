import random
import numpy as np
import torch
import logging
import sys
from pathlib import Path
from typing import Optional

def set_seed(seed: int = 42):
    """
    固定隨機種子以確保實驗可重複性。

    Args:
        seed (int): 隨機種子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir: Optional[Path] = None):
    """
    設定日誌記錄器，可同時輸出到控制台和檔案。

    Args:
        log_dir (Optional[Path]): 日誌檔案儲存的目錄。如果提供，
                                  則會建立名為 'run.log' 的日誌檔案。
    """
    # 設置基本配置，這會影響 root logger
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 清除所有現有的 handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "run.log"))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
