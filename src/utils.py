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
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger = logging.getLogger("esg_promise")
    logger.setLevel(logging.INFO)
    
    # 清除已經存在的 handlers，避免重複記錄
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台輸出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 檔案輸出
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    # 避免將日誌傳播到 root logger
    logger.propagate = False
