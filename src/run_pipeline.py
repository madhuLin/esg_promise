import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro

from configs.label_maps import NA_LABEL
from src.utils import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineArguments:
    """完整預測流程的參數。"""
    test_data_path: Path = tyro.conf.arg(help="原始測試資料的 JSON 檔案路徑。")
    output_path: Path = tyro.conf.arg(help="儲存最終預測結果的 JSON 檔案路徑。")
    
    # 四個任務的模型路徑
    promise_model_path: Path = tyro.conf.arg(help="Promise 任務的模型路徑。")
    evidence_model_path: Path = tyro.conf.arg(help="Evidence 任務的模型路徑。")
    clarity_model_path: Path = tyro.conf.arg(help="Clarity 任務的模型路徑。")
    timeline_model_path: Path = tyro.conf.arg(help="Timeline 任務的模型路徑。")
    
    batch_size: int = 32

def run_single_task_prediction(
    task: str,
    model_path: Path,
    input_path: Path,
    output_path: Path,
    batch_size: int
):
    """執行單一任務的預測子程序。"""
    cmd = [
        "python", "-m", "src.predict",
        "--task", task,
        "--model-path", str(model_path),
        "--input-path", str(input_path),
        "--output-path", str(output_path),
        "--batch-size", str(batch_size),
    ]
    logger.info(f"執行指令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        logger.error(f"'{task}' 任務預測失敗。")
        logger.error("stdout:\n" + result.stdout)
        logger.error("stderr:\n" + result.stderr)
        raise RuntimeError(f"'{task}' 任務預測失敗。")
    
    logger.info(f"'{task}' 任務預測成功，結果儲存於 {output_path}")

def apply_hierarchical_logic(data: list) -> list:
    """
    在預測結果上應用階層式約束。

    Args:
        data (list): 包含所有預測結果的資料列表。

    Returns:
        list: 套用邏輯後的資料列表。
    """
    for sample in data:
        # 規則 1: 如果 promise = No，則 evidence / clarity / timeline = N/A
        if sample.get("predicted_promise") == "No":
            sample["predicted_evidence"] = NA_LABEL
            sample["predicted_clarity"] = NA_LABEL
            sample["predicted_timeline"] = NA_LABEL
        # 規則 2: 如果 evidence = No (且 promise = Yes)，則 clarity = N/A
        elif sample.get("predicted_evidence") == "No":
            sample["predicted_clarity"] = NA_LABEL
    return data

def main(args: PipelineArguments):
    """執行完整的階層式預測流程。"""
    setup_logging()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # --- 步驟 1: 預測 Promise Status ---
        promise_preds_path = temp_path / "1_promise_preds.json"
        run_single_task_prediction(
            task="promise",
            model_path=args.promise_model_path,
            input_path=args.test_data_path,
            output_path=promise_preds_path,
            batch_size=args.batch_size
        )
        
        # --- 步驟 2: 預測 Evidence Status 和 Timeline ---
        # 這兩個任務都依賴 promise=Yes，可以並行，但此處為簡化採序列執行
        evidence_preds_path = temp_path / "2_evidence_preds.json"
        run_single_task_prediction(
            task="evidence",
            model_path=args.evidence_model_path,
            input_path=promise_preds_path, # 使用上一階段的輸出
            output_path=evidence_preds_path,
            batch_size=args.batch_size
        )
        
        timeline_preds_path = temp_path / "3_timeline_preds.json"
        run_single_task_prediction(
            task="timeline",
            model_path=args.timeline_model_path,
            input_path=evidence_preds_path, # 使用上一階段的輸出
            output_path=timeline_preds_path,
            batch_size=args.batch_size
        )
        
        # --- 步驟 3: 預測 Clarity (Evidence Quality) ---
        clarity_preds_path = temp_path / "4_clarity_preds.json"
        run_single_task_prediction(
            task="clarity",
            model_path=args.clarity_model_path,
            input_path=timeline_preds_path, # 使用上一階段的輸出
            output_path=clarity_preds_path,
            batch_size=args.batch_size
        )
        
        # --- 步驟 4: 載入最終結果並應用階層邏輯 ---
        logger.info("所有任務預測完成，正在套用階層邏輯...")
        with open(clarity_preds_path, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
            
        processed_data = apply_hierarchical_logic(final_data)
        
        # 清理欄位，只保留 id, data, 和最終預測
        cleaned_data = []
        for sample in processed_data:
            cleaned_sample = {
                "id": sample.get("id"),
                "data": sample.get("data"),
                "promise_status": sample.get("predicted_promise"),
                "evidence_status": sample.get("predicted_evidence"),
                "evidence_quality": sample.get("predicted_clarity"),
                "verification_timeline": sample.get("predicted_timeline"),
            }
            cleaned_data.append(cleaned_sample)

        # 儲存最終結果
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"完整預測流程結束，最終結果已儲存至 {args.output_path}")

if __name__ == "__main__":
    tyro.cli(main)
