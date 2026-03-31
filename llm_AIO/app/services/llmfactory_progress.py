"""
从 LlamaFactory / HuggingFace Trainer 的 output_dir 读取 trainer_state.json，供训练进度 API 使用。
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def infer_model_display_name(output_dir: str, task_type: str = "") -> str:
    """
    供前端展示用：与任务 output_dir 的末级目录名一致（服务端自动生成时为
    {模型标识}_{lora|qlora|full}_YYYYMMDD_HHMMSS，例如 Qwen3-0.6B_lora_20260323_102510），
    与 GET /trained-models 等列表中的目录名一致，便于对照。task_type 仅保留参数兼容，不参与计算。
    """
    _ = task_type
    base = Path(output_dir).name
    if not base:
        base = os.path.basename(str(output_dir).rstrip("/\\"))
    return base or "unknown"


def load_training_progress(
    output_dir: str,
    *,
    log_history_limit: int = 100,
) -> Dict[str, Any]:
    """
    读取 trainer_state.json，返回扁平字段（不含任务表信息）。
    log_history 仅返回尾部若干条，避免响应过大。
    """
    out: Dict[str, Any] = {
        "trainer_state_found": False,
        "global_step": None,
        "max_steps": None,
        "epoch": None,
        "num_train_epochs": None,
        "latest_loss": None,
        "learning_rate": None,
        "progress_ratio": None,
        "log_history": None,
        "best_model_checkpoint": None,
        "message": None,
    }

    root = Path(output_dir)
    if not root.is_dir():
        out["message"] = "训练输出目录尚不存在或已被移除"
        return out

    state_file = root / "trainer_state.json"
    if not state_file.is_file():
        out["message"] = "尚未生成 trainer_state.json（训练未写入进度或刚开始）"
        return out

    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        out["message"] = "无法解析 trainer_state.json: %s" % str(e)
        return out

    if not isinstance(state, dict):
        out["message"] = "trainer_state.json 格式异常"
        return out

    out["trainer_state_found"] = True
    out["global_step"] = state.get("global_step")
    out["max_steps"] = state.get("max_steps")
    out["epoch"] = state.get("epoch")
    out["best_model_checkpoint"] = state.get("best_model_checkpoint")

    log_history: List[Dict[str, Any]] = state.get("log_history") or []
    if not isinstance(log_history, list):
        log_history = []

    ta_path = root / "training_args.json"
    if ta_path.is_file():
        try:
            with open(ta_path, "r", encoding="utf-8") as tf:
                ta = json.load(tf)
            if isinstance(ta, dict) and ta.get("num_train_epochs") is not None:
                out["num_train_epochs"] = ta.get("num_train_epochs")
        except (json.JSONDecodeError, OSError):
            pass

    limit = max(0, int(log_history_limit))
    tail: List[Dict[str, Any]] = log_history[-limit:] if limit else list(log_history)
    out["log_history"] = tail

    for entry in reversed(log_history):
        if isinstance(entry, dict) and "loss" in entry:
            out["latest_loss"] = entry.get("loss")
            out["learning_rate"] = entry.get("learning_rate")
            break

    gs = out["global_step"]
    ms = out["max_steps"]
    if ms is not None and isinstance(ms, (int, float)) and int(ms) > 0 and isinstance(gs, (int, float)):
        out["progress_ratio"] = min(1.0, float(gs) / float(ms))

    if out["progress_ratio"] is None:
        nte = out["num_train_epochs"]
        ep = out["epoch"]
        if nte is not None and isinstance(nte, (int, float)) and float(nte) > 0:
            if ep is not None and isinstance(ep, (int, float)):
                out["progress_ratio"] = min(1.0, float(ep) / float(nte))

    return out
