"""
llmfactory 集成接口：训练、合并、推理 API 等。
推理 API 通过主服务 8000 代理，统一入口为 /api/playground/llmfactory/v1/*
"""
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.database import get_convert_db, ConvertSessionLocal
from app.datetime_utils import serialize_datetime_for_api_response, utc_now
from app.schemas import (
    LoraTrainRequest,
    QLoraTrainRequest,
    FullTrainRequest,
    MergeAdapterRequest,
    StartInferenceApiRequest,
    LlmFactoryTaskResponse,
    TrainingJobItem,
    TrainingProgressResponse,
    QLoRASupportCheckRequest,
    QLoRASupportCheckResponse,
)
from app.services import llmfactory_service
from app.services import llmfactory_progress
from app.services import gpu_vram_guard
from app.config import get_settings
from convert_url import TrainingJob, DatasetMetadata
from app.services.qlora_compat import check_qlora_support

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/playground/llmfactory", tags=["llmfactory"])


def _inference_state_file() -> str:
    """
    推理子进程状态落盘文件路径。
    用于跨多 worker / --reload / 重启后仍能 stop/proxy 到正确的端口与 PID。
    """
    return os.path.join(tempfile.gettempdir(), "llm_aio_llmfactory_inference_state.json")


def _load_inference_state() -> dict:
    try:
        p = _inference_state_file()
        if not os.path.isfile(p):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_inference_state(state: dict) -> None:
    try:
        p = _inference_state_file()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state or {}, f, ensure_ascii=False)
        os.replace(tmp, p)
    except Exception:
        # 落盘失败不应影响主流程
        pass


def _clear_inference_state() -> None:
    try:
        p = _inference_state_file()
        if os.path.isfile(p):
            os.remove(p)
    except Exception:
        pass


def _ensure_llmfactory_heavy_gpu_budget() -> None:
    """
    若配置了 LLMFACTORY_MIN_FREE_VRAM_MIB>0，在训练/合并/推理启动前检查 GPU 空闲显存；
    不足则 HTTP 503。查询不到时：严格模式拒绝，否则跳过（无 NVIDIA 环境可继续）。
    """
    s = get_settings()
    min_mib = int(getattr(s, "LLMFACTORY_MIN_FREE_VRAM_MIB", 0) or 0)
    gpu_index = int(getattr(s, "LLMFACTORY_GPU_VRAM_GUARD_INDEX", 0) or 0)
    strict = str(getattr(s, "LLMFACTORY_GPU_VRAM_GUARD_STRICT", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    gpu_vram_guard.llmfactory_check_min_free_vram(
        min_free_mib=min_mib,
        gpu_index=gpu_index,
        strict_when_unknown=strict,
    )


def _get_llmfactory_config():
    """从配置获取 llmfactory 相关设置"""
    settings = get_settings()
    config = {}
    cmd_prefix = getattr(settings, "LLMFACTORY_COMMAND_PREFIX", None)
    if cmd_prefix:
        config["command_prefix"] = cmd_prefix
    return config


def _get_data_type_by_dataset_id(db, dataset_id: Optional[str]) -> Optional[str]:
    """根据 dataset_id 查询训练集的 data_type（用于模型类型：用什么数据集训练的）。"""
    if not dataset_id or not db:
        return None
    meta = db.query(DatasetMetadata).filter_by(id=dataset_id).first()
    return meta.data_type if meta else None


def _sanitize_dirname(name: str) -> str:
    """将模型名等整理为合法目录名：去掉/替换非法字符。"""
    if not name:
        return "model"
    s = re.sub(r'[/\\:*?"<>|\s]+', "_", name.strip())
    return s or "model"


def _resolve_train_params(
    request: Union[LoraTrainRequest, QLoraTrainRequest, FullTrainRequest],
    db=None,
    task_type: str = None,
):
    """
    解析训练参数。未传 output_dir 且传入 task_type 时，自动生成：
    训练输出根目录 / {模型名}_{训练方式}_{时间戳}，例如 train_output/Qwen3-0.6B_lora_20260310_083000
    """
    settings = get_settings()
    if getattr(request, "model_id", None):
        model_path = _resolve_model_id(request.model_id)
    else:
        model_path = request.model_path or getattr(settings, "LLMFACTORY_DEFAULT_MODEL_PATH", "") or None
    dataset = request.dataset or getattr(settings, "LLMFACTORY_DEFAULT_DATASET", "") or None
    dataset_dir = request.dataset_dir or getattr(settings, "LLMFACTORY_DEFAULT_DATASET_DIR", "") or None
    output_dir = request.output_dir or getattr(settings, "LLMFACTORY_DEFAULT_OUTPUT_DIR", "") or None
    output_dir = (output_dir or "").strip() or None

    # 若传入 dataset_id，从已上传训练集解析 dataset_dir 和 dataset
    if getattr(request, "dataset_id", None) and db:
        from app.services import dataset_service
        dd, dn = dataset_service.prepare_dataset_for_training(db, request.dataset_id)
        if dd and dn:
            dataset_dir, dataset = dd, dn
        else:
            raise HTTPException(status_code=400, detail="dataset_id 无效或数据集文件不可用")

    missing = []
    if not model_path:
        missing.append("model_path")
    if not dataset:
        missing.append("dataset 或 dataset_id")
    if not dataset_dir:
        missing.append("dataset_dir 或 dataset_id")
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"缺少必要参数（请传入 model_id 或 model_path，以及 dataset_id 或 dataset/dataset_dir，或在 .env 中配置 LLMFACTORY_DEFAULT_*）: {', '.join(missing)}",
        )

    base = _ensure_training_output_base()  # train_output/SFT
    if output_dir and not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(os.getcwd(), output_dir))
    # 统一：训练结果必须落在「SFT/模型名_训练方式_时间戳」子目录下，不直接写在根目录或 SFT 根下
    if output_dir and task_type:
        try:
            real_out = os.path.realpath(output_dir)
            real_base = os.path.realpath(base)
            real_root = os.path.realpath(_get_training_output_base())
            if real_out == real_base or real_out == real_root:
                output_dir = None  # 用户只传了根目录或 SFT 根，改为自动生成子目录
        except OSError:
            pass
    if not output_dir and task_type:
        model_name = _sanitize_dirname(
            getattr(request, "model_id", None) or (os.path.basename(model_path.rstrip(os.sep)) if model_path else "model")
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base, f"{model_name}_{task_type}_{ts}")

    if not output_dir:
        raise HTTPException(
            status_code=400,
            detail="缺少 output_dir（未传且未指定 task_type 时无法自动生成）",
        )
    return model_path, dataset, dataset_dir, output_dir


def _resolve_adapter_path_safe(adapter_path: str, db=None) -> str:
    """
    将 adapter_path 解析为仅允许「微调结果保存目录」下的路径，禁止任意服务器路径。
    - 若为 task_id（在 training_jobs 中存在）：使用该任务的 output_dir（且必须在 train_output 下）。
    - 否则视为「目录名」或相对标识：解析为 train_output/SFT/{目录名}，禁止 .. 与绝对路径穿越。
    - 若传入的已是绝对路径：仅当其实路径落在 train_output 下时才允许，否则 400。
    返回解析后的绝对路径；若非法则抛出 HTTPException。
    """
    adapter_path = (adapter_path or "").strip()
    if not adapter_path:
        raise HTTPException(status_code=400, detail="缺少 adapter_path（请传入 task_id 或 GET /trained-models 返回的目录名，如 Qwen2-0.5B_lora_20260311_142141）")

    root = _get_training_output_base()
    sft_base = _get_sft_output_base()

    # 1) 尝试按 task_id 查找任务记录，使用其 output_dir
    if db:
        job = db.query(TrainingJob).filter_by(id=adapter_path).first()
        if job and job.output_dir:
            real = os.path.normpath(os.path.abspath(job.output_dir))
            if not (real == root or real.startswith(root + os.sep)):
                raise HTTPException(
                    status_code=400,
                    detail="该任务 output_dir 不在允许的 train_output 目录下，无法用于合并",
                )
            if not os.path.isdir(real):
                raise HTTPException(status_code=404, detail="该任务对应的输出目录不存在或已被删除")
            return real

    # 2) 非绝对路径：视为「目录名」，解析到 train_output/SFT/{目录名}（禁止 ..）
    if not os.path.isabs(adapter_path):
        # 只取最后一段，防止 SFT/../etc 之类
        name = os.path.basename(adapter_path.replace("\\", "/"))
        if not name or name in (".", ".."):
            raise HTTPException(status_code=400, detail="adapter_path 请传入有效的目录名或 task_id")
        resolved = os.path.normpath(os.path.join(sft_base, _sanitize_dirname(name)))
        # 由 sft_base + name 拼接，必在 SFT 下；仅校验目录存在
        if not os.path.isdir(resolved):
            raise HTTPException(
                status_code=404,
                detail="未找到该微调结果目录（train_output/SFT/%s），请确认目录名或使用 GET /trained-models 查看" % name,
            )
        return resolved

    # 3) 绝对路径：仅当在 train_output 下才允许
    real = os.path.normpath(os.path.realpath(adapter_path))
    if not (real == root or real.startswith(root + os.sep)):
        raise HTTPException(
            status_code=400,
            detail="adapter_path 仅允许为 task_id、或 GET /trained-models 中的目录名，或 train_output 下的路径；禁止传入其他服务器路径",
        )
    if not os.path.isdir(real):
        raise HTTPException(status_code=404, detail="指定的适配器目录不存在")
    return real


def _resolve_merge_params(request: MergeAdapterRequest, db=None):
    """
    解析合并参数。model_id 与 model_path 二选一。
    adapter_path 仅允许：task_id（对应任务 output_dir）、或微调目录名（解析到 train_output/SFT/xxx）、或 train_output 下的路径；禁止任意服务器路径。
    export_dir 不传时默认为 train_output/merged/{适配器目录名}_merged。
    """
    settings = get_settings()
    if request.model_id:
        model_path = _resolve_model_id(request.model_id)
    else:
        model_path = request.model_path or getattr(settings, "LLMFACTORY_DEFAULT_MODEL_PATH", "") or None
    raw_adapter = request.adapter_path or getattr(settings, "LLMFACTORY_DEFAULT_ADAPTER_PATH", "") or None
    export_dir = (request.export_dir or "").strip() or getattr(settings, "LLMFACTORY_DEFAULT_EXPORT_DIR", "") or None

    if not model_path:
        raise HTTPException(status_code=400, detail="缺少 model_id 或 model_path（请传入或在 .env 配置 LLMFACTORY_DEFAULT_MODEL_PATH）")

    adapter_path = _resolve_adapter_path_safe(raw_adapter, db=db)

    if not export_dir:
        root = _get_training_output_base()
        adapter_basename = os.path.basename(adapter_path.rstrip(os.sep)) or "adapter"
        merged_name = _sanitize_dirname(adapter_basename) + "_merged"
        export_dir = os.path.join(root, MERGED_SUBDIR, merged_name)
        os.makedirs(os.path.dirname(export_dir), exist_ok=True)

    if export_dir and not os.path.isabs(export_dir):
        export_dir = os.path.normpath(os.path.join(os.getcwd(), export_dir))

    return model_path, adapter_path, export_dir


def _resolve_inference_params(request: StartInferenceApiRequest):
    """解析推理 API 启动参数。model_id 与 model_path 二选一，model_id 在服务端解析为路径。"""
    settings = get_settings()
    if request.model_id:
        model_path = _resolve_model_id(request.model_id)
    else:
        model_path = request.model_path or getattr(settings, "LLMFACTORY_DEFAULT_MODEL_PATH", "") or None
    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="缺少 model_id 或 model_path（请传入或在 .env 中配置 LLMFACTORY_DEFAULT_MODEL_PATH）",
        )
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail="model_path not exists!",
        )
    return model_path


def _is_model_dir(dir_path: str) -> bool:
    """判断目录是否为 Hugging Face / llm-factory 模型目录（含 config.json）"""
    return os.path.isfile(os.path.join(dir_path, "config.json"))


def _read_model_type(config_path: str) -> str:
    """从 config.json 读取 model_type（若有），用于前端展示，不暴露路径。"""
    try:
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("model_type") or (data.get("architectures") or [None])[0] or "") or ""
    except Exception:
        return ""


# model_type（或 architectures[0]）到 LLaMA-Factory template 的默认映射，用于推理时「按模型自动选模板」
MODEL_TYPE_TO_TEMPLATE = {
    "qwen2": "qwen",
    "qwen2forcausallm": "qwen",
    "qwen3": "qwen3_nothink",
    "qwen3forcausallm": "qwen3_nothink",
    "qwen": "qwen",
    "llama2": "llama2",
    "llama3": "llama3",
    "llama4": "llama4",
    "llama2forcausallm": "llama2",
    "llama3forcausallm": "llama3",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "deepseek3": "deepseek3",
    "deepseekr1": "deepseekr1",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma3",
    "glm4": "glm4",
    "yi": "llama3",
    "qwen2_vl": "qwen2_vl",
    "qwen3_vl": "qwen3_vl",
}


def _infer_template_for_model(model_path: str) -> str:
    """
    根据模型目录下 config.json 的 model_type/architectures 推断默认 template。
    若无法推断或不在映射表，返回 qwen3_nothink；调用方仍可传显式 template 覆盖。
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return "qwen3_nothink"
    raw = _read_model_type(config_path).strip()
    if not raw:
        return "qwen3_nothink"
    key = raw.lower().replace("-", "_")
    if key in MODEL_TYPE_TO_TEMPLATE:
        return MODEL_TYPE_TO_TEMPLATE[key]
    for prefix, template in MODEL_TYPE_TO_TEMPLATE.items():
        if key.startswith(prefix) or prefix in key:
            return template
    return "qwen3_nothink"


def _get_local_models_map():
    """
    扫描 LLMFACTORY_MODELS_DIR 下有效模型，返回：
    - 供列表接口用的元数据列表 [{id, name, model_type}]，不包含路径；
    - id -> 绝对路径 的映射，仅用于服务端根据 model_id 解析路径。
    """
    settings = get_settings()
    models_dir = getattr(settings, "LLMFACTORY_MODELS_DIR", "") or ""
    meta_list: list[dict] = []
    id_to_path: dict[str, str] = {}
    if not models_dir or not os.path.isdir(models_dir):
        return meta_list, id_to_path
    seen_realpath: dict[str, tuple[str, str]] = {}
    try:
        for name in sorted(os.listdir(models_dir)):
            if name.startswith("."):
                continue
            path = os.path.join(models_dir, name)
            if not os.path.isdir(path) or not _is_model_dir(path):
                continue
            real_path = os.path.realpath(path)
            abs_path = os.path.abspath(path)
            if real_path not in seen_realpath:
                seen_realpath[real_path] = (name, abs_path)
            elif os.path.islink(path):
                seen_realpath[real_path] = (name, abs_path)
    except OSError:
        return meta_list, id_to_path
    for name, abs_path in sorted(seen_realpath.values(), key=lambda x: x[0]):
        config_path = os.path.join(abs_path, "config.json")
        model_type = _read_model_type(config_path) if os.path.isfile(config_path) else ""
        meta_list.append({"id": name, "name": name, "model_type": model_type})
        id_to_path[name] = abs_path
    return meta_list, id_to_path


def _resolve_model_id(model_id: str) -> str:
    """
    根据 model_id 解析为服务器本地路径，仅后端使用。
    解析顺序：
    1) 优先在 LLMFACTORY_MODELS_DIR 下查找（GET /models 的来源）
    2) 若未找到，再尝试视为合并后模型目录名：train_output/merged/{model_id}
    未找到则抛出 HTTPException。
    """
    _, id_to_path = _get_local_models_map()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id 不能为空")

    # 1) 普通模型（来自 LLMFACTORY_MODELS_DIR）
    path = id_to_path.get(model_id)
    if path:
        return path

    # 2) 尝试作为「合并后模型 id」（来自 GET /merged-models 的 id/name）
    merged_base = _get_merged_output_base()
    candidate = os.path.normpath(os.path.join(merged_base, model_id))
    try:
        if os.path.isdir(candidate):
            return candidate
    except OSError:
        pass

    raise HTTPException(
        status_code=400,
        detail=(
            f"未找到模型 id: {model_id}，请使用 GET /api/playground/llmfactory/models "
            f"或 GET /api/playground/llmfactory/merged-models 返回的 id"
        ),
    )


@router.get("/models", response_model=dict)
async def list_local_models():
    """
    列出本地已下载的模型（llm-factory 可用）。
    仅返回抽象元数据：id（用于训练/合并/推理时传参）、name（展示名）、model_type（来自 config.json）。
    不返回任何服务器物理路径，前端用 id 调用其它接口即可。
    """
    settings = get_settings()
    models_dir = getattr(settings, "LLMFACTORY_MODELS_DIR", "") or ""
    if not models_dir:
        return {"models": [], "message": "请在 .env 中配置 LLMFACTORY_MODELS_DIR（本地模型文件根目录）"}
    if not os.path.isdir(models_dir):
        return {"models": [], "message": "配置的模型目录不存在或不可读"}
    try:
        meta_list, _ = _get_local_models_map()
        return {"models": meta_list, "message": ""}
    except OSError as e:
        return {"models": [], "message": f"读取目录失败: {e!s}"}


@router.post("/models/qlora-support", response_model=QLoRASupportCheckResponse)
async def check_model_qlora_support(body: QLoRASupportCheckRequest):
    """
    检查某个模型是否可用 QLoRA 训练。
    - 若模型已是 GPTQ/AWQ/AQLM/GGUF 等量化/特定推理格式，通常不应再启用 QLoRA（再量化）。
    """
    model_path = None
    if body.model_id:
        model_path = _resolve_model_id(body.model_id)
    else:
        model_path = body.model_path
    if not model_path:
        raise HTTPException(status_code=400, detail="缺少 model_id 或 model_path")
    result = check_qlora_support(model_path)
    return QLoRASupportCheckResponse(**result.to_dict())


@router.get("/models/qlora-support", response_model=dict)
async def list_models_qlora_support():
    """
    扫描本地模型目录（GET /models 同源），返回每个模型是否支持 QLoRA 及原因。
    """
    try:
        meta_list, id_to_path = _get_local_models_map()
        items = []
        unsupported = []
        for m in meta_list:
            mid = m["id"]
            path = id_to_path.get(mid) or ""
            r = check_qlora_support(path)
            item = {"id": mid, "name": m.get("name", mid), "model_type": m.get("model_type", ""), **r.to_dict()}
            items.append(item)
            if not r.supported:
                unsupported.append(
                    {"id": mid, "name": m.get("name", mid), "model_type": m.get("model_type", ""), "reasons": r.reasons}
                )

        return {
            "models": items,
            "unsupported_models": unsupported,
            "message": "supported=false 一般表示模型已是量化/特定推理格式（如 GPTQ/AWQ/AQLM/GGUF），不建议/不支持再做 QLoRA。",
        }
    except OSError as e:
        return {"models": [], "unsupported_models": [], "message": f"读取目录失败: {e!s}"}
    except Exception as e:
        return {"models": [], "unsupported_models": [], "message": f"发生未知错误: {e!s}"}


@router.get("/templates", response_model=dict)
async def list_llmfactory_templates():
    """
    列出微调/合并可用的 template（与 LlamaFactory 的 template_register 一致）。
    训练、合并、推理请求中的 template 必须在此列表中；常用如 qwen3_nothink、qwen3、llama3 等。
    """
    templates = llmfactory_service.get_llmfactory_templates()
     # 过滤掉占位项 "-"（LlamaFactory 中表示无指定模板），便于前端展示
    templates = [t for t in templates if t and t != "-"]
    return {"templates": templates, "default": "qwen3_nothink"}


def _update_training_job_status(job_id: str, status: str, error_message: str = None):
    """在后台任务中更新训练任务状态（使用独立 session）"""
    db = ConvertSessionLocal()
    try:
        job = db.query(TrainingJob).filter_by(id=job_id).first()
        if job:
            job.status = status
            if error_message is not None:
                job.error_message = error_message[:4096] if error_message else None
            db.commit()
    except Exception as e:
        logger.exception(f"更新训练任务状态失败: {e}")
        db.rollback()
    finally:
        db.close()


# 微调输出放在 train_output/SFT 下；合并输出放在 train_output/merged 下
SFT_SUBDIR = "SFT"
MERGED_SUBDIR = "merged"


def _get_training_output_base() -> str:
    """返回训练结果根目录（用于校验、下载允许范围等）。未配置时默认为当前目录下的 train_output。"""
    settings = get_settings()
    base = (
        getattr(settings, "LLMFACTORY_OUTPUT_BASE", "") or
        getattr(settings, "LLMFACTORY_DEFAULT_OUTPUT_DIR", "") or
        "train_output"
    )
    base = os.path.realpath(os.path.abspath(base))
    return base


def _get_sft_output_base() -> str:
    """返回微调输出根目录 train_output/SFT，若不存在则创建。用于生成训练 output_dir 与列出已训练模型。"""
    root = _get_training_output_base()
    sft_base = os.path.join(root, SFT_SUBDIR)
    os.makedirs(sft_base, exist_ok=True)
    return sft_base


def _ensure_training_output_base() -> str:
    """返回微调输出根目录 train_output/SFT，若不存在则创建（与 _get_sft_output_base 一致，供列表/解析使用）。"""
    return _get_sft_output_base()


def _get_merged_output_base() -> str:
    """返回合并输出根目录 train_output/merged，若不存在则创建。"""
    root = _get_training_output_base()
    merged_base = os.path.join(root, MERGED_SUBDIR)
    os.makedirs(merged_base, exist_ok=True)
    return merged_base


def _is_output_dir_allowed(output_dir: str) -> bool:
    """校验 output_dir 是否在允许的根目录下，避免路径穿越。"""
    if not output_dir or not os.path.isdir(output_dir):
        return False
    base = _get_training_output_base()
    try:
        real_out = os.path.realpath(output_dir)
        return real_out == base or real_out.startswith(base + os.sep)
    except OSError:
        return False


def _is_output_path_under_training_base(output_dir: str) -> bool:
    """
    规范化路径是否落在训练输出根目录下（目录可以尚未创建）。
    用于进度查询等只读接口；下载仍使用 _is_output_dir_allowed（要求目录存在）。
    """
    if not output_dir or not str(output_dir).strip():
        return False
    base = os.path.realpath(_get_training_output_base())
    try:
        candidate = os.path.realpath(os.path.abspath(output_dir))
    except OSError:
        candidate = os.path.normpath(os.path.abspath(output_dir))
    sep = os.sep
    return candidate == base or candidate.startswith(base + sep)


def _run_lora_train_task(model_path, dataset, dataset_dir, output_dir, request: LoraTrainRequest, job_id: str = None):
    """后台执行 LoRA 训练。失败时将 stderr 写入任务 error_message。"""
    try:
        cfg = _get_llmfactory_config()
        result = llmfactory_service.sft_lora_train(
            model_path=model_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            template=request.template,
            lora_rank=request.lora_rank,
            lora_target=request.lora_target,
            learning_rate=request.learning_rate,
            num_train_epochs=request.num_train_epochs,
            bf16=request.bf16,
            deepspeed_config=request.deepspeed_config,
            config=cfg,
        )
        code = result[0] if isinstance(result, tuple) else result
        stderr_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        logger.info(f"LoRA train finished with return_code={code}")
        # 训练失败时清理输出目录，避免留下失败任务的空/不完整目录
        if code != 0 and output_dir and os.path.isdir(output_dir):
            try:
                shutil.rmtree(output_dir)
            except OSError:
                pass
        if job_id:
            err = (stderr_msg or f"exit code {code}") if code != 0 else None
            _update_training_job_status(job_id, "success" if code == 0 else "failed", err)
    except Exception as e:
        logger.exception(f"LoRA train failed: {e}")
        if job_id:
            _update_training_job_status(job_id, "failed", str(e))


def _run_qlora_train_task(model_path, dataset, dataset_dir, output_dir, request: QLoraTrainRequest, job_id: str = None):
    """后台执行 QLoRA 训练。失败时将 stderr 写入任务 error_message。"""
    try:
        cfg = _get_llmfactory_config()
        result = llmfactory_service.sft_qlora_train(
            model_path=model_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            template=request.template,
            lora_rank=request.lora_rank,
            lora_target=request.lora_target,
            quantization_bit=request.quantization_bit,
            quantization_method=request.quantization_method,
            double_quantization=request.double_quantization,
            learning_rate=request.learning_rate,
            num_train_epochs=request.num_train_epochs,
            bf16=request.bf16,
            deepspeed_config=request.deepspeed_config,
            config=cfg,
        )
        code = result[0] if isinstance(result, tuple) else result
        stderr_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        logger.info(f"QLoRA train finished with return_code={code}")
        if code != 0 and output_dir and os.path.isdir(output_dir):
            try:
                shutil.rmtree(output_dir)
            except OSError:
                pass
        if job_id:
            err = (stderr_msg or f"exit code {code}") if code != 0 else None
            _update_training_job_status(job_id, "success" if code == 0 else "failed", err)
    except Exception as e:
        logger.exception(f"QLoRA train failed: {e}")
        if job_id:
            _update_training_job_status(job_id, "failed", str(e))


def _run_full_train_task(model_path, dataset, dataset_dir, output_dir, request: FullTrainRequest, job_id: str = None):
    """后台执行全量训练。失败时将 stderr 写入任务 error_message。"""
    try:
        cfg = _get_llmfactory_config()
        result = llmfactory_service.sft_full_train(
            model_path=model_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            template=request.template,
            learning_rate=request.learning_rate,
            num_train_epochs=request.num_train_epochs,
            bf16=request.bf16,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            deepspeed_config=request.deepspeed_config,
            config=cfg,
        )
        code = result[0] if isinstance(result, tuple) else result
        stderr_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        logger.info(f"Full train finished with return_code={code}")
        if code != 0 and output_dir and os.path.isdir(output_dir):
            try:
                shutil.rmtree(output_dir)
            except OSError:
                pass
        if job_id:
            err = (stderr_msg or f"exit code {code}") if code != 0 else None
            _update_training_job_status(job_id, "success" if code == 0 else "failed", err)
    except Exception as e:
        logger.exception(f"Full train failed: {e}")
        if job_id:
            _update_training_job_status(job_id, "failed", str(e))


def _run_merge_task(model_path, adapter_path, export_dir, request: MergeAdapterRequest, job_id: str | None = None):
    """后台执行合并。若提供 job_id，则在完成后更新合并任务状态。"""
    try:
        cfg = _get_llmfactory_config()
        result = llmfactory_service.merge_adapter(
            model_path=model_path,
            adapter_path=adapter_path,
            export_dir=export_dir,
            template=request.template,
            export_size=request.export_size,
            export_device=request.export_device,
            config=cfg,
        )
        code = result[0] if isinstance(result, tuple) else result
        stderr_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        logger.info(f"Merge finished with return_code={code}")
        # 合并失败时清理导出目录，避免残留不完整模型
        if code != 0 and export_dir and os.path.isdir(export_dir):
            try:
                shutil.rmtree(export_dir)
            except OSError:
                pass
        if job_id:
            err = (stderr_msg or f"exit code {code}") if code != 0 else None
            _update_training_job_status(job_id, "success" if code == 0 else "failed", err)
    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        if job_id:
            _update_training_job_status(job_id, "failed", str(e))


async def _run_lora_train_task_async(model_path, dataset, dataset_dir, output_dir, request: LoraTrainRequest, job_id: str = None):
    """在线程池中执行 LoRA 训练，避免阻塞事件循环"""
    await asyncio.to_thread(
        _run_lora_train_task, model_path, dataset, dataset_dir, output_dir, request, job_id
    )


async def _run_qlora_train_task_async(model_path, dataset, dataset_dir, output_dir, request: QLoraTrainRequest, job_id: str = None):
    """在线程池中执行 QLoRA 训练"""
    await asyncio.to_thread(
        _run_qlora_train_task, model_path, dataset, dataset_dir, output_dir, request, job_id
    )


async def _run_full_train_task_async(model_path, dataset, dataset_dir, output_dir, request: FullTrainRequest, job_id: str = None):
    """在线程池中执行全量训练"""
    await asyncio.to_thread(
        _run_full_train_task, model_path, dataset, dataset_dir, output_dir, request, job_id
    )


async def _run_merge_task_async(model_path, adapter_path, export_dir, request: MergeAdapterRequest, job_id: str | None = None):
    """在线程池中执行合并"""
    await asyncio.to_thread(
        _run_merge_task, model_path, adapter_path, export_dir, request, job_id
    )


@router.post("/train/lora", response_model=LlmFactoryTaskResponse)
async def train_lora(
    request: LoraTrainRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_convert_db),
):
    """
    LoRA 微调训练（后台异步执行）。
    支持 dataset_id 使用已上传训练集；否则使用 dataset/dataset_dir 或 .env 默认值。
    """
    _ensure_llmfactory_heavy_gpu_budget()
    model_path, dataset, dataset_dir, output_dir = _resolve_train_params(request, db, task_type="lora")
    job = TrainingJob(output_dir=output_dir, task_type="lora", status="running")
    job.dataset_id = getattr(request, "dataset_id", None)
    job.data_type = _get_data_type_by_dataset_id(db, getattr(request, "dataset_id", None))
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(
        _run_lora_train_task_async, model_path, dataset, dataset_dir, output_dir, request, job.id
    )
    return LlmFactoryTaskResponse(
        success=True,
        message="LoRA 训练任务已提交，正在后台执行。请使用 job_id 在 GET /train/jobs 中查看该任务状态。",
        job_id=job.id,
    )


@router.post("/train/qlora", response_model=LlmFactoryTaskResponse)
async def train_qlora(
    request: QLoraTrainRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_convert_db),
):
    """
    QLoRA 微调训练（量化 + LoRA，省显存，后台异步执行）。
    支持 dataset_id 使用已上传训练集。
    """
    _ensure_llmfactory_heavy_gpu_budget()
    model_path, dataset, dataset_dir, output_dir = _resolve_train_params(request, db, task_type="qlora")
    job = TrainingJob(output_dir=output_dir, task_type="qlora", status="running")
    job.dataset_id = getattr(request, "dataset_id", None)
    job.data_type = _get_data_type_by_dataset_id(db, getattr(request, "dataset_id", None))
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(
        _run_qlora_train_task_async, model_path, dataset, dataset_dir, output_dir, request, job.id
    )
    return LlmFactoryTaskResponse(
        success=True,
        message="QLoRA 训练任务已提交，正在后台执行。请使用 job_id 在 GET /train/jobs 中查看该任务状态。",
        job_id=job.id,
    )


@router.post("/train/full", response_model=LlmFactoryTaskResponse)
async def train_full(
    request: FullTrainRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_convert_db),
):
    """
    全量微调训练（后台异步执行）。
    数据集：一、使用已上传训练集（dataset_id）；二、使用本地 dataset / dataset_dir。
    模型：model_id（推荐）与 model_path 二选一，与 LoRA 一致。
    """
    _ensure_llmfactory_heavy_gpu_budget()
    model_path, dataset, dataset_dir, output_dir = _resolve_train_params(request, db, task_type="full")
    job = TrainingJob(output_dir=output_dir, task_type="full", status="running")
    job.dataset_id = getattr(request, "dataset_id", None)
    job.data_type = _get_data_type_by_dataset_id(db, getattr(request, "dataset_id", None))
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(
        _run_full_train_task_async, model_path, dataset, dataset_dir, output_dir, request, job.id
    )
    return LlmFactoryTaskResponse(
        success=True,
        message="全量训练任务已提交，正在后台执行。请使用 job_id 在 GET /train/jobs 中查看该任务状态。",
        job_id=job.id,
    )


@router.post("/merge", response_model=LlmFactoryTaskResponse)
async def merge_adapter(
    request: MergeAdapterRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_convert_db),
):
    """
    合并 LoRA 适配器与基础模型（后台异步执行）。
    adapter_path 仅允许传 task_id 或 GET /trained-models 的目录名，由服务端解析到 train_output/SFT 下，禁止任意服务器路径。
    """
    _ensure_llmfactory_heavy_gpu_budget()
    model_path, adapter_path, export_dir = _resolve_merge_params(request, db=db)
    # 复用 TrainingJob 表，task_type=merge
    job = TrainingJob(output_dir=export_dir, task_type="merge", status="running")
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(_run_merge_task_async, model_path, adapter_path, export_dir, request, job.id)
    return LlmFactoryTaskResponse(
        success=True,
        message="合并任务已提交，正在后台执行。请使用 job_id 在 GET /merge/jobs 中查看该任务状态。",
        job_id=job.id,
    )


@router.get("/train/jobs", response_model=dict)
async def list_training_jobs(
    status: str = None,
    include_running: bool = True,
    page: int = 1,
    per_page: int = 20,
    db=Depends(get_convert_db),
):
    """
    查看已提交的训练任务列表，包含完成状态（成功/失败）。
    - status: 可选，逗号分隔筛选 status，如 success,failed 或 running
    - include_running: 是否包含进行中的任务，默认 true
    - page, per_page: 分页
    """
    q = db.query(TrainingJob)
    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        if statuses:
            q = q.filter(TrainingJob.status.in_(statuses))
    if not include_running:
        q = q.filter(TrainingJob.status.in_(["success", "failed"]))
    total = q.count()
    q = q.order_by(TrainingJob.updated_at.desc())
    offset = (page - 1) * per_page
    jobs = q.offset(offset).limit(per_page).all()
    items = []
    for j in jobs:
        d = j.to_dict()
        if j.status == "success" and _is_output_dir_allowed(j.output_dir):
            d["download_url"] = "/api/playground/llmfactory/train/jobs/%s/download" % j.id
        items.append(TrainingJobItem(**d))
    pages = (total + per_page - 1) // per_page if total else 0
    return {
        "jobs": items,
        "pagination": {"page": page, "per_page": per_page, "total": total, "pages": pages},
    }


@router.get("/train/jobs/{job_id}/progress", response_model=TrainingProgressResponse)
async def get_training_job_progress(
    job_id: str,
    log_limit: int = Query(100, ge=1, le=500, description="log_history 返回的最大条数（从文件尾部截取）"),
    db=Depends(get_convert_db),
):
    """
    查询微调任务训练进度，供前端轮询展示。

    数据来源：任务 `output_dir` 下的 `trainer_state.json`（HuggingFace Trainer / LlamaFactory 写入）。
    合并类任务（task_type=merge）无步进信息，仅返回 job 状态与说明。

    前端建议：对 `job_status=running` 每 2~5 秒请求一次；`success`/`failed` 后停止轮询。
    """
    job = db.query(TrainingJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not _is_output_path_under_training_base(job.output_dir):
        raise HTTPException(
            status_code=403,
            detail="该任务输出路径不在允许的目录范围内",
        )

    model_display_name = llmfactory_progress.infer_model_display_name(job.output_dir, job.task_type)

    if job.task_type == "merge":
        return TrainingProgressResponse(
            job_id=job.id,
            task_type=job.task_type,
            job_status=job.status,
            model_display_name=model_display_name,
            output_dir=job.output_dir,
            error_message=job.error_message,
            trainer_state_found=False,
            message="合并任务无 Trainer 步进信息，请根据 job_status 判断完成情况",
        )

    data = llmfactory_progress.load_training_progress(
        job.output_dir, log_history_limit=log_limit
    )
    return TrainingProgressResponse(
        job_id=job.id,
        task_type=job.task_type,
        job_status=job.status,
        model_display_name=model_display_name,
        output_dir=job.output_dir,
        error_message=job.error_message,
        trainer_state_found=data["trainer_state_found"],
        global_step=data.get("global_step"),
        max_steps=data.get("max_steps"),
        epoch=data.get("epoch"),
        num_train_epochs=data.get("num_train_epochs"),
        latest_loss=data.get("latest_loss"),
        learning_rate=data.get("learning_rate"),
        progress_ratio=data.get("progress_ratio"),
        log_history=data.get("log_history"),
        best_model_checkpoint=data.get("best_model_checkpoint"),
        message=data.get("message"),
    )


@router.get("/merge/jobs", response_model=dict)
async def list_merge_jobs(
    status: str = None,
    include_running: bool = True,
    page: int = 1,
    per_page: int = 20,
    db=Depends(get_convert_db),
):
    """
    查看合并任务列表（task_type=merge），包含完成状态（成功/失败）。
    - status: 可选，逗号分隔筛选 status，如 success,failed 或 running
    - include_running: 是否包含进行中的任务，默认 true
    - page, per_page: 分页
    """
    q = db.query(TrainingJob).filter(TrainingJob.task_type == "merge")
    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        if statuses:
            q = q.filter(TrainingJob.status.in_(statuses))
    if not include_running:
        q = q.filter(TrainingJob.status.in_(["success", "failed"]))
    total = q.count()
    q = q.order_by(TrainingJob.updated_at.desc())
    offset = (page - 1) * per_page
    jobs = q.offset(offset).limit(per_page).all()
    items = []
    for j in jobs:
        d = j.to_dict()
        items.append(TrainingJobItem(**d))
    pages = (total + per_page - 1) // per_page if total else 0
    return {
        "jobs": items,
        "pagination": {"page": page, "per_page": per_page, "total": total, "pages": pages},
    }


@router.get("/train/jobs/{job_id}/download")
async def download_training_result(
    job_id: str,
    background_tasks: BackgroundTasks,
    db=Depends(get_convert_db),
):
    """
    将指定训练任务的结果目录打包为 zip 供用户下载到本地。
    仅支持状态为 success 的任务；output_dir 必须在配置允许的根目录下。
    """
    job = db.query(TrainingJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if job.status != "success":
        raise HTTPException(
            status_code=400,
            detail="仅支持下载已成功完成的任务，当前状态: %s" % job.status,
        )
    output_dir = job.output_dir
    if not _is_output_dir_allowed(output_dir):
        raise HTTPException(
            status_code=403,
            detail="该任务输出目录不在允许的下载范围内",
        )
    out_path = Path(output_dir)
    if not out_path.is_dir():
        raise HTTPException(status_code=404, detail="训练结果目录不存在或已被移除")

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = tmp.name
    try:
        tmp.close()
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in out_path.rglob("*"):
                if f.is_file():
                    arcname = f.relative_to(out_path)
                    zf.write(f, arcname)
        filename = f"train_result_{job_id}.zip"
        background_tasks.add_task(_remove_download_temp_file, tmp_path)
        return FileResponse(
            tmp_path,
            media_type="application/zip",
            filename=filename,
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.exception("打包训练结果失败")
        raise HTTPException(status_code=500, detail="打包训练结果失败: %s" % str(e))


def _remove_download_temp_file(path: str):
    """下载完成后删除临时 zip 文件。"""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@router.delete("/train/jobs/{job_id}", status_code=204)
async def delete_training_job(
    job_id: str,
    db=Depends(get_convert_db),
):
    """
    删除单条训练任务记录（仅删数据库记录，不删除磁盘上的 output_dir 目录）。
    """
    job = db.query(TrainingJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    db.delete(job)
    db.commit()
    return None


@router.delete("/train/jobs", response_model=dict)
async def clear_training_jobs(
    clear_all: bool = False,
    db=Depends(get_convert_db),
):
    """
    清空历史训练记录。
    - 不传参数或 clear_all=false：不执行删除，返回 deleted=0。
    - clear_all=true：删除所有训练任务记录（仅删库，不删磁盘上的输出目录）。
    """
    if not clear_all:
        return {"deleted": 0, "message": "请使用 clear_all=true 确认清空所有训练记录"}
    count = db.query(TrainingJob).delete()
    db.commit()
    return {"deleted": count, "message": "已清空所有训练记录"}


# 训练输出子目录命名格式: {模型名}_{lora|qlora|full}_{YYYYmmdd}_{HHMMSS}
_TRAINED_DIR_PATTERN = re.compile(r"^(.+)_(lora|qlora|full)_(\d{8})_(\d{6})$")


def _parse_trained_dir_name(dir_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    解析训练输出目录名，格式为 {模型名}_{训练方式}_{YYYYmmdd}_{HHMMSS}。
    返回 (基础模型名, 训练方法, 创建时间 ISO 字符串)，无法解析时返回 (None, None, None)。
    """
    m = _TRAINED_DIR_PATTERN.match(dir_name)
    if not m:
        return None, None, None
    model_name, task_type, date_part, time_part = m.groups()
    try:
        dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
        created_at = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        created_at = None
    return model_name or None, task_type or None, created_at


def _read_adapter_meta(output_dir: str) -> Dict[str, Any]:
    """从训练输出目录读取 adapter_config.json，获取基础模型路径等。返回 dict，键为 base_model_name_or_path、peft_type 等。"""
    result = {}
    config_path = os.path.join(output_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        return result
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            base = data.get("base_model_name_or_path") or data.get("base_model")
            if base:
                result["base_model_name_or_path"] = base
            peft = data.get("peft_type") or data.get("finetuning_type")
            if peft:
                result["peft_type"] = peft
    except (json.JSONDecodeError, OSError):
        pass
    return result


def _get_data_type_for_output_dir(db, output_dir: str) -> Optional[str]:
    """根据训练输出目录从任务记录中查询 data_type（用什么数据集训练的），用作模型类型。"""
    if not db or not output_dir:
        return None
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.output_dir == output_dir)
        .order_by(TrainingJob.updated_at.desc())
        .first()
    )
    return job.data_type if job and job.data_type else None


def _get_task_id_for_output_dir(db, output_dir: str) -> Optional[str]:
    """根据训练输出目录从任务记录中查询 job id（task_id）。"""
    if not db or not output_dir:
        return None
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.output_dir == output_dir)
        .order_by(TrainingJob.updated_at.desc())
        .first()
    )
    return job.id if job else None


@router.get("/trained-models", response_model=dict)
async def list_trained_models(db=Depends(get_convert_db)):
    """
    列出训练输出目录下已训练好的模型。
    返回：模型名称、基础模型、模型类型、训练方法、创建时间、output_dir 等。
    """
    base = _ensure_training_output_base()
    models = []
    try:
        for name in sorted(os.listdir(base)):
            if name.startswith("."):
                continue
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            abs_path = os.path.normpath(os.path.abspath(path))
            parsed_base, parsed_task, created_at = _parse_trained_dir_name(name)
            meta = _read_adapter_meta(abs_path)
            base_model = meta.get("base_model_name_or_path")
            if not base_model and parsed_base:
                base_model = parsed_base
            elif base_model and os.path.sep in base_model:
                base_model = os.path.basename(base_model.rstrip(os.path.sep))
            # 模型类型 = 用什么数据集训练的（data_type），从任务记录中查；无记录时留空
            model_type = _get_data_type_for_output_dir(db, abs_path) or ""
            task_id = _get_task_id_for_output_dir(db, abs_path)
            item = {
                "id": name,
                "name": name,
                "output_dir": abs_path,
                "model_name": name,
                "base_model": base_model or parsed_base or "",
                "model_type": model_type,
                "training_method": parsed_task or "",
                "created_at": created_at or "",
                "task_id": task_id,
            }
            models.append(item)
    except OSError as e:
        logger.warning("list trained models: %s", e)
    return {
        "output_base": base,
        "models": models,
        "message": "列出的是训练输出根目录下的子目录，可作为 adapter_path 用于合并或推理。",
    }


@router.get("/merged-models", response_model=dict)
async def list_merged_models():
    """
    列出合并后的模型（train_output/merged 下的子目录）。
    返回：id/name、output_dir、source_adapter（对应合并前的适配器目录名）等，可作为推理的 model_path。
    """
    base = _get_merged_output_base()
    models = []
    try:
        for name in sorted(os.listdir(base)):
            if name.startswith("."):
                continue
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            abs_path = os.path.normpath(os.path.abspath(path))
            # 合并目录名为 xxx_merged，来源适配器名为 xxx
            source_adapter = name[:-7] if name.endswith("_merged") else name
            try:
                mtime = os.path.getmtime(path)
                created_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                created_at = ""
            item = {
                "id": name,
                "name": name,
                "output_dir": abs_path,
                "source_adapter": source_adapter,
                "created_at": created_at,
            }
            models.append(item)
    except OSError as e:
        logger.warning("list merged models: %s", e)
    return {
        "output_base": base,
        "models": models,
        "message": "列出的是合并输出目录下的子目录，可作为 model_path 用于推理或部署。",
    }


@router.get("/merged-models/{model_id}/download")
async def download_merged_model(
    model_id: str,
    background_tasks: BackgroundTasks,
):
    """
    将指定合并后模型目录打包为 zip 供用户下载到本地。
    model_id 为 GET /merged-models 返回的 id/name（目录名）。
    """
    if not model_id or not (model_id := model_id.strip()):
        raise HTTPException(status_code=400, detail="缺少 model_id（请传入 GET /merged-models 返回的 id/name）")

    base = _get_merged_output_base()
    name = os.path.basename(model_id.replace("\\", "/"))
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="model_id 无效")

    target = os.path.normpath(os.path.join(base, _sanitize_dirname(name)))
    try:
        target_real = os.path.realpath(target)
        base_real = os.path.realpath(base)
    except OSError:
        raise HTTPException(status_code=400, detail="路径解析失败")

    if target_real != base_real and not target_real.startswith(base_real + os.sep):
        raise HTTPException(status_code=403, detail="仅允许下载 train_output/merged 下的合并模型目录")
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail="未找到该合并模型目录")

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = tmp.name
    try:
        tmp.close()
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in Path(target).rglob("*"):
                if f.is_file():
                    arcname = f.relative_to(target)
                    zf.write(f, arcname)
        filename = f"merged_model_{name}.zip"
        background_tasks.add_task(_remove_download_temp_file, tmp_path)
        return FileResponse(
            tmp_path,
            media_type="application/zip",
            filename=filename,
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.exception("打包合并模型失败")
        raise HTTPException(status_code=500, detail="打包合并模型失败: %s" % str(e))


@router.delete("/merged-models/{model_id}", status_code=204)
async def delete_merged_model(model_id: str):
    """
    删除合并后的模型：删除 train_output/merged 下对应目录。
    model_id 为 GET /merged-models 返回的 id/name（目录名）。
    不涉及数据库记录。
    """
    if not model_id or not (model_id := model_id.strip()):
        raise HTTPException(status_code=400, detail="缺少 model_id（请传入 GET /merged-models 返回的 id/name）")
    name = os.path.basename(model_id.replace("\\", "/"))
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="model_id 无效")

    base = _get_merged_output_base()
    target = os.path.normpath(os.path.join(base, _sanitize_dirname(name)))
    try:
        target_real = os.path.realpath(target)
        base_real = os.path.realpath(base)
    except OSError:
        raise HTTPException(status_code=400, detail="路径解析失败")

    if target_real != base_real and not target_real.startswith(base_real + os.sep):
        raise HTTPException(status_code=403, detail="仅允许删除 train_output/merged 下的合并模型目录")
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail="未找到该合并模型目录")
    try:
        shutil.rmtree(target)
    except OSError as e:
        logger.exception("delete merged model dir failed: %s", e)
        raise HTTPException(status_code=500, detail="删除目录失败: %s" % str(e))
    return None


@router.delete("/trained-models/{model_id}", status_code=204)
async def delete_trained_model(model_id: str, db=Depends(get_convert_db)):
    """
    删除已训练模型：删除 train_output/SFT 下对应目录及数据库中关联的任务记录。
    model_id 为 GET /trained-models 返回的 id/name（目录名），如 Qwen2-0.5B_lora_20260311_142141。
    """
    if not model_id or not (model_id := model_id.strip()):
        raise HTTPException(status_code=400, detail="缺少 model_id（请传入 GET /trained-models 返回的 id/name）")
    name = os.path.basename(model_id.replace("\\", "/"))
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="model_id 无效")
    sft_base = _get_sft_output_base()
    target = os.path.normpath(os.path.join(sft_base, _sanitize_dirname(name)))
    try:
        target_real = os.path.realpath(target)
        base_real = os.path.realpath(sft_base)
    except OSError:
        raise HTTPException(status_code=400, detail="路径解析失败")
    if target_real != base_real and not target_real.startswith(base_real + os.sep):
        raise HTTPException(status_code=403, detail="仅允许删除 train_output/SFT 下的已训练模型目录")
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail="未找到该已训练模型目录")
    try:
        shutil.rmtree(target)
    except OSError as e:
        logger.exception("delete trained model dir failed: %s", e)
        raise HTTPException(status_code=500, detail="删除目录失败: %s" % str(e))
    # 删除该 output_dir 对应的训练任务记录
    db.query(TrainingJob).filter(TrainingJob.output_dir == target_real).delete(synchronize_session=False)
    db.commit()
    return None


@router.post("/train/lora/sync", response_model=LlmFactoryTaskResponse)
async def train_lora_sync(request: LoraTrainRequest, db=Depends(get_convert_db)):
    """
    LoRA 微调训练（同步执行，等待完成后返回）。
    适用于快速测试，支持 dataset_id。
    """
    _ensure_llmfactory_heavy_gpu_budget()
    model_path, dataset, dataset_dir, output_dir = _resolve_train_params(request, db, task_type="lora")
    job = TrainingJob(output_dir=output_dir, task_type="lora", status="running")
    job.dataset_id = getattr(request, "dataset_id", None)
    job.data_type = _get_data_type_by_dataset_id(db, getattr(request, "dataset_id", None))
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        cfg = _get_llmfactory_config()
        result = llmfactory_service.sft_lora_train(
            model_path=model_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            template=request.template,
            lora_rank=request.lora_rank,
            lora_target=request.lora_target,
            learning_rate=request.learning_rate,
            num_train_epochs=request.num_train_epochs,
            bf16=request.bf16,
            deepspeed_config=request.deepspeed_config,
            config=cfg,
        )
        code = result[0] if isinstance(result, tuple) else result
        stderr_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        job.status = "success" if code == 0 else "failed"
        if code != 0:
            job.error_message = (stderr_msg or f"exit code {code}")[:4096]
            # 同步训练失败时清理输出目录
            if output_dir and os.path.isdir(output_dir):
                try:
                    shutil.rmtree(output_dir)
                except OSError:
                    pass
        db.commit()
        msg = "训练完成" if code == 0 else (stderr_msg or f"训练退出，返回码 {code}")
        return LlmFactoryTaskResponse(
            success=(code == 0),
            message=msg[:4096] if len(msg) > 4096 else msg,
            return_code=code,
            job_id=job.id,
        )
    except HTTPException:
        job.status = "failed"
        job.error_message = "HTTPException"
        db.commit()
        raise
    except Exception as e:
        logger.exception(f"LoRA train failed: {e}")
        job.status = "failed"
        job.error_message = str(e)[:4096]
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/qlora/sync", response_model=LlmFactoryTaskResponse)
async def train_qlora_sync(request: QLoraTrainRequest, db=Depends(get_convert_db)):
    """
    QLoRA 微调训练（同步执行，等待完成后返回）。
    """
    _ensure_llmfactory_heavy_gpu_budget()
    model_path, dataset, dataset_dir, output_dir = _resolve_train_params(request, db, task_type="qlora")
    job = TrainingJob(output_dir=output_dir, task_type="qlora", status="running")
    job.dataset_id = getattr(request, "dataset_id", None)
    job.data_type = _get_data_type_by_dataset_id(db, getattr(request, "dataset_id", None))
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        cfg = _get_llmfactory_config()
        result = llmfactory_service.sft_qlora_train(
            model_path=model_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            template=request.template,
            lora_rank=request.lora_rank,
            lora_target=request.lora_target,
            quantization_bit=request.quantization_bit,
            quantization_method=request.quantization_method,
            double_quantization=request.double_quantization,
            learning_rate=request.learning_rate,
            num_train_epochs=request.num_train_epochs,
            bf16=request.bf16,
            deepspeed_config=request.deepspeed_config,
            config=cfg,
        )
        code = result[0] if isinstance(result, tuple) else result
        stderr_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        job.status = "success" if code == 0 else "failed"
        if code != 0:
            job.error_message = (stderr_msg or f"exit code {code}")[:4096]
            if output_dir and os.path.isdir(output_dir):
                try:
                    shutil.rmtree(output_dir)
                except OSError:
                    pass
        db.commit()
        msg = "训练完成" if code == 0 else (stderr_msg or f"训练退出，返回码 {code}")
        return LlmFactoryTaskResponse(
            success=(code == 0),
            message=msg[:4096] if len(msg) > 4096 else msg,
            return_code=code,
            job_id=job.id,
        )
    except HTTPException:
        job.status = "failed"
        job.error_message = "HTTPException"
        db.commit()
        raise
    except Exception as e:
        logger.exception(f"QLoRA train failed: {e}")
        job.status = "failed"
        job.error_message = str(e)[:4096]
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/full/sync", response_model=LlmFactoryTaskResponse)
async def train_full_sync(request: FullTrainRequest, db=Depends(get_convert_db)):
    """
    全量微调训练（同步执行，等待完成后返回）。
    数据集：一、dataset_id（已上传）；二、dataset / dataset_dir（本地）。模型：model_id 或 model_path。
    """
    _ensure_llmfactory_heavy_gpu_budget()
    model_path, dataset, dataset_dir, output_dir = _resolve_train_params(request, db, task_type="full")
    job = TrainingJob(output_dir=output_dir, task_type="full", status="running")
    job.dataset_id = getattr(request, "dataset_id", None)
    job.data_type = _get_data_type_by_dataset_id(db, getattr(request, "dataset_id", None))
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        cfg = _get_llmfactory_config()
        result = llmfactory_service.sft_full_train(
            model_path=model_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            template=request.template,
            learning_rate=request.learning_rate,
            num_train_epochs=request.num_train_epochs,
            bf16=request.bf16,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            deepspeed_config=request.deepspeed_config,
            config=cfg,
        )
        code = result[0] if isinstance(result, tuple) else result
        stderr_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        job.status = "success" if code == 0 else "failed"
        if code != 0:
            job.error_message = (stderr_msg or f"exit code {code}")[:4096]
            if output_dir and os.path.isdir(output_dir):
                try:
                    shutil.rmtree(output_dir)
                except OSError:
                    pass
        db.commit()
        msg = "训练完成" if code == 0 else (stderr_msg or f"训练退出，返回码 {code}")
        return LlmFactoryTaskResponse(
            success=(code == 0),
            message=msg[:4096] if len(msg) > 4096 else msg,
            return_code=code,
            job_id=job.id,
        )
    except HTTPException:
        job.status = "failed"
        job.error_message = "HTTPException"
        db.commit()
        raise
    except Exception as e:
        logger.exception(f"Full train failed: {e}")
        job.status = "failed"
        job.error_message = str(e)[:4096]
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/merge/sync", response_model=LlmFactoryTaskResponse)
async def merge_adapter_sync(request: MergeAdapterRequest, db=Depends(get_convert_db)):
    """
    合并适配器（同步执行）。adapter_path 仅允许 task_id 或微调目录名，由服务端解析。
    """
    _ensure_llmfactory_heavy_gpu_budget()
    # 记录为合并任务
    model_path, adapter_path, export_dir = _resolve_merge_params(request, db=db)
    job = TrainingJob(output_dir=export_dir, task_type="merge", status="running")
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        cfg = _get_llmfactory_config()
        result = llmfactory_service.merge_adapter(
            model_path=model_path,
            adapter_path=adapter_path,
            export_dir=export_dir,
            template=request.template,
            export_size=request.export_size,
            export_device=request.export_device,
            config=cfg,
        )
        code = result[0] if isinstance(result, tuple) else result
        stderr_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        job.status = "success" if code == 0 else "failed"
        if code != 0:
            job.error_message = (stderr_msg or f"exit code {code}")[:4096]
            if export_dir and os.path.isdir(export_dir):
                try:
                    shutil.rmtree(export_dir)
                except OSError:
                    pass
        db.commit()
        msg = "合并完成" if code == 0 else (stderr_msg or f"合并退出，返回码 {code}")
        return LlmFactoryTaskResponse(
            success=(code == 0),
            message=msg[:4096] if len(msg) > 4096 else msg,
            return_code=code,
            job_id=job.id,
        )
    except HTTPException:
        job.status = "failed"
        job.error_message = "HTTPException"
        db.commit()
        raise
    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        job.status = "failed"
        job.error_message = str(e)[:4096]
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/start", response_model=dict)
async def start_inference_api(
    body: StartInferenceApiRequest,
    request: Request,
):
    """
    启动推理 API 服务（OpenAI 兼容格式）。
    推理服务在内部端口运行，对外统一通过主服务 8000 的 /api/playground/llmfactory/v1 访问。
    返回的 api_url 为 8000 端口地址，无需再访问 8001。
    model_path 不传时使用 .env 中的 LLMFACTORY_DEFAULT_MODEL_PATH。
    """
    try:
        _ensure_llmfactory_heavy_gpu_budget()
        model_path = _resolve_inference_params(body)
        # 未传模板或传 auto 时，按模型 config.json 的 model_type 自动选择模板
        template = (body.template or "").strip()
        if template.lower() in ("", "auto"):
            template = _infer_template_for_model(model_path)
        else:
            template = body.template
        cfg = _get_llmfactory_config()
        process, port, start_cmd = llmfactory_service.start_inference_api(
            model_path=model_path,
            adapter_path=body.adapter_path,
            template=template,
            api_port=None,  # 使用内部端口，由 8000 代理
            cuda_devices=body.cuda_devices,
            bf16=body.bf16,
            infer_dtype=body.infer_dtype,
            config=cfg,
        )
        pid = getattr(process, "pid", None) if process is not None else None
        base_url = str(request.base_url).rstrip("/")
        request.app.state.llmfactory_inference_port = port
        request.app.state.llmfactory_inference_pid = pid
        request.app.state.llmfactory_inference_start_cmd = start_cmd
        _save_inference_state(
            {
                "pid": pid,
                "port": port,
                "start_cmd": start_cmd,
                "saved_at": serialize_datetime_for_api_response(utc_now()),
            }
        )
        return {
            "success": True,
            "message": "推理 API 服务已启动，请使用下方 api_url（8000 端口）调用",
            "api_url": f"{base_url}/api/playground/llmfactory/v1",
            "pid": pid,
            "start_cmd": start_cmd,
            "note": "进程在后台运行，重启 llm_AIO 后需重新启动",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Start inference API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/stop", response_model=dict)
async def stop_inference_api(request: Request):
    """
    关闭已启动的推理 API 子进程，释放 GPU 显存。
    """
    pid = getattr(request.app.state, "llmfactory_inference_pid", None)
    if not pid:
        state = _load_inference_state()
        pid = state.get("pid")
        if pid:
            request.app.state.llmfactory_inference_pid = pid
            request.app.state.llmfactory_inference_port = state.get("port")
            request.app.state.llmfactory_inference_start_cmd = state.get("start_cmd")

    if not pid:
        return {
            "success": True,
            "message": "推理服务未在运行，无需关闭",
        }
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Stop inference process {pid}: {e}")
    request.app.state.llmfactory_inference_pid = None
    request.app.state.llmfactory_inference_port = None
    _clear_inference_state()
    return {
        "success": True,
        "message": f"已发送关闭信号给推理进程 (PID {pid})，显存将逐步释放",
    }


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_llmfactory_inference(request: Request, path: str):
    """
    将 8000 上的 /api/playground/llmfactory/v1/* 代理到推理子进程（内部端口）。
    需先调用 POST /api/playground/llmfactory/api/start 启动推理服务。
    """
    port = getattr(request.app.state, "llmfactory_inference_port", None)
    if not port:
        state = _load_inference_state()
        port = state.get("port")
        if port:
            request.app.state.llmfactory_inference_port = port
            request.app.state.llmfactory_inference_pid = state.get("pid")
            request.app.state.llmfactory_inference_start_cmd = state.get("start_cmd")
    if not port:
        raise HTTPException(
            status_code=503,
            detail="推理服务未启动，请先调用 POST /api/playground/llmfactory/api/start",
        )
    backend = f"http://127.0.0.1:{port}"
    url = f"{backend}/v1/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = dict(request.headers)
    for h in ("host", "connection", "content-length"):
        headers.pop(h.lower(), None)
    try:
        body = await request.body()
    except Exception:
        body = b""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
        except httpx.ConnectError as e:
            raise HTTPException(
                status_code=503,
                detail=f"无法连接推理服务 (127.0.0.1:{port})，请确认已启动或重试",
            ) from e
    # 4xx/5xx 时在主服务日志中记录推理子进程返回的 body，便于排查
    if resp.status_code >= 400 and resp.content:
        try:
            body_preview = resp.content[:500].decode("utf-8", errors="replace")
            logger.warning(
                "推理子进程返回 %s: %s",
                resp.status_code,
                body_preview + ("..." if len(resp.content) > 500 else ""),
            )
        except Exception:
            pass
    from starlette.responses import Response as StarletteResponse
    return StarletteResponse(
        status_code=resp.status_code,
        content=resp.content,
        headers=dict(resp.headers),
    )                                                                                              