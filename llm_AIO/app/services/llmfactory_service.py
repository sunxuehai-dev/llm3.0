"""
llmfactory 服务层：封装 FactoryBackend 的调用，供 API 路由使用。
训练/合并失败时在 llm_AIO 侧自行执行命令并捕获 stderr，返回 (returncode, stderr_text)，不修改 llmfactory 包。
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# 将 llmfactory 目录加入 Python 路径（项目根为 LLM_AIO，llmfactory 为兄弟目录）
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_llmfactory_path = _project_root / "llmfactory"
if str(_llmfactory_path) not in sys.path:
    sys.path.insert(0, str(_llmfactory_path))

from src.FactoryBackend.LlamaConfig import LlamaConfig
from src.FactoryBackend.TaskGeneral import FactoryCli
from src.FactoryBackend import FineTuning as ft
from src.FactoryBackend import Stage as st
from src.FactoryBackend import InitTask as it


def get_llmfactory_templates() -> list:
    """返回微调/合并可用的 template 列表（与 LlamaFactory 的 template_register 一致）。"""
    return list(getattr(it.InitModel, "template_register", []))


def _get_command_prefix(config: Dict[str, str]) -> str:
    """获取 llamafactory-cli 的命令前缀（venv bin 路径）"""
    prefix = config.get("command_prefix", "")
    if prefix:
        return prefix
    # 默认使用 llmfactory 下的 .venv
    default_venv = _llmfactory_path / ".venv" / "bin"
    if default_venv.exists():
        return str(default_venv.resolve())
    # Windows
    default_venv_win = _llmfactory_path / ".venv" / "Scripts"
    if default_venv_win.exists():
        return str(default_venv_win.resolve())
    return ""


def _extract_key_error_message(stderr_text: str, max_len: int = 1200) -> str:
    """
    从完整 stderr 中只保留关键报错：最后一个 Traceback 块 + 异常类型与信息。
    避免把大量 [INFO] 日志写进任务 error_message。
    """
    if not stderr_text or len(stderr_text) <= max_len:
        return stderr_text
    # 从最后一个 "Traceback (most recent call last)" 开始截取
    tb_marker = "Traceback (most recent call last):"
    idx = stderr_text.rfind(tb_marker)
    if idx >= 0:
        excerpt = stderr_text[idx:].strip()
    else:
        # 没有 Traceback 则保留末尾（错误常在最后）
        excerpt = stderr_text
    if len(excerpt) > max_len:
        excerpt = "...\n" + excerpt[-max_len:]
    return excerpt.strip()


def _run_cli_capture_stderr(cli: FactoryCli) -> Tuple[int, Optional[str]]:
    """
    在 llm_AIO 侧执行 FactoryCli 生成的命令并捕获 stderr，不修改 llmfactory 源码。
    返回 (returncode, stderr_text)；成功时 stderr_text 为 None，失败时为精简后的关键报错（约 1200 字符）。
    """
    cli._gen_llamafactory_cmd()
    env = os.environ.copy()
    env.update(getattr(cli, "_env_var", {}))
    process = subprocess.Popen(
        cli._cmd,
        stdin=None,
        stdout=None,
        stderr=subprocess.PIPE,
        env=env,
    )
    returncode = process.wait()
    stderr_out = None
    if returncode != 0 and process.stderr:
        try:
            raw = process.stderr.read().decode("utf-8", errors="replace").strip()
            stderr_out = _extract_key_error_message(raw, max_len=1200)
        except Exception:
            stderr_out = "(无法读取 stderr)"
    return (returncode, stderr_out)


def sft_lora_train(
    model_path: str,
    dataset: str,
    dataset_dir: str,
    output_dir: str,
    template: str = "qwen3_nothink",
    lora_rank: int = 8,
    lora_target: str = "all",
    learning_rate: float = 1e-4,
    num_train_epochs: float = 5.0,
    bf16: bool = True,
    deepspeed_config: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    LoRA 微调训练。
    返回: (subprocess 返回码, 失败时的 stderr 文本或 None)，0 表示成功。
    """
    config = config or {}
    lora_sft = ft.LlamaConfig()

    stage = st.SetStage(lora_sft)
    if deepspeed_config:
        stage.set_sft(Path(deepspeed_config))
    else:
        stage.set_sft()

    finetuning = ft.SetFinetuning(lora_sft)
    finetuning.set_lora(rank=lora_rank, target=lora_target)

    model = it.InitModel(lora_sft, model_path=Path(model_path))
    model.set_template(template)

    it.InitDataset(lora_sft, dataset=dataset, dataset_dir=Path(dataset_dir))

    train_strategy = it.Train(lora_sft)
    train_strategy.set_learning_rate(learning_rate)
    train_strategy.set_num_train_epochs(num_train_epochs)
    train_strategy.set_bf16(bf16)

    output = it.Output(lora_sft)
    output.set_output_dir(Path(output_dir))

    prefix = _get_command_prefix(config)
    cli = FactoryCli(lora_sft, command_type="train", command_prefix=prefix)
    for k, v in config.get("env_vars", {}).items():
        cli.add_env_var(k, str(v))
    return _run_cli_capture_stderr(cli)


def sft_qlora_train(
    model_path: str,
    dataset: str,
    dataset_dir: str,
    output_dir: str,
    template: str = "qwen3_nothink",
    lora_rank: int = 8,
    lora_target: str = "all",
    quantization_bit: int = 4,
    quantization_method: str = "bnb",
    double_quantization: bool = False,
    learning_rate: float = 1e-4,
    num_train_epochs: float = 5.0,
    bf16: bool = True,
    deepspeed_config: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    QLoRA 微调训练（量化 + LoRA，省显存）。
    返回: (subprocess 返回码, 失败时的 stderr 文本或 None)，0 表示成功。
    """
    config = config or {}
    qlora_sft = ft.LlamaConfig()

    stage = st.SetStage(qlora_sft)
    if deepspeed_config:
        stage.set_sft(Path(deepspeed_config))
    else:
        stage.set_sft()

    finetuning = ft.SetFinetuning(qlora_sft)
    finetuning.set_lora(rank=lora_rank, target=lora_target)

    # QLoRA: 启用量化
    qlora_sft.quantization_bit = quantization_bit
    qlora_sft.quantization_method = quantization_method
    qlora_sft.double_quantization = double_quantization

    model = it.InitModel(qlora_sft, model_path=Path(model_path))
    model.set_template(template)

    it.InitDataset(qlora_sft, dataset=dataset, dataset_dir=Path(dataset_dir))

    train_strategy = it.Train(qlora_sft)
    train_strategy.set_learning_rate(learning_rate)
    train_strategy.set_num_train_epochs(num_train_epochs)
    train_strategy.set_bf16(bf16)

    output = it.Output(qlora_sft)
    output.set_output_dir(Path(output_dir))

    prefix = _get_command_prefix(config)
    cli = FactoryCli(qlora_sft, command_type="train", command_prefix=prefix)
    for k, v in config.get("env_vars", {}).items():
        cli.add_env_var(k, str(v))
    return _run_cli_capture_stderr(cli)


def sft_full_train(
    model_path: str,
    dataset: str,
    dataset_dir: str,
    output_dir: str,
    template: str = "qwen3_nothink",
    learning_rate: float = 1e-4,
    num_train_epochs: float = 1.0,
    bf16: bool = True,
    gradient_accumulation_steps: int = 2,
    deepspeed_config: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    全量微调训练。
    返回: (subprocess 返回码, 失败时的 stderr 文本或 None)。
    """
    config = config or {}
    full_sft = ft.LlamaConfig()

    stage = st.SetStage(full_sft)
    if deepspeed_config:
        stage.set_sft(Path(deepspeed_config))
    else:
        stage.set_sft()

    finetuning = ft.SetFinetuning(full_sft)
    finetuning.set_full()

    model = it.InitModel(full_sft, model_path=Path(model_path))
    model.set_template(template)

    it.InitDataset(full_sft, dataset=dataset, dataset_dir=Path(dataset_dir))

    train_strategy = it.Train(full_sft)
    train_strategy.set_learning_rate(learning_rate)
    train_strategy.set_num_train_epochs(num_train_epochs)
    train_strategy.set_bf16(bf16)
    train_strategy.set_gradient_accumulation_steps(gradient_accumulation_steps)

    output = it.Output(full_sft)
    output.set_output_dir(Path(output_dir))

    prefix = _get_command_prefix(config)
    cli = FactoryCli(full_sft, command_type="train", command_prefix=prefix)
    for k, v in config.get("env_vars", {}).items():
        cli.add_env_var(k, str(v))
    return _run_cli_capture_stderr(cli)


def merge_adapter(
    model_path: str,
    adapter_path: str,
    export_dir: str,
    template: str = "qwen3_nothink",
    export_size: int = 5,
    export_device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    合并 LoRA 适配器与基础模型。
    返回: (subprocess 返回码, 失败时的 stderr 文本或 None)。
    """
    config = config or {}
    merge_cfg = ft.LlamaConfig()
    merge_cfg.reset_to_none()

    model = it.InitModel(merge_cfg, model_path=Path(model_path))
    model.set_adapter_name_or_path(Path(adapter_path))
    model.set_template(template)

    export = it.Export(merge_cfg)
    export.set_export_dir(Path(export_dir))
    export.set_export_size(export_size)
    export.set_export_device(export_device)

    prefix = _get_command_prefix(config)
    cli = FactoryCli(merge_cfg, command_type="export", command_prefix=prefix)
    for k, v in config.get("env_vars", {}).items():
        cli.add_env_var(k, str(v))
    # 与训练保持一致：由 llm_AIO 捕获 stderr 并精简后返回
    return _run_cli_capture_stderr(cli)


# 推理子进程使用的内部端口（与主服务 8000 隔离，对外通过 8000 代理访问）
INFERENCE_INTERNAL_PORT = "18001"


def _build_inference_api_cmd(
    *,
    cli_path: str,
    model_path: str,
    adapter_path: Optional[str],
    template: str,
    infer_dtype: Optional[str] = None,
    trust_remote_code: bool = True,
) -> list[str]:
    """构建推理 api 子进程命令行，与原先 FactoryCli + LlamaConfig 行为一致（含 trust_remote_code）。"""
    cmd = [
        cli_path,
        "api",
        "--model_name_or_path",
        model_path,
        "--infer_backend",
        "huggingface",
        "--template",
        template,
        "--trust_remote_code",
        "true" if trust_remote_code else "false",
    ]
    if adapter_path:
        cmd.extend(["--adapter_name_or_path", adapter_path])
    if infer_dtype is not None:
        cmd.extend(["--infer_dtype", infer_dtype])
    return cmd


def _start_inference_api_with_infer_dtype(
    model_path: str,
    adapter_path: Optional[str],
    template: str,
    port: str,
    cuda_devices: str,
    infer_dtype: str,
    config: Dict[str, Any],
) -> Tuple[subprocess.Popen, str]:
    """
    在 llm_AIO 侧直接构建并执行带 --infer_dtype 的推理命令（不修改 llmfactory）。
    用于规避 LLaMA-Factory 不读环境变量时无法传推理精度的问题。
    """
    prefix = _get_command_prefix(config).rstrip("/")
    cli_path = f"{prefix}/llamafactory-cli" if prefix else "llamafactory-cli"
    cmd = _build_inference_api_cmd(
        cli_path=cli_path,
        model_path=model_path,
        adapter_path=adapter_path,
        template=template,
        infer_dtype=infer_dtype,
    )
    env = os.environ.copy()
    env["API_PORT"] = port
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    for k, v in config.get("env_vars", {}).items():
        env[k] = str(v)
    import logging
    logging.getLogger(__name__).info("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, env=env)
    return process, port


def start_inference_api(
    model_path: str,
    adapter_path: Optional[str] = None,
    template: str = "qwen3_nothink",
    api_port: Optional[str] = None,
    cuda_devices: str = "0",
    bf16: bool = False,
    infer_dtype: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    启动推理 API 服务（OpenAI 兼容）。
    api_port 为 None 时使用内部端口 INFERENCE_INTERNAL_PORT，由主服务 8000 代理对外提供。
    当调用方传入 infer_dtype 或 bf16 时，在 llm_AIO 侧直接拼命令并传入 --infer_dtype，以规避
    子进程不读环境变量导致的 CUBLAS_STATUS_INVALID_VALUE（可传 infer_dtype=float32）。
    返回: (subprocess.Popen 进程对象, 实际监听端口)，需由调用方管理生命周期。
    """
    config = config or {}
    port = api_port if api_port else INFERENCE_INTERNAL_PORT

    # 需要传推理精度时，由 llm_AIO 直接拼命令并加 --infer_dtype（子进程不读 env 时仍生效）
    chosen_dtype = None
    if infer_dtype is not None and infer_dtype in ("auto", "float16", "bfloat16", "float32"):
        chosen_dtype = infer_dtype
    elif bf16 is True:
        chosen_dtype = "bfloat16"

    if chosen_dtype is not None:
        process, port = _start_inference_api_with_infer_dtype(
            model_path=model_path,
            adapter_path=adapter_path,
            template=template,
            port=port,
            cuda_devices=cuda_devices,
            infer_dtype=chosen_dtype,
            config=config,
        )
        return process, port, " ".join(
            _build_inference_api_cmd(
                cli_path=(f"{_get_command_prefix(config).rstrip('/')}/llamafactory-cli" if _get_command_prefix(config) else "llamafactory-cli"),
                model_path=model_path,
                adapter_path=adapter_path,
                template=template,
                infer_dtype=chosen_dtype,
            )
        )

    # 不指定 infer_dtype 时，也不要走 FactoryCli.server()（它不返回 Popen，拿不到 pid，api/stop 失效）。
    prefix = _get_command_prefix(config).rstrip("/")
    cli_path = f"{prefix}/llamafactory-cli" if prefix else "llamafactory-cli"
    cmd = _build_inference_api_cmd(
        cli_path=cli_path,
        model_path=model_path,
        adapter_path=adapter_path,
        template=template,
        infer_dtype=None,
    )
    env = os.environ.copy()
    env["API_PORT"] = port
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    for k, v in config.get("env_vars", {}).items():
        env[k] = str(v)
    import logging
    logging.getLogger(__name__).info("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, env=env)
    return process, port, " ".join(cmd)


def list_available_datasets(dataset_dir: str) -> Dict[str, Any]:
    """
    列出指定目录下可用的数据集。
    
    Args:
        dataset_dir: 数据集目录路径
    
    Returns:
        包含数据集信息的字典
    """
    try:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            return {
                "success": False,
                "message": f"数据集目录不存在: {dataset_dir}",
                "datasets": []
            }
        
        datasets = []
        for item in dataset_path.iterdir():
            if item.is_dir():
                # 检查是否为数据集目录
                if (item / "dataset_info.json").exists() or any(
                    ext in str(f) for f in item.iterdir() if f.is_file()
                    for ext in [".json", ".jsonl", ".csv", ".txt"]
                ):
                    datasets.append({
                        "name": item.name,
                        "path": str(item.resolve()),
                        "type": "directory"
                    })
            elif item.is_file():
                # 检查是否为数据集文件
                if any(ext in str(item) for ext in [".json", ".jsonl", ".csv", ".txt"]):
                    datasets.append({
                        "name": item.name,
                        "path": str(item.resolve()),
                        "type": "file"
                    })
        
        return {
            "success": True,
            "message": "",
            "datasets": datasets,
            "dataset_dir": str(dataset_path.resolve())
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"列出数据集失败: {str(e)}",
            "datasets": []
        }


def validate_dataset_format(dataset_path: str, data_format: str) -> Dict[str, Any]:
    """
    验证数据集格式是否正确。
    
    Args:
        dataset_path: 数据集路径
        data_format: 数据格式 (alpaca, sharegpt, text_to_image, image_classification, text_classification)
    
    Returns:
        验证结果
    """
    try:
        import json
        import csv
        
        path = Path(dataset_path)
        if not path.exists():
            return {
                "success": False,
                "message": f"数据集不存在: {dataset_path}"
            }
        
        if path.is_file():
            # 验证文件格式
            ext = path.suffix.lower()
            
            if ext == ".json":
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data_format == "alpaca":
                        # 验证 alpaca 格式
                        if isinstance(data, list):
                            for item in data:
                                if not isinstance(item, dict):
                                    return {
                                        "success": False,
                                        "message": "Alpaca 格式错误: 列表项必须是字典"
                                    }
                                if "instruction" not in item:
                                    return {
                                        "success": False,
                                        "message": "Alpaca 格式错误: 缺少 instruction 字段"
                                    }
                        else:
                            return {
                                "success": False,
                                "message": "Alpaca 格式错误: 数据必须是列表"
                            }
                    elif data_format == "sharegpt":
                        # 验证 sharegpt 格式
                        if isinstance(data, list):
                            for item in data:
                                if not isinstance(item, dict):
                                    return {
                                        "success": False,
                                        "message": "ShareGPT 格式错误: 列表项必须是字典"
                                    }
                                if "conversations" not in item:
                                    return {
                                        "success": False,
                                        "message": "ShareGPT 格式错误: 缺少 conversations 字段"
                                    }
                                if not isinstance(item["conversations"], list):
                                    return {
                                        "success": False,
                                        "message": "ShareGPT 格式错误: conversations 必须是列表"
                                    }
                        else:
                            return {
                                "success": False,
                                "message": "ShareGPT 格式错误: 数据必须是列表"
                            }
                    elif data_format == "text_to_image":
                        # 验证文生图格式
                        if isinstance(data, list):
                            for item in data:
                                if not isinstance(item, dict):
                                    return {
                                        "success": False,
                                        "message": "Text-to-Image 格式错误: 列表项必须是字典"
                                    }
                                if "prompt" not in item:
                                    return {
                                        "success": False,
                                        "message": "Text-to-Image 格式错误: 缺少 prompt 字段"
                                    }
                                if "image" not in item:
                                    return {
                                        "success": False,
                                        "message": "Text-to-Image 格式错误: 缺少 image 字段"
                                    }
                        else:
                            return {
                                "success": False,
                                "message": "Text-to-Image 格式错误: 数据必须是列表"
                            }
            
            elif ext == ".jsonl":
                # 验证 jsonl 格式
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                json.loads(line)
                            except json.JSONDecodeError:
                                return {
                                    "success": False,
                                    "message": f"JSONL 格式错误: 第 {i+1} 行不是有效的 JSON"
                                }
            
            elif ext == ".csv":
                # 验证 csv 格式
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        return {
                            "success": False,
                            "message": "CSV 格式错误: 缺少表头"
                        }
            
            elif ext == ".txt":
                # 简单验证 txt 格式
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        return {
                            "success": False,
                            "message": "TXT 格式错误: 文件为空"
                        }
        
        return {
            "success": True,
            "message": "数据集格式验证成功",
            "dataset_path": str(path.resolve()),
            "data_format": data_format
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"验证数据集格式失败: {str(e)}"
        }


def get_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """
    获取数据集的详细信息。
    
    Args:
        dataset_path: 数据集路径
    
    Returns:
        包含数据集信息的字典
    """
    try:
        import json
        import os
        
        path = Path(dataset_path)
        if not path.exists():
            return {
                "success": False,
                "message": f"数据集不存在: {dataset_path}"
            }
        
        dataset_info = {
            "path": str(path.resolve()),
            "type": "directory" if path.is_dir() else "file",
            "size": 0,
            "files": [],
            "format": None
        }
        
        # 计算数据集大小
        if path.is_dir():
            total_size = 0
            files = []
            for root, dirs, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        files.append({
                            "name": filename,
                            "path": file_path,
                            "size": file_size
                        })
            dataset_info["size"] = total_size
            dataset_info["files"] = files
        else:
            dataset_info["size"] = os.path.getsize(path)
            dataset_info["files"] = [{
                "name": path.name,
                "path": str(path.resolve()),
                "size": dataset_info["size"]
            }]
        
        # 检测数据集格式
        if path.is_file():
            ext = path.suffix.lower()
            if ext == ".json":
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            if data and isinstance(data[0], dict):
                                if "instruction" in data[0]:
                                    dataset_info["format"] = "alpaca"
                                elif "conversations" in data[0]:
                                    dataset_info["format"] = "sharegpt"
                                elif "prompt" in data[0] and "image" in data[0]:
                                    dataset_info["format"] = "text_to_image"
                                elif "text" in data[0] and "label" in data[0]:
                                    dataset_info["format"] = "text_classification"
                except Exception:
                    pass
            elif ext == ".jsonl":
                dataset_info["format"] = "jsonl"
            elif ext == ".csv":
                dataset_info["format"] = "csv"
            elif ext == ".txt":
                dataset_info["format"] = "txt"
        
        return {
            "success": True,
            "message": "获取数据集信息成功",
            "dataset_info": dataset_info
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"获取数据集信息失败: {str(e)}"
        }

