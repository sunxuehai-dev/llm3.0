import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QLoRASupportResult:
    supported: bool
    reasons: List[str]
    detected: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported": self.supported,
            "reasons": self.reasons,
            "detected": self.detected,
        }


_QUANTIZED_HINT_FILES = (
    "quantize_config.json",  # GPTQ 常见
    "quant_config.json",
    "gptq_config.json",
    "awq_config.json",
    "awq_model.safetensors",  # 部分 AWQ 仓库会有
)


def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _detect_quantization_from_config(cfg: dict) -> Tuple[bool, Dict[str, Any], List[str]]:
    detected: Dict[str, Any] = {}
    reasons: List[str] = []

    # 1) transformers / HF 常见：config.json 里可能包含 quantization_config
    qcfg = cfg.get("quantization_config")
    if isinstance(qcfg, dict) and qcfg:
        detected["quantization_config"] = qcfg
        reasons.append("config.json 中包含 quantization_config（模型看起来已是量化权重，不建议再做 QLoRA 量化）")

    # 2) 一些仓库会把量化方法写在字段里（不统一，尽量兜底）
    for k in ("quant_method", "quantization_method", "quantize_method"):
        v = cfg.get(k)
        if isinstance(v, str) and v:
            detected[k] = v
            if "量化" not in "".join(reasons):
                reasons.append(f"config.json 中检测到字段 {k}={v}（疑似已量化模型）")

    # 3) 通过模型名/架构字段粗略判断（如包含 GPTQ/AWQ/AQLM 等关键字）
    name_like = " ".join(
        str(x)
        for x in [
            cfg.get("_name_or_path", ""),
            cfg.get("model_type", ""),
            " ".join(cfg.get("architectures", []) or []),
        ]
        if x
    )
    name_like_lower = name_like.lower()
    for kw in ("gptq", "awq", "aqlm", "gguf", "exl2", "exllama"):
        if kw in name_like_lower:
            detected["name_like_quant_hint"] = name_like
            reasons.append(f"config.json 字段包含关键字 {kw}（通常代表已量化/特定推理格式，不能作为 QLoRA 的基础模型）")
            break

    return (len(reasons) > 0, detected, reasons)


def check_qlora_support(model_path: str) -> QLoRASupportResult:
    """
    判断某个“基础模型目录”是否适合使用 QLoRA（在 llmfactory 这里，QLoRA=再量化 + LoRA）。

    结论偏保守：如果模型看起来已经是 GPTQ/AWQ/AQLM/GGUF 等量化格式，则直接判定不支持。
    """
    reasons: List[str] = []
    detected: Dict[str, Any] = {"model_path": model_path}

    if not model_path or not os.path.exists(model_path):
        return QLoRASupportResult(
            supported=False,
            reasons=["model_path 不存在"],
            detected=detected,
        )

    if os.path.isfile(model_path):
        # 单文件通常是 gguf / onnx 等，不是 HF 目录
        ext = os.path.splitext(model_path)[1].lower()
        detected["is_file"] = True
        detected["file_ext"] = ext
        return QLoRASupportResult(
            supported=False,
            reasons=["QLoRA 需要 HuggingFace 模型目录（含 config.json）；单文件模型不支持"],
            detected=detected,
        )

    # 目录模型必须含 config.json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        # gguf 常见：目录里只有 *.gguf
        names = _safe_listdir(model_path)
        if any(n.lower().endswith(".gguf") for n in names):
            detected["found_gguf"] = True
            return QLoRASupportResult(
                supported=False,
                reasons=["检测到 GGUF（llama.cpp）格式；QLoRA 训练链路不支持 GGUF 作为基础模型"],
                detected=detected,
            )
        return QLoRASupportResult(
            supported=False,
            reasons=["未发现 config.json（不是标准 HuggingFace 模型目录），无法使用 QLoRA"],
            detected=detected,
        )

    cfg = _read_json(config_path) or {}
    detected["model_type"] = cfg.get("model_type") or ""
    detected["architectures"] = cfg.get("architectures") or []

    # 先用 config.json 的信息判断“已量化”
    has_quant, q_detected, q_reasons = _detect_quantization_from_config(cfg)
    if q_detected:
        detected.update(q_detected)
    reasons.extend(q_reasons)

    # 再用目录文件名判断（量化模型常带这些文件）
    names = _safe_listdir(model_path)
    detected["files_sample"] = sorted(names)[:50]

    for fn in _QUANTIZED_HINT_FILES:
        if fn in names:
            reasons.append(f"模型目录包含 `{fn}`（常见于已量化模型），不能作为 QLoRA 的基础模型")

    if any(n.lower().endswith(".gguf") for n in names):
        reasons.append("模型目录包含 .gguf 文件（llama.cpp 量化格式），不能作为 QLoRA 的基础模型")

    # 常见的“量化权重命名”线索（保守）
    lowered = " ".join(n.lower() for n in names)
    for kw in ("gptq", "awq", "aqlm", "exl2", "exllama"):
        if kw in lowered:
            reasons.append(f"模型文件名包含关键字 {kw}（疑似量化/特定推理格式），不能作为 QLoRA 的基础模型")
            break

    supported = len(reasons) == 0
    if supported:
        detected["note"] = "未检测到“已量化模型”特征；理论上可用于 QLoRA（仍取决于具体依赖与硬件环境）"

    return QLoRASupportResult(supported=supported, reasons=reasons, detected=detected)

