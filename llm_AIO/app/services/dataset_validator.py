"""
LlamaFactory 数据集格式校验
支持 Alpaca、ShareGPT 及预训练/偏好训练格式
"""
import json
import re
from typing import Tuple, List, Optional, Any


# LlamaFactory 支持的数据格式
FORMAT_ALPACA = "alpaca"
FORMAT_SHAREGPT = "sharegpt"

# 数据类型（对应前端选项）
DATA_TYPE_TEXT_CONVERSATION = "text_conversation"
DATA_TYPE_TEXT2IMAGE = "text2image"
DATA_TYPE_IMAGE_CLASSIFICATION = "image_classification"
DATA_TYPE_TEXT_CLASSIFICATION = "text_classification"
DATA_TYPE_PRETRAIN = "pretrain"
DATA_TYPE_DPO = "dpo"
DATA_TYPE_KTO = "kto"
DATA_TYPE_TOOL_CALL = "tool_call"

# 数据集类型
DATASET_TYPE_TRAIN = "train"
DATASET_TYPE_INFERENCE = "inference"

# 数据用途
DATA_USAGE_SFT = "sft"
DATA_USAGE_DPO = "dpo"
DATA_USAGE_KTO = "kto"
DATA_USAGE_PRETRAIN = "pretrain"


def _validate_alpaca_record(record: dict, idx: int) -> Optional[str]:
    """校验单条 Alpaca 格式记录"""
    if not isinstance(record, dict):
        return f"记录 {idx}: 必须是对象"
    # instruction + output 必填
    if "instruction" not in record and "output" not in record:
        # 兼容 columns 映射后的 prompt/response
        if "prompt" not in record and "response" not in record:
            return f"记录 {idx}: Alpaca 格式需要 instruction+output 或 prompt+response"
    return None


def _validate_sharegpt_record(record: dict, idx: int) -> Optional[str]:
    """校验单条 ShareGPT 格式记录"""
    if not isinstance(record, dict):
        return f"记录 {idx}: 必须是对象"
    # messages 必填
    messages = record.get("messages") or record.get("conversations")
    if not messages:
        return f"记录 {idx}: ShareGPT 格式需要 messages 或 conversations 数组"
    if not isinstance(messages, list):
        return f"记录 {idx}: messages 必须是数组"
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"记录 {idx} 消息 {i}: 必须是对象"
        if "role" not in msg and "from" not in msg:
            return f"记录 {idx} 消息 {i}: 需要 role 或 from"
        if "content" not in msg and "value" not in msg:
            return f"记录 {idx} 消息 {i}: 需要 content 或 value"
    return None


def validate_dataset_format(
    content: bytes,
    filename: str,
    data_format: str,
) -> Tuple[bool, Optional[str], Optional[List[dict]]]:
    """
    校验数据集格式是否符合 LlamaFactory 要求
    :return: (是否有效, 错误信息, 解析后的记录列表用于抽样)
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    text = content.decode("utf-8", errors="replace")

    records: List[dict] = []

    try:
        if ext == "json":
            data = json.loads(text)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
            else:
                return False, "JSON 根节点应为数组或对象", None

        elif ext == "jsonl":
            # 流式校验：只解析前 100 条，避免大文件全量加载
            lines = text.strip().split("\n")
            max_validate = 100
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                    if len(records) >= max_validate:
                        break
                except json.JSONDecodeError as e:
                    return False, f"JSONL 第 {i+1} 行解析失败: {e}", None
            if not records:
                return False, "JSONL 无有效行", None

        elif ext == "csv":
            import csv
            import io
            reader = csv.DictReader(io.StringIO(text))
            records = list(reader)
            if not records:
                return False, "CSV 文件无有效数据", None

        else:
            return False, f"不支持的文件格式: {ext}，仅支持 json/jsonl/csv", None
    except json.JSONDecodeError as e:
        return False, f"JSON 解析失败: {e}", None
    except Exception as e:
        return False, f"解析失败: {e}", None

    if not records:
        return False, "数据集为空", None

    # 格式校验
    validate_fn = _validate_alpaca_record if data_format == FORMAT_ALPACA else _validate_sharegpt_record
    for i, rec in enumerate(records[:100]):  # 只校验前 100 条
        err = validate_fn(rec, i)
        if err:
            return False, err, None

    return True, None, records[:5]  # 返回前 5 条供展示


def sanitize_dataset_name(name: str) -> str:
    """将用户输入的名称转为 dataset_info 可用的键（仅字母数字下划线）"""
    s = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", name)
    s = re.sub(r"[\s-]+", "_", s)
    return s.strip("_").lower()[:64] or "dataset"


def make_dataset_info_name(metadata_id: str, display_name: str) -> str:
    """生成 dataset_info.json 中的唯一键"""
    slug = sanitize_dataset_name(display_name)
    return f"user_{metadata_id[:12]}_{slug}"[:80]
