from pathlib import Path
from typing import Optional
from .Response import Response
from .HandlerData import HandlerData
from .ResponseBackend import ResponseQwen3


def gen_distill_dataset(
        *,
        dataset_path_or_name: str,
        output_path: Path = Path('./distill.json').resolve(),
        data_type: str,
        base_url: str = "",
        api_key: str = "",
        model_name_or_path: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False,
    ):
    """
    这两个用于连接本地的Openai兼容模型接口
    base_url: str = "",
    api_key: str = "",

    model_name_or_path直连huggingface, 或者本地模型
    镜像: export HF_ENDPOINT=https://hf-mirror.com
    """
    # Check if model is Qwen (case-insensitive)
    if "qwen" in model_name_or_path.lower():
        ai = ResponseQwen3(
            model=model_name_or_path,
            base_url=base_url,
            api_key=api_key,
            enable_thinking=enable_thinking
        )
    else:
        ai = Response(
            model=model_name_or_path, 
            base_url=base_url, 
            api_key=api_key)

    if temperature is not None:
        ai.set_temperature(temperature)
    if max_tokens is not None:
        ai.set_max_tokens(max_tokens)

    data = HandlerData(
        type=data_type,
        dataset_path_or_name=dataset_path_or_name)
    
    data.gen(output_path, ai)
