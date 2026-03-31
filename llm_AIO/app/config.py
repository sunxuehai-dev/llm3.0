import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # App Info
    APP_NAME: str = os.getenv("APP_NAME", "Universal Model Playground Gateway & User Management System")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # API JSON 中 datetime 字段的展示时区（IANA，如 Asia/Shanghai）。
    # 内部仍以 UTC 存储与比较；仅影响响应里的字符串形式。
    # 设为 UTC 或留空则输出带 Z 的 UTC；国内直连展示可保留默认 Asia/Shanghai。
    API_RESPONSE_DATETIME_TZ: str = os.getenv("API_RESPONSE_DATETIME_TZ", "Asia/Shanghai")
    
    # Databases
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./user_management.db")
    CONVERT_URL_DATABASE_URL: str = os.getenv("CONVERT_URL_DATABASE_URL", "sqlite:///./videos.db")

    # llmfactory 微调集成
    LLMFACTORY_COMMAND_PREFIX: str = os.getenv("LLMFACTORY_COMMAND_PREFIX", "")
    # 本地模型文件根目录，其下每个子目录视为一个模型（用于「查看本地模型」接口）
    LLMFACTORY_MODELS_DIR: str = os.getenv("LLMFACTORY_MODELS_DIR", "")
    LLMFACTORY_DEFAULT_MODEL_PATH: str = os.getenv("LLMFACTORY_DEFAULT_MODEL_PATH", "")
    LLMFACTORY_DEFAULT_DATASET: str = os.getenv("LLMFACTORY_DEFAULT_DATASET", "")
    LLMFACTORY_DEFAULT_DATASET_DIR: str = os.getenv("LLMFACTORY_DEFAULT_DATASET_DIR", "")
    # 训练输出根目录，未配置时默认为当前工作目录下的 train_output
    LLMFACTORY_DEFAULT_OUTPUT_DIR: str = os.getenv("LLMFACTORY_DEFAULT_OUTPUT_DIR", "train_output")
    # 允许下载的训练结果根目录，未设置时使用 LLMFACTORY_DEFAULT_OUTPUT_DIR 或当前工作目录
    LLMFACTORY_OUTPUT_BASE: str = os.getenv("LLMFACTORY_OUTPUT_BASE", "")
    LLMFACTORY_DEFAULT_ADAPTER_PATH: str = os.getenv("LLMFACTORY_DEFAULT_ADAPTER_PATH", "")
    LLMFACTORY_DEFAULT_EXPORT_DIR: str = os.getenv("LLMFACTORY_DEFAULT_EXPORT_DIR", "")
    # 高显存操作前检查：要求指定 GPU 空闲显存不低于该值（MiB，与 nvidia-smi 一致）。0 表示不检查。
    LLMFACTORY_MIN_FREE_VRAM_MIB: int = int(os.getenv("LLMFACTORY_MIN_FREE_VRAM_MIB", "0") or 0)
    # 检查哪块卡（物理索引，与 nvidia-smi -i 一致），默认 0。
    LLMFACTORY_GPU_VRAM_GUARD_INDEX: int = int(os.getenv("LLMFACTORY_GPU_VRAM_GUARD_INDEX", "0") or 0)
    # true：无法通过 nvidia-smi 查询时也拒绝；false：查询不到则跳过检查（便于无 GPU / CI）。
    LLMFACTORY_GPU_VRAM_GUARD_STRICT: str = os.getenv("LLMFACTORY_GPU_VRAM_GUARD_STRICT", "false")


def get_settings() -> Settings:
    return Settings()
