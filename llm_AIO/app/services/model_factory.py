from app.adapters.base import BaseAdapter
from app.adapters.aliyun import AliyunAdapter
from app.adapters.deepseek import DeepSeekAdapter
from app.adapters.doubao import DoubaoAdapter
from app.adapters.local import LocalAdapter

class ModelFactory:
    @staticmethod
    def get_adapter(provider: str) -> BaseAdapter:
        if provider == "aliyun":
            return AliyunAdapter()
        elif provider == "deepseek":
            return DeepSeekAdapter()
        elif provider == "doubao":
            return DoubaoAdapter()
        elif provider == "local":
            return LocalAdapter()
        else:
            raise ValueError(f"Unknown provider: {provider}")
