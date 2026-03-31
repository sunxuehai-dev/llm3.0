from abc import ABC, abstractmethod
from typing import AsyncGenerator
from app.models.chat import ChatRequest, ChatResponseChunk

class BaseAdapter(ABC):
    @abstractmethod
    async def chat(self, request: ChatRequest) -> AsyncGenerator[ChatResponseChunk, None]:
        """
        Stream chat responses from the provider.
        Must yield ChatResponseChunk objects.
        """
        pass
