import os
import json
import httpx
from typing import AsyncGenerator
from app.adapters.base import BaseAdapter
from app.models.chat import ChatRequest, ChatResponseChunk

class DeepSeekAdapter(BaseAdapter):
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/chat/completions"
        if not self.api_key:
            print("Warning: DEEPSEEK_API_KEY not found.")

    async def chat(self, request: ChatRequest) -> AsyncGenerator[ChatResponseChunk, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": request.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.config.temperature,
            "top_p": request.config.top_p,
            "max_tokens": request.config.max_tokens,
            "stream": True
        }

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", self.base_url, headers=headers, json=payload, timeout=60.0) as response:
                    if response.status_code != 200:
                        error_text = await response.read()
                        yield ChatResponseChunk(content=f"Error {response.status_code}: {error_text.decode()}", finish_reason="error")
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    reasoning_content = delta.get("reasoning_content", None)
                                    finish_reason = data["choices"][0].get("finish_reason")
                                    
                                    if content or reasoning_content or finish_reason:
                                        yield ChatResponseChunk(
                                            content=content, 
                                            reasoning_content=reasoning_content,
                                            finish_reason=finish_reason
                                        )
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                yield ChatResponseChunk(content=f"Connection Error: {str(e)}", finish_reason="error")
