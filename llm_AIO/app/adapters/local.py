import os
import json
import httpx
import base64
from typing import AsyncGenerator
from app.adapters.base import BaseAdapter
from app.models.chat import ChatRequest, ChatResponseChunk

class LocalAdapter(BaseAdapter):
    def __init__(self):
        # Default to Ollama port, but allow override
        self.base_url = os.getenv("LOCAL_MODEL_URL", "http://localhost:11434/v1/chat/completions")
        self.api_key = "local" # Dummy key usually required by libraries, but we use raw HTTP

    async def _process_messages(self, messages):
        """Pre-process messages to handle images for Ollama"""
        processed_messages = []
        for msg in messages:
            content = msg.content
            # Check if content is complex (list)
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict):
                        # Handle frontend custom format: {"image": "url_or_base64"}
                        if "image" in item:
                            image_data = item["image"]
                            if image_data.startswith("http"):
                                try:
                                    async with httpx.AsyncClient() as client:
                                        resp = await client.get(image_data)
                                        if resp.status_code == 200:
                                            b64_img = base64.b64encode(resp.content).decode('utf-8')
                                            # Ollama OpenAI compat expects data URI
                                            new_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
                                except Exception as e:
                                    print(f"Error fetching image for local model: {e}")
                            else:
                                # It's already a base64 string or data URI
                                new_content.append({"type": "image_url", "image_url": {"url": image_data}})
                        
                        # Handle frontend custom format: {"text": "..."}
                        elif "text" in item and "type" not in item:
                            new_content.append({"type": "text", "text": item["text"]})
                            
                        # Handle standard OpenAI format: {"type": "image_url", ...}
                        elif item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                            # If it's a URL, download and convert to base64
                            if image_url.startswith("http"):
                                try:
                                    async with httpx.AsyncClient() as client:
                                        resp = await client.get(image_url)
                                        if resp.status_code == 200:
                                            b64_img = base64.b64encode(resp.content).decode('utf-8')
                                            # Ollama OpenAI compat expects data URI
                                            item["image_url"]["url"] = f"data:image/jpeg;base64,{b64_img}"
                                except Exception as e:
                                    print(f"Error fetching image for local model: {e}")
                            new_content.append(item)
                        else:
                            # Fallback for other dict structures
                            new_content.append(item)
                    else:
                        new_content.append(item)
                processed_messages.append({"role": msg.role, "content": new_content})
            else:
                processed_messages.append({"role": msg.role, "content": content})
        return processed_messages

    async def _call_model_sync(self, model: str, messages: list) -> str:
        """Call a model and collect the full response (non-streaming)"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Handle OpenAI format
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("message", {}).get("content", "")
                # Handle Ollama native format
                if "message" in data:
                    return data.get("message", {}).get("content", "")
            return ""

    async def _translate_to_chinese(self, english_text: str) -> str:
        """Translate English text to Chinese using Qwen 2.5 7B"""
        messages = [
            {"role": "system", "content": "你是一个专业翻译。将用户提供的英文精准翻译成中文。只输出翻译结果，不要解释。"},
            {"role": "user", "content": english_text}
        ]
        return await self._call_model_sync("qwen2.5:7b", messages)


    async def chat(self, request: ChatRequest) -> AsyncGenerator[ChatResponseChunk, None]:
        headers = {
            "Content-Type": "application/json"
        }
        
        # Process images (convert URLs to Base64)
        processed_messages = await self._process_messages(request.messages)

        # Detect llava-cn: use llava internally, then translate
        is_llava_cn = request.model == "llava-cn"
        actual_model = "llava" if is_llava_cn else request.model

        payload = {
            "model": actual_model,
            "messages": processed_messages,
            "temperature": request.config.temperature,
            "top_p": request.config.top_p,
            "stream": not is_llava_cn  # Non-streaming for llava-cn to collect full response
        }
        
        # Add optional parameters if provided
        if request.config.top_k is not None:
            payload["top_k"] = request.config.top_k
        if request.config.max_tokens is not None:
            payload["max_tokens"] = request.config.max_tokens
        if request.config.repetition_penalty is not None:
            payload["repeat_penalty"] = request.config.repetition_penalty  # Ollama uses repeat_penalty
        if request.config.seed is not None:
            payload["seed"] = request.config.seed
        if request.config.presence_penalty is not None:
            payload["presence_penalty"] = request.config.presence_penalty
        if request.config.frequency_penalty is not None:
            payload["frequency_penalty"] = request.config.frequency_penalty

        # Special path for llava-cn: collect full response, translate, then yield
        if is_llava_cn:
            async with httpx.AsyncClient(timeout=120.0) as client:
                try:
                    response = await client.post(self.base_url, headers=headers, json=payload)
                    if response.status_code != 200:
                        yield ChatResponseChunk(content=f"LLaVA Error: {response.text}", finish_reason="error")
                        return
                    
                    data = response.json()
                    english_text = ""
                    if "choices" in data and len(data["choices"]) > 0:
                        english_text = data["choices"][0].get("message", {}).get("content", "")
                    elif "message" in data:
                        english_text = data.get("message", {}).get("content", "")
                    
                    if english_text:
                        # Translate to Chinese
                        chinese_text = await self._translate_to_chinese(english_text)
                        yield ChatResponseChunk(content=chinese_text, finish_reason="stop")
                    else:
                        yield ChatResponseChunk(content="无法获取图片描述", finish_reason="error")
                except Exception as e:
                    yield ChatResponseChunk(content=f"LLaVA-CN Error: {str(e)}", finish_reason="error")
            return

        async with httpx.AsyncClient() as client:
            try:
                # Connection might fail if local server isn't running
                async with client.stream("POST", self.base_url, headers=headers, json=payload, timeout=60.0) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield ChatResponseChunk(content=f"Local Error {response.status_code}: {error_text.decode()}", finish_reason="error")
                        return

                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        try:
                            text = chunk.decode("utf-8", errors="ignore")
                            buffer += text
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if line.startswith("data: "):
                                    data_str = line[6:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data = json.loads(data_str)
                                        if "message" in data: # Ollama native format
                                            content = data.get("message", {}).get("content", "")
                                            if content:
                                                yield ChatResponseChunk(content=content)
                                            if data.get("done"):
                                                break
                                        elif "choices" in data and len(data["choices"]) > 0: # OpenAI format
                                            delta = data["choices"][0].get("delta", {})
                                            content = delta.get("content", "")
                                            finish_reason = data["choices"][0].get("finish_reason")
                                            if content or finish_reason:
                                                yield ChatResponseChunk(content=content, finish_reason=finish_reason)
                                    except json.JSONDecodeError:
                                        pass
                        except Exception:
                            pass
            except httpx.ConnectError:
                yield ChatResponseChunk(content="Error: Could not connect to local model server. Is Ollama/vLLM running?", finish_reason="error")
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield ChatResponseChunk(content=f"Local Adapter Error: {str(e)}", finish_reason="error")
