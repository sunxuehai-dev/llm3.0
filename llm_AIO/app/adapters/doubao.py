import os
import json
import httpx
from typing import AsyncGenerator, List, Union, Any
from app.adapters.base import BaseAdapter
from app.models.chat import ChatRequest, ChatResponseChunk

# 尝试导入 Ark，防止未安装报错
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    Ark = None

class DoubaoAdapter(BaseAdapter):
    def __init__(self):
        self.api_key = os.getenv("VOLCENGINE_API_KEY") 
        if not self.api_key:
             self.api_key = os.getenv("ARK_API_KEY")

        self.base_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

        if Ark:
            self.client = Ark(
                api_key=self.api_key,
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            )
        else:
            self.client = None
            print("Warning: volcenginesdkarkruntime not installed. Using httpx fallback.")

    async def chat(self, request: ChatRequest) -> AsyncGenerator[ChatResponseChunk, None]:
        if not self.api_key:
             yield ChatResponseChunk(content="Error: VOLCENGINE_API_KEY or ARK_API_KEY not found.", finish_reason="stop")
             return

        # Common message formatting for Doubao Multimodal
        formatted_messages = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                formatted_messages.append(msg.dict())
            elif isinstance(msg.content, list):
                # Handle multimodal content (image, video)
                new_content = []
                for item in msg.content:
                    if item.get("type") == "text":
                        new_content.append({"type": "text", "text": item.get("text")}) # or input_text based on strictly needed, though text usually works
                    elif item.get("type") == "image_url":
                        # 兼容 image / image_url：前端可能用 image 传 URL
                        img_val = item.get("image_url") or item.get("image")
                        img_url = (img_val.get("url", img_val) if isinstance(img_val, dict) else img_val) if img_val else None
                        if img_url:
                            new_content.append({"type": "image_url", "image_url": {"url": img_url} if isinstance(img_url, str) else img_val})
                        else:
                            new_content.append(item)
                    elif item.get("type") == "video_url":
                        # 兼容 video / video_url：前端用 video 传 base64/URL，API 要求 video_url
                        vid_val = item.get("video_url") or item.get("video")
                        vid_url = (vid_val.get("url", vid_val) if isinstance(vid_val, dict) else vid_val) if vid_val else None
                        if vid_url:
                            new_content.append({"type": "video_url", "video_url": {"url": vid_url} if isinstance(vid_url, str) else vid_val})
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                
                formatted_messages.append({"role": msg.role, "content": new_content})

        # 1. SDK Mode
        if self.client:
            try:
                 call_kwargs = {
                    "model": request.model,
                    "messages": formatted_messages,
                    "stream": True
                 }

                 # These models do not support parameter adjustments, omit them entirely to prevent API rejection
                 if request.model not in ["doubao-lite-32k-character-250228", "doubao-1-5-lite-32k-250115"]:
                     # Only include parameters if they are not None:
                     if request.config.temperature is not None:
                         call_kwargs["temperature"] = request.config.temperature
                     if request.config.top_p is not None:
                         call_kwargs["top_p"] = request.config.top_p
                     if request.config.max_tokens is not None:
                         call_kwargs["max_tokens"] = request.config.max_tokens
                     if request.config.frequency_penalty is not None:
                         call_kwargs["frequency_penalty"] = request.config.frequency_penalty
                 
                 # Inject reasoning_effort and thinking if available in config
                 if hasattr(request.config, "reasoning_effort") and request.config.reasoning_effort:
                     call_kwargs["reasoning_effort"] = request.config.reasoning_effort
                 if hasattr(request.config, "enable_thinking") and request.config.enable_thinking is not None:
                     call_kwargs["thinking"] = {"type": "enabled" if request.config.enable_thinking else "disabled"}
                     
                 stream = self.client.chat.completions.create(**call_kwargs)


                 for chunk in stream:
                     if not chunk.choices:
                         continue
                     
                     delta = chunk.choices[0].delta
                     content = delta.content
                     finish_reason = chunk.choices[0].finish_reason
                     
                     yield ChatResponseChunk(content=content or "", finish_reason=finish_reason)

            except Exception as e:
                yield ChatResponseChunk(content=f"Doubao/Ark SDK Error: {str(e)}", finish_reason="stop")
            return

        # 2. HTTPX Fallback Mode (When SDK is missing)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # In raw HTTPX mode, strictly adhere to Ark's explicit input_image / input_video schema if needed
        ark_raw_messages = []
        for msg in formatted_messages:
            if isinstance(msg["content"], list):
                new_content = []
                for item in msg["content"]:
                    if item.get("type") == "text":
                        new_content.append({"type": "text", "text": item.get("text")}) # using text instead of input_text as ark supports standard text too
                    elif item.get("type") == "image_url":
                        img_val = item.get("image_url") or item.get("image")
                        img_url = (img_val.get("url", img_val) if isinstance(img_val, dict) else img_val) if img_val else None
                        if img_url:
                            new_content.append({"type": "image_url", "image_url": {"url": img_url} if isinstance(img_url, str) else img_val})
                        else:
                            new_content.append(item)
                    elif item.get("type") == "video_url":
                        vid_val = item.get("video_url") or item.get("video")
                        vid_url = (vid_val.get("url", vid_val) if isinstance(vid_val, dict) else vid_val) if vid_val else None
                        if vid_url:
                            new_content.append({"type": "video_url", "video_url": {"url": vid_url} if isinstance(vid_url, str) else vid_val})
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                ark_raw_messages.append({"role": msg["role"], "content": new_content})
            else:
                 ark_raw_messages.append(msg)
                 
        payload = {
            "model": request.model,
            "messages": ark_raw_messages,
            "stream": True
        }

        if request.model not in ["doubao-lite-32k-character-250228", "doubao-1-5-lite-32k-250115"]:
            if request.config.temperature is not None:
                payload["temperature"] = request.config.temperature
            if request.config.top_p is not None:
                payload["top_p"] = request.config.top_p
            if request.config.max_tokens is not None:
                payload["max_tokens"] = request.config.max_tokens
            if request.config.frequency_penalty is not None:
                payload["frequency_penalty"] = request.config.frequency_penalty

        if hasattr(request.config, "reasoning_effort") and request.config.reasoning_effort:
            payload["reasoning_effort"] = request.config.reasoning_effort
            
        if hasattr(request.config, "enable_thinking") and request.config.enable_thinking is not None:
             payload["thinking"] = {"type": "enabled" if request.config.enable_thinking else "disabled"}

        try:
            # Force no proxy for domestic endpoints if user has global proxy set
            async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
                async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield ChatResponseChunk(content=f"Error {response.status_code}: {error_text.decode('utf-8')}", finish_reason="stop")
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    reasoning_content = delta.get("reasoning_content", None)
                                    finish_reason = choices[0].get("finish_reason", None)
                                    
                                yield ChatResponseChunk(content=content, reasoning_content=reasoning_content, finish_reason=finish_reason)
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
             yield ChatResponseChunk(content=f"Doubao/HTTPX Fallback Error: {str(e)}", finish_reason="stop")


    async def generate_video(self, request: Any):
        """Asynchronous video generation using Volcengine's Tasks API"""
        if not self.api_key:
            return {"error": "Error: VOLCENGINE_API_KEY or ARK_API_KEY not found."}
            
        task_url = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Build Doubao's highly specific payload
        content_array = []
        if getattr(request, 'prompt', None):
            content_array.append({
                "type": "text",
                "text": request.prompt
            })
        if getattr(request, 'image_url', None):
            content_array.append({
                "type": "image_url",
                "image_url": {
                    "url": request.image_url
                }
            })
            
        payload = {
            "model": request.model,
            "content": content_array
        }
        
        # Parse extra config options
        config = getattr(request, 'config', None)
        if config:
            # Resolution & Ratio
            if hasattr(config, "resolution"):
                # E.g. "1280x720 (720P 16:9)"
                res_str = config.resolution.lower()
                if "1080p" in res_str:
                    payload["resolution"] = "1080p"
                elif "480p" in res_str:
                    payload["resolution"] = "480p"
                else:
                    payload["resolution"] = "720p" # Default
                    
            if getattr(config, "ratio", None):
                payload["ratio"] = config.ratio
            else:
                # Fallback from old resolution string combination
                if hasattr(config, "resolution"):
                    res_str = config.resolution.lower()
                    if "16:9" in res_str: payload["ratio"] = "16:9"
                    elif "9:16" in res_str: payload["ratio"] = "9:16"
                    elif "1:1" in res_str: payload["ratio"] = "1:1"
                    elif "4:3" in res_str: payload["ratio"] = "4:3"
                    elif "3:4" in res_str: payload["ratio"] = "3:4"
                    elif "21:9" in res_str: payload["ratio"] = "21:9"
                    else: payload["ratio"] = "16:9"
            
            # Duration
            if hasattr(config, "duration"):
                payload["duration"] = int(config.duration)
                
            # Extra Doubao params mappings
            if getattr(config, "seed", None) is not None:
                payload["seed"] = config.seed
            if getattr(config, "camera_fixed", None) is not None:
                payload["camera_fixed"] = config.camera_fixed
            if getattr(config, "watermark", None) is not None:
                payload["watermark"] = config.watermark
        
        print(f"[DoubaoVideo] Sending Payload: {json.dumps(payload, ensure_ascii=False)}")
        
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            try:
                # 1. Submit Job
                submit_resp = await client.post(task_url, json=payload, headers=headers)
                
                if submit_resp.status_code == 200:
                    submit_data = submit_resp.json()
                    task_id = submit_data.get("id")
                    
                    if not task_id:
                         return {"error": f"Failed to get task ID. Response: {submit_data}"}
                         
                    print(f"[DoubaoVideo] Task Submitted: {task_id}. Polling for completion...")
                    
                    # 2. Poll for Completion
                    import asyncio
                    status = "queued"
                    while status in ["queued", "running", "PENDING", "RUNNING"]:
                        await asyncio.sleep(5)
                        
                        poll_url = f"{task_url}/{task_id}"
                        poll_resp = await client.get(poll_url, headers=headers)
                        
                        if poll_resp.status_code == 200:
                            poll_data = poll_resp.json()
                            status = poll_data.get("status", "unknown").lower()
                            print(f"[DoubaoVideo] Task {task_id} status: {status}")
                            
                            if status in ["succeed", "succeeded"]:
                                content = poll_data.get("content", {})
                                video_url = content.get("video_url")
                                return {"output": {"video_url": video_url}}
                            elif status in ["failed", "canceled"]:
                                return {"error": f"Video Generation Failed: {poll_data.get('error', 'Unknown error')}"}
                        else:
                            return {"error": f"Task Check Failed: {poll_resp.status_code} - {poll_resp.text}"}
                            
                    return {"error": f"Unknown or timeout status: {status}"}
                else:
                    return {"error": f"API Error {submit_resp.status_code}: {submit_resp.text}"}
                    
            except Exception as e:
                import traceback
                return {"error": f"{type(e).__name__}: {str(e)}", "traceback": traceback.format_exc()}

