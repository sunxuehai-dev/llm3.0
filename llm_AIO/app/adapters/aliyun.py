import os
import dashscope
from typing import AsyncGenerator, List, Any, Union
from http import HTTPStatus
from app.adapters.base import BaseAdapter
from app.models.chat import ChatRequest, ChatResponseChunk


def _normalize_multimodal_content(content: Union[str, List[Any]]) -> Union[str, List[dict]]:
    """将前端/通用格式的 content 转为 DashScope 要求的带 type 的多模态项。
    DashScope 要求：content 为列表时，每项须包含 type 字段，取值为 'text'|'image'|'audio'|'video'|'image_hw'。
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    normalized = []
    for item in content:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        # 已有合法 type 则直接保留（仅补全 type 若为别名）
        if "type" in item and item["type"] in ("text", "image", "audio", "video", "image_hw"):
            normalized.append(dict(item))
            continue
        # 无 type：按字段推断（兼容 image/video 及 OpenAI 格式 image_url/video_url）
        if "text" in item:
            normalized.append({"type": "text", "text": item["text"]})
        elif "image" in item:
            normalized.append({"type": "image", "image": item["image"]})
        elif "image_url" in item:
            # OpenAI 格式: image_url 可能为 {"url": "..."} 或直接字符串
            img_val = item["image_url"]
            url = img_val.get("url", img_val) if isinstance(img_val, dict) else img_val
            if url:
                normalized.append({"type": "image", "image": url})
        elif "audio" in item:
            normalized.append({"type": "audio", "audio": item["audio"]})
        elif "video" in item:
            normalized.append({"type": "video", "video": item["video"]})
        elif "video_url" in item:
            # OpenAI 格式: video_url 可能为 {"url": "..."} 或直接字符串
            vid_val = item["video_url"]
            url = vid_val.get("url", vid_val) if isinstance(vid_val, dict) else vid_val
            if url:
                normalized.append({"type": "video", "video": url})
        elif "image_hw" in item:
            normalized.append({"type": "image_hw", "image_hw": item["image_hw"]})
    return normalized


class AliyunAdapter(BaseAdapter):
    def __init__(self):
        # Force unset all proxy/VPN variables to avoid SSL/Connection errors
        proxy_keys = ["NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]
        for key in proxy_keys:
            if key in os.environ:
                del os.environ[key]

        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("Warning: DASHSCOPE_API_KEY not found.")
        dashscope.api_key = self.api_key

    async def chat(self, request: ChatRequest) -> AsyncGenerator[ChatResponseChunk, None]:
        # 多模态接口要求 content 为列表时每项带 type（text/image/audio/video/image_hw），此处统一规范化
        messages = [
            {"role": msg.role, "content": _normalize_multimodal_content(msg.content)}
            for msg in request.messages
        ]
        
        try:
            # --- Dedicated OpenAI-Compatible path for qwen3.5 / deep reasoning models ---
            if "qwen3.5" in request.model.lower():
                import httpx
                import json
                
                # Transform messages for OpenAI format (image_url/video_url instead of image/video keys)
                oai_messages = []
                for msg in messages:
                    if isinstance(msg["content"], list):
                        new_content = []
                        for item in msg["content"]:
                            if item["type"] == "text":
                                new_content.append({"type": "text", "text": item["text"]})
                            elif item["type"] == "image":
                                new_content.append({"type": "image_url", "image_url": {"url": item["image"]}})
                            elif item["type"] == "video":
                                new_content.append({"type": "video_url", "video_url": {"url": item["video"]}})
                            else:
                                new_content.append(item)
                        oai_messages.append({"role": msg["role"], "content": new_content})
                    else:
                        oai_messages.append({"role": msg["role"], "content": msg["content"]})
                        
                payload = {
                    "model": request.model,
                    "messages": oai_messages,
                    "stream": True,
                    "temperature": request.config.temperature if getattr(request.config, "temperature", None) is not None else 0.8,
                }
                if getattr(request.config, "top_p", None) is not None:
                     payload["top_p"] = request.config.top_p
                     
                if getattr(request.config, "enable_thinking", False):
                     # Crucial for Qwen 3.5 deep thinking capabilities
                     payload["extra_body"] = {"enable_thinking": True}
                     
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                    async with client.stream("POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            err_text = await response.aread()
                            yield ChatResponseChunk(content=f"API Error {response.status_code}: {err_text.decode()}", finish_reason="error")
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
                                        content = delta.get("content")
                                        content = content if content is not None else ""
                                        reasoning_content = delta.get("reasoning_content")
                                        finish_reason = choices[0].get("finish_reason")
                                        yield ChatResponseChunk(content=content, reasoning_content=reasoning_content, finish_reason=finish_reason)
                                except json.JSONDecodeError:
                                    continue
                return # End Qwen 3.5 branch execution
            # --- End Dedicated Path ---

            # --- qwen3-omni-flash: 语音输出需使用 compatible-mode + modalities/audio ---
            if "qwen3-omni-flash" in request.model or "qwen-omni-flash" in request.model.lower():
                import httpx
                import json

                def _parse_modalities(modalities) -> list:
                    """解析 modalities：支持 ["text","audio"] 或 "text,audio (文本+音频)" 等格式"""
                    if modalities is None:
                        return ["text"]
                    if isinstance(modalities, list):
                        return modalities if modalities else ["text"]
                    s = str(modalities)
                    if "audio" in s.lower():
                        return ["text", "audio"]
                    return ["text"]

                def _get_voice(config) -> str:
                    """获取音色：config.voice 或 config.audio.voice"""
                    v = getattr(config, "voice", None)
                    if v:
                        return v
                    if config.audio and isinstance(config.audio, dict):
                        return config.audio.get("voice", "Cherry")
                    return "Cherry"

                modalities = _parse_modalities(request.config.modalities)
                voice = _get_voice(request.config)
                audio_config = {"voice": voice, "format": "wav"} if "audio" in modalities else None

                oai_messages = []
                for msg in messages:
                    if isinstance(msg["content"], list):
                        new_content = []
                        for item in msg["content"]:
                            if item.get("type") == "text":
                                new_content.append({"type": "text", "text": item.get("text", "")})
                            elif item.get("type") == "image":
                                new_content.append({"type": "image_url", "image_url": {"url": item["image"]}})
                            elif item.get("type") == "video":
                                new_content.append({"type": "video_url", "video_url": {"url": item["video"]}})
                            elif item.get("type") == "audio":
                                new_content.append({"type": "audio", "audio": item["audio"]})
                            else:
                                new_content.append(item)
                        oai_messages.append({"role": msg["role"], "content": new_content})
                    else:
                        oai_messages.append({"role": msg["role"], "content": msg["content"]})

                payload = {
                    "model": request.model,
                    "messages": oai_messages,
                    "stream": True,
                    "modalities": modalities,
                    "stream_options": {"include_usage": True},
                }
                if audio_config:
                    payload["audio"] = audio_config
                if getattr(request.config, "temperature", None) is not None:
                    payload["temperature"] = request.config.temperature
                if getattr(request.config, "top_p", None) is not None:
                    payload["top_p"] = request.config.top_p
                if getattr(request.config, "top_k", None) is not None:
                    payload["top_k"] = request.config.top_k

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                    async with client.stream("POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            err_text = await response.aread()
                            yield ChatResponseChunk(content=f"API Error {response.status_code}: {err_text.decode()}", finish_reason="error")
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
                                        content = delta.get("content")
                                        content = content if content is not None else ""
                                        finish_reason = choices[0].get("finish_reason")
                                        audio_content = None
                                        if isinstance(delta.get("audio"), dict):
                                            audio_content = delta["audio"]
                                        yield ChatResponseChunk(content=content, finish_reason=finish_reason, audio=audio_content)
                                except json.JSONDecodeError:
                                    continue
                return
            # --- End qwen3-omni-flash path ---

            # Omni models (qwen-audio, qwen-omni) ARE multimodal and should use MultiModalConversation or specific handling
            is_multimodal = ("vl" in request.model or "audio" in request.model or "omni" in request.model)
            
            if is_multimodal:
                from dashscope import MultiModalConversation
                # 前端未传 modalities 时默认为 ["text"]，否则 Omni 可能不流式返回或只返回 [DONE]
                modalities = request.config.modalities if request.config.modalities is not None else ["text"]
                responses = MultiModalConversation.call(
                    model=request.model,
                    messages=messages,
                    stream=True,
                    incremental_output=True,
                    top_p=request.config.top_p,
                    top_k=request.config.top_k,
                    enable_search=request.config.enable_search,
                    modalities=modalities,
                    result_format="message",
                )
                # Note: Voice might need to be passed if supported by SDK. 
                # Checking SDK signature, usually it's part of generation config or similar?
                # For MultiModalConversation, 'voice' might not be a direct arg in all versions?
                # But 'modalities' is definitely needed.
            elif "image" in request.model and "omni" not in request.model:
                from dashscope import ImageSynthesis
                # Extract prompt from last user message
                prompt = messages[-1]['content']
                if isinstance(prompt, list): # Handle multimodal content if any
                    for item in prompt:
                        if isinstance(item, dict) and 'text' in item:
                            prompt = item['text']
                            break
                        elif isinstance(item, str):
                            prompt = item
                            break
                
                # Image Synthesis is typically synchronous or task-based
                # We will wait for it (sync call) and yield the result
                rsp = ImageSynthesis.call(
                    model=request.model,
                    prompt=prompt,
                    n=1,
                    size=request.config.size or '1024x1024',
                    style=request.config.style or '<auto>'
                )
                
                if rsp.status_code == HTTPStatus.OK:
                    # Result is typically in rsp.output.results[0].url
                    if rsp.output and rsp.output.results:
                        img_url = rsp.output.results[0].url
                        # Markdown format for image
                        markdown_response = f"![Generated Image]({img_url})"
                        yield ChatResponseChunk(content=markdown_response, finish_reason="stop")
                    else:
                         yield ChatResponseChunk(content="Error: No image URL in response", finish_reason="error")
                else:
                     yield ChatResponseChunk(content=f"Error: {rsp.message}", finish_reason="error")
                return # End generator

            elif "wan" in request.model or "video" in request.model:
                from dashscope import VideoSynthesis
                # Extract prompt
                prompt = messages[-1]['content']
                if isinstance(prompt, list): 
                    for item in prompt:
                        if isinstance(item, dict) and 'text' in item:
                            prompt = item['text']
                            break
                        elif isinstance(item, str):
                            prompt = item
                            break
                
                # Extract Image URL if I2V
                img_url_param = None
                # Check messages for image in the last message content
                last_content = messages[-1]['content']
                if isinstance(last_content, list):
                    for item in last_content:
                        if isinstance(item, dict) and 'image' in item:
                            img_url_param = item['image']
                            break

                # For now, let's support T2V (Text to Video) via prompt.
                # If I2V is needed, we need to locate the image. 
                # The prompt might contain it?
                
                # Video Synthesis call
                rsp = VideoSynthesis.call(
                    model=request.model,
                    prompt=prompt,
                    img_url=img_url_param,
                    size=request.config.resolution or "1280x720",
                    # debug: duration might be int/str
                )
                
                # Handle Job
                if rsp.status_code == HTTPStatus.OK:
                    # Video is ASHNC. We get a task_id.
                    if rsp.output and rsp.output.task_id:
                        task_id = rsp.output.task_id
                        yield ChatResponseChunk(content=f"Generating Task ID: {task_id} ...", finish_reason=None)
                        
                        # Poll
                        import time
                        import json
                        status = "PENDING"
                        while status in ["PENDING", "RUNNING", "QUEUED"]:
                            time.sleep(5)
                            task = VideoSynthesis.fetch(task_id)
                            if task.status_code != HTTPStatus.OK:
                                yield ChatResponseChunk(content=f"Error checking task: {task.message}", finish_reason="error")
                                return
                            
                            status = task.output.task_status
                            if status == "SUCCEEDED":
                                video_url = task.output.video_url
                                yield ChatResponseChunk(content=f"\n![Generated Video]({video_url})", finish_reason="stop")
                                return
                            elif status in ["FAILED", "CANCELED"]:
                                yield ChatResponseChunk(content=f"Video Generation Failed: {task.output.message}", finish_reason="error")
                                return
                    else:
                         yield ChatResponseChunk(content="Error: No Task ID in response", finish_reason="error")
                else:
                     yield ChatResponseChunk(content=f"Error: {rsp.message}", finish_reason="error")
                return

            else:
                responses = dashscope.Generation.call(
                    model=request.model,
                    messages=messages,
                    result_format='message',
                    stream=True,
                    incremental_output=True,
                    temperature=request.config.temperature,
                    top_p=request.config.top_p,
                    top_k=request.config.top_k,
                    repetition_penalty=request.config.repetition_penalty,
                    enable_search=request.config.enable_search,
                    enable_thinking=request.config.enable_thinking,
                    # Omni Params
                    modalities=request.config.modalities,
                    audio=request.config.audio
                )

            for response in responses:
                if response.status_code != HTTPStatus.OK:
                    yield ChatResponseChunk(content=f"Error: {response.message}", finish_reason="error")
                    continue
                # 标准结构：output.choices[0].message.content
                if response.output and response.output.choices:
                    choice = response.output.choices[0]
                    reasoning_content = None
                    finish_reason = None
                    content = ""
                    
                    # Handle diverse response structures (Dict vs Object)
                    if isinstance(choice, dict):
                        message = choice.get('message', {})
                        raw_content = message.get('content', '')
                        reasoning_content = message.get('reasoning_content')
                        finish_reason = choice.get('finish_reason')
                    else:
                        message = getattr(choice, 'message', None)
                        raw_content = getattr(message, 'content', '') if message else ''
                        reasoning_content = getattr(message, 'reasoning_content', None) if message else None
                        finish_reason = getattr(choice, 'finish_reason', None)
                        
                    # Handle Diverse Content Structure (Text & Audio)
                    audio_content = None
                    if isinstance(raw_content, list):
                        content_parts = []
                        for item in raw_content:
                            if isinstance(item, dict):
                                if 'text' in item:
                                    content_parts.append(item.get('text', ''))
                                if 'audio' in item:
                                    audio_temp = item['audio']
                                    if audio_temp:
                                        audio_content = audio_temp
                            elif isinstance(item, str):
                                content_parts.append(item)
                        content = "".join(content_parts)
                    elif raw_content is None:
                        content = ""
                    else:
                        content = str(raw_content)

                    # Debug Logging
                    with open("debug_aliyun.log", "a", encoding="utf-8") as f:
                        f.write(f"--- Chunk ---\n")
                        try:
                            f.write(f"Choice Content: {str(choice)}\n")
                        except Exception as e:
                            f.write(f"Logging Error: {e}\n")

                    if not audio_content:
                        try:
                            if isinstance(choice, dict):
                                delta = choice.get('delta', {})
                                if isinstance(delta, dict) and 'audio' in delta:
                                    audio_content = delta['audio']
                            elif hasattr(choice, 'delta'):
                                d = getattr(choice, 'delta', None)
                                if d and hasattr(d, 'audio'):
                                    audio_content = d.audio
                        except Exception:
                            pass
                    
                    yield ChatResponseChunk(
                        content=content, 
                        reasoning_content=reasoning_content,
                        finish_reason=finish_reason,
                        audio=audio_content
                    )
                else:
                    # output 无 choices 时尝试从 output.message / output.content 取（部分 Omni 流式结构）
                    content = ""
                    if response.output:
                        out = response.output
                        if hasattr(out, 'message') and getattr(out.message, 'content', None):
                            content = str(out.message.content) if out.message.content else ""
                        elif hasattr(out, 'content') and out.content:
                            content = str(out.content)
                    if content or (response.output and getattr(response.output, 'choices', None) is None):
                        yield ChatResponseChunk(content=content or "(empty)", finish_reason="stop")
        except Exception as e:
            print(f"Aliyun Adapter Error: {e}")
            yield ChatResponseChunk(content=f"Internal Error: {str(e)}", finish_reason="error")
