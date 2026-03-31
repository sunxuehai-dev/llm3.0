import os
import httpx
from app.models.multimodal import VideoGenRequest

class AliyunVideoAdapter:
    def __init__(self):
        # Force unset all proxy variables to respect system VPN / Direct Connection
        proxy_keys = ["NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]
        for key in proxy_keys:
            if key in os.environ:
                 del os.environ[key]
        
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"

    async def generate_video(self, request: VideoGenRequest):
        # Using Direct API via httpx as SDK support for Wanx 2.6 is missing/unclear
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable",
            "Content-Type": "application/json"
        }

        # Wanx 2.6 I2V payload structure
        input_data = {}
        if request.prompt:
             # Truncate prompt if needed (limit 1500 for wan2.6)
             input_data["prompt"] = request.prompt[:1500]
        if request.image_url:
            input_data["img_url"] = request.image_url
        if request.audio_url:
             input_data["audio_url"] = request.audio_url

        # Parameters
        # Parameters
        size = request.config.resolution.split(' ')[0]
        if "x" in size:
            size = size.replace("x", "*") # Aliyun often prefers 1280*720

        parameters = {
            "duration": request.config.duration,
            "size": size,
            "prompt_extend": request.config.prompt_extend
        }
        
        # ... (rest of params)
        
        # negative_prompt
        if request.negative_prompt:
             parameters["negative_prompt"] = request.negative_prompt[:500]

        # shot_type (only if prompt_extend is True, though model might ignore it if False)
        if request.config.prompt_extend and request.config.shot_type:
             parameters["shot_type"] = request.config.shot_type

        # audio (generate_audio) - for wan2.6 models and others that support it
        if "wan" in request.model and request.config.generate_audio:
             parameters["audio"] = True
             
        # template (if video effects)
        if request.template:
             parameters["template"] = request.template
             # if template is set, prompt is ignored (as per doc), but we still pass input_data parameters as structured.

        payload = {
            "model": request.model,
            "input": input_data,
            "parameters": parameters,
        }
        
        print(f"[AliyunVideo] Sending Payload: {payload}")

        async with httpx.AsyncClient() as client:
            try:
                # 1. Submit Job
                response = await client.post(self.base_url, json=payload, headers=headers)
                
                if response.status_code == 200:
                   data = response.json()
                   if "output" in data and "task_id" in data["output"]:
                       task_id = data["output"]["task_id"]
                       print(f"[AliyunVideo] Task Submitted: {task_id}. Polling for completion...")
                       
                       # 2. Poll for Completion
                       import asyncio
                       status = "PENDING"
                       while status in ["PENDING", "RUNNING", "QUEUED"]:
                           await asyncio.sleep(5) # Wait 5 seconds
                           
                           task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                           task_resp = await client.get(task_url, headers=headers)
                           
                           if task_resp.status_code == 200:
                               task_data = task_resp.json()
                               status = task_data["output"]["task_status"]
                               
                               if status == "SUCCEEDED":
                                   video_url = task_data["output"]["video_url"]
                                   # Return in a format that looks like immediate success with URL
                                   return {"output": {"video_url": video_url}}
                               elif status in ["FAILED", "CANCELED"]:
                                   return {"error": f"Video Generation Failed: {task_data['output']['message']}"}
                           else:
                               return {"error": f"Task Check Failed: {task_resp.text}"}
                               
                       return {"error": "Timeout or Unknown Status"}
                   
                   return {"data": data}
                else:
                    return {"error": f"API Error {response.status_code}: {response.text}"}
            except Exception as e:
                import traceback
                return {"error": f"{type(e).__name__}: {str(e)}", "traceback": traceback.format_exc()}
