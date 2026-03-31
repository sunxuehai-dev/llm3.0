import os
import dashscope
from app.models.multimodal import ImageGenRequest

class AliyunImageAdapter:
    def __init__(self):
        # Force unset NO_PROXY to respect system VPN
        if "NO_PROXY" in os.environ: del os.environ["NO_PROXY"]
        if "no_proxy" in os.environ: del os.environ["no_proxy"]
        
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key

    async def generate(self, request: ImageGenRequest):
        # Implementation of Aliyun Wanx API call
        # https://help.aliyun.com/zh/dashscope/developer-reference/api-details-9
        
        try:
            model = request.model

            # --- QWEN-IMAGE-MAX (MultiModalConversation API) ---
            if model == "qwen-image-max":
                from dashscope import MultiModalConversation

                messages = [
                    {
                        "role": "user",
                        "content": [{"text": request.prompt}]
                    }
                ]
                
                # Max typical size format
                size = request.config.size or "1024*1024"
                if "x" in size:
                    size = size.replace("x", "*")
                
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "result_format": "message",
                    "stream": False,
                    "size": size
                }
                
                if request.config.negative_prompt:
                    kwargs['negative_prompt'] = request.config.negative_prompt
                if request.config.prompt_extend is not None:
                    kwargs['prompt_extend'] = request.config.prompt_extend
                    
                import asyncio
                loop = asyncio.get_event_loop()
                
                def run_multimodal_image():
                    return MultiModalConversation.call(**kwargs)
                    
                rsp = await loop.run_in_executor(None, run_multimodal_image)
                
                if rsp.status_code == 200:
                    try:
                        # Extract the image URL from the message content
                        content = rsp.output.choices[0].message.content
                        for item in content:
                            if "image" in item:
                                return {"data": [{"url": item["image"]}]}
                        return {"error": "No image found in MultiModal response"}
                    except Exception as e:
                        return {"error": f"Failed to parse MultiModal image response: {e}"}
                else:
                    return {"error": rsp.message}

            # --- LEGACY MODELS (ImageSynthesis API) ---
            else:
                # Size format normalization
                size = request.config.size
                if "qwen-image" in model and "x" in size:
                    size = size.replace("x", "*")
                    
                call_args = {
                    "model": model,
                    "prompt": request.prompt,
                    "n": 1,
                    "size": size
                }

                if request.config.negative_prompt:
                    call_args['negative_prompt'] = request.config.negative_prompt
                
                if request.config.prompt_extend is not None:
                    call_args['prompt_extend'] = request.config.prompt_extend

                import asyncio
                loop = asyncio.get_event_loop()
                
                def run_synthesis():
                    return dashscope.ImageSynthesis.call(**call_args)
                    
                rsp = await loop.run_in_executor(None, run_synthesis)
                
                if rsp.status_code == 200:
                    # Return list of image URLs in OpenAI format
                    return {"data": [{"url": res['url']} for res in rsp.output.results]}
                else:
                    return {"error": rsp.message}
                    
        except Exception as e:
            return {"error": str(e)}
