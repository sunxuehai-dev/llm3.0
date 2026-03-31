import os
import httpx
from typing import Dict, Any, List
from app.models.multimodal import ImageGenRequest

class DoubaoImageAdapter:
    def __init__(self):
        self.api_key = os.getenv("VOLCENGINE_API_KEY") 
        if not self.api_key:
             self.api_key = os.getenv("ARK_API_KEY")
             
        # Support Image Generations
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

    async def generate(self, request: ImageGenRequest) -> Dict[str, Any]:
        """
        Calls the Volcano Engine /api/v3/images/generations endpoint.
        """
        if not self.api_key:
            return {"error": "Volcano Engine API key not configured"}
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "AIO-Gateway-ImageGen/1.0"
        }

        # Build base payload
        payload = {
            "model": request.model,
            "prompt": request.prompt
        }
        
        # Format images if present (i2i / i2t2i)
        # Seedream supports URLs or Base64 (data:image/png;base64,...)
        if request.image:
            if isinstance(request.image, list):
                payload["image"] = request.image
            else:
                payload["image"] = [request.image]

        # Config parameter mapping for sizes and other Seedream features
        if hasattr(request.config, "size") and request.config.size:
             payload["size"] = request.config.size

        if hasattr(request.config, "sequential_image_generation") and request.config.sequential_image_generation:
             payload["sequential_image_generation"] = request.config.sequential_image_generation
             
        if hasattr(request.config, "stream") and request.config.stream is not None:
             payload["stream"] = request.config.stream
             
        if hasattr(request.config, "watermark") and request.config.watermark is not None:
             payload["watermark"] = request.config.watermark
             
        if hasattr(request.config, "seed") and request.config.seed is not None:
             payload["seed"] = request.config.seed
             
        # Map style parameter if present
        if hasattr(request.config, "style") and request.config.style:
             payload["style"] = request.config.style
             
        # Map prompt_extend parameter if present
        if hasattr(request.config, "prompt_extend") and request.config.prompt_extend is not None:
             payload["prompt_extend"] = request.config.prompt_extend

        if hasattr(request.config, "response_format") and request.config.response_format:
             payload["response_format"] = request.config.response_format
        else:
             # Default to URL per official docs for easier mapping
             payload["response_format"] = "url"

        try:
            # Volcano Engine endpoint sometimes drops connections without a User-Agent or fails SSL in certain environments
            async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                # We expect either a JSON payload format or streaming
                if response.status_code == 200:
                    resp_data = response.json()
                    if "data" in resp_data:
                        return {"data": resp_data["data"]}
                    else:
                        return resp_data
                else:
                    return {"error": f"Doubao API error: {response.status_code} - {response.text}"}
                    
        except Exception as e:
            return {"error": f"Failed to call Doubao API: {str(e)}"}
