import re
import base64
import io
import uuid
import os
from sqlalchemy.orm import Session
from convert_url import FileConverter

def get_converter(file_category: str, db: Session):
    """Get FileConverter instance"""
    storage_type = os.getenv('STORAGE_TYPE', 'local')
    server_url = os.getenv('SERVER_URL', 'http://localhost:8000')
    
    config = {
        'server_url': server_url
    }
    
    if storage_type == 'oss':
        config.update({
            'endpoint': os.getenv('OSS_ENDPOINT'),
            'access_key_id': os.getenv('OSS_ACCESS_KEY_ID'),
            'access_key_secret': os.getenv('OSS_ACCESS_KEY_SECRET'),
            'bucket_name': os.getenv('OSS_BUCKET_NAME'),
            'region': os.getenv('OSS_REGION')
        })
    
    return FileConverter(
        session=db,
        file_category=file_category,
        storage_type=storage_type,
        **config
    )

async def convert_base64_to_url(base64_str: str, db: Session, default_ext: str = "png") -> str:
    """
    单个 Base64 字符串转 URL
    支持格式: 'data:image/png;base64,xxxx...' 或纯 base64 字符串 (需指定 default_ext)
    """
    if not base64_str or not isinstance(base64_str, str):
        return base64_str
        
    # Check if it's data URI
    base64_pattern = re.compile(r'^data:(image|video|audio)/([a-zA-Z0-9]+);base64,(.+)$')
    match = base64_pattern.match(base64_str)
    
    file_type = "image" # Default fallback
    ext = default_ext
    b64_data = base64_str
    
    if match:
        file_type, ext, b64_data = match.groups()
    elif len(base64_str) > 100: # Simple heuristic check if it's likely base64 without prefix
        # Assume it is raw base64, use default_ext and type
        pass
    else:
        # Not base64, maybe already a URL
        return base64_str

    try:
        # Decode
        file_data = base64.b64decode(b64_data)
        
        # Create file-like object
        file_obj = io.BytesIO(file_data)
        filename = f"auto_{uuid.uuid4().hex[:8]}.{ext}"
        file_obj.name = filename
        
        # Determine category for converter
        # file_type from match is usually singular (image/video/audio)
        converter = get_converter(file_type, db)
        
        # Upload
        file_record, error = await converter.upload(file_obj, user_id="auto_convert")
        
        if error:
            print(f"Error converting base64: {error}")
            return base64_str
            
        # Get URL
        url = converter.get_url(file_record.id)
        if url:
            print(f"Converted Base64 to URL: {url}")
            return url
            
    except Exception as e:
        print(f"Exception during base64 conversion: {e}")
        
    return base64_str

async def process_chat_request_base64(request, db: Session):
    """
    Process ChatRequest messages for Base64 content.
    Mutates the request object in place.
    支持: 顶层 image/video/audio 的 base64，以及嵌套 image_url.url / video_url.url 的 base64。
    """
    base64_pattern = re.compile(r'^data:(image|video|audio)/([a-zA-Z0-9]+);base64,(.+)$')

    async def _convert_if_base64(val, db, default_ext="png"):
        if isinstance(val, str) and val.startswith("data:"):
            match = base64_pattern.match(val)
            if match:
                return await convert_base64_to_url(val, db, default_ext=match.group(2) or default_ext)
        return val

    for message in request.messages:
        content = message.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # 1. 顶层 key: image, video, audio, file
                    for key, value in item.items():
                        if isinstance(value, str) and value.startswith('data:'):
                            match = base64_pattern.match(value)
                            if match:
                                new_url = await convert_base64_to_url(value, db, default_ext=match.group(2) or "png")
                                if new_url != value:
                                    item[key] = new_url
                    # 2. 嵌套 image_url.url / video_url.url（OpenAI 格式）
                    for nest_key in ("image_url", "video_url"):
                        nest = item.get(nest_key)
                        if isinstance(nest, dict) and "url" in nest:
                            url_val = nest["url"]
                            if isinstance(url_val, str) and url_val.startswith("data:"):
                                new_url = await _convert_if_base64(url_val, db)
                                if new_url != url_val:
                                    nest["url"] = new_url
