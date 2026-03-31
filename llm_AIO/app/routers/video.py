from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.models.multimodal import VideoGenRequest
from app.adapters.aliyun_video import AliyunVideoAdapter
from app.database import get_convert_db
from app.services.file_processing import convert_base64_to_url

router = APIRouter(prefix="/api/playground/videos", tags=["video"])

@router.post("/generations")
async def generate_video(request: VideoGenRequest, db: Session = Depends(get_convert_db)):
    # 非本地模型：将上传的 base64 转为 URL 再传给下游
    if request.provider != "local" and request.image_url and request.image_url.startswith("data:"):
        request.image_url = await convert_base64_to_url(request.image_url, db, default_ext="png")

    if request.provider == "aliyun":
        adapter = AliyunVideoAdapter()
        return await adapter.generate_video(request)
    elif request.provider == "doubao":
        # Lazy import to avoid circular dependency or unnecessary load context
        from app.adapters.doubao import DoubaoAdapter
        adapter = DoubaoAdapter()
        return await adapter.generate_video(request)
    else:
        raise HTTPException(status_code=400, detail="Unknown provider")