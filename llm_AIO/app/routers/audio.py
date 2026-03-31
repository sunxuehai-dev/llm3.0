from fastapi import APIRouter, HTTPException, Response, Depends
from sqlalchemy.orm import Session

from app.models.multimodal import AudioSpeechRequest, AudioTranscriptionRequest
from app.adapters.aliyun_audio import AliyunAudioAdapter
from app.adapters.aliyun_audio import AliyunAudioAdapter
from app.adapters.xunfei_audio import XunfeiAudioAdapter
from app.adapters.local_audio import LocalAudioAdapter
from app.database import get_convert_db
from app.services.file_processing import convert_base64_to_url

router = APIRouter(prefix="/api/playground/audio", tags=["audio"])

@router.post("/speech")
async def text_to_speech(request: AudioSpeechRequest):
    """
    文本转语音（TTS）
    """
    if request.provider == "aliyun":
        adapter = AliyunAudioAdapter()
        result = await adapter.speech(request)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        # Return audio file to browser
        return Response(content=result["audio_data"], media_type=result["content_type"])
        
    elif request.provider == "local":
        # adapter = LocalAudioAdapter()
        return {"error": "Local TTS not implemented yet"}
    else:
        raise HTTPException(status_code=400, detail="Unknown provider")


@router.post("/transcription")
async def audio_transcription(request: AudioTranscriptionRequest, db: Session = Depends(get_convert_db)):
    """
    语音识别（ASR）- 将音频转换为文字
    """
    # 非本地模型：将上传的 base64 转为 URL 再传给下游
    if request.provider != "local" and request.input and request.input.startswith("data:"):
        request.input = await convert_base64_to_url(request.input, db, default_ext=request.config.format or "wav")

    if request.provider == "aliyun":
        adapter = AliyunAudioAdapter()
        result = await adapter.transcribe(request)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 返回识别结果
        return {
            "text": result.get("text", ""),
            "raw": result.get("raw", "")
        }

    elif request.provider == "xunfei":
        adapter = XunfeiAudioAdapter()
        result = await adapter.transcribe(request)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return {
            "text": result.get("text", ""),
            "raw": result.get("raw", "")
        }
        
    elif request.provider == "local":
        adapter = LocalAudioAdapter()
        result = await adapter.transcribe(request)
        
        if "error" in result:
             # Handle 500 scenarios vs dependency missing
             detail = result["error"]
             status = 500
             if "Missing dependency" in detail:
                 status = 503
             raise HTTPException(status_code=status, detail=detail)

        return {
            "text": result.get("text", ""),
            "raw": result.get("raw", "")
        }
    else:
        raise HTTPException(status_code=400, detail="Unknown provider")
