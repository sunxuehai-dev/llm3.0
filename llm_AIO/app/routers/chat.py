import json
import base64
import re
import io
import os
import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session
from app.models.chat import ChatRequest
from app.services.model_factory import ModelFactory
from app.database import get_convert_db
from convert_url import FileConverter
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/api/playground", tags=["chat"])

from app.services.file_processing import process_chat_request_base64

@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_convert_db)):
    try:
        # Pre-process Base64 content
        if request.provider != "local":
            await process_chat_request_base64(request, db)
        
        adapter = ModelFactory.get_adapter(request.provider)
        
        # 非流式：后端先拼接完整内容，再一次性返回给前端
        if not request.config.stream:
            full_content = ""
            last_audio = None
            finish_reason = None
            async for chunk in adapter.chat(request):
                if chunk.content:
                    full_content += chunk.content
                if chunk.audio:
                    last_audio = chunk.audio
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
            response_data = {
                "id": f"chatcmpl-{request.provider}",
                "object": "chat.completion",
                "created": 0,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                        },
                        "finish_reason": finish_reason or "stop",
                    }
                ],
            }
            if last_audio is not None:
                response_data["choices"][0]["message"]["audio"] = last_audio
            return Response(
                content=json.dumps(response_data, ensure_ascii=False),
                media_type="application/json; charset=utf-8",
            )
        
        # 流式：逐 chunk 推送 SSE
        async def event_generator():
            async for chunk in adapter.chat(request):
                response_data = {
                    "id": f"chatcmpl-{request.provider}",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": chunk.finish_reason
                        }
                    ]
                }
                if chunk.content:
                    response_data["choices"][0]["delta"]["content"] = chunk.content
                if chunk.audio:
                    response_data["choices"][0]["delta"]["audio"] = chunk.audio
                data = json.dumps(response_data, ensure_ascii=False)
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=headers,
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except NotImplementedError as nie:
        raise HTTPException(status_code=501, detail=str(nie))
    except Exception as e:
        # In a real app, log this error
        print(f"Error processing chat request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
