from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.code_online_service import chat_with_llm, run_python_code, run_python_code_in_thread


router = APIRouter(prefix="/api/playground/code-online", tags=["code-online"])


class RunCodeRequest(BaseModel):
    code: str = Field(..., description="需要执行的 Python 代码")
    env: str = Field(default="default", description="本地 Python 环境名（由 CODE_ONLINE_* 配置映射）")
    timeout_seconds: int = Field(default=10, ge=1, le=120, description="执行超时时间（秒）")


class RunCodeResponse(BaseModel):
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None


class ChatWithLlmRequest(BaseModel):
    input: str = Field(..., description="用户需求/问题")
    code: Optional[str] = Field(default=None, description="可选：当前代码内容（用于结合分析）")
    error: Optional[str] = Field(default=None, description="可选：代码运行报错信息（用于结合分析）")


class ChatWithLlmResponse(BaseModel):
    success: bool
    reply: Optional[str] = None
    error: Optional[str] = None


def _code_run_enabled() -> bool:
    raw = os.getenv("CODE_ONLINE_ENABLE_RUN_CODE", "true").strip().lower()
    return raw in ("true", "1", "t", "yes", "y", "on")


@router.post("/run-code", response_model=RunCodeResponse)
async def run_code(req: RunCodeRequest):
    if not _code_run_enabled():
        raise HTTPException(status_code=403, detail="code-online 的代码执行功能已被禁用")

    if not req.code.strip():
        raise HTTPException(status_code=400, detail="代码不能为空")

    # 避免 subprocess 阻塞事件循环
    result = await run_python_code_in_thread(req.code, env=req.env, timeout_seconds=req.timeout_seconds)
    return RunCodeResponse(**result)


@router.post("/chat-with-llm", response_model=ChatWithLlmResponse)
async def chat_with_llm_api(req: ChatWithLlmRequest):
    result = await chat_with_llm(user_input=req.input, code=req.code, error=req.error)
    return ChatWithLlmResponse(**result)

