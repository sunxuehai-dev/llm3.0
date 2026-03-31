"""
OpenAI 风格对话接口，对接 LangGraph RAG（与 Knowledge_retrive_and_search/main.py 行为对齐）。
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.permissions import Permission, require_permission
from app.security import get_current_user
from app.services.rag import (
    get_rag_dependencies,
    get_rag_init_error,
    is_rag_ready,
)
from app.services.rag.kb_control import (
    get_vector_kb_control_state,
    is_vector_kb_disabled,
    set_vector_kb_disabled,
)
from app.services.rag.rag_llms import LLMInitializationError
from app.services.rag.response_format import format_rag_response
from app.services.rag.vector_store import (
    delete_rag_chroma_collection,
    delete_rag_chunks_by_source_file,
    ingest_pdf_to_chroma,
    list_rag_vector_index_summary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-retrieval", tags=["Knowledge Retrieval (RAG)"])


class RagChatMessage(BaseModel):
    role: str
    content: str


class RagChatCompletionRequest(BaseModel):
    messages: List[RagChatMessage]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None


class RagChatCompletionResponseChoice(BaseModel):
    index: int
    message: RagChatMessage
    finish_reason: Optional[str] = None


class RagChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[RagChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


class RagVectorKbSettingsBody(BaseModel):
    """教师/管理员：启用或暂停向量知识库（入库与对话）。"""

    vector_kb_disabled: bool = Field(
        ..., description="true=暂停向量知识库；false=恢复"
    )
    note: Optional[str] = Field(
        None, max_length=2000, description="可选说明，展示给排查或前端提示"
    )


def _raise_if_vector_kb_disabled(db: Session) -> None:
    if not is_vector_kb_disabled(db):
        return
    st = get_vector_kb_control_state(db)
    msg = "向量知识库已由管理员暂停，暂不可入库与对话。"
    if st.get("note"):
        msg += f" 说明：{st['note']}"
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=msg,
    )


async def handle_non_stream_response(
    user_input: str, graph: Any, tool_config: Any, config: dict
) -> JSONResponse:
    content = None
    try:
        events = graph.stream(
            {
                "messages": [{"role": "user", "content": user_input}],
                "rewrite_count": 0,
            },
            config,
        )
        for event in events:
            for value in event.values():
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("RAG: 响应中无有效 messages")
                    continue
                last_message = value["messages"][-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            logger.info("RAG: 调用工具 %s", tool_call["name"])
                    continue
                if hasattr(last_message, "content"):
                    content = last_message.content
                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        logger.info("RAG: 工具输出 [%s]", last_message.name)
                    else:
                        logger.info("RAG: 模型最终回复")
    except Exception as e:
        logger.error("RAG: 处理非流式响应失败: %s", e)

    formatted = str(format_rag_response(content)) if content else "No response generated"
    try:
        resp = RagChatCompletionResponse(
            choices=[
                RagChatCompletionResponseChoice(
                    index=0,
                    message=RagChatMessage(role="assistant", content=formatted),
                    finish_reason="stop",
                )
            ]
        )
    except Exception:
        resp = RagChatCompletionResponse(
            choices=[
                RagChatCompletionResponseChoice(
                    index=0,
                    message=RagChatMessage(
                        role="assistant", content="Error generating response"
                    ),
                    finish_reason="error",
                )
            ]
        )
    return JSONResponse(content=resp.model_dump())


async def handle_stream_response(user_input: str, graph: Any, config: dict) -> StreamingResponse:
    async def generate_stream():
        try:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            stream_data = graph.stream(
                {
                    "messages": [{"role": "user", "content": user_input}],
                    "rewrite_count": 0,
                },
                config,
                stream_mode="messages",
            )
            for message_chunk, metadata in stream_data:
                try:
                    node_name = metadata.get("langgraph_node") if metadata else None
                    if node_name in ("generate", "agent"):
                        if hasattr(message_chunk, "content"):
                            chunk = message_chunk.content
                        else:
                            chunk = str(message_chunk) if message_chunk else ""
                        if not isinstance(chunk, str):
                            chunk = str(chunk) if chunk else ""
                        payload = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                except Exception as ex:
                    logger.error("RAG: 流式分块错误: %s", ex)
            done = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error("RAG: 流生成失败: %s", e)
            yield f"data: {json.dumps({'error': 'Stream processing failed'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _require_rag():
    if not is_rag_ready():
        detail = get_rag_init_error() or "RAG 未初始化，请检查依赖与 RAG_* 环境变量"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


@router.get("/status")
async def rag_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.VIEW_KNOWLEDGE_BASE)
    ctrl = get_vector_kb_control_state(db)
    return {
        "ready": is_rag_ready(),
        "error": get_rag_init_error(),
        "vector_kb_disabled": ctrl["vector_kb_disabled"],
        "vector_kb_note": ctrl.get("note"),
        "vector_kb_updated_at": ctrl.get("updated_at"),
    }


@router.get("/library")
async def rag_library(
    metadata_scan_limit: int = 20000,
    current_user: User = Depends(get_current_user),
):
    """
    列出当前 RAG 向量库（Chroma）的持久化位置、块数及已入库文档来源文件名汇总。
    不依赖 LangGraph 是否 ready；仅只读打开本地 Chroma 目录。
    """
    require_permission(current_user.role, Permission.VIEW_KNOWLEDGE_BASE)
    if metadata_scan_limit < 1:
        raise HTTPException(status_code=400, detail="metadata_scan_limit 须 >= 1")
    if metadata_scan_limit > 100_000:
        raise HTTPException(status_code=400, detail="metadata_scan_limit 过大，最大 100000")
    data = list_rag_vector_index_summary(metadata_scan_limit=metadata_scan_limit)
    data["storage_hint"] = (
        "向量与分块元数据在服务端 persist_directory 下；"
        "通过 ingest-pdf 上传的 PDF 默认不落盘保留原文件，仅写入向量与元数据。"
    )
    return data


@router.post("/chat/completions")
async def rag_chat_completions(
    request: RagChatCompletionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.VIEW_KNOWLEDGE_BASE)
    _raise_if_vector_kb_disabled(db)
    _require_rag()
    graph, tool_config = get_rag_dependencies()

    if not request.messages or not request.messages[-1].content:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    user_input = request.messages[-1].content
    uid = request.userId or str(current_user.id)
    conv = request.conversationId or "default"
    config = {
        "configurable": {
            "thread_id": f"{uid}@@{conv}",
            "user_id": uid,
        }
    }

    if request.stream:
        return await handle_stream_response(user_input, graph, config)
    return await handle_non_stream_response(user_input, graph, tool_config, config)


@router.post("/ingest-pdf")
async def rag_ingest_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """将上传的 PDF 写入当前 RAG Chroma 库（与离线 vectorSave_langchain 逻辑一致）。"""
    require_permission(current_user.role, Permission.MANAGE_KNOWLEDGE_BASE)
    _raise_if_vector_kb_disabled(db)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    import tempfile

    suffix = f"_{uuid.uuid4().hex}.pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        body = await file.read()
        tmp.write(body)
        tmp_path = tmp.name
    try:
        try:
            ingest_pdf_to_chroma(tmp_path, source_display_name=file.filename)
        except LLMInitializationError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"嵌入模型不可用: {e}",
            ) from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {"ok": True, "message": "已向量化并写入 Chroma", "filename": file.filename}


@router.get("/admin/vector-settings")
async def rag_get_vector_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """教师/管理员查看向量知识库暂停开关。"""
    require_permission(current_user.role, Permission.MANAGE_KNOWLEDGE_BASE)
    return get_vector_kb_control_state(db)


@router.put("/admin/vector-settings")
async def rag_put_vector_settings(
    body: RagVectorKbSettingsBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """教师/管理员启用或暂停向量知识库（暂停后无法入库与对话，直至重新启用）。"""
    require_permission(current_user.role, Permission.MANAGE_KNOWLEDGE_BASE)
    state, audit = set_vector_kb_disabled(
        db,
        disabled=body.vector_kb_disabled,
        note=body.note,
        updated_by_id=current_user.id,
    )
    logger.info("RAG: %s (user_id=%s)", audit, current_user.id)
    return {"ok": True, "message": audit, **state}


@router.delete("/library/by-source")
async def rag_delete_library_by_source(
    source_file: str = Query(..., description="入库时的文件名，如 doc.pdf"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    按来源文件名删除向量分块（不删其它文档）。
    典型「编辑」流程：先删再重新 POST ingest-pdf。
    """
    require_permission(current_user.role, Permission.MANAGE_KNOWLEDGE_BASE)
    result = delete_rag_chunks_by_source_file(source_file)
    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "删除失败"),
        )
    return {
        "ok": True,
        "source_file": source_file,
        "deleted_count": result.get("deleted_count", 0),
    }


@router.delete("/library")
async def rag_delete_library_all(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除当前 Chroma 集合（清空向量知识库）。教师/管理员。"""
    require_permission(current_user.role, Permission.MANAGE_KNOWLEDGE_BASE)
    result = delete_rag_chroma_collection()
    if not result.get("ok") and result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )
    return {
        "ok": bool(result.get("ok")),
        "message": result.get("message") or result.get("error"),
        "collection_name": result.get("collection_name"),
        "persist_directory_resolved": result.get("persist_directory_resolved"),
    }
