"""领域 RAG：LangGraph 编排 + Chroma 检索 + 可选联网搜索。"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_graph: Any = None
_tool_config: Any = None
_init_error: Optional[str] = None


def is_rag_ready() -> bool:
    return _graph is not None and _tool_config is not None


def get_rag_dependencies() -> Tuple[Any, Any]:
    if not is_rag_ready():
        raise RuntimeError(_init_error or "RAG 未初始化")
    return _graph, _tool_config


def get_rag_init_error() -> Optional[str]:
    return _init_error


def initialize_rag() -> None:
    global _graph, _tool_config, _init_error
    _init_error = None
    flag = os.getenv("RAG_ENABLED", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        logger.info("RAG: 已关闭 (RAG_ENABLED)")
        return
    try:
        from app.services.rag.langgraph_agent import (
            ConnectionPoolError,
            ToolConfig,
            create_graph,
            save_graph_visualization,
        )
        from app.services.rag.rag_config import RagConfig
        from app.services.rag.rag_llms import get_llm
        from app.services.rag.tools_config import get_tools

        llm_chat, llm_embedding = get_llm(RagConfig.LLM_TYPE)
        if RagConfig.ENABLE_WEB_SEARCH:
            tools = get_tools(llm_embedding, llm_chat)
        else:
            tools = get_tools(llm_embedding, None)
        tool_cfg = ToolConfig(tools)

        db_uri = RagConfig.LANGGRAPH_SQLITE_URI
        if db_uri.startswith("sqlite:///"):
            rel = db_uri.replace("sqlite:///", "", 1).lstrip("./")
            parent = os.path.dirname(rel)
            if parent:
                os.makedirs(parent, exist_ok=True)

        _graph = create_graph(db_uri, llm_chat, llm_embedding, tool_cfg)
        _tool_config = tool_cfg

        if RagConfig.SAVE_GRAPH_VIZ:
            path = RagConfig.GRAPH_VIZ_PATH
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            save_graph_visualization(_graph, path)

        logger.info("RAG: LangGraph 初始化完成")
    except ConnectionPoolError as e:
        _graph = None
        _tool_config = None
        _init_error = str(e)
        logger.warning("RAG: 检查点/存储初始化失败: %s", e)
    except Exception as e:
        _graph = None
        _tool_config = None
        _init_error = str(e)
        logger.warning("RAG: 初始化失败: %s", e)


__all__ = [
    "initialize_rag",
    "is_rag_ready",
    "get_rag_dependencies",
    "get_rag_init_error",
]
