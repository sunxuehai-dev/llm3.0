"""RAG / LangGraph 子系统配置（环境变量可覆盖）。"""
import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent


class RagConfig:
    PROMPT_TEMPLATE_TXT_AGENT = str(_PKG / "prompts" / "prompt_template_agent.txt")
    PROMPT_TEMPLATE_TXT_GRADE = str(_PKG / "prompts" / "prompt_template_grade.txt")
    PROMPT_TEMPLATE_TXT_REWRITE = str(_PKG / "prompts" / "prompt_template_rewrite.txt")
    PROMPT_TEMPLATE_TXT_GENERATE = str(_PKG / "prompts" / "prompt_template_generate.txt")

    CHROMADB_DIRECTORY = os.getenv("RAG_CHROMADB_DIRECTORY", "chroma_db_rag")
    CHROMADB_COLLECTION_NAME = os.getenv("RAG_CHROMADB_COLLECTION", "demo001")

    LOG_FILE = os.getenv("RAG_LOG_FILE", "logs/rag_app.log")
    MAX_BYTES = 5 * 1024 * 1024
    BACKUP_COUNT = 3

    LLM_TYPE = os.getenv("RAG_LLM_TYPE", "qwen")

    LANGGRAPH_SQLITE_URI = os.getenv(
        "RAG_LANGGRAPH_SQLITE", "sqlite:///./data/langgraph_rag.db"
    )

    SAVE_GRAPH_VIZ = os.getenv("RAG_SAVE_GRAPH_VIZ", "").lower() in ("1", "true", "yes")
    GRAPH_VIZ_PATH = os.getenv("RAG_GRAPH_VIZ_PATH", "data/rag_graph.png")

    RETRIEVER_DESCRIPTION = os.getenv(
        "RAG_RETRIEVER_DESCRIPTION",
        "领域知识库检索工具，从本地向量库中检索与用户问题相关的文档片段。",
    )

    ENABLE_WEB_SEARCH = os.getenv("RAG_ENABLE_WEB_SEARCH", "true").lower() in (
        "1",
        "true",
        "yes",
    )
