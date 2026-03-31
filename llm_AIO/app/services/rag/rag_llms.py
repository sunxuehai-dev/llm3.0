import logging
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "chat_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1",
    },
    "oneapi": {
        "base_url": os.getenv("RAG_ONEAPI_BASE_URL", "http://127.0.0.1:3000/v1"),
        "api_key": os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "chat_model": os.getenv("RAG_ONEAPI_CHAT_MODEL", "qwen-max"),
        "embedding_model": os.getenv("RAG_ONEAPI_EMBEDDING_MODEL", "text-embedding-v1"),
    },
    "ollama": {
        "base_url": os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"),
        "api_key": "ollama",
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:32b"),
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest"),
    },
}

DEFAULT_LLM_TYPE = "qwen"
DEFAULT_TEMPERATURE = 0.0


class LLMInitializationError(Exception):
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(
                f"不支持的 RAG LLM 类型: {llm_type}. 可选: {list(MODEL_CONFIGS.keys())}"
            )

        config = MODEL_CONFIGS[llm_type]

        if llm_type == "ollama":
            os.environ.setdefault("OPENAI_API_KEY", "NA")

        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=60,
            max_retries=2,
        )

        llm_embedding = OpenAIEmbeddings(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["embedding_model"],
            deployment=config["embedding_model"],
            check_embedding_ctx_length=False,
        )

        logger.info("RAG: 已初始化 chat/embedding (%s)", llm_type)
        return llm_chat, llm_embedding
    except ValueError as e:
        raise LLMInitializationError(str(e)) from e
    except LLMInitializationError:
        raise
    except Exception as e:
        raise LLMInitializationError(str(e)) from e


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning("RAG LLM 使用默认类型重试: %s", e)
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise
