from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

try:
    from langchain_core.tools import create_retriever_tool
except ImportError:  # 极旧环境
    from langchain.tools.retriever import create_retriever_tool  # type: ignore

from app.services.rag.rag_config import RagConfig


def get_tools(llm_embedding, llm_chat: ChatOpenAI | None = None):
    vectorstore = Chroma(
        persist_directory=RagConfig.CHROMADB_DIRECTORY,
        collection_name=RagConfig.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description=RagConfig.RETRIEVER_DESCRIPTION,
    )
    tools = [retriever_tool]
    if llm_chat:
        from app.services.rag.search_tool import create_search_tool

        tools.append(create_search_tool(llm_chat))
    return tools
