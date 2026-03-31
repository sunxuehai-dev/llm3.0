# 功能说明：工具配置文件
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from utils.config import Config


def get_tools(llm_embedding, llm_chat: ChatOpenAI = None):
    """
    创建并返回工具列表

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储
        llm_chat: 聊天模型实例，用于搜索工具改写结果

    Returns:
        list: 工具列表
    """

    # 创建 Chroma 向量存储实例
    vectorstore = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )
    # 将向量存储转换为检索器
    retriever = vectorstore.as_retriever()
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是健康档案查询工具，搜索并返回有关用户的健康档案信息。"
    )

    # 创建互联网搜索工具
    web_search_tool = None
    if llm_chat:
        from utils.search_tool import create_search_tool
        web_search_tool = create_search_tool(llm_chat)

    # 返回工具列表
    tools = [retriever_tool]
    if web_search_tool:
        tools.append(web_search_tool)
    
    return tools