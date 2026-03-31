# 功能说明：工具配置文件（包含搜索工具）
# 在原有工具基础上添加互联网搜索工具
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from utils.config import Config
from utils.search_tool import create_search_tool


def get_tools_with_search(llm_embedding, llm_chat: ChatOpenAI = None):
    """
    创建并返回工具列表（包含互联网搜索工具）

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储
        llm_chat: 聊天模型实例，用于搜索结果改写

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

    # 自定义 multiply 工具
    @tool
    def multiply(a: float, b: float) -> float:
        """这是计算两个数的乘积的工具，返回最终的计算结果"""
        return a * b

    # 创建互联网搜索工具
    search_tool = None
    if llm_chat:
        search_tool = create_search_tool(llm_chat)

    # 返回工具列表
    tools = [retriever_tool, multiply]
    if search_tool:
        tools.append(search_tool)

    return tools


# 保持与原有接口兼容
def get_tools(llm_embedding):
    """
    创建并返回工具列表（原有接口，保持兼容）

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储

    Returns:
        list: 工具列表
    """
    from langchain_chroma import Chroma
    from langchain.tools.retriever import create_retriever_tool
    from langchain_core.tools import tool
    from utils.config import Config

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

    # 自定义 multiply 工具
    @tool
    def multiply(a: float, b: float) -> float:
        """这是计算两个数的乘积的工具，返回最终的计算结果"""
        return a * b

    # 返回工具列表
    return [retriever_tool, multiply]
