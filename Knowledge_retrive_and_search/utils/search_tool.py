# 功能说明：互联网搜索工具，调用搜索引擎并使用大模型改写结果
import logging
from typing import Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 搜索结果改写提示模板
SEARCH_REWRITE_PROMPT = """你是一个专业的信息整理助手。请根据以下搜索结果，用简洁清晰的语言回答用户的问题。

用户问题：{query}

搜索结果：
{search_results}

请将搜索结果进行整理和改写，要求：
1. 提取关键信息，去除无关内容
2. 用通俗易懂的语言表达
3. 保持信息的准确性
4. 如果搜索结果不相关或不足，请如实说明

改写后的回答："""


class WebSearcher:
    """互联网搜索器类"""
    
    def __init__(self):
        """初始化搜索器"""
        self.search_engine = None
        self._init_search_engine()
    
    def _init_search_engine(self):
        """初始化搜索引擎"""
        try:
            # 优先使用新版 ddgs 库
            try:
                from ddgs import DDGS
                self.search_engine = DDGS(timeout=30)
                logger.info("DDGS 搜索引擎初始化成功")
            except ImportError:
                # 回退到旧版 duckduckgo-search 库
                from duckduckgo_search import DDGS
                self.search_engine = DDGS(timeout=30)
                logger.info("DuckDuckGo 搜索引擎初始化成功")
        except ImportError:
            logger.warning("搜索库未安装，请运行: pip install ddgs 或 pip install duckduckgo-search")
            self.search_engine = None
        except Exception as e:
            logger.error(f"搜索引擎初始化失败: {e}")
            self.search_engine = None
    
    def search(self, query: str, max_results: int = 5) -> str:
        """
        执行搜索并返回结果
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            搜索结果字符串
        """
        if not self.search_engine:
            return "搜索引擎未初始化，无法执行搜索。请安装 duckduckgo-search 库。"
        
        try:
            logger.info(f"正在搜索: {query}")
            
            # 使用 DuckDuckGo 搜索
            # 尝试不同的 API 调用方式
            try:
                # 新版 ddgs API
                results = list(self.search_engine.text(
                    query,
                    max_results=max_results
                ))
            except TypeError:
                # 旧版 duckduckgo-search API
                results = list(self.search_engine.text(
                    keywords=query,
                    max_results=max_results
                ))
            
            if not results:
                return "未找到相关搜索结果。"
            
            # 格式化搜索结果
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', '无标题')
                body = result.get('body', '无内容')
                href = result.get('href', '')
                formatted_results.append(f"【结果{i}】{title}\n{body}\n来源: {href}")
            
            search_text = "\n\n".join(formatted_results)
            logger.info(f"搜索完成，找到 {len(results)} 条结果")
            
            return search_text
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return f"搜索过程中出错: {str(e)}"


# 全局搜索器实例
_searcher = None


def get_searcher() -> WebSearcher:
    """获取搜索器单例"""
    global _searcher
    if _searcher is None:
        _searcher = WebSearcher()
    return _searcher


def create_search_tool(llm_chat: ChatOpenAI):
    """
    创建互联网搜索工具
    
    Args:
        llm_chat: 聊天模型实例，用于改写搜索结果
        
    Returns:
        LangChain Tool 实例
    """
    
    @tool
    def web_search(query: str) -> str:
        """
        这是互联网搜索工具，用于搜索简体中文互联网上的信息。
        当用户询问的问题超出健康档案范围，或需要查询最新信息时使用此工具。
        搜索结果会经过大模型改写，返回便于理解的内容。
        
        Args:
            query: 搜索查询关键词或问题
            
        Returns:
            改写后的搜索结果
        """
        logger.info(f"执行互联网搜索: {query}")
        
        # 1. 执行搜索
        searcher = get_searcher()
        raw_results = searcher.search(query, max_results=5)
        
        if raw_results.startswith("搜索引擎未初始化") or raw_results.startswith("搜索过程中出错"):
            return raw_results
        
        # 2. 使用大模型改写结果
        try:
            prompt = ChatPromptTemplate.from_template(SEARCH_REWRITE_PROMPT)
            chain = prompt | llm_chat
            
            response = chain.invoke({
                "query": query,
                "search_results": raw_results
            })
            
            rewritten_result = response.content
            logger.info(f"搜索结果改写完成")
            
            return rewritten_result
            
        except Exception as e:
            logger.error(f"改写搜索结果失败: {e}")
            # 如果改写失败，返回原始搜索结果
            return f"搜索结果（改写失败，返回原始结果）：\n\n{raw_results}"
    
    return web_search


# 用于测试的独立搜索函数
def test_search(query: str):
    """测试搜索功能"""
    searcher = get_searcher()
    results = searcher.search(query)
    print(f"搜索结果:\n{results}")
    return results


if __name__ == "__main__":
    # 测试搜索功能
    test_search("Python 编程语言")
