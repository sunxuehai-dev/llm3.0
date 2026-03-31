import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

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
    def __init__(self):
        self.search_engine = None
        self._init_search_engine()

    def _init_search_engine(self):
        try:
            try:
                from ddgs import DDGS

                self.search_engine = DDGS(timeout=30)
                logger.info("RAG: DDGS 搜索引擎已就绪")
            except ImportError:
                from duckduckgo_search import DDGS

                self.search_engine = DDGS(timeout=30)
                logger.info("RAG: duckduckgo_search 已就绪")
        except ImportError:
            logger.warning("RAG: 未安装 ddgs / duckduckgo-search，联网搜索不可用")
            self.search_engine = None
        except Exception as e:
            logger.error("RAG: 搜索引擎初始化失败: %s", e)
            self.search_engine = None

    def search(self, query: str, max_results: int = 5) -> str:
        if not self.search_engine:
            return "搜索引擎未初始化，无法执行搜索。请安装 ddgs 或 duckduckgo-search。"

        try:
            logger.info("RAG: 搜索: %s", query)
            try:
                results = list(self.search_engine.text(query, max_results=max_results))
            except TypeError:
                results = list(
                    self.search_engine.text(keywords=query, max_results=max_results)
                )

            if not results:
                return "未找到相关搜索结果。"

            formatted = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                body = result.get("body", "无内容")
                href = result.get("href", "")
                formatted.append(f"【结果{i}】{title}\n{body}\n来源: {href}")

            return "\n\n".join(formatted)
        except Exception as e:
            logger.error("RAG: 搜索失败: %s", e)
            return f"搜索过程中出错: {str(e)}"


_searcher: WebSearcher | None = None


def get_searcher() -> WebSearcher:
    global _searcher
    if _searcher is None:
        _searcher = WebSearcher()
    return _searcher


def create_search_tool(llm_chat: "ChatOpenAI"):
    @tool
    def web_search(query: str) -> str:
        """
        互联网搜索工具，用于查询向量库以外的实时信息或通用知识。
        结果会经大模型整理后返回。
        """
        logger.info("RAG: web_search: %s", query)
        searcher = get_searcher()
        raw = searcher.search(query, max_results=5)
        if raw.startswith("搜索引擎未初始化") or raw.startswith("搜索过程中出错"):
            return raw
        try:
            prompt = ChatPromptTemplate.from_template(SEARCH_REWRITE_PROMPT)
            chain = prompt | llm_chat
            response = chain.invoke({"query": query, "search_results": raw})
            return response.content
        except Exception as e:
            logger.error("RAG: 改写搜索结果失败: %s", e)
            return f"搜索结果（改写失败，返回原始结果）：\n\n{raw}"

    return web_search
