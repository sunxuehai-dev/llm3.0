# 导入日志模块，用于记录程序运行时的信息
import logging
import threading
# 导入UUID模块，用于生成唯一标识符
import uuid
# 从html模块导入escape函数，用于转义HTML特殊字符
from html import escape
# 从typing模块导入类型提示工具
from typing import Literal, Annotated, Sequence, Optional
# 从typing_extensions导入TypedDict，用于定义类型化的字典
from typing_extensions import TypedDict
# 导入LangChain的提示模板类
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 导入LangChain的消息基类
from langchain_core.messages import BaseMessage
# 导入消息处理函数，用于追加消息
from langgraph.graph.message import add_messages
# 导入预构建的工具条件和工具节点
from langgraph.prebuilt import tools_condition, ToolNode
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
# 导入状态图和起始/结束节点的定义
from langgraph.graph import StateGraph, START, END
# 导入基础存储接口
from langgraph.store.base import BaseStore
# 导入可运行配置类
from langchain_core.runnables import RunnableConfig
# 导入SQLite检查点保存类
from langgraph.checkpoint.sqlite import SqliteSaver
# 导入SQLite存储类
from langgraph.store.sqlite import SqliteStore
# 导入Pydantic的基类和字段定义工具
from pydantic import BaseModel, Field

from app.services.rag.rag_config import RagConfig

logger = logging.getLogger(__name__)


# 定义消息状态类，使用TypedDict进行类型注解
class MessagesState(TypedDict):
    # 定义messages字段，类型为消息序列，使用add_messages处理追加
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 定义relevance_score字段，用于存储文档相关性评分
    relevance_score: Annotated[Optional[str], "Relevance score of retrieved documents, 'yes' or 'no'"]
    # 定义rewrite_count字段，用于跟踪问题重写的次数，达到次数退出graph的递归循环
    rewrite_count: Annotated[int, "Number of times query has been rewritten"]

# 定义工具配置管理类，用于管理工具及其路由配置
class ToolConfig:
    # 初始化方法，接收工具列表并设置相关属性
    def __init__(self, tools):
        # 将传入的工具列表存储到实例变量 self.tools 中
        self.tools = tools
        # 创建一个集合，包含所有工具的名称，使用集合推导式从 tools 中提取 name 属性
        self.tool_names = {tool.name for tool in tools}
        # 调用内部方法 _build_routing_config，动态生成工具路由配置并存储到 self.tool_routing_config
        self.tool_routing_config = self._build_routing_config(tools)
        # 记录日志，输出初始化完成的工具名称集合和路由配置，便于调试和验证
        logger.info(f"Initialized ToolConfig with tools: {self.tool_names}, routing: {self.tool_routing_config}")

    # 内部方法，用于根据工具定义动态构建路由配置
    def _build_routing_config(self, tools):
        # 创建一个空字典，用于存储工具名称到目标节点的映射
        routing_config = {}
        # 遍历传入的工具列表，逐个处理每个工具
        for tool in tools:
            # 将工具名称转换为小写，确保匹配时忽略大小写
            tool_name = tool.name.lower()
            # 检查工具名称中是否包含 "retrieve"，用于判断是否为检索类工具
            if "retrieve" in tool_name:
                # 如果是检索类工具，将其路由目标设置为 "grade_documents"（需要评分）
                routing_config[tool_name] = "grade_documents"
                # 记录调试日志，说明该工具被路由到 "grade_documents"，并标注为检索工具
                logger.debug(f"Tool '{tool_name}' routed to 'grade_documents' (retrieval tool)")
            # 如果工具名称不包含 "retrieve"
            else:
                # 将其路由目标设置为 "generate"（直接生成结果）
                routing_config[tool_name] = "generate"
                # 记录调试日志，说明该工具被路由到 "generate"，并标注为非检索工具
                logger.debug(f"Tool '{tool_name}' routed to 'generate' (non-retrieval tool)")
        # 检查路由配置字典是否为空（即没有工具被处理）
        if not routing_config:
            # 如果为空，记录警告日志，提示工具列表可能为空或未正确处理
            logger.warning("No tools provided or routing config is empty")
        # 返回生成的路由配置字典
        return routing_config

    # 获取工具列表的方法，返回存储在实例中的 tools
    def get_tools(self):
        # 直接返回 self.tools，提供外部访问工具列表的接口
        return self.tools

    # 获取工具名称集合的方法，返回存储在实例中的 tool_names
    def get_tool_names(self):
        # 直接返回 self.tool_names，提供外部访问工具名称集合的接口
        return self.tool_names

    # 获取工具路由配置的方法，返回动态生成的路由配置
    def get_tool_routing_config(self):
        # 直接返回 self.tool_routing_config，提供外部访问路由配置的接口
        return self.tool_routing_config

# 文档相关性评分
class DocumentRelevanceScore(BaseModel):
    # 定义binary_score字段，表示相关性评分，取值为"yes"或"no"
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# 自定义异常，表示数据库连接池初始化或状态异常
class ConnectionPoolError(Exception):
    """自定义异常，表示数据库连接池初始化或状态异常"""
    pass

# 重定义ToolNode，支持并发处理工具调用
class ParallelToolNode(ToolNode):
    # 初始化方法，继承自ToolNode，接收工具列表和最大线程数参数
    def __init__(self, tools, max_workers: int = 5):
        # 调用父类ToolNode的初始化方法，传入工具列表
        super().__init__(tools)
        # 设置实例变量max_workers，定义线程池的最大工作线程数，默认为5
        self.max_workers = max_workers  # 线程池最大工作线程数

    # 定义私有方法，用于执行单个工具调用，返回ToolMessage对象
    def _run_single_tool(self, tool_call: dict, tool_map: dict) -> ToolMessage:
        """执行单个工具调用"""
        # 使用try-except块捕获工具执行中的异常
        try:
            # 从tool_call字典中提取工具名称
            tool_name = tool_call["name"]
            # 从tool_map中获取对应的工具实例，若不存在则返回None
            tool = tool_map.get(tool_name)
            # 检查工具是否存在，若不存在则抛出ValueError异常
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            # 调用工具的invoke方法，传入工具参数，执行工具逻辑
            result = tool.invoke(tool_call["args"])
            # 创建并返回ToolMessage对象，包含工具执行结果、调用ID和工具名称
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name
            )
        # 捕获所有异常，记录错误并返回包含错误信息的ToolMessage
        except Exception as e:
            # 记录工具执行失败的错误日志，包含工具名称和异常信息
            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
            # 返回包含错误内容的ToolMessage对象，用于状态更新
            return ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call["id"],
                name=tool_call.get("name", "unknown")
            )

    # 定义可调用方法，使实例可直接调用，实现并行执行所有工具调用
    def __call__(self, state: dict) -> dict:
        """并行执行所有工具调用"""
        # 记录日志，表示开始处理工具调用
        logger.info("ParallelToolNode processing tool calls")
        # 从状态字典中获取最后一条消息
        last_message = state["messages"][-1]
        # 获取最后一条消息中的工具调用列表，若不存在则返回空列表
        tool_calls = getattr(last_message, "tool_calls", [])
        # 检查工具调用列表是否为空，若为空则记录警告并返回空消息列表
        if not tool_calls:
            logger.warning("No tool calls found in state")
            return {"messages": []}

        # 创建工具名称到工具实例的映射字典，便于快速查找
        tool_map = {tool.name: tool for tool in self.tools}
        # 初始化结果列表，用于存储所有工具调用的返回消息
        results = []

        # 使用线程池管理并行任务，max_workers控制最大并发线程数
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 使用字典推导式提交所有工具调用任务到线程池，返回future到tool_call的映射
            future_to_tool = {
                executor.submit(self._run_single_tool, tool_call, tool_map): tool_call
                for tool_call in tool_calls
            }
            # 遍历已完成的future对象，按完成顺序收集结果
            for future in as_completed(future_to_tool):
                # 使用try-except块捕获线程执行中的异常
                try:
                    # 获取future的结果，即工具调用的返回消息
                    result = future.result()
                    # 将结果添加到结果列表
                    results.append(result)
                # 捕获线程执行失败的异常，记录错误并添加错误消息到结果
                except Exception as e:
                    # 记录工具执行失败的错误日志，包含异常信息
                    logger.error(f"Tool execution failed: {e}")
                    # 获取失败任务对应的tool_call
                    tool_call = future_to_tool[future]
                    # 创建包含错误信息的ToolMessage并添加到结果列表
                    results.append(ToolMessage(
                        content=f"Unexpected error: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name", "unknown")
                    ))

        # 记录日志，表示完成所有工具调用，包含调用数量
        logger.info(f"Completed {len(results)} tool calls")
        # 返回更新后的状态字典，包含所有工具调用的结果消息
        return {"messages": results}


# 定义获取最新问题的辅助函数
def get_latest_question(state: MessagesState) -> Optional[str]:
    """从状态中安全地获取最新用户问题。

    Args:
        state: 当前对话状态，包含消息历史。

    Returns:
        Optional[str]: 最新问题的内容，如果无法获取则返回 None。
    """
    try:
        # 检查状态是否包含消息列表且不为空
        if not state.get("messages") or not isinstance(state["messages"], (list, tuple)) or len(state["messages"]) == 0:
            logger.warning("No valid messages found in state for getting latest question")
            return None

        # 从后向前遍历消息，找到最近的 HumanMessage（用户输入）
        for message in reversed(state["messages"]):
            if message.__class__.__name__ == "HumanMessage" and hasattr(message, "content"):
                return message.content

        # 如果没有找到 HumanMessage，返回 None
        logger.info("No HumanMessage found in state")
        return None

    except Exception as e:
        logger.error(f"Error getting latest question: {e}")
        return None


# 定义线程内的持久化存储消息过滤函数
def filter_messages(messages: list) -> list:
    """过滤消息列表，仅保留 AIMessage 和 HumanMessage 类型消息"""
    # 过滤出 AIMessage 和 HumanMessage 类型的消息
    filtered = [msg for msg in messages if msg.__class__.__name__ in ['AIMessage', 'HumanMessage']]
    # 如果过滤后的消息超过N条，返回最后N条，否则返回过滤后的完整列表
    return filtered[-5:] if len(filtered) > 5 else filtered


# 定义跨线程的持久化存储的存储和过滤函数
def store_memory(question: BaseMessage, config: RunnableConfig, store: BaseStore) -> str:
    """存储用户输入中的记忆信息。

    Args:
        question: 用户输入的消息。
        config: 运行时配置。
        store: 数据存储实例。

    Returns:
        str: 用户相关的记忆信息字符串。
    """
    namespace = ("memories", config["configurable"]["user_id"])
    try:
        # 在跨线程存储数据库中搜索相关记忆
        memories = store.search(namespace, query=str(question.content))
        user_info = "\n".join([d.value["data"] for d in memories])

        # 如果包含“记住”，存储新记忆
        if "记住" in question.content.lower():
            memory = escape(question.content)
            store.put(namespace, str(uuid.uuid4()), {"data": memory})
            logger.info(f"Stored memory: {memory}")

        return user_info
    except Exception as e:
        logger.error(f"Error in store_memory: {e}")
        return ""


# 定义创建处理链的函数
def create_chain(llm_chat, template_file: str, structured_output=None):
    """创建 LLM 处理链，加载提示模板并绑定模型，使用缓存避免重复读取文件。

    Args:
        llm_chat: 语言模型实例。
        template_file: 提示模板文件路径。
        structured_output: 可选的结构化输出模型。

    Returns:
        Runnable: 配置好的处理链。

    Raises:
        FileNotFoundError: 如果模板文件不存在。
    """
    # 定义静态缓存和锁（仅在函数第一次调用时初始化）
    if not hasattr(create_chain, "prompt_cache"):
        # 缓存字典
        create_chain.prompt_cache = {}
        # 线程锁 确保缓存的读写是线程安全的
        create_chain.lock = threading.Lock()

    try:
        # 先检查缓存，无锁访问
        if template_file in create_chain.prompt_cache:
            prompt_template = create_chain.prompt_cache[template_file]
            logger.info(f"Using cached prompt template for {template_file}")
        else:
            # 使用锁保护缓存访问
            with create_chain.lock:
                # 检查缓存中是否已有该模板
                if template_file not in create_chain.prompt_cache:
                    logger.info(f"Loading and caching prompt template from {template_file}")
                    # 从文件加载提示模板并存入缓存
                    create_chain.prompt_cache[template_file] = PromptTemplate.from_file(template_file, encoding="utf-8")
                # 从缓存中获取提示模板
                prompt_template = create_chain.prompt_cache[template_file]

        # 创建聊天提示模板，使用模板内容
        prompt = ChatPromptTemplate.from_messages([("human", prompt_template.template)])
        # 返回提示模板与LLM的组合链，若有结构化输出则绑定
        return prompt | (llm_chat.with_structured_output(structured_output) if structured_output else llm_chat)
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise



# 定义 Node agent分诊函数
def agent(state: MessagesState, config: RunnableConfig, *, store: BaseStore, llm_chat, tool_config: ToolConfig) -> dict:
    """代理函数，根据用户问题决定是否调用工具或结束。

    Args:
        state: 当前对话状态。
        config: 运行时配置。
        store: 数据存储实例。
        llm_chat: Chat模型。
        tool_config: 工具配置参数。

    Returns:
        dict: 更新后的对话状态。
    """
    # 记录代理开始处理查询
    logger.info("Agent processing user query")
    # 定义存储命名空间，使用用户ID
    namespace = ("memories", config["configurable"]["user_id"])
    # 尝试执行以下代码块
    try:
        # 获取最后一条消息即用户问题
        question = state["messages"][-1]
        logger.info(f"agent question:{question}")

        # 自定义跨线程持久化存储记忆并获取相关信息
        user_info = store_memory(question, config, store)
        # 自定义线程内存储逻辑 过滤消息
        messages = filter_messages(state["messages"])

        # 将工具绑定到 LLM
        llm_chat_with_tool = llm_chat.bind_tools(tool_config.get_tools())

        # 创建代理处理链
        agent_chain = create_chain(llm_chat_with_tool, RagConfig.PROMPT_TEMPLATE_TXT_AGENT)
        # 调用代理链处理消息
        response = agent_chain.invoke({"question": question,"messages": messages, "userInfo": user_info})
        # logger.info(f"Agent response: {response}")
        # 返回更新后的对话状态
        return {"messages": [response]}
    # 捕获异常
    except Exception as e:
        # 记录错误日志
        logger.error(f"Error in agent processing: {e}")
        # 返回错误消息
        return {"messages": [{"role": "system", "content": "处理请求时出错"}]}


# 定义 Node grade_documents相关性评估函数
def grade_documents(state: MessagesState, llm_chat) -> dict:
    """评估检索到的文档内容与问题的相关性，并将评分结果存储在状态中。

    Args:
        state: 当前对话状态，包含消息历史。

    Returns:
        dict: 更新后的状态，包含评分结果。
    """
    logger.info("Grading documents for relevance")
    if not state.get("messages"):
        logger.error("Messages state is empty")
        return {
            "messages": [{"role": "system", "content": "状态为空，无法评分"}],
            "relevance_score": None
        }

    try:
        # # 获取用户的最新问题
        question = get_latest_question(state)
        # 获取最后一条消息作为上下文(因为调用工具输出的内容写入到state的最新消息中)
        context = state["messages"][-1].content
        # logger.info(f"Evaluating relevance - Question: {question}, Context: {context}")

        # 创建评分处理链
        grade_chain = create_chain(llm_chat, RagConfig.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        # 调用评分链评估相关性
        scored_result = grade_chain.invoke({"question": question, "context": context})
        # logger.info(f"scored_result:{scored_result}")
        # 获取评分结果
        score = scored_result.binary_score
        logger.info(f"Document relevance score: {score}")

        # 返回更新后的状态，包括评分结果
        return {
            # 保持消息不变
            "messages": state["messages"],
            # 存储评分结果
            "relevance_score": score
        }
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error: {e}")
        return {
            "messages": [{"role": "system", "content": "无法评分文档"}],
            "relevance_score": None
        }
    except Exception as e:
        logger.error(f"Unexpected error in grading: {e}")
        return {
            "messages": [{"role": "system", "content": "评分过程中出错"}],
            "relevance_score": None
        }


# 查询重写
def rewrite(state: MessagesState, llm_chat) -> dict:
    """重写用户查询以改进问题。

    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
    # 记录开始重写查询
    logger.info("Rewriting query")
    # 尝试执行以下代码块
    try:
        # 获取用户的最新问题
        question = get_latest_question(state)
        # 创建重写处理链
        rewrite_chain = create_chain(llm_chat, RagConfig.PROMPT_TEMPLATE_TXT_REWRITE)
        # 调用重写链生成新查询
        response = rewrite_chain.invoke({"question": question})
        # logger.info(f"rewrite question:{response}")
        # 重写次数+1
        rewrite_count = state.get("rewrite_count", 0) + 1
        logger.info(f"Rewrite count: {rewrite_count}")
        # 返回更新后的对话状态
        return {"messages": [response], "rewrite_count": rewrite_count}
    # 捕获索引或键错误
    except (IndexError, KeyError) as e:
        # 记录错误日志
        logger.error(f"Message access error in rewrite: {e}")
        # 返回错误消息
        return {"messages": [{"role": "system", "content": "无法重写查询"}]}


# 定义Node 生成回复函数
def generate(state: MessagesState, llm_chat) -> dict:
    """基于工具返回的内容生成最终回复。

    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
    # 记录开始生成回复
    logger.info("Generating final response")
    # 尝试执行以下代码块
    try:
        # 获取用户的最新问题
        question = get_latest_question(state)
        # 获取最后一条消息作为上下文(因为调用工具输出的内容写入到state的最新消息中)
        context = state["messages"][-1].content
        # logger.info(f"generate - Question: {question}, Context: {context}")
        # 创建生成处理链
        generate_chain = create_chain(llm_chat, RagConfig.PROMPT_TEMPLATE_TXT_GENERATE)
        # 调用生成链生成回复
        response = generate_chain.invoke({"context": context, "question": question})
        # 返回更新后的消息状态
        return {"messages": [response]}
    # 捕获索引或键错误
    except (IndexError, KeyError) as e:
        # 记录错误日志
        logger.error(f"Message access error in generate: {e}")
        # 返回错误消息
        return {"messages": [{"role": "system", "content": "无法生成回复"}]}


# 定义Edge 根据工具调用的结果动态决定下一步路由
def route_after_tools(state: MessagesState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    """
    根据工具调用的结果动态决定下一步路由，使用配置字典支持多工具并包含容错处理。

    Args:
        state: 当前对话状态，包含消息历史和可能的工具调用结果。
        tool_config: 工具配置参数。

    Returns:
        Literal["generate", "grade_documents"]: 下一步的目标节点。
    """
    # 检查状态是否包含消息列表，若为空则记录错误并默认路由到 generate
    if not state.get("messages") or not isinstance(state["messages"], list):
        logger.error("Messages state is empty or invalid, defaulting to generate")
        return "generate"

    try:
        # 获取状态中的最后一条消息，用于判断工具调用来源
        last_message = state["messages"][-1]

        # 检查消息是否具有 name 属性，若无则路由到 generate
        if not hasattr(last_message, "name") or last_message.name is None:
            logger.info("Last message has no name attribute, routing to generate")
            return "generate"

        # 检查消息是否来自已注册的工具
        tool_name = last_message.name
        if tool_name not in tool_config.get_tool_names():
            logger.info(f"Unknown tool {tool_name}, routing to generate")
            return "generate"

        # 根据配置字典决定路由，若无配置则默认路由到 generate
        target = tool_config.get_tool_routing_config().get(tool_name, "generate")
        logger.info(f"Tool {tool_name} routed to {target} based on config")
        return target

    except IndexError:
        # 捕获消息列表为空或索引错误的异常，记录错误并默认路由到 generate
        logger.error("No messages available in state, defaulting to generate")
        return "generate"
    except AttributeError:
        # 捕获消息对象属性访问错误的异常，记录错误并默认路由到 generate
        logger.error("Invalid message object, defaulting to generate")
        return "generate"
    except Exception as e:
        # 捕获其他未预期的异常，记录详细错误信息并默认路由到 generate
        logger.error(f"Unexpected error in route_after_tools: {e}, defaulting to generate")
        return "generate"


# 定义Edge 根据状态中的评分结果决定下一步路由
def route_after_grade(state: MessagesState) -> Literal["generate", "rewrite"]:
    """
    根据状态中的评分结果决定下一步路由，包含增强的状态校验和容错处理。

    Args:
        state: 当前对话状态，预期包含 messages 和 relevance_score 字段。

    Returns:
        Literal["generate", "rewrite"]: 下一步的目标节点。
    """
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到 rewrite
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to rewrite")
        return "rewrite"

    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to rewrite")
        return "rewrite"

    # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to rewrite")
        return "rewrite"

    # 获取状态中的 relevance_score，若不存在则返回 None
    relevance_score = state.get("relevance_score")
    # 获取状态中的 rewrite_count
    rewrite_count = state.get("rewrite_count", 0)
    logger.info(f"Routing based on relevance_score: {relevance_score}, rewrite_count: {rewrite_count}")

    # 如果重写次数超过 3 次，强制路由到 generate
    if rewrite_count >= 3:
        logger.info("Max rewrite limit reached, proceeding to generate")
        return "generate"

    try:
        # 检查 relevance_score 是否为有效字符串，若不是则视为无效评分
        if not isinstance(relevance_score, str):
            logger.warning(f"Invalid relevance_score type: {type(relevance_score)}, defaulting to rewrite")
            return "rewrite"

        # 如果评分结果为 "yes"，表示文档相关，路由到 generate 节点
        if relevance_score.lower() == "yes":
            logger.info("Documents are relevant, proceeding to generate")
            return "generate"

        # 如果评分结果为 "no" 或其他值（包括空字符串），路由到 rewrite 节点
        logger.info("Documents are not relevant or scoring failed, proceeding to rewrite")
        return "rewrite"

    except AttributeError:
        # 捕获 relevance_score 不支持 lower() 方法的异常（例如 None），默认路由到 rewrite
        logger.error("relevance_score is not a string or is None, defaulting to rewrite")
        return "rewrite"
    except Exception as e:
        # 捕获其他未预期的异常，记录详细错误信息并默认路由到 rewrite
        logger.error(f"Unexpected error in route_after_grade: {e}, defaulting to rewrite")
        return "rewrite"


# 保存状态图的可视化表示
def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    """保存状态图的可视化表示。

    Args:
        graph: 状态图实例。
        filename: 保存文件路径。
    """
    # 尝试执行以下代码块
    try:
        # 以二进制写模式打开文件
        with open(filename, "wb") as f:
            # 将状态图转换为Mermaid格式的PNG并写入文件
            f.write(graph.get_graph().draw_mermaid_png())
        # 记录保存成功的日志
        logger.info(f"Graph visualization saved as {filename}")
    # 捕获IO错误
    except IOError as e:
        # 记录警告日志
        logger.warning(f"Failed to save graph visualization: {e}")


# 创建并配置状态图
def create_graph(db_path: str, llm_chat, llm_embedding, tool_config: ToolConfig) -> StateGraph:
    """创建并配置状态图。

    Args:
        db_path: SQLite数据库文件路径。
        llm_chat: Chat模型。
        llm_embedding: Embedding模型。
        tool_config: 工具配置参数。

    Returns:
        StateGraph: 编译后的状态图。
    """
    # 线程内持久化存储 - 使用 SQLite
    try:
        # 创建SQLite检查点保存实例
        import sqlite3
        db_file = db_path.replace('sqlite:///', '')
        # 使用隔离级别 None 以支持自动提交
        conn = sqlite3.connect(db_file, check_same_thread=False, isolation_level=None)
        # 启用WAL模式以支持并发访问
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=60000')  # 60秒超时
        conn.execute('PRAGMA synchronous=NORMAL')  # 平衡性能和安全
        conn.execute('PRAGMA cache_size=-64000')  # 64MB 缓存
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        logger.info("SQLite checkpointer initialized")
    except Exception as e:
        logger.error(f"Failed to setup SqliteSaver: {e}")
        raise ConnectionPoolError(f"检查点初始化失败: {str(e)}")

    # 跨线程持久化存储 - 使用同一个SQLite连接
    try:
        # 创建SQLite存储实例（复用同一连接）
        store = SqliteStore(conn)
        store.setup()
        logger.info("SQLite store initialized")
    except Exception as e:
        logger.error(f"Failed to setup SqliteStore: {e}")
        raise ConnectionPoolError(f"存储初始化失败: {str(e)}")

    # 创建状态图实例，使用MessagesState作为状态类型
    workflow = StateGraph(MessagesState)
    # 添加代理节点
    workflow.add_node("agent", lambda state, config: agent(state, config, store=store, llm_chat=llm_chat, tool_config=tool_config))
    # 添加工具节点，使用并行工具节点
    workflow.add_node("call_tools", ParallelToolNode(tool_config.get_tools(), max_workers=5))
    # 添加重写节点
    workflow.add_node("rewrite", lambda state: rewrite(state,llm_chat=llm_chat))
    # 添加生成节点
    workflow.add_node("generate", lambda state: generate(state, llm_chat=llm_chat))
    # 添加文档相关性评分节点
    workflow.add_node("grade_documents", lambda state: grade_documents(state, llm_chat=llm_chat))

    # 添加从起始到代理的边
    workflow.add_edge(START, end_key="agent")
    # 添加代理的条件边，根据工具调用的工具名称决定下一步路由
    workflow.add_conditional_edges(source="agent", path=tools_condition, path_map={"tools": "call_tools", END: END})
    # 添加检索的条件边，根据工具调用的结果动态决定下一步路由
    workflow.add_conditional_edges(source="call_tools", path=lambda state: route_after_tools(state, tool_config),path_map={"generate": "generate", "grade_documents": "grade_documents"})
    # 添加检索的条件边，根据状态中的评分结果决定下一步路由
    workflow.add_conditional_edges(source="grade_documents", path=route_after_grade, path_map={"generate": "generate", "rewrite": "rewrite"})
    # 添加从生成到结束的边
    workflow.add_edge(start_key="generate", end_key=END)
    # 添加从重写到代理的边
    workflow.add_edge(start_key="rewrite", end_key="agent")

    # 编译状态图，绑定检查点和存储
    return workflow.compile(checkpointer=checkpointer, store=store)