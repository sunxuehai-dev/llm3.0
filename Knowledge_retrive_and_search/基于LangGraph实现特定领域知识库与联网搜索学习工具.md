# 基于LangGraph实现特定领域知识库与联网搜索学习工具

## 一、项目概述

### 1.1 项目简介

本项目是一个基于 LangGraph 框架实现的智能问答系统，集成了**特定领域知识库检索（RAG）**和**互联网搜索**两大核心能力。系统通过 Agent 分诊机制，自动识别用户意图并选择合适的工具进行回答，实现知识库问答与联网搜索的无缝切换。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **知识库问答** | 基于 ChromaDB 向量数据库，支持特定领域知识检索 |
| **联网搜索** | 集成 DuckDuckGo 搜索引擎，支持中文互联网搜索 |
| **智能分诊** | Agent 自动识别用户意图，选择合适工具 |
| **文档相关性评分** | 对检索结果进行相关性评估，确保回答质量 |
| **查询重写** | 当检索结果不相关时，自动优化查询重新检索 |
| **流式输出** | 支持流式响应，实时展示生成内容 |
| **Web界面** | 基于 Gradio 构建友好的用户交互界面 |

### 1.3 项目流程

```
用户问题
    │
    ▼
  agent (意图识别)
    │
    ├── 直接回答 ──────────────────► __end__
    │
    ├── 知识库检索 (retrieve)
    │       │
    │       ▼
    │   grade_documents (相关性评分)
    │       │
    │       ├── 相关 ──► generate ──► __end__
    │       │
    │       └── 不相关 ──► rewrite ──► agent (重试)
    │
    └── 联网搜索 (web_search)
            │
            ▼
        generate ──► __end__
```

### 1.4 技术栈

| 类别 | 技术 |
|------|------|
| 框架 | LangGraph, LangChain |
| 向量数据库 | ChromaDB |
| 关系数据库 | SQLite |
| 大模型 | 通义千问 (Qwen) / OpenAI / Ollama |
| Web框架 | FastAPI, Gradio |
| 搜索引擎 | DuckDuckGo |

---

## 二、项目环境准备

### 2.1 创建 Conda 环境

```bash
conda create -n L1-project-2 python=3.11
conda activate L1-project-2
```

### 2.2 安装项目依赖

```bash
# LangGraph 核心
pip install langgraph==0.2.74
pip install langchain-openai==0.3.6
pip install langchain-community==0.3.19
pip install langchain-chroma==0.2.2
pip install langchain-text-splitters
pip install langchain-core

# 文档处理
pip install pypdf
pip install pdfminer
pip install pdfminer.six
pip install nltk==3.9.1

# 数据库
pip install sqlite3

# 日志
pip install concurrent-log-handler==0.9.25

# Web 界面
pip install gradio
pip install fastapi
pip install uvicorn

# 网络搜索
pip install ddgs
pip install duckduckgo-search

# 其他
pip install requests
pip install tenacity
```

### 2.3 环境变量配置

在系统环境变量中设置以下变量：

```bash
# 阿里通义千问 API Key（必需）
DASHSCOPE_API_KEY=your_dashscope_api_key

# OpenAI API Key（可选）
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 三、项目配置管理

### 3.1 全局配置类 (utils/config.py)

```python
import os

class Config:
    """统一的配置类，集中管理所有常量"""
    
    # Prompt 文件路径
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # 日志持久化存储
    LOG_FILE = "output/app.log"
    MAX_BYTES = 5*1024*1024
    BACKUP_COUNT = 3

    # LLM 类型配置
    # openai: GPT模型, qwen: 通义千问, oneapi: OneAPI方案, ollama: 本地模型
    LLM_TYPE = "qwen"

    # API 服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012
```

### 3.2 大模型配置 (utils/llms.py)

支持多种大模型服务商：

| 类型 | 服务商 | Chat 模型 | Embedding 模型 |
|------|--------|-----------|----------------|
| openai | OpenAI | gpt-4o | text-embedding-3-small |
| qwen | 阿里通义千问 | qwen-max | text-embedding-v1 |
| oneapi | OneAPI | qwen-max | text-embedding-v1 |
| ollama | 本地模型 | qwen2.5:32b | bge-m3:latest |

---

## 四、项目目录结构

```
L1-Project-2/
├── chromaDB/                    # 向量数据库存储目录
├── data/                        # SQLite 数据库目录
├── input/                       # 输入文档目录
│   ├── deepseek-v3-1-4.pdf     # 英文测试文档
│   └── 健康档案.pdf              # 中文测试文档
├── output/                      # 输出目录
│   └── app.log                  # 应用日志
├── prompts/                     # 提示模板目录
│   ├── prompt_template_agent.txt
│   ├── prompt_template_generate.txt
│   ├── prompt_template_grade.txt
│   └── prompt_template_rewrite.txt
├── utils/                       # 工具模块
│   ├── config.py               # 配置管理
│   ├── llms.py                 # 大模型配置
│   ├── tools_config.py         # 工具配置
│   ├── search_tool.py          # 网络搜索工具
│   ├── pdfSplitTest_Ch.py      # 中文PDF解析
│   └── pdfSplitTest_En.py      # 英文PDF解析
├── main.py                      # FastAPI 后端服务
├── webUI.py                     # Gradio Web界面
├── ragAgent.py                  # LangGraph 状态图定义
├── vectorSave.py                # 原始向量灌库脚本
├── vectorSave_langchain.py      # LangChain 向量灌库脚本
├── apiTest.py                   # API 测试脚本
├── graph.png                    # 状态图可视化
└── docker-compose.yml           # Docker 配置
```

---

## 五、快速开始

### 5.1 文档向量灌库

使用 LangChain 框架进行文档处理和向量存储：

```bash
conda activate L1-project-2
python vectorSave_langchain.py
```

**输出示例：**

```
使用 LLM 类型: qwen
正在加载 PDF 文件: input/健康档案.pdf
成功加载 6 页文档
正在分割文档，chunk_size=1000, chunk_overlap=200
文档分割完成，共 7 个文本块
正在创建向量存储，集合名称: demo001
向量存储创建成功，共 7 个文档
检索完成，返回 5 个结果
```

### 5.2 启动后端服务

```bash
conda activate L1-project-2
python main.py
```

**输出示例：**

```
Start the server on port 8012
成功初始化 qwen LLM
Tool 'retrieve' routed to 'grade_documents' (retrieval tool)
Tool 'web_search' routed to 'generate' (non-retrieval tool)
Initialized ToolConfig with tools: {'web_search', 'retrieve'}
SQLite checkpointer initialized
SQLite store initialized
Graph visualization saved as graph.png
Application startup complete.
Uvicorn running on http://0.0.0.0:8012
```

### 5.3 启动 Web 界面

```bash
conda activate L1-project-2
python webUI.py
```

**访问地址：** http://127.0.0.1:7863

---

## 六、工具开发

### 6.1 工具列表

| 工具名称 | 功能描述 | 路由目标 |
|---------|----------|----------|
| `retrieve` | 健康档案知识库检索 | grade_documents |
| `web_search` | 互联网搜索（支持中文） | generate |

### 6.2 知识库检索工具

```python
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
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
```

### 6.3 网络搜索工具

网络搜索工具基于 DuckDuckGo 搜索引擎，支持中文搜索，并使用大模型对搜索结果进行改写：

```python
from utils.search_tool import create_search_tool

# 创建搜索工具（需要传入 LLM 用于改写结果）
search_tool = create_search_tool(llm_chat)
```

**搜索工具特性：**

- 免费使用，无需 API Key
- 支持中文搜索
- 自动使用大模型改写搜索结果
- 返回便于用户理解的内容

---

## 七、核心功能详解

### 7.1 Agent 分诊节点

Agent 负责分析用户意图，决定是否调用工具：

```python
def agent(state: MessagesState, config: RunnableConfig, *, store: BaseStore, llm_chat, tool_config: ToolConfig) -> dict:
    """
    代理函数，根据用户问题决定是否调用工具或结束。
    
    工作流程：
    1. 获取用户问题
    2. 检索用户记忆（跨线程持久化）
    3. 过滤历史消息
    4. 调用 LLM 进行意图识别
    5. 返回响应（可能包含工具调用）
    """
```

### 7.2 文档相关性评分

对检索到的文档进行相关性评估：

```python
def grade_documents(state: MessagesState, llm_chat) -> dict:
    """
    评估检索到的文档内容与问题的相关性。
    
    返回：
    - relevance_score: "yes" 或 "no"
    - "yes": 文档相关，路由到 generate
    - "no": 文档不相关，路由到 rewrite
    """
```

### 7.3 查询重写

当检索结果不相关时，优化用户查询：

```python
def rewrite(state: MessagesState, llm_chat) -> dict:
    """
    重写用户查询以改进问题。
    
    特点：
    - 最多重写 3 次
    - 重写后返回 agent 重新检索
    """
```

### 7.4 内容生成

生成最终回复：

```python
def generate(state: MessagesState, llm_chat) -> dict:
    """
    基于工具返回的内容生成最终回复。
    
    支持两种来源：
    1. 知识库检索结果
    2. 网络搜索结果
    """
```

---

## 八、API 接口

### 8.1 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 对话补全接口 |

### 8.2 请求示例

```python
import requests
import json

url = "http://127.0.0.1:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "messages": [{"role": "user", "content": "张三九的基本信息是什么？"}],
    "stream": True,  # 流式输出
    "userId": "user001",
    "conversationId": "conv001"
}

response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### 8.3 响应格式

**非流式响应：**

```json
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1234567890,
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "回答内容..."
        },
        "finish_reason": "stop"
    }]
}
```

**流式响应：**

```
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {"content": "回答"}, "finish_reason": null}]}
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {"content": "内容"}, "finish_reason": null}]}
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {}, "finish_reason": "stop"}]}
```

---

## 九、Web 界面使用

### 9.1 用户注册登录

1. 访问 http://127.0.0.1:7863
2. 点击"注册"按钮
3. 输入用户名和密码
4. 注册成功后使用账号登录

### 9.2 开始对话

1. 登录后点击"新建会话"
2. 在输入框输入问题
3. 按回车或点击发送

### 9.3 测试问题示例

**知识库问答：**

- "张三九的基本信息是什么？"
- "张三九有什么过敏史？"
- "张三九的体检结果如何？"

**联网搜索：**

- "今天有什么新闻？"
- "人工智能最新发展"
- "Python 编程语言特点"

---

## 十、状态图可视化

![graph](./graph.png)

**节点说明：**

| 节点 | 功能 |
|------|------|
| `__start__` | 开始节点 |
| `agent` | 意图识别，决定是否调用工具 |
| `call_tools` | 执行工具调用 |
| `grade_documents` | 文档相关性评分 |
| `rewrite` | 查询重写 |
| `generate` | 生成最终回复 |
| `__end__` | 结束节点 |

---

## 十一、常见问题

### 11.1 环境变量未设置

**错误信息：** `DASHSCOPE_API_KEY not found`

**解决方案：** 在系统环境变量中设置 `DASHSCOPE_API_KEY`

### 11.2 向量数据库为空

**错误信息：** 检索结果为空

**解决方案：** 运行 `python vectorSave_langchain.py` 进行向量灌库

### 11.3 端口被占用

**错误信息：** `Cannot find empty port in range: 8012-8012`

**解决方案：** 修改 `utils/config.py` 中的 `PORT` 配置，或关闭占用端口的程序

### 11.4 Gradio 版本兼容性

**错误信息：** `data incompatible with messages format`

**解决方案：** 本项目已适配 Gradio 6.0+ 版本，使用 `{"role": "user", "content": "..."}` 格式

---

## 十二、扩展开发

### 12.1 添加新工具

1. 在 `utils/tools_config.py` 中定义新工具
2. 更新 `ToolConfig._build_routing_config()` 方法添加路由规则
3. 重启服务

### 12.2 更换大模型

修改 `utils/config.py` 中的 `LLM_TYPE`：

```python
LLM_TYPE = "qwen"  # 可选: openai, qwen, oneapi, ollama
```

### 12.3 更换向量数据库

修改 `utils/tools_config.py` 中的向量存储配置，支持：
- ChromaDB
- FAISS
- Pinecone
- Milvus

---

## 十三、项目依赖版本

```
langgraph==0.2.74
langchain-openai==0.3.6
langchain-community==0.3.19
langchain-chroma==0.2.2
langchain-text-splitters
langchain-core
gradio>=6.0
fastapi
uvicorn
ddgs
duckduckgo-search
pypdf
pdfminer
pdfminer.six
nltk==3.9.1
concurrent-log-handler==0.9.25
requests
tenacity
```

---

## 十四、参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 官方文档](https://python.langchain.com/)
- [Gradio 官方文档](https://www.gradio.app/)
- [通义千问 API 文档](https://help.aliyun.com/zh/dashscope/)
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)
