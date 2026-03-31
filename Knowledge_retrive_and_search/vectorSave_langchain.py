# 功能说明：使用 LangChain 框架内置的文本分割器与数据库检索器进行向量存储
# 与项目其他脚本接口适配，使用 utils/config.py 和 utils/llms.py 中的配置
import os
import logging
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from utils.config import Config
from utils.llms import get_llm, LLMInitializationError

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 文本语言类型配置
TEXT_LANGUAGE = 'Chinese'  # 'Chinese' 或 'English'
INPUT_PDF = "input/健康档案.pdf"
# TEXT_LANGUAGE = 'English'
# INPUT_PDF = "input/deepseek-v3-1-4.pdf"

# 指定文件中待处理的页码，全部页码则填 None
PAGE_NUMBERS = None
# PAGE_NUMBERS = [2, 3]


class LangChainVectorStore:
    """使用 LangChain 内置组件的向量存储管理类"""
    
    def __init__(self, embedding_model: Embeddings, 
                 persist_directory: str = None,
                 collection_name: str = None):
        """
        初始化向量存储管理器
        
        Args:
            embedding_model: LangChain Embeddings 实例
            persist_directory: 向量数据库持久化目录
            collection_name: 集合名称
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or Config.CHROMADB_DIRECTORY
        self.collection_name = collection_name or Config.CHROMADB_COLLECTION_NAME
        self.vectorstore = None
        
    def load_pdf(self, file_path: str, page_numbers: Optional[List[int]] = None) -> List[Document]:
        """
        使用 LangChain PyPDFLoader 加载 PDF 文件
        
        Args:
            file_path: PDF 文件路径
            page_numbers: 指定页码列表，None 表示全部页码
            
        Returns:
            Document 列表
        """
        logger.info(f"正在加载 PDF 文件: {file_path}")
        
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # 如果指定了页码，过滤出指定页面
            if page_numbers:
                # PyPDFLoader 的页面索引从 0 开始
                documents = [doc for i, doc in enumerate(documents) if i + 1 in page_numbers]
                logger.info(f"已过滤指定页码: {page_numbers}")
            
            logger.info(f"成功加载 {len(documents)} 页文档")
            return documents
            
        except Exception as e:
            logger.error(f"加载 PDF 文件失败: {e}")
            raise
    
    def split_documents(self, documents: List[Document], 
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200,
                       language: str = 'Chinese') -> List[Document]:
        """
        使用 LangChain RecursiveCharacterTextSplitter 分割文档
        
        Args:
            documents: 待分割的文档列表
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            language: 语言类型，用于选择合适的分隔符
            
        Returns:
            分割后的文档列表
        """
        logger.info(f"正在分割文档，chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        # 根据语言选择分隔符
        if language == 'Chinese':
            # 中文分隔符优先级：句号 > 感叹号 > 问号 > 分号 > 逗号 > 换行 > 空白
            separators = ["。", "！", "？", "；", "，", "\n", " ", ""]
        else:
            # 英文使用默认分隔符
            separators = None
        
        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            is_separator_regex=False
        )
        
        # 分割文档
        split_documents = text_splitter.split_documents(documents)
        logger.info(f"文档分割完成，共 {len(split_documents)} 个文本块")
        
        return split_documents
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        创建 Chroma 向量存储并持久化
        
        Args:
            documents: 文档列表
            
        Returns:
            Chroma 向量存储实例
        """
        logger.info(f"正在创建向量存储，集合名称: {self.collection_name}")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            logger.info(f"向量存储创建成功，共 {len(documents)} 个文档")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"创建向量存储失败: {e}")
            raise
    
    def load_vectorstore(self) -> Chroma:
        """
        加载已存在的向量存储
        
        Returns:
            Chroma 向量存储实例
        """
        logger.info(f"正在加载向量存储，集合名称: {self.collection_name}")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            logger.info("向量存储加载成功")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        使用 LangChain 内置检索器进行相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        logger.info(f"正在检索: {query}")
        
        try:
            results = self.vectorstore.similarity_search(query, k=top_k)
            logger.info(f"检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
    
    def search_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        带相似度分数的检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            (Document, score) 元组列表
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        logger.info(f"正在检索（带分数）: {query}")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            logger.info(f"检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
    
    def get_retriever(self, search_kwargs: dict = None):
        """
        获取 LangChain 检索器，可用于 RAG 链
        
        Args:
            search_kwargs: 检索参数，如 {"k": 5}
            
        Returns:
            VectorStoreRetriever 实例
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        search_kwargs = search_kwargs or {"k": 5}
        retriever = self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )
        logger.info("检索器创建成功")
        return retriever


def vectorStoreSave():
    """
    主函数：使用 LangChain 内置组件进行文档处理和向量存储
    与项目其他脚本接口适配
    """
    global TEXT_LANGUAGE, INPUT_PDF, PAGE_NUMBERS
    
    try:
        # 1. 初始化嵌入模型（与项目其他脚本适配）
        logger.info(f"使用 LLM 类型: {Config.LLM_TYPE}")
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        
        # 2. 创建向量存储管理器
        vector_store = LangChainVectorStore(
            embedding_model=llm_embedding,
            persist_directory=Config.CHROMADB_DIRECTORY,
            collection_name=Config.CHROMADB_COLLECTION_NAME
        )
        
        # 3. 加载 PDF 文件
        documents = vector_store.load_pdf(INPUT_PDF, PAGE_NUMBERS)
        
        # 4. 分割文档
        split_docs = vector_store.split_documents(
            documents,
            chunk_size=1000,
            chunk_overlap=200,
            language=TEXT_LANGUAGE
        )
        
        # 5. 创建向量存储
        vector_store.create_vectorstore(split_docs)
        
        # 6. 检索测试
        if TEXT_LANGUAGE == 'Chinese':
            test_query = "张三九的基本信息是什么"
        else:
            test_query = "How many parameters does deepseek V3 have"
        
        # 使用 LangChain 内置检索器进行检索
        results = vector_store.search(test_query, top_k=5)
        
        logger.info(f"检索查询: {test_query}")
        for i, doc in enumerate(results):
            logger.info(f"结果 {i+1}: {doc.page_content[:200]}...")
        
        # 7. 获取检索器（可用于 RAG 链）
        retriever = vector_store.get_retriever(search_kwargs={"k": 5})
        logger.info(f"检索器类型: {type(retriever)}")
        
        return vector_store
        
    except LLMInitializationError as e:
        logger.error(f"LLM 初始化失败: {e}")
        raise
    except Exception as e:
        logger.error(f"向量存储处理失败: {e}")
        raise


if __name__ == "__main__":
    # 执行文档处理和向量存储
    vectorStoreSave()
