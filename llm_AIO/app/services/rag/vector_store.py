"""PDF 分块与 Chroma 向量库构建（供离线/管理端脚本或后续接口复用）。"""
import logging
import os
from typing import Any, List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.rag.rag_config import RagConfig
from app.services.rag.rag_llms import get_llm, LLMInitializationError

logger = logging.getLogger(__name__)


class LangChainVectorStore:
    def __init__(
        self,
        embedding_model: Embeddings,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or RagConfig.CHROMADB_DIRECTORY
        self.collection_name = collection_name or RagConfig.CHROMADB_COLLECTION_NAME
        self.vectorstore: Chroma | None = None

    def load_pdf(
        self, file_path: str, page_numbers: Optional[List[int]] = None
    ) -> List[Document]:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if page_numbers:
            documents = [
                doc for i, doc in enumerate(documents) if i + 1 in page_numbers
            ]
        return documents

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        language: str = "Chinese",
    ) -> List[Document]:
        if language == "Chinese":
            separators = ["。", "！", "？", "；", "，", "\n", " ", ""]
        else:
            separators = None
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            is_separator_regex=False,
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )
        return self.vectorstore

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.vectorstore:
            self.load_vectorstore()
        assert self.vectorstore is not None
        return self.vectorstore.similarity_search(query, k=top_k)

    def get_retriever(self, search_kwargs: dict | None = None):
        if not self.vectorstore:
            self.load_vectorstore()
        assert self.vectorstore is not None
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs or {"k": 5})


def list_rag_vector_index_summary(metadata_scan_limit: int = 20000) -> dict[str, Any]:
    """
    读取 Chroma 持久化目录，返回集合块数与来源文件名汇总（不调用嵌入 API）。
    旧数据若无 source_file，则回退使用 PyPDF 写入的 source 路径 basename。
    """
    persist = RagConfig.CHROMADB_DIRECTORY
    name = RagConfig.CHROMADB_COLLECTION_NAME
    abs_persist = os.path.abspath(persist)
    out: dict[str, Any] = {
        "persist_directory": persist,
        "persist_directory_resolved": abs_persist,
        "collection_name": name,
        "chunk_count": 0,
        "unique_sources": [],
        "scanned_metadata_rows": 0,
        "has_chunks_without_source_metadata": False,
    }
    if not os.path.isdir(abs_persist):
        out["note"] = "持久化目录尚不存在（可能从未成功入库）。"
        return out
    try:
        import chromadb
    except ImportError:
        out["error"] = "未安装 chromadb，无法枚举索引"
        return out
    try:
        client = chromadb.PersistentClient(path=abs_persist)
    except Exception as e:
        logger.warning("RAG: chromadb PersistentClient 失败: %s", e)
        out["error"] = str(e)
        return out
    try:
        coll = client.get_collection(name=name)
    except Exception:
        out["note"] = "集合不存在（可能尚未成功入库）。"
        return out
    try:
        n = coll.count()
    except Exception as e:
        out["error"] = f"count 失败: {e}"
        return out
    out["chunk_count"] = n
    scan = min(max(0, metadata_scan_limit), n) if n else 0
    out["scanned_metadata_rows"] = scan
    if scan == 0:
        return out
    try:
        batch = coll.get(include=["metadatas"], limit=scan)
    except Exception as e:
        out["error"] = f"读取元数据失败: {e}"
        return out
    metas = batch.get("metadatas") or []
    sources: set[str] = set()
    for m in metas:
        if not m:
            out["has_chunks_without_source_metadata"] = True
            continue
        label = m.get("source_file") or m.get("source")
        if label:
            sources.add(os.path.basename(str(label)))
        else:
            out["has_chunks_without_source_metadata"] = True
    out["unique_sources"] = sorted(sources)
    if n > scan:
        out["note_truncated"] = (
            f"仅扫描前 {scan} 条记录的元数据用于汇总来源；chunk 总数为 {n}。"
        )
    return out


def ingest_pdf_to_chroma(
    pdf_path: str,
    language: str = "Chinese",
    page_numbers: Optional[List[int]] = None,
    source_display_name: Optional[str] = None,
) -> LangChainVectorStore:
    llm_chat, llm_embedding = get_llm(RagConfig.LLM_TYPE)
    _ = llm_chat
    store = LangChainVectorStore(embedding_model=llm_embedding)
    docs = store.load_pdf(pdf_path, page_numbers)
    chunks = store.split_documents(
        docs, chunk_size=1000, chunk_overlap=200, language=language
    )
    display = source_display_name or os.path.basename(pdf_path)
    for c in chunks:
        c.metadata = dict(c.metadata)
        c.metadata["source_file"] = display
    store.create_vectorstore(chunks)
    logger.info("RAG: 已向量化写入 Chroma，块数=%s", len(chunks))
    return store


def delete_rag_chroma_collection() -> dict[str, Any]:
    """删除当前配置下的整个 Chroma 集合（向量知识库清空）。"""
    persist = os.path.abspath(RagConfig.CHROMADB_DIRECTORY)
    name = RagConfig.CHROMADB_COLLECTION_NAME
    out: dict[str, Any] = {
        "collection_name": name,
        "persist_directory_resolved": persist,
        "ok": False,
    }
    try:
        import chromadb
    except ImportError:
        out["error"] = "未安装 chromadb"
        return out
    if not os.path.isdir(persist):
        out["ok"] = True
        out["message"] = "持久化目录不存在，无需删除"
        return out
    try:
        client = chromadb.PersistentClient(path=persist)
        try:
            client.delete_collection(name)
            out["ok"] = True
            out["message"] = f"已删除集合 {name}"
        except Exception as e:
            err = str(e).lower()
            if "does not exist" in err or "not found" in err or "no collection" in err:
                out["ok"] = True
                out["message"] = "集合本不存在"
            else:
                raise
    except Exception as e:
        logger.warning("RAG: delete collection failed: %s", e)
        out["error"] = str(e)
    return out


def delete_rag_chunks_by_source_file(source_file: str) -> dict[str, Any]:
    """按入库时的 source_file（上传文件名）删除匹配分块；便于「删掉再传」完成替换。"""
    source_file = (source_file or "").strip()
    if not source_file:
        return {"ok": False, "error": "source_file 不能为空", "deleted_count": 0}
    persist = os.path.abspath(RagConfig.CHROMADB_DIRECTORY)
    name = RagConfig.CHROMADB_COLLECTION_NAME
    out: dict[str, Any] = {"source_file": source_file, "deleted_count": 0, "ok": False}
    try:
        import chromadb
    except ImportError:
        out["error"] = "未安装 chromadb"
        return out
    try:
        client = chromadb.PersistentClient(path=persist)
        coll = client.get_collection(name=name)
    except Exception:
        out["ok"] = True
        out["message"] = "集合不存在，未删除任何分块"
        return out

    ids_to_delete: list[str] = []
    try:
        got = coll.get(where={"source_file": source_file}, include=[])
        ids_to_delete = list(got.get("ids") or [])
    except Exception as e:
        logger.info("RAG: where 删除回退为全表扫描: %s", e)
        try:
            n = coll.count()
            cap = min(n, 100_000)
            if cap <= 0:
                out["ok"] = True
                return out
            got = coll.get(include=["metadatas"], limit=cap)
            all_ids = got.get("ids") or []
            metas = got.get("metadatas") or []
            base = os.path.basename(source_file)
            for i, m in enumerate(metas):
                if not m or i >= len(all_ids):
                    continue
                sf = m.get("source_file")
                src = m.get("source")
                if sf == source_file or (
                    src and os.path.basename(str(src)) == base
                ):
                    ids_to_delete.append(all_ids[i])
        except Exception as e2:
            out["error"] = str(e2)
            return out

    if ids_to_delete:
        coll.delete(ids=ids_to_delete)
    out["deleted_count"] = len(ids_to_delete)
    out["ok"] = True
    return out


__all__ = [
    "LangChainVectorStore",
    "ingest_pdf_to_chroma",
    "list_rag_vector_index_summary",
    "delete_rag_chroma_collection",
    "delete_rag_chunks_by_source_file",
    "LLMInitializationError",
]
