"""
领域 RAG（knowledge-retrieval）接口测试。

使用 FastAPI TestClient + dependency_overrides 模拟登录用户，避免强依赖真实账号；
对 LangGraph / 外部 LLM 使用 mock，默认不发起真实推理。

运行（项目根目录）:
  python -m unittest tests.test_rag_retrieval -v
"""

from __future__ import annotations

import os
import sys
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

try:
    from app.main import app
    from app.security import get_current_user

    _APP_IMPORT_OK = True
except ImportError:
    app = None  # type: ignore
    get_current_user = None  # type: ignore
    _APP_IMPORT_OK = False


def _fake_user(role: str = "super_admin", uid: int = 1):
    return SimpleNamespace(
        id=uid,
        username="rag_test_user",
        role=role,
        is_active=True,
    )


@unittest.skipUnless(_APP_IMPORT_OK, "无法导入 app.main（请安装完整依赖，如 ijson、langchain 等）")
class TestRagRetrievalAPI(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides.clear()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_status_without_auth(self):
        r = self.client.get("/api/v1/knowledge-retrieval/status")
        self.assertIn(r.status_code, (401, 403))

    def test_status_authenticated_shape(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("teacher")
        r = self.client.get("/api/v1/knowledge-retrieval/status")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertIn("ready", data)
        self.assertIn("error", data)
        self.assertIsInstance(data["ready"], bool)
        self.assertIn("vector_kb_disabled", data)

    def test_chat_empty_messages_400(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("student")
        with patch("app.routers.rag_retrieval.is_rag_ready", return_value=True), patch(
            "app.routers.rag_retrieval.get_rag_dependencies",
            return_value=(MagicMock(), MagicMock()),
        ):
            r = self.client.post(
                "/api/v1/knowledge-retrieval/chat/completions",
                json={"messages": []},
            )
        self.assertEqual(r.status_code, 400)

    def test_chat_rag_not_ready_503(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("student")
        with patch("app.routers.rag_retrieval.is_rag_ready", return_value=False), patch(
            "app.routers.rag_retrieval.get_rag_init_error",
            return_value="unit: RAG off",
        ):
            r = self.client.post(
                "/api/v1/knowledge-retrieval/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
        self.assertEqual(r.status_code, 503)
        self.assertIn("detail", r.json())

    def test_chat_mock_graph_returns_200(self):
        """图 stream 为空时，处理器得到默认文案，仍返回 200 JSON。"""
        mock_graph = MagicMock()
        mock_graph.stream.return_value = []
        mock_tc = MagicMock()
        mock_tc.get_tool_names.return_value = set()

        app.dependency_overrides[get_current_user] = lambda: _fake_user("super_admin")
        with patch("app.routers.rag_retrieval.is_rag_ready", return_value=True), patch(
            "app.routers.rag_retrieval.get_rag_dependencies",
            return_value=(mock_graph, mock_tc),
        ):
            r = self.client.post(
                "/api/v1/knowledge-retrieval/chat/completions",
                json={"messages": [{"role": "user", "content": "ping"}]},
            )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body.get("object"), "chat.completion")
        self.assertTrue(body.get("choices"))
        msg = body["choices"][0]["message"]
        self.assertEqual(msg.get("role"), "assistant")
        self.assertIsInstance(msg.get("content"), str)

    def test_ingest_pdf_student_403(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("student")
        r = self.client.post(
            "/api/v1/knowledge-retrieval/ingest-pdf",
            files={"file": ("doc.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        self.assertEqual(r.status_code, 403)

    def test_ingest_non_pdf_400(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("super_admin")
        r = self.client.post(
            "/api/v1/knowledge-retrieval/ingest-pdf",
            files={"file": ("note.txt", b"hello", "text/plain")},
        )
        self.assertEqual(r.status_code, 400)

    def test_ingest_pdf_mock_success(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("super_admin")
        with patch("app.routers.rag_retrieval.ingest_pdf_to_chroma") as m:
            r = self.client.post(
                "/api/v1/knowledge-retrieval/ingest-pdf",
                files={"file": ("doc.pdf", b"%PDF-1.4 minimal\n", "application/pdf")},
            )
        self.assertEqual(r.status_code, 200, r.text)
        m.assert_called_once()
        data = r.json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("filename"), "doc.pdf")
        kwargs = m.call_args.kwargs
        self.assertEqual(kwargs.get("source_display_name"), "doc.pdf")

    def test_library_without_auth(self):
        r = self.client.get("/api/v1/knowledge-retrieval/library")
        self.assertIn(r.status_code, (401, 403))

    def test_library_authenticated_shape(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("student")
        fake = {
            "persist_directory": "chroma_db_rag",
            "persist_directory_resolved": "/tmp/x",
            "collection_name": "demo001",
            "chunk_count": 3,
            "unique_sources": ["a.pdf"],
            "scanned_metadata_rows": 3,
            "has_chunks_without_source_metadata": False,
        }
        with patch(
            "app.routers.rag_retrieval.list_rag_vector_index_summary",
            return_value=fake,
        ):
            r = self.client.get("/api/v1/knowledge-retrieval/library")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["chunk_count"], 3)
        self.assertEqual(data["unique_sources"], ["a.pdf"])
        self.assertIn("storage_hint", data)

    def test_library_scan_limit_400(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("teacher")
        r = self.client.get(
            "/api/v1/knowledge-retrieval/library",
            params={"metadata_scan_limit": 0},
        )
        self.assertEqual(r.status_code, 400)

    def test_vector_settings_student_403(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("student")
        r = self.client.put(
            "/api/v1/knowledge-retrieval/admin/vector-settings",
            json={"vector_kb_disabled": True},
        )
        self.assertEqual(r.status_code, 403)

    def test_delete_library_student_403(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("student")
        r = self.client.delete("/api/v1/knowledge-retrieval/library")
        self.assertEqual(r.status_code, 403)

    def test_delete_by_source_student_403(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user("student")
        r = self.client.delete(
            "/api/v1/knowledge-retrieval/library/by-source",
            params={"source_file": "x.pdf"},
        )
        self.assertEqual(r.status_code, 403)


class TestRagResponseFormat(unittest.TestCase):
    """仅依赖 app.services.rag.response_format，无需拉起完整 FastAPI。"""

    def test_format_paragraphs(self):
        from app.services.rag.response_format import format_rag_response

        out = format_rag_response("第一段\n\n第二段")
        self.assertIn("第一段", out)
        self.assertIn("第二段", out)

    def test_format_code_fence(self):
        from app.services.rag.response_format import format_rag_response

        raw = "说明\n```\ncode\n```\n"
        out = format_rag_response(raw)
        self.assertIn("```", out)


if __name__ == "__main__":
    unittest.main()
