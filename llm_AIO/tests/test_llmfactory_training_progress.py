"""
微调训练进度：load_training_progress 单元测试 + GET .../train/jobs/{job_id}/progress 接口测试。

在导入 `app.routers.llmfactory` 之前注入 `app.services.llmfactory_service` 与 `ijson` 的 stub，
避免拉起 llmfactory 源码树中的可选/编译依赖（如 ijson、fast_converter），使本测试可在精简环境中运行。

运行（项目根目录 llm_AIO）:
  python -m unittest tests.test_llmfactory_training_progress -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
import warnings
from unittest.mock import MagicMock, patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if "ijson" not in sys.modules:
    sys.modules["ijson"] = MagicMock()

if "app.services.llmfactory_service" not in sys.modules:
    _lf_stub = types.ModuleType("app.services.llmfactory_service")

    def _templates():
        return []

    _lf_stub.get_llmfactory_templates = _templates
    _lf_stub.sft_lora_train = lambda *a, **k: (0, None)
    _lf_stub.sft_qlora_train = lambda *a, **k: (0, None)
    _lf_stub.sft_full_train = lambda *a, **k: (0, None)
    _lf_stub.merge_adapter = lambda *a, **k: (0, None)
    _lf_stub.start_inference_api = lambda *a, **k: (None, 0, "")
    sys.modules["app.services.llmfactory_service"] = _lf_stub

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from convert_url import Base as ConvertBase, TrainingJob
from app.database import get_convert_db
from app.routers.llmfactory import router as llmfactory_router
from app.services.llmfactory_progress import infer_model_display_name, load_training_progress


def _mini_llmfactory_app() -> FastAPI:
    app = FastAPI()
    app.include_router(llmfactory_router)
    return app


class TestLoadTrainingProgress(unittest.TestCase):
    """直接测试 trainer_state.json 解析逻辑（不启动 HTTP）。"""

    def test_missing_dir(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "nope")
            out = load_training_progress(missing)
        self.assertFalse(out["trainer_state_found"])
        self.assertIsNotNone(out["message"])

    def test_missing_trainer_state(self):
        with tempfile.TemporaryDirectory() as td:
            out = load_training_progress(td)
        self.assertFalse(out["trainer_state_found"])
        self.assertIn("trainer_state", out["message"] or "")

    def test_parses_state_and_progress(self):
        with tempfile.TemporaryDirectory() as td:
            state = {
                "global_step": 100,
                "max_steps": 1000,
                "epoch": 0.5,
                "log_history": [
                    {"step": 50, "loss": 0.9, "learning_rate": 1e-4, "epoch": 0.25},
                    {"step": 100, "loss": 0.3, "learning_rate": 9e-5, "epoch": 0.5},
                ],
            }
            with open(os.path.join(td, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump(state, f)
            out = load_training_progress(td, log_history_limit=1)
        self.assertTrue(out["trainer_state_found"])
        self.assertEqual(out["global_step"], 100)
        self.assertEqual(out["max_steps"], 1000)
        self.assertAlmostEqual(out["latest_loss"], 0.3)
        self.assertAlmostEqual(out["progress_ratio"], 0.1)
        self.assertEqual(len(out["log_history"]), 1)
        self.assertEqual(out["log_history"][0]["step"], 100)

    def test_training_args_num_epochs_fallback_ratio(self):
        with tempfile.TemporaryDirectory() as td:
            state = {
                "global_step": 50,
                "max_steps": -1,
                "epoch": 1.0,
                "log_history": [{"loss": 0.1, "epoch": 1.0}],
            }
            with open(os.path.join(td, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump(state, f)
            with open(os.path.join(td, "training_args.json"), "w", encoding="utf-8") as f:
                json.dump({"num_train_epochs": 4}, f)
            out = load_training_progress(td)
        self.assertTrue(out["trainer_state_found"])
        self.assertEqual(out["num_train_epochs"], 4)
        self.assertAlmostEqual(out["progress_ratio"], 0.25)

    def test_infer_model_display_name_is_output_basename(self):
        d = os.path.join("/tmp", "SFT", "Qwen3-0.6B_lora_20260323_102510")
        self.assertEqual(infer_model_display_name(d, "lora"), "Qwen3-0.6B_lora_20260323_102510")

    def test_infer_model_display_name_merge_uses_basename(self):
        d = os.path.join("/out", "merged", "foo_bar_merged")
        self.assertEqual(infer_model_display_name(d, "merge"), "foo_bar_merged")


class TestTrainingProgressEndpoint(unittest.TestCase):
    """GET /api/playground/llmfactory/train/jobs/{job_id}/progress"""

    def setUp(self):
        # convert_url 中 TrainingJob 等仍用 datetime.utcnow 作 Column default；Python 3.12+ 弃用 utcnow，
        # SQLAlchemy 在 flush/commit 时会触发 DeprecationWarning。unittest 会单独配置 warnings，模块级 filter 常被覆盖，
        # 故在会写入 ORM 的用例里用 catch_warnings 局部忽略 DeprecationWarning。
        self._warn_cm = warnings.catch_warnings()
        self._warn_cm.__enter__()
        warnings.simplefilter("ignore", DeprecationWarning)

        self._td = tempfile.TemporaryDirectory()
        self.train_root = self._td.name
        self.output_dir = os.path.join(self.train_root, "SFT", "Qwen3_ut_lora_20260322_120000")
        os.makedirs(self.output_dir, exist_ok=True)
        state = {
            "global_step": 100,
            "max_steps": 1000,
            "epoch": 0.5,
            "log_history": [
                {"step": 50, "loss": 0.8, "learning_rate": 1e-4, "epoch": 0.25},
                {"step": 100, "loss": 0.4, "learning_rate": 9e-5, "epoch": 0.5},
            ],
        }
        with open(os.path.join(self.output_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f)

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        ConvertBase.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        self.db = Session()
        self.db.add(
            TrainingJob(
                id="job-progress-ut",
                output_dir=self.output_dir,
                task_type="lora",
                status="running",
            )
        )
        self.db.commit()

        self.app = _mini_llmfactory_app()

        def _override_convert_db():
            try:
                yield self.db
            finally:
                pass

        self.app.dependency_overrides[get_convert_db] = _override_convert_db
        self.client = TestClient(self.app)
        self._patch = patch(
            "app.routers.llmfactory._get_training_output_base",
            return_value=self.train_root,
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.app.dependency_overrides.clear()
        self.db.close()
        self._td.cleanup()
        self._warn_cm.__exit__(None, None, None)

    def test_progress_200(self):
        r = self.client.get(
            "/api/playground/llmfactory/train/jobs/job-progress-ut/progress",
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["trainer_state_found"])
        self.assertEqual(data["global_step"], 100)
        self.assertEqual(data["job_status"], "running")
        self.assertEqual(data["model_display_name"], "Qwen3_ut_lora_20260322_120000")
        self.assertAlmostEqual(data["progress_ratio"], 0.1)
        self.assertEqual(len(data["log_history"]), 2)

    def test_log_limit_query(self):
        r = self.client.get(
            "/api/playground/llmfactory/train/jobs/job-progress-ut/progress?log_limit=1",
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["log_history"]), 1)

    def test_job_not_found_404(self):
        r = self.client.get(
            "/api/playground/llmfactory/train/jobs/does-not-exist/progress",
        )
        self.assertEqual(r.status_code, 404)

    def test_merge_job_no_trainer_state(self):
        merge_out = os.path.join(self.train_root, "merged", "demo_merged")
        os.makedirs(merge_out, exist_ok=True)
        self.db.add(
            TrainingJob(
                id="job-merge-ut",
                output_dir=merge_out,
                task_type="merge",
                status="running",
            )
        )
        self.db.commit()
        r = self.client.get(
            "/api/playground/llmfactory/train/jobs/job-merge-ut/progress",
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertFalse(data["trainer_state_found"])
        self.assertEqual(data["model_display_name"], "demo_merged")
        self.assertIn("合并", data.get("message") or "")


if __name__ == "__main__":
    unittest.main()
