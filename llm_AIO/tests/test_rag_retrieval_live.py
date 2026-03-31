"""
领域 RAG 接口联调（HTTP 请求真实服务）。

直接运行（无需事先 export）:
  python tests/test_rag_retrieval_live.py

默认行为（见下方「脚本内默认配置」）:
  - 默认连本机 127.0.0.1:8000，且不在本机自动起 uvicorn（适合 systemd 已托管 API）。
  - 需要本机临时起 uvicorn 时：把 DEFAULT_RAG_LIVE_AUTO_SERVER 改为 True，或设环境变量
    RAG_LIVE_AUTO_SERVER=1。

跳过本次执行（如 CI）:
  RAG_LIVE_TEST=0 python tests/test_rag_retrieval_live.py

可选覆盖（仍可用环境变量，优先级高于脚本内常量）:
  RAG_LIVE_BASE / RAG_LIVE_AUTO_SERVER / RAG_LIVE_USER / RAG_LIVE_PASSWORD

systemd：也可在 unit 里写 Environment=RAG_LIVE_BASE=...，与脚本默认值二选一即可。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# 脚本内默认配置（按服务器实际情况改这里即可，不必 export）
# -----------------------------------------------------------------------------
DEFAULT_RAG_LIVE_BASE = "http://127.0.0.1:8000/api/v1"
# False：只连已有服务（推荐 systemctl 部署）。True：本机连不上时再子进程起 uvicorn。
DEFAULT_RAG_LIVE_AUTO_SERVER = False
DEFAULT_RAG_LIVE_USER = "admin"
DEFAULT_RAG_LIVE_PASSWORD = "admin123"
# -----------------------------------------------------------------------------


def _env_or_default(key: str, default: str) -> str:
    v = os.environ.get(key)
    if v is not None and str(v).strip() != "":
        return str(v).strip()
    return default


BASE = _env_or_default("RAG_LIVE_BASE", DEFAULT_RAG_LIVE_BASE).rstrip("/")


def _parse_base(api_v1_base: str) -> tuple[str, str, int]:
    """返回 (origin 用于 /health, hostname, port)。"""
    p = urlparse(api_v1_base)
    scheme = p.scheme or "http"
    host = p.hostname or "localhost"
    port = p.port
    if port is None:
        port = 443 if scheme == "https" else 80
    origin = f"{scheme}://{host}:{port}"
    return origin, host, port


def _loopback_host(host: str) -> bool:
    return host.lower() in ("localhost", "127.0.0.1", "::1")


def _auto_server_enabled() -> bool:
    raw = os.environ.get("RAG_LIVE_AUTO_SERVER")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_RAG_LIVE_AUTO_SERVER
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _wait_health(origin: str, timeout: float = 90.0) -> bool:
    url = f"{origin.rstrip('/')}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.getcode() == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    return False


def _spawn_uvicorn(port: int) -> subprocess.Popen:
    bind = os.environ.get("RAG_LIVE_UVICORN_HOST", "127.0.0.1")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        bind,
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )


def main() -> int:
    if os.environ.get("RAG_LIVE_TEST", "").strip().lower() in ("0", "false", "no", "skip"):
        print("跳过：RAG_LIVE_TEST=0")
        return 0

    try:
        import requests
    except ImportError:
        print("需要: pip install requests")
        return 1

    origin, host, port = _parse_base(BASE)
    user = _env_or_default("RAG_LIVE_USER", DEFAULT_RAG_LIVE_USER)
    password = _env_or_default("RAG_LIVE_PASSWORD", DEFAULT_RAG_LIVE_PASSWORD)

    proc: subprocess.Popen | None = None

    def login():
        return requests.post(
            f"{BASE}/auth/login",
            data={"username": user, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )

    try:
        try:
            lr = login()
        except requests.exceptions.ConnectionError as e:
            if _loopback_host(host) and _auto_server_enabled():
                print(f"无法连接 {origin}，尝试自动启动 uvicorn（端口 {port}）…")
                proc = _spawn_uvicorn(port)
                time.sleep(0.5)
                if proc.poll() is not None:
                    err = (proc.stderr.read() or b"").decode("utf-8", errors="replace")
                    print("uvicorn 已退出，启动失败。stderr:\n", err[-4000:] or "(空)")
                    return 1
                if not _wait_health(origin):
                    err = (proc.stderr.read() or b"").decode("utf-8", errors="replace")
                    print("等待 /health 超时。uvicorn stderr（节选）:\n", err[-4000:] or "(空)")
                    return 1
                print("服务已就绪:", origin)
                try:
                    lr = login()
                except requests.exceptions.ConnectionError as e2:
                    print("自动启动后仍无法登录:", e2)
                    return 1
                except requests.exceptions.Timeout:
                    print("自动启动后登录请求超时。")
                    return 1
            else:
                print("连接失败（Connection refused / 无法连接）。")
                print(f"  当前 RAG_LIVE_BASE = {BASE}")
                if not _auto_server_enabled():
                    print("  当前为「不自动起服务」模式。请确认 systemd 服务已启动，或修改脚本顶部 DEFAULT_RAG_LIVE_BASE。")
                else:
                    print("  非本机地址时请启动远端服务，或修改 DEFAULT_RAG_LIVE_BASE / RAG_LIVE_BASE。")
                print("  手动示例：cd 项目根 && uvicorn app.main:app --host 0.0.0.0 --port 8000")
                print(f"  底层错误: {e}")
                return 1
        except requests.exceptions.Timeout:
            print(f"连接超时（{BASE}），请检查网络或防火墙。")
            return 1

        if lr.status_code != 200:
            print("登录失败:", lr.status_code, lr.text[:500])
            return 1
        token = lr.json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}

        st = requests.get(f"{BASE}/knowledge-retrieval/status", headers=h, timeout=15)
        print("GET status:", st.status_code, st.text[:800])
        if st.status_code != 200:
            return 1

        cc = requests.post(
            f"{BASE}/knowledge-retrieval/chat/completions",
            headers={**h, "Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "conversationId": "live-test",
            },
            timeout=120,
        )
        print("POST chat/completions:", cc.status_code)
        try:
            print(json.dumps(cc.json(), ensure_ascii=False, indent=2)[:2000])
        except Exception:
            print(cc.text[:2000])

        if cc.status_code == 503:
            print("提示: RAG 未就绪时返回 503 为预期，请检查 RAG_* 与向量库。")
        return 0 if cc.status_code in (200, 503) else 1
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
