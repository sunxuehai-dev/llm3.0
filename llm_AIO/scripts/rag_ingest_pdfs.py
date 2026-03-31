#!/usr/bin/env python3
"""
将目录下所有 PDF 批量上传到网关 RAG（POST /api/v1/knowledge-retrieval/ingest-pdf）。

需要账号具备 MANAGE_KNOWLEDGE_BASE（如 admin / teacher）。

示例（原 Knowledge_retrive_and_search/input 目录）:
  python scripts/rag_ingest_pdfs.py --dir /path/to/input

本机默认连 127.0.0.1:8000；账号默认 admin / admin123，可在下方常量或参数中修改。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 可按部署修改（参数可覆盖）
DEFAULT_API_BASE = os.environ.get("RAG_LIVE_BASE", "http://127.0.0.1:8000/api/v1").rstrip("/")
DEFAULT_USER = os.environ.get("RAG_LIVE_USER", "admin")
DEFAULT_PASSWORD = os.environ.get("RAG_LIVE_PASSWORD", "admin123")


def login(base: str, user: str, password: str) -> str:
    import requests

    r = requests.post(
        f"{base}/auth/login",
        data={"username": user, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def ingest_one(base: str, token: str, pdf_path: Path) -> tuple[bool, str]:
    import requests

    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{base}/knowledge-retrieval/ingest-pdf",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=600,
        )
    if r.status_code == 200:
        return True, r.text
    return False, f"{r.status_code} {r.text[:500]}"


def main() -> int:
    p = argparse.ArgumentParser(description="批量上传 PDF 到 RAG Chroma")
    p.add_argument("--dir", type=Path, required=True, help="含 PDF 的目录（如原项目 input/）")
    p.add_argument("--base", default=DEFAULT_API_BASE, help="API 前缀，含 /api/v1")
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--password", default=DEFAULT_PASSWORD)
    args = p.parse_args()

    d = args.dir.resolve()
    if not d.is_dir():
        print("目录不存在:", d, file=sys.stderr)
        return 1

    unique = sorted({x.resolve() for x in d.rglob("*.pdf") if x.is_file()})

    if not unique:
        print("未找到 PDF:", d)
        return 1

    try:
        import requests
    except ImportError:
        print("需要: pip install requests", file=sys.stderr)
        return 1

    base = args.base.rstrip("/")
    print("登录:", base, "用户:", args.user)
    try:
        token = login(base, args.user, args.password)
    except Exception as e:
        print("登录失败:", e, file=sys.stderr)
        return 1

    ok_n = 0
    for i, path in enumerate(unique, 1):
        print(f"[{i}/{len(unique)}] 上传 {path.name} ...", flush=True)
        ok, msg = ingest_one(base, token, path)
        if ok:
            print("  ->", msg[:200])
            ok_n += 1
        else:
            print("  -> 失败", msg, file=sys.stderr)

    print(f"完成: 成功 {ok_n}/{len(unique)}")
    return 0 if ok_n == len(unique) else 1


if __name__ == "__main__":
    sys.exit(main())
