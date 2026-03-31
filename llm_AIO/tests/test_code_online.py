"""
代码在线（code-online）接口测试

测试脚本风格与本仓库其它 tests 保持一致：
- 通过 requests 调用运行中的服务（默认 http://localhost:8000）
- 不依赖 pytest
"""

import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import requests


# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "http://localhost:8000"

# 测试结果输出目录
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


def print_result(test_name: str, success: bool, message: str = "", data: Optional[Dict[str, Any]] = None):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    if data is not None:
        print(f"      响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
    print()


def test_health_check() -> bool:
    """测试健康检查"""
    test_name = "健康检查"
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print_result(test_name, True, "服务正常", data)
            return True
        print_result(test_name, False, f"状态码: {resp.status_code}")
        return False
    except Exception as e:
        print_result(test_name, False, f"无法连接到服务器: {str(e)}")
        return False


def _post_json(path: str, payload: Dict[str, Any], timeout: int = 20) -> Tuple[int, Dict[str, Any]]:
    resp = requests.post(
        f"{BASE_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {"raw": resp.text}


def test_run_code_success() -> bool:
    """测试 run-code 成功路径"""
    test_name = "run-code 成功"
    payload = {
        "code": "print('hello from code-online')",
        "env": "default",
        "timeout_seconds": 5,
    }
    try:
        status_code, data = _post_json("/api/playground/code-online/run-code", payload, timeout=15)
        if status_code == 200 and data.get("success") is True:
            output = data.get("output") or ""
            ok = "hello from code-online" in output
            print_result(test_name, ok, "stdout 包含预期内容" if ok else "stdout 不包含预期内容", data)
            return ok

        # 如果 run-code 被禁用（例如 CODE_ONLINE_ENABLE_RUN_CODE=false）
        if status_code == 403:
            print_result(test_name, True, "run-code 已在服务端禁用（403），跳过断言", data)
            return True

        print_result(test_name, False, f"非预期响应（status_code={status_code}）", data)
        return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False


def test_run_code_error_line_fix() -> bool:
    """
    测试 run-code 的错误行号修正：
    - 服务端会自动在用户代码前插入一行编码声明（added_lines=1）
    - Python traceback 通常会报告真实临时代码行号（用户代码原第一行 -> 报 line 2）
    - 服务端应将其修正为 line 1
    """
    test_name = "run-code 错误行号修正"
    payload = {
        "code": "1/0",
        "env": "default",
        "timeout_seconds": 5,
    }
    try:
        status_code, data = _post_json("/api/playground/code-online/run-code", payload, timeout=15)
        if status_code == 403:
            print_result(test_name, True, "run-code 已在服务端禁用（403），跳过断言", data)
            return True

        if status_code != 200:
            print_result(test_name, False, f"非预期响应（status_code={status_code}）", data)
            return False

        if data.get("success") is not True:
            error_msg = data.get("error") or ""
            ok = ("ZeroDivisionError" in error_msg) and ("line 1" in error_msg)
            print_result(test_name, ok, "stderr 包含 ZeroDivisionError 且行号修正为 line 1" if ok else "stderr 未符合预期", data)
            return ok

        print_result(test_name, False, "预期失败但实际返回 success=true", data)
        return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False


def test_chat_with_llm_no_code() -> bool:
    """测试 chat-with-llm 的普通问答路径（不带 code/error）"""
    test_name = "chat-with-llm（无 code/error）"
    payload = {
        "input": "请用一句话解释什么是 Python。"
    }
    try:
        status_code, data = _post_json("/api/playground/code-online/chat-with-llm", payload, timeout=60)

        # 没配 DASHSCOPE_API_KEY 或网络/额度问题：这里允许跳过
        if status_code != 200:
            msg = str(data)
            if "DASHSCOPE_API_KEY" in msg or "大模型调用失败" in msg or "未设置" in msg:
                print_result(test_name, True, f"无法调用大模型（status_code={status_code}），跳过", data)
                return True
            print_result(test_name, False, f"非预期状态码: {status_code}", data)
            return False

        if data.get("success") is True:
            reply = data.get("reply") or ""
            ok = len(reply.strip()) > 0
            print_result(test_name, ok, "reply 非空" if ok else "reply 为空", data)
            return ok

        # success=false：同样按错误信息决定是否跳过
        msg = str(data)
        if "DASHSCOPE_API_KEY" in msg or "未设置" in msg or "大模型调用失败" in msg:
            print_result(test_name, True, "大模型未就绪，跳过", data)
            return True

        print_result(test_name, False, "模型调用返回失败", data)
        return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("code-online 接口测试")
    print("=" * 60)
    print()

    if not test_health_check():
        print("❌ 服务未启动，请先启动: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    print("-" * 60)
    test_run_code_success()
    print("-" * 60)
    test_run_code_error_line_fix()
    print("-" * 60)
    test_chat_with_llm_no_code()
    print("-" * 60)

    print("测试完成")


if __name__ == "__main__":
    main()

