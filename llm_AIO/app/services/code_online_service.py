import asyncio
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, Optional

import httpx


def _parse_python_envs_from_env() -> Dict[str, str]:
    """
    读取本地 Python 解释器映射表。

    支持两种配置方式（二选一，JSON 优先）：
    - `CODE_ONLINE_LOCAL_PYTHON_ENVS_JSON`：形如 {"python3.10":"D:\\path\\python.exe"}
    - `CODE_ONLINE_PYTHON_ENVS`：形如 "python3.10=D:\\path\\python.exe;python3.11=C:\\Python311\\python.exe"
    """
    default_python = os.getenv("CODE_ONLINE_DEFAULT_PYTHON", "python")
    mapping: Dict[str, str] = {"default": default_python}

    raw_json = os.getenv("CODE_ONLINE_LOCAL_PYTHON_ENVS_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(k, str) and isinstance(v, str) and v.strip():
                        mapping[k.strip()] = v.strip()
            return mapping
        except json.JSONDecodeError:
            # fallthrough to other env format
            pass

    raw_pairs = os.getenv("CODE_ONLINE_PYTHON_ENVS", "").strip()
    if raw_pairs:
        for part in raw_pairs.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                mapping[k] = v

    return mapping


def get_python_command(env: str) -> str:
    """根据环境名称获取对应的 Python 执行命令。"""
    env_name = (env or "").strip() or "default"
    mapping = _parse_python_envs_from_env()
    return mapping.get(env_name, mapping.get("default", "python"))


def fix_error_line_number(error_msg: str, added_lines: int) -> str:
    """
    修正错误信息中的行号。
    :param error_msg: 原始错误信息
    :param added_lines: 后端添加的行数（通常是 1 行编码声明）
    """
    if added_lines == 0:
        return error_msg

    pattern = re.compile(r"(line\s+)(\d+)")

    def replace_line_num(match: re.Match) -> str:
        original_num = int(match.group(2))
        fixed_num = max(1, original_num - added_lines)
        return f"{match.group(1)}{fixed_num}"

    return pattern.sub(replace_line_num, error_msg)


def run_python_code(code: str, env: str = "default", timeout_seconds: int = 10) -> Dict[str, str | bool]:
    """
    在指定本地 python 解释器下执行临时代码。
    返回：
      - {"success": True, "output": stdout}
      - {"success": False, "error": stderr_or_reason}
    """
    if not code:
        return {"success": False, "error": "代码不能为空"}

    added_lines = 0

    # 自动添加 UTF-8 编码声明（仅当用户未添加时）
    if not code.startswith("# -*- coding:") and not code.startswith("# coding="):
        code = "# -*- coding: utf-8 -*-\n" + code
        added_lines = 1

    python_command = get_python_command(env)
    temp_file_path: str = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_file_path = f.name

        result = subprocess.run(
            [python_command, temp_file_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )

        if result.returncode == 0:
            return {"success": True, "output": result.stdout}

        error_msg = result.stderr
        fixed_error_msg = fix_error_line_number(error_msg, added_lines)

        if "No such file or directory" in fixed_error_msg or "系统找不到指定的文件" in fixed_error_msg:
            fixed_error_msg = f"⚠️ 环境 {env} 的Python解释器路径不存在！请检查后端 CODE_ONLINE_* 配置。"

        return {"success": False, "error": fixed_error_msg}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"代码运行超时（最大{timeout_seconds}秒）"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass


async def chat_with_llm(
    user_input: str,
    code: Optional[str] = None,
    error: Optional[str] = None,
    model: str = "qwen-plus",
    temperature: float = 0.7,
    timeout_seconds: int = 30,
) -> Dict[str, str | bool]:
    """
    调用 DashScope OpenAI-compatible 的 chat/completions 接口。
    返回：
      - {"success": True, "reply": content}
      - {"success": False, "error": ...}
    """
    if not user_input:
        return {"success": False, "error": "input 不能为空"}

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "error": "未设置 DASHSCOPE_API_KEY，请在后端 .env 中配置后重试"}

    if code and error:
        prompt = f"""
用户需求：{user_input}
当前代码：
{code}
运行错误信息：
{error}

请基于以上信息回答用户问题，要求：
1. 如果是代码错误，明确指出问题原因和修复方案
2. 如果是生成代码，给出可运行的完整代码
3. 如果是优化建议，给出具体的优化点和代码示例
4. 回答语言为中文，简洁易懂
        """
    else:
        prompt = f"""
用户需求：{user_input}

请回答用户的问题，要求：
1. 回答语言为中文，简洁易懂
2. 按照用户需求直接回复，不要提及任何未提供的代码内容
        """

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=False) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return {"success": False, "error": "大模型返回结果为空（choices 为空）"}
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            return {"success": True, "reply": content}
    except httpx.HTTPError as e:
        return {"success": False, "error": f"大模型调用失败：{str(e)}"}


async def run_python_code_in_thread(code: str, env: str = "default", timeout_seconds: int = 10) -> Dict[str, str | bool]:
    """用于路由层避免阻塞事件循环。"""
    return await asyncio.to_thread(run_python_code, code, env, timeout_seconds)

