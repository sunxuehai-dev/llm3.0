"""RAG 对话输出排版（与独立服务 main.format_response 一致）。"""
from __future__ import annotations

import re
from typing import Optional


def format_rag_response(response: Optional[str]) -> str:
    if not response:
        return ""
    paragraphs = re.split(r"\n{2,}", response)
    formatted_paragraphs = []
    for para in paragraphs:
        if "```" in para:
            parts = para.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = "".join(parts)
        else:
            para = para.replace(". ", ".\n")
        formatted_paragraphs.append(para.strip())
    return "\n\n".join(formatted_paragraphs)
