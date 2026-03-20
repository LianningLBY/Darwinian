"""
LLM 输出 JSON 解析工具

部分模型（如 MiniMax）会在 JSON 外包裹 markdown 代码块，
本模块统一处理后再解析，所有 Agent 通过此函数解析 LLM 输出。
"""

from __future__ import annotations

import json
import re


def parse_llm_json(content: str) -> dict:
    """
    解析 LLM 返回的 JSON 字符串，兼容以下格式：
    1. 纯 JSON 字符串
    2. ```json ... ``` 包裹的代码块
    3. ``` ... ``` 包裹的代码块（无语言标识）
    4. 首尾有多余空白或说明文字

    Raises:
        json.JSONDecodeError: 无法解析时抛出，附带原始内容以便调试
    """
    text = content.strip()

    # 剥离推理模型的 <think>...</think> 块（MiniMax / DeepSeek-R1 等）
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # 尝试剥离 markdown 代码块
    # 匹配 ```json ... ``` 或 ``` ... ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        text = code_block.group(1).strip()

    # 如果仍不是 { 或 [ 开头，尝试找到第一个 { 的位置
    if text and text[0] not in ("{", "["):
        brace_pos = text.find("{")
        bracket_pos = text.find("[")
        start = min(
            brace_pos if brace_pos != -1 else len(text),
            bracket_pos if bracket_pos != -1 else len(text),
        )
        if start < len(text):
            text = text[start:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 修复非法反斜杠转义（LLM 在 JSON 字符串中写了 LaTeX 公式如 \mathcal, \sum 等）
    # JSON 只允许 \" \\ \/ \b \f \n \r \t \uXXXX，其余 \X 均非法
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"LLM 输出无法解析为 JSON。原始内容（前 500 字符）：\n{content[:500]}",
            e.doc,
            e.pos,
        ) from e
