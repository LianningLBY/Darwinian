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
    except json.JSONDecodeError:
        pass

    # 修复 JSON 字符串值中的字面量控制字符（换行/回车/制表符）
    # 原因：LLM 生成代码时常把多行代码直接写入 JSON 字符串，而非用 \n 转义
    fixed2 = _escape_control_chars_in_strings(fixed)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    # 修复截断的 JSON（LLM 超出 token 上限导致输出中途中断）
    # 强制关闭未完成的字符串和嵌套结构
    fixed3 = _repair_truncated_json(fixed2)
    try:
        return json.loads(fixed3)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"LLM 输出无法解析为 JSON。原始内容（前 500 字符）：\n{content[:500]}",
            e.doc,
            e.pos,
        ) from e


def _repair_truncated_json(text: str) -> str:
    """
    修复因 LLM token 截断导致的不完整 JSON。
    逐字符扫描，跟踪字符串/对象/数组的开闭状态，
    在末尾补全缺失的引号和括号。
    """
    result: list[str] = []
    in_string = False
    escape_next = False
    stack: list[str] = []   # 记录未关闭的 '{' 或 '['

    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue

        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if not in_string:
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()

        result.append(ch)

    # 补全截断处
    if in_string:
        result.append('"')   # 关闭未完成的字符串

    # 关闭未完成的嵌套结构（从内到外）
    for opener in reversed(stack):
        result.append("}" if opener == "{" else "]")

    return "".join(result)


def _escape_control_chars_in_strings(text: str) -> str:
    """
    逐字符扫描 JSON，将字符串值内部的字面量控制字符（\\n \\r \\t）转义。
    不改变字符串之外的任何字符，保证 JSON 结构完整。
    """
    result: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue

        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if in_string:
            if ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
            else:
                result.append(ch)
        else:
            result.append(ch)

    return "".join(result)
