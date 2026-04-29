"""parse_llm_json 单元测试"""

import json
import pytest
from darwinian.utils.json_parser import parse_llm_json


def test_plain_json():
    result = parse_llm_json('{"verdict": "PASS", "feedback": "ok"}')
    assert result["verdict"] == "PASS"


def test_json_with_markdown_block():
    content = '```json\n{"verdict": "PASS", "feedback": "ok"}\n```'
    result = parse_llm_json(content)
    assert result["verdict"] == "PASS"


def test_json_with_plain_code_block():
    content = '```\n{"verdict": "PASS"}\n```'
    result = parse_llm_json(content)
    assert result["verdict"] == "PASS"


def test_json_with_leading_text():
    content = '以下是结果：\n{"verdict": "PASS"}'
    result = parse_llm_json(content)
    assert result["verdict"] == "PASS"


def test_json_with_whitespace():
    result = parse_llm_json('  \n  {"key": "value"}  \n  ')
    assert result["key"] == "value"


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("这不是 JSON")


# ===========================================================================
# R10-Pri-4: <think> 块处理（特别是 unclosed think 防误抓）
# ===========================================================================

def test_closed_think_block_stripped():
    """正常情况：<think>...</think> 后跟 JSON"""
    content = "<think>let me reason</think>\n{\"verdict\": \"PASS\"}"
    result = parse_llm_json(content)
    assert result["verdict"] == "PASS"


def test_unclosed_think_finds_json_after():
    """unclosed <think> 但 JSON 真在后面（≥2 个 ASCII key:value）"""
    content = (
        "<think>\nstill thinking..."
        "\nlots of reasoning\n\n"
        '{"verdict": "PASS", "feedback": "ok"}'
    )
    result = parse_llm_json(content)
    assert result["verdict"] == "PASS"


def test_unclosed_think_with_chinese_field_does_not_misfire():
    """v2 LIVE 实测 bug: <think> 内中文字段 '**用户的研究方向**:'  含 ':' 触发误抓.
    R10-Pri-4 fix: 要求 ≥2 个 ASCII identifier "key": 模式才认作 JSON,
    所以中文 markdown 字段不会被误判.
    """
    content = (
        "<think>\nLet me analyze this request carefully.\n"
        "**用户的研究方向**: encrypted traffic\n"
        "**约束**:\n"
        "- GPU: 4× RTX PRO 6000\n"
        "- 时长: 7 days"
    )
    # 应该 fail（没有真 JSON），不该把 thinking 当 JSON 解析成功
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json(content)


def test_unclosed_think_real_json_after_still_works():
    """R10-Pri-4 回归保护：unclosed <think> + 紧跟合法 JSON，仍要正确解出
    （防 fix 过度收紧把真 JSON 也漏掉）"""
    content = (
        "<think>\n"
        "Let me reason in detail about this task.\n"
        '\n{"verdict": "PASS", "feedback": "good", "score": 0.9}'
    )
    result = parse_llm_json(content)
    assert result["verdict"] == "PASS"
