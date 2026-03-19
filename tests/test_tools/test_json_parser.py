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
