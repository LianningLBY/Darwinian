"""
phenomenon_miner 单元测试

不打真实 LLM —— mock invoke_with_retry，验证：
- 单论文挖掘正常返 list[Phenomenon]
- type 校验（unexplained_trend / surprising_result 接受，cross_paper_contradiction 在
  单论文路径里被过滤，其他类型直接丢）
- 缺 description / supporting_quote 的条目被丢弃
- max_per_paper 截断
- LLM 失败 / JSON 解析失败 → 安全返 []
- batch_mine_phenomena 对每篇调一次，full_text_provider 异常吞掉
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.phenomenon_miner import (
    mine_phenomena,
    batch_mine_phenomena,
    _build_user_message,
)
from darwinian.state import Phenomenon


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _good_response(phenomena_list: list[dict]):
    return MagicMock(content=_json.dumps({"phenomena": phenomena_list}))


def _ph(type="surprising_result",
        description="LayerSkip 在 layer 16 上 PPL 8.12 而 baseline 1900",
        quote="surprisingly, layer 16 yields acceptable PPL while ...",
        question="why does layer 16 alone tolerate quantization?"):
    return {
        "type": type,
        "description": description,
        "supporting_quote": quote,
        "suggested_question": question,
    }


# ---------------------------------------------------------------------------
# mine_phenomena 单论文
# ---------------------------------------------------------------------------

class TestMinePhenomenaHappyPath:
    def test_basic_extraction(self):
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph()])):
            phenomena = mine_phenomena(
                paper_id="arxiv:2404.16710",
                title="LayerSkip",
                full_text="...",
                abstract="abs",
                llm=MagicMock(),
            )
        assert len(phenomena) == 1
        ph = phenomena[0]
        assert ph.type == "surprising_result"
        assert "PPL 8.12" in ph.description
        assert "surprisingly" in ph.supporting_quote
        assert ph.paper_ids == ["arxiv:2404.16710"]
        assert ph.suggested_question.startswith("why does")

    def test_unexplained_trend_type_accepted(self):
        item = _ph(type="unexplained_trend",
                   description="trend X observed but not explained",
                   quote="we leave investigation of X to future work")
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([item])):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert len(phenomena) == 1
        assert phenomena[0].type == "unexplained_trend"

    def test_max_per_paper_caps(self):
        items = [_ph(description=f"phenomenon {i}", quote=f"quote {i}")
                 for i in range(10)]
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response(items)):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock(),
                                        max_per_paper=3)
        assert len(phenomena) == 3

    def test_long_quote_truncated_to_500_chars(self):
        long_quote = "x" * 1000
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph(quote=long_quote)])):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert len(phenomena[0].supporting_quote) == 500


class TestMinePhenomenaValidation:
    def test_invalid_type_filtered(self):
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph(type="invalid_type")])):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert phenomena == []

    def test_cross_paper_contradiction_filtered_in_single_paper_mode(self):
        """单论文挖掘不应抽 cross_paper_contradiction（这类需多篇交叉）"""
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph(type="cross_paper_contradiction")])):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert phenomena == []

    def test_empty_description_dropped(self):
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph(description="")])):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert phenomena == []

    def test_empty_quote_dropped(self):
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph(quote="")])):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert phenomena == []

    def test_non_dict_entries_skipped(self):
        bad_resp = MagicMock(content=_json.dumps({
            "phenomena": ["not a dict", 42, None, _ph()]
        }))
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=bad_resp):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert len(phenomena) == 1   # 只保留合法的 dict


class TestMinePhenomenaExceptions:
    def test_llm_failure_returns_empty(self):
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   side_effect=ConnectionError("net")):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert phenomena == []

    def test_unparseable_json_returns_empty(self):
        bad = MagicMock(content="not json at all")
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=bad):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert phenomena == []

    def test_missing_phenomena_field_returns_empty(self):
        resp = MagicMock(content=_json.dumps({"foo": "bar"}))
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=resp):
            phenomena = mine_phenomena("p", "t", "ft", "ab", MagicMock())
        assert phenomena == []

    def test_empty_text_returns_empty_without_llm_call(self):
        """无 full_text 也无 abstract → 不调 LLM"""
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry") as mock_llm:
            phenomena = mine_phenomena("p", "t", "", "", MagicMock())
        assert phenomena == []
        assert mock_llm.call_count == 0


# ---------------------------------------------------------------------------
# batch_mine_phenomena
# ---------------------------------------------------------------------------

class TestBatchMine:
    def test_calls_per_paper(self):
        papers = [
            {"paper_id": "p1", "title": "T1", "abstract": "A1"},
            {"paper_id": "p2", "title": "T2", "abstract": "A2"},
        ]
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph()])) as mock_llm:
            phenomena = batch_mine_phenomena(papers, MagicMock())
        assert mock_llm.call_count == 2
        # 每篇 1 条 phenomenon → 总 2 条
        assert len(phenomena) == 2
        # paper_ids 正确归属
        assert phenomena[0].paper_ids == ["p1"]
        assert phenomena[1].paper_ids == ["p2"]

    def test_full_text_provider_called(self):
        papers = [{"paper_id": "p1", "title": "T", "abstract": "A"}]
        provider = MagicMock(return_value="full text content")
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph()])):
            batch_mine_phenomena(papers, MagicMock(), full_text_provider=provider)
        provider.assert_called_once_with("p1")

    def test_full_text_provider_exception_swallowed(self):
        """provider 抛异常时 → 走 abstract-only 模式，仍调 LLM"""
        papers = [{"paper_id": "p1", "title": "T", "abstract": "A"}]
        provider = MagicMock(side_effect=Exception("net"))
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   return_value=_good_response([_ph()])) as mock_llm:
            phenomena = batch_mine_phenomena(papers, MagicMock(),
                                              full_text_provider=provider)
        assert mock_llm.call_count == 1
        assert len(phenomena) == 1

    def test_partial_failure_continues(self):
        """3 篇 paper，第 2 个 LLM 挂 → 第 1 和 3 仍正常"""
        papers = [{"paper_id": f"p{i}", "title": "T", "abstract": "A"} for i in range(3)]
        responses = [
            _good_response([_ph()]),
            ConnectionError("net"),
            _good_response([_ph()]),
        ]
        idx = {"i": 0}
        def fake(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            if isinstance(r, Exception):
                raise r
            return r
        with patch("darwinian.agents.phenomenon_miner.invoke_with_retry",
                   side_effect=fake):
            phenomena = batch_mine_phenomena(papers, MagicMock())
        assert len(phenomena) == 2


# ---------------------------------------------------------------------------
# prompt 构造
# ---------------------------------------------------------------------------

class TestUserMessage:
    def test_includes_paper_id_title(self):
        msg = _build_user_message("arxiv:1234", "MyPaper", "abstract", "full text")
        assert "arxiv:1234" in msg
        assert "MyPaper" in msg
        assert "abstract" in msg
        assert "full text" in msg

    def test_full_text_truncated(self):
        long_full = "x" * 50000
        msg = _build_user_message("p", "t", "a", long_full)
        # 截到 20K（留 100 char 余量给 prompt boilerplate 偶含 x 如 "experiments"）
        assert msg.count("x") <= 20100
        # 但绝不会超过 25K
        assert len(msg) < 25000

    def test_no_full_text_falls_back(self):
        msg = _build_user_message("p", "t", "abstract only", "")
        assert "只有 abstract" in msg
