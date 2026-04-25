"""
paper_evidence_extractor 单元测试

不打真实 LLM：mock invoke_with_retry 验证：
- PaperEvidence 拼装
- 6 项校验各自能抓出错
- 校验失败带反馈重试
- batch 接口透传 full_text_provider
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.tools.paper_evidence_extractor import (
    extract_evidence,
    batch_extract_evidence,
    _build_evidence,
    _validate,
    _has_number,
    _VALID_RELATIONS,
)
from darwinian.state import PaperEvidence, QuantitativeClaim


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _good_response(**overrides):
    """构造一个能通过所有校验的 LLM 输出 dict"""
    base = {
        "short_name": "LayerSkip",
        "venue": "ACL 2024",
        "year": 2024,
        "category": "Layer-skipping self-speculative methods",
        "method_names": ["layer dropout", "early exit"],
        "datasets": ["MATH", "GSM8K"],
        "metrics": ["speedup", "accuracy"],
        "quantitative_claims": [
            {"metric_name": "speedup", "metric_value": "2.16x", "setting": "Llama-3.1-8B"},
            {"metric_name": "accuracy", "metric_value": "85.3%", "setting": "MATH"},
        ],
        "headline_result": "2.16-2.62x speedup on Llama-3.1-8B",
        "limitations": ["requires re-training"],
        "relation_to_direction": "baseline",
    }
    base.update(overrides)
    return base


def _mock_llm(response_dict):
    """构造一个 mock llm，invoke 返回带 content=JSON dump 的 response"""
    fake_resp = MagicMock(content=_json.dumps(response_dict))
    return fake_resp


# ---------------------------------------------------------------------------
# _has_number 启发函数
# ---------------------------------------------------------------------------

class TestHasNumber:
    def test_simple_int(self):
        assert _has_number("2x speedup") is True

    def test_decimal(self):
        assert _has_number("85.3% accuracy") is True

    def test_range(self):
        assert _has_number("2.16-2.62x") is True

    def test_no_number(self):
        assert _has_number("improves performance") is False

    def test_empty(self):
        assert _has_number("") is False
        assert _has_number(None) is False


# ---------------------------------------------------------------------------
# _build_evidence 拼装
# ---------------------------------------------------------------------------

class TestBuildEvidence:
    def test_complete_response(self):
        evidence = _build_evidence(
            _good_response(),
            paper_id="arxiv:2404.16710",
            title="LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding",
            full_text="some full text",
        )
        assert evidence.paper_id == "arxiv:2404.16710"
        assert evidence.short_name == "LayerSkip"
        assert evidence.venue == "ACL 2024"
        assert evidence.year == 2024
        assert evidence.category.startswith("Layer-skipping")
        assert len(evidence.quantitative_claims) == 2
        assert evidence.quantitative_claims[0].metric_value == "2.16x"
        assert evidence.full_text_used is True

    def test_missing_full_text_marks_false(self):
        evidence = _build_evidence(_good_response(), "arxiv:x", "T", full_text="")
        assert evidence.full_text_used is False

    def test_missing_optional_fields(self):
        """LLM 漏字段时安全返默认值"""
        minimal = {"short_name": "X", "headline_result": "1x"}
        evidence = _build_evidence(minimal, "arxiv:x", "T", full_text="")
        assert evidence.method_names == []
        assert evidence.datasets == []
        assert evidence.year == 0


# ---------------------------------------------------------------------------
# _validate 6 种 error code
# ---------------------------------------------------------------------------

class TestValidate:
    def test_good_evidence_no_errors(self):
        evidence = _build_evidence(_good_response(), "arxiv:x", "T", full_text="")
        assert _validate(evidence) == []

    def test_missing_short_name(self):
        bad = _good_response(short_name="")
        evidence = _build_evidence(bad, "arxiv:x", "T", full_text="")
        codes = [e[0] for e in _validate(evidence)]
        assert "MISSING_SHORT_NAME" in codes

    def test_headline_no_number(self):
        bad = _good_response(headline_result="improves performance significantly")
        evidence = _build_evidence(bad, "arxiv:x", "T", full_text="")
        codes = [e[0] for e in _validate(evidence)]
        assert "HEADLINE_NO_NUMBER" in codes

    def test_no_quant_claims(self):
        bad = _good_response(quantitative_claims=[])
        evidence = _build_evidence(bad, "arxiv:x", "T", full_text="")
        codes = [e[0] for e in _validate(evidence)]
        assert "NO_QUANT_CLAIMS" in codes

    def test_quant_value_no_number(self):
        bad = _good_response(quantitative_claims=[
            {"metric_name": "speedup", "metric_value": "fast", "setting": "x"},
        ])
        evidence = _build_evidence(bad, "arxiv:x", "T", full_text="")
        codes = [e[0] for e in _validate(evidence)]
        assert "QUANT_VALUE_NO_NUMBER" in codes

    def test_invalid_relation(self):
        bad = _good_response(relation_to_direction="some random thing")
        evidence = _build_evidence(bad, "arxiv:x", "T", full_text="")
        codes = [e[0] for e in _validate(evidence)]
        assert "INVALID_RELATION" in codes

    def test_all_5_relation_enum_values_valid(self):
        for relation in _VALID_RELATIONS:
            ev = _build_evidence(_good_response(relation_to_direction=relation),
                                  "arxiv:x", "T", full_text="")
            codes = [e[0] for e in _validate(ev)]
            assert "INVALID_RELATION" not in codes


# ---------------------------------------------------------------------------
# extract_evidence 端到端
# ---------------------------------------------------------------------------

class TestExtractEvidence:
    def test_happy_path_first_attempt(self):
        with patch("darwinian.tools.paper_evidence_extractor.invoke_with_retry",
                   return_value=_mock_llm(_good_response())):
            ev = extract_evidence(
                paper_id="arxiv:2404.16710",
                title="LayerSkip: ...",
                abstract="We present X.",
                direction="LLM inference acceleration",
                llm=MagicMock(),
            )
        assert ev is not None
        assert ev.short_name == "LayerSkip"
        assert ev.full_text_used is False  # 没传 full_text

    def test_full_text_marks_used(self):
        with patch("darwinian.tools.paper_evidence_extractor.invoke_with_retry",
                   return_value=_mock_llm(_good_response())):
            ev = extract_evidence(
                paper_id="arxiv:x",
                title="T",
                abstract="A",
                direction="d",
                llm=MagicMock(),
                full_text="## METHOD\n...",
            )
        assert ev.full_text_used is True

    def test_validation_fail_then_retry_success(self):
        """第 1 次缺数字 → 反馈 → 第 2 次修正"""
        bad = _good_response(headline_result="improves things")
        good = _good_response()
        responses = [_mock_llm(bad), _mock_llm(good)]
        idx = {"i": 0}

        def fake_invoke(llm, messages, **kwargs):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r

        with patch("darwinian.tools.paper_evidence_extractor.invoke_with_retry",
                   side_effect=fake_invoke):
            with patch("time.sleep"):
                ev = extract_evidence(
                    paper_id="arxiv:x", title="T", abstract="A",
                    direction="d", llm=MagicMock(), max_retries=2,
                )
        assert ev is not None
        assert "speedup" in ev.headline_result
        assert idx["i"] == 2

    def test_all_retries_fail_returns_last(self):
        bad = _good_response(short_name="")
        with patch("darwinian.tools.paper_evidence_extractor.invoke_with_retry",
                   return_value=_mock_llm(bad)):
            with patch("time.sleep"):
                ev = extract_evidence(
                    paper_id="arxiv:x", title="T", abstract="A",
                    direction="d", llm=MagicMock(), max_retries=2,
                )
        # 仍返回最后一次（让调用方决定怎么用）
        assert ev is not None
        assert ev.short_name == ""

    def test_unparseable_returns_none(self):
        fake_resp = MagicMock(content="not json at all")
        with patch("darwinian.tools.paper_evidence_extractor.invoke_with_retry",
                   return_value=fake_resp):
            with patch("time.sleep"):
                ev = extract_evidence(
                    paper_id="arxiv:x", title="T", abstract="A",
                    direction="d", llm=MagicMock(), max_retries=1,
                )
        assert ev is None

    def test_empty_paper_id_returns_none(self):
        ev = extract_evidence(paper_id="", title="T", abstract="A",
                              direction="d", llm=MagicMock())
        assert ev is None

    def test_empty_title_returns_none(self):
        ev = extract_evidence(paper_id="x", title="", abstract="A",
                              direction="d", llm=MagicMock())
        assert ev is None


# ---------------------------------------------------------------------------
# batch_extract_evidence
# ---------------------------------------------------------------------------

class TestBatchExtractEvidence:
    def test_iterates_papers(self):
        papers = [
            {"paper_id": "arxiv:1", "title": "T1", "abstract": "A1"},
            {"paper_id": "arxiv:2", "title": "T2", "abstract": "A2"},
        ]
        with patch("darwinian.tools.paper_evidence_extractor.invoke_with_retry",
                   return_value=_mock_llm(_good_response())):
            out = batch_extract_evidence(papers, direction="d", llm=MagicMock())
        assert len(out) == 2

    def test_full_text_provider_called_per_paper(self):
        papers = [{"paper_id": "p1", "title": "T", "abstract": "A"}]
        provider = MagicMock(return_value="## METHOD\n...")
        with patch("darwinian.tools.paper_evidence_extractor.invoke_with_retry",
                   return_value=_mock_llm(_good_response())):
            out = batch_extract_evidence(papers, direction="d", llm=MagicMock(),
                                          full_text_provider=provider)
        provider.assert_called_once_with("p1")
        assert out[0].full_text_used is True

    def test_provider_exception_falls_back_to_abstract(self):
        """provider 抛错时降级到 abstract-only，不阻塞"""
        papers = [{"paper_id": "p1", "title": "T", "abstract": "A"}]
        provider = MagicMock(side_effect=RuntimeError("network"))
        with patch("darwinian.tools.paper_evidence_extractor.invoke_with_retry",
                   return_value=_mock_llm(_good_response())):
            out = batch_extract_evidence(papers, direction="d", llm=MagicMock(),
                                          full_text_provider=provider)
        assert len(out) == 1
        assert out[0].full_text_used is False
