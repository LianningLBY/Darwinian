"""
novelty_booster 单元测试

不打真实 LLM / S2 —— mock 全部，验证：
- 单轮：query 抽取 / S2 召回 / overlap 评估 / refinement
- overlap_level 校验（none/partial/substantial/identical）
- 收敛逻辑（partial 即停 vs identical 必触发 refine）
- max_rounds 截断
- LLM/S2 各种异常吞掉不阻塞
- 最终 proposal.novelty_assessment 被填回
- novelty_score clamp [0,1]
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.novelty_booster import (
    boost_novelty,
    _extract_queries,
    _search_prior_work,
    _assess_overlap,
    _refine_for_novelty,
)
from darwinian.state import (
    AbstractionBranch,
    NoveltyAssessment,
    ResearchProposal,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _proposal(title="QuantSpec: Quantized Draft for Speculative Decoding",
              motivation="orig motivation", method="orig method") -> ResearchProposal:
    return ResearchProposal(
        skeleton=AbstractionBranch(
            name="x", description="x", algorithm_logic="x", math_formulation="x",
        ),
        title=title,
        elevator_pitch="pitch",
        motivation=motivation,
        proposed_method=method,
        technical_details="details",
    )


def _query_response(queries):
    return MagicMock(content=_json.dumps({"queries": queries}))


def _assess_response(level="substantial", score=0.4, **kwargs):
    payload = {
        "overlap_level": level,
        "closest_paper_id": kwargs.get("paper_id", "S2_X"),
        "closest_title": kwargs.get("title", "Some Prior Work"),
        "overlap_summary": kwargs.get("summary", "shares method X"),
        "differentiation_gap": kwargs.get("gap", "ours needs to focus on Y"),
        "novelty_score": score,
    }
    return MagicMock(content=_json.dumps(payload))


def _refine_response(motivation="REFINED motivation", method="REFINED method"):
    return MagicMock(content=_json.dumps({
        "motivation": motivation,
        "proposed_method": method,
    }))


# ---------------------------------------------------------------------------
# _extract_queries
# ---------------------------------------------------------------------------

class TestExtractQueries:
    def test_happy_path(self):
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_query_response(["query A", "query B"])):
            qs = _extract_queries(_proposal(), MagicMock())
        assert qs == ["query A", "query B"]

    def test_caps_at_2(self):
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_query_response(["q1", "q2", "q3", "q4"])):
            qs = _extract_queries(_proposal(), MagicMock())
        assert len(qs) == 2

    def test_strip_and_filter(self):
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_query_response(["  ok  ", "", None, "ok2"])):
            qs = _extract_queries(_proposal(), MagicMock())
        assert qs == ["ok", "ok2"]

    def test_llm_failure_falls_back_to_title(self):
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   side_effect=ConnectionError("net")):
            qs = _extract_queries(_proposal(title="MyTitle"), MagicMock())
        assert qs == ["MyTitle"]

    def test_no_title_no_queries(self):
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   side_effect=ConnectionError("net")):
            qs = _extract_queries(_proposal(title=""), MagicMock())
        assert qs == []


# ---------------------------------------------------------------------------
# _search_prior_work
# ---------------------------------------------------------------------------

class TestSearchPriorWork:
    def test_combines_results_and_dedupes(self):
        def fake_search(q, limit):
            if q == "q1":
                return [{"paperId": "P1"}, {"paperId": "P2"}]
            return [{"paperId": "P2"}, {"paperId": "P3"}]   # P2 dup
        with patch("darwinian.agents.novelty_booster.search_papers",
                   side_effect=fake_search):
            pool = _search_prior_work(["q1", "q2"], 5)
        ids = {p["paperId"] for p in pool}
        assert ids == {"P1", "P2", "P3"}

    def test_caps_at_8(self):
        many = [{"paperId": f"P{i}"} for i in range(20)]
        with patch("darwinian.agents.novelty_booster.search_papers",
                   return_value=many):
            pool = _search_prior_work(["q"], 20)
        assert len(pool) == 8

    def test_s2_failure_swallowed(self):
        with patch("darwinian.agents.novelty_booster.search_papers",
                   side_effect=ConnectionError("net")):
            pool = _search_prior_work(["q"], 5)
        assert pool == []

    def test_empty_query_skipped(self):
        with patch("darwinian.agents.novelty_booster.search_papers") as mock:
            _search_prior_work(["", None], 5)
        assert mock.call_count == 0


# ---------------------------------------------------------------------------
# _assess_overlap
# ---------------------------------------------------------------------------

class TestAssessOverlap:
    def test_happy_path(self):
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_assess_response(level="substantial", score=0.4)):
            na = _assess_overlap(_proposal(), [{"paperId": "P1"}],
                                  "x", MagicMock())
        assert na.overlap_level == "substantial"
        assert na.novelty_score == 0.4
        assert na.closest_work_paper_id == "S2_X"

    def test_invalid_overlap_level_returns_none(self):
        bad = MagicMock(content=_json.dumps({
            "overlap_level": "fuzzy", "novelty_score": 0.5,
        }))
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=bad):
            na = _assess_overlap(_proposal(), [], "x", MagicMock())
        assert na is None

    def test_score_clamp_above_1(self):
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_assess_response(score=2.5)):
            na = _assess_overlap(_proposal(), [], "x", MagicMock())
        assert na.novelty_score == 1.0

    def test_score_clamp_below_0(self):
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_assess_response(score=-0.5)):
            na = _assess_overlap(_proposal(), [], "x", MagicMock())
        assert na.novelty_score == 0.0

    def test_unparseable_returns_none(self):
        bad = MagicMock(content="not json")
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=bad):
            na = _assess_overlap(_proposal(), [], "x", MagicMock())
        assert na is None

    def test_uppercase_overlap_level_normalized(self):
        bad = MagicMock(content=_json.dumps({
            "overlap_level": "PARTIAL", "novelty_score": 0.7,
        }))
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=bad):
            na = _assess_overlap(_proposal(), [], "x", MagicMock())
        assert na.overlap_level == "partial"


# ---------------------------------------------------------------------------
# _refine_for_novelty
# ---------------------------------------------------------------------------

class TestRefineForNovelty:
    def test_returns_new_proposal(self):
        orig = _proposal(motivation="OLD M", method="OLD METHOD")
        na = NoveltyAssessment(overlap_level="substantial",
                                differentiation_gap="add Z")
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_refine_response("NEW M", "NEW METHOD")):
            new = _refine_for_novelty(orig, na, MagicMock())
        assert new.motivation == "NEW M"
        assert new.proposed_method == "NEW METHOD"
        # 其他字段保留
        assert new.title == orig.title

    def test_empty_motivation_returns_none(self):
        orig = _proposal()
        na = NoveltyAssessment(overlap_level="substantial")
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_refine_response(motivation="")):
            assert _refine_for_novelty(orig, na, MagicMock()) is None

    def test_llm_failure_returns_none(self):
        orig = _proposal()
        na = NoveltyAssessment(overlap_level="substantial")
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   side_effect=ConnectionError("net")):
            assert _refine_for_novelty(orig, na, MagicMock()) is None


# ---------------------------------------------------------------------------
# boost_novelty 端到端
# ---------------------------------------------------------------------------

class TestBoostNoveltyEndToEnd:
    def test_converged_first_round_when_partial(self):
        """第 1 轮就 partial → 立刻收敛，不 refine"""
        orig = _proposal()
        # mock: extract_queries 返 1 条 / search 返 1 篇 / assess 返 partial
        responses = [
            _query_response(["q"]),
            _assess_response(level="partial", score=0.7),
        ]
        idx = {"i": 0}
        def fake_invoke(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   side_effect=fake_invoke), \
             patch("darwinian.agents.novelty_booster.search_papers",
                   return_value=[{"paperId": "P1", "title": "X", "abstract": "y"}]):
            new, result = boost_novelty(orig, "dir", MagicMock(), max_rounds=3)
        assert result.converged is True
        assert result.rounds_taken == 1
        # 没 refine → motivation 保持原样
        assert new.motivation == orig.motivation
        # assessment 写回 proposal
        assert new.novelty_assessment is not None
        assert new.novelty_assessment.overlap_level == "partial"

    def test_refines_then_converges(self):
        """第 1 轮 substantial → refine → 第 2 轮 partial 收敛"""
        orig = _proposal(motivation="ORIG")
        responses = [
            _query_response(["q1"]),
            _assess_response(level="substantial", score=0.4),
            _refine_response("REFINED", "NEW METHOD"),
            _query_response(["q2"]),
            _assess_response(level="partial", score=0.7),
        ]
        idx = {"i": 0}
        def fake_invoke(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   side_effect=fake_invoke), \
             patch("darwinian.agents.novelty_booster.search_papers",
                   return_value=[{"paperId": "P1"}]):
            new, result = boost_novelty(orig, "dir", MagicMock(), max_rounds=3)
        assert result.converged is True
        assert result.rounds_taken == 2
        assert len(result.revisions_log) == 1
        # motivation 已被 refine 改写
        assert new.motivation == "REFINED"

    def test_max_rounds_reached_unconverged(self):
        """3 轮都 substantial → max_rounds 截断，converged=False"""
        orig = _proposal()
        # 每轮 = query + assess + refine 共 3 次 invoke_with_retry
        responses = []
        for r in range(3):
            responses.extend([
                _query_response([f"q{r}"]),
                _assess_response(level="substantial", score=0.4),
                _refine_response(f"M{r}", f"P{r}"),
            ])
        idx = {"i": 0}
        def fake_invoke(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   side_effect=fake_invoke), \
             patch("darwinian.agents.novelty_booster.search_papers",
                   return_value=[{"paperId": "P1"}]):
            new, result = boost_novelty(orig, "dir", MagicMock(), max_rounds=3)
        assert result.converged is False
        assert result.rounds_taken == 3
        # final assessment 仍是最后一轮的
        assert result.final_assessment.overlap_level == "substantial"

    def test_no_prior_work_recalled_treated_as_novel(self):
        """S2 召回 0 篇 → 视为 none (novel)，立刻收敛"""
        orig = _proposal()
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   return_value=_query_response(["q"])), \
             patch("darwinian.agents.novelty_booster.search_papers",
                   return_value=[]):
            new, result = boost_novelty(orig, "dir", MagicMock(), max_rounds=3)
        assert result.converged is True
        assert result.final_assessment.overlap_level == "none"
        assert result.rounds_taken == 1

    def test_assessment_failure_treated_as_partial_and_stops(self):
        """LLM assess 失败 → 保守标 partial，停止迭代（不阻塞 pipeline）"""
        orig = _proposal()
        bad_assess = MagicMock(content="not json")
        responses = [_query_response(["q"]), bad_assess]
        idx = {"i": 0}
        def fake_invoke(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   side_effect=fake_invoke), \
             patch("darwinian.agents.novelty_booster.search_papers",
                   return_value=[{"paperId": "P1"}]):
            new, result = boost_novelty(orig, "dir", MagicMock(), max_rounds=3)
        # 保守 partial → converged=True
        assert result.converged is True
        assert result.final_assessment.overlap_level == "partial"

    def test_custom_convergence_levels(self):
        """convergence_levels 自定义为只接受 'none' → partial 也得继续 refine"""
        orig = _proposal()
        responses = [
            _query_response(["q"]),
            _assess_response(level="partial", score=0.7),
            _refine_response("M", "P"),
            _query_response(["q"]),
            _assess_response(level="none", score=0.95),
        ]
        idx = {"i": 0}
        def fake_invoke(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r
        with patch("darwinian.agents.novelty_booster.invoke_with_retry",
                   side_effect=fake_invoke), \
             patch("darwinian.agents.novelty_booster.search_papers",
                   return_value=[{"paperId": "P1"}]):
            new, result = boost_novelty(orig, "dir", MagicMock(),
                                         max_rounds=3,
                                         convergence_levels={"none"})
        assert result.converged is True
        assert result.rounds_taken == 2
        assert result.final_assessment.overlap_level == "none"
