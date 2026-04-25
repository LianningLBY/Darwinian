"""
theoretical_critic_node 单元测试

重点覆盖 v2 改造后的能力：
- pre-check：所有 branch existing_combination=True → 直接 NOT_NOVEL（省 LLM）
- _render_branches_for_critic：把 ConceptGraph + existing_combination 信号塞给 LLM
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.theoretical_critic import (
    _render_branches_for_critic,
    theoretical_critic_node,
)
from darwinian.state import (
    AbstractionBranch,
    ConceptGraph,
    CriticVerdict,
    Entity,
    Hypothesis,
    LimitationRef,
    PaperInfo,
    ResearchState,
)


def _branch(name="B", cited=None, lim_id="", exists=False, refs=None):
    return AbstractionBranch(
        name=name,
        description="x",
        algorithm_logic="x",
        math_formulation="x",
        cited_entity_names=cited or [],
        solved_limitation_id=lim_id,
        existing_combination=exists,
        existing_combination_refs=refs or [],
    )


def _state(branches, graph=None):
    h = Hypothesis(core_problem="real research problem", abstraction_tree=branches)
    return ResearchState(
        research_direction="x", current_hypothesis=h, concept_graph=graph,
    )


# ---------------------------------------------------------------------------
# pre-check：全部 existing_combination=True 直接 NOT_NOVEL
# ---------------------------------------------------------------------------

class TestExistingCombinationPrecheck:
    def test_all_existing_skips_llm(self):
        branches = [
            _branch("B1", exists=True, refs=["paperA", "paperB"]),
            _branch("B2", exists=True, refs=["paperC"]),
        ]
        state = _state(branches)
        with patch("darwinian.agents.theoretical_critic.invoke_with_retry") as mock_llm:
            out = theoretical_critic_node(state, MagicMock())
        # 没调 LLM
        assert mock_llm.call_count == 0
        assert out["critic_verdict"] == CriticVerdict.NOT_NOVEL
        # feedback 里包含至少一个命中的 paperId
        assert "paperA" in out["critic_feedback"] or "paperB" in out["critic_feedback"] \
            or "paperC" in out["critic_feedback"]

    def test_partial_existing_still_uses_llm(self):
        """一个 branch existing=True 一个 False → 不进 pre-check，正常走 LLM"""
        branches = [
            _branch("B1", exists=True, refs=["paperA"]),
            _branch("B2", exists=False),
        ]
        state = _state(branches)
        fake_resp = MagicMock(content=(
            '{"verdict":"NOT_NOVEL","selected_branch_name":null,'
            '"feedback":"x","novelty_concern":"","error_keywords":[]}'
        ))
        with patch("darwinian.agents.theoretical_critic.invoke_with_retry",
                   return_value=fake_resp) as mock_llm:
            out = theoretical_critic_node(state, MagicMock())
        # 有调 LLM
        assert mock_llm.call_count == 1
        assert out["critic_verdict"] == CriticVerdict.NOT_NOVEL

    def test_none_existing_normal_flow(self):
        """都 existing=False → 走 LLM 正常流"""
        branches = [
            _branch("B1", cited=["adam"], exists=False),
            _branch("B2", cited=["resnet"], exists=False),
        ]
        state = _state(branches)
        fake_resp = MagicMock(content=(
            '{"verdict":"PASS","selected_branch_name":"B1",'
            '"feedback":"good","novelty_concern":"","error_keywords":[]}'
        ))
        with patch("darwinian.agents.theoretical_critic.invoke_with_retry",
                   return_value=fake_resp) as mock_llm:
            out = theoretical_critic_node(state, MagicMock())
        assert mock_llm.call_count == 1
        assert out["critic_verdict"] == CriticVerdict.PASS


# ---------------------------------------------------------------------------
# _render_branches_for_critic：把 ConceptGraph 信号塞给 LLM
# ---------------------------------------------------------------------------

class TestRenderBranchesForCritic:
    def test_includes_existing_combination_marker(self):
        b1 = _branch("B1", cited=["a"], exists=False)
        b2 = _branch("B2", cited=["b", "c"], exists=True, refs=["paperX"])
        out = _render_branches_for_critic([b1, b2], graph=None)
        assert "B1" in out and "B2" in out
        assert "S2 未命中" in out  # b1 标记
        assert "已存在" in out      # b2 标记
        assert "paperX" in out      # b2 命中 ref

    def test_resolves_solved_limitation_text(self):
        graph = ConceptGraph(
            limitations=[LimitationRef(id="abc12345", text="慢得离谱", source_paper_id="p7")],
        )
        b = _branch("B1", lim_id="abc12345")
        out = _render_branches_for_critic([b], graph=graph)
        assert "慢得离谱" in out
        assert "p7" in out

    def test_unknown_limitation_id_marked(self):
        graph = ConceptGraph()
        b = _branch("B1", lim_id="bogus")
        out = _render_branches_for_critic([b], graph=graph)
        assert "未指定" in out

    def test_empty_branches_returns_empty(self):
        out = _render_branches_for_critic([], graph=None)
        assert out == ""


# ---------------------------------------------------------------------------
# 现有保护路径不被打破
# ---------------------------------------------------------------------------

class TestExistingProtections:
    def test_empty_tree_returns_math_error(self):
        state = ResearchState(
            research_direction="x",
            current_hypothesis=Hypothesis(core_problem="real problem", abstraction_tree=[]),
        )
        out = theoretical_critic_node(state, MagicMock())
        assert out["critic_verdict"] == CriticVerdict.MATH_ERROR

    def test_meta_complaint_returns_math_error(self):
        h = Hypothesis(
            core_problem="文献检索不可用，无法基于这些数据分析",
            abstraction_tree=[_branch("B1")],
        )
        state = ResearchState(research_direction="x", current_hypothesis=h)
        out = theoretical_critic_node(state, MagicMock())
        assert out["critic_verdict"] == CriticVerdict.MATH_ERROR

    def test_raises_without_hypothesis(self):
        state = ResearchState(research_direction="x")
        with pytest.raises(ValueError):
            theoretical_critic_node(state, MagicMock())
