"""
proposal_elaborator 单元测试

不打真实 LLM：mock invoke_with_retry，验证：
- 完整 ResearchProposal 拼装正确
- 6 项硬约束校验各自能抓出错
- 校验失败时带反馈重试
- 最终通过校验时 fits_resource_budget=True
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.proposal_elaborator import (
    elaborate_proposal,
    proposal_elaborator_node,
    _build_proposal,
    _validate,
)
from darwinian.state import (
    AbstractionBranch,
    ConceptGraph,
    Hypothesis,
    LimitationRef,
    MethodologyPhase,
    PaperInfo,
    ResearchProposal,
    ResearchState,
)


# ---------------------------------------------------------------------------
# 工具：构造 fixtures
# ---------------------------------------------------------------------------

def _skeleton():
    return AbstractionBranch(
        name="QuantSkip-test",
        description="test",
        algorithm_logic="x",
        math_formulation="y",
        cited_entity_names=["adam", "resnet"],
        solved_limitation_id="L001",
    )


def _graph_with_papers(n_papers=10):
    papers = [
        PaperInfo(paper_id=f"arxiv:240{i}.1234", title=f"Paper {i}",
                  year=2024, citation_count=100 - i)
        for i in range(n_papers)
    ]
    return ConceptGraph(
        papers=papers,
        limitations=[LimitationRef(id="L001", text="slow convergence", source_paper_id="arxiv:2401.1234")],
    )


def _good_llm_response(graph):
    """构造一个能通过所有校验的 LLM 输出"""
    paper_ids = [p.paper_id for p in graph.papers[:6]]
    return {
        "title": "QuantSkip: Do Layers Tolerate ?",
        "elevator_pitch": "An empirical study of layer tolerance",
        "challenges": "Mixed-precision quantization sensitivity",
        "existing_methods": "Layer-skipping methods exist (DEL, CLaSp)",
        "motivation": "DEL achieves 2.16-2.62x speedup on Llama-3.1-8B; QSpec gets 1.64-1.80x; SpecAttn 2.81x. These suggest...",
        "proposed_method": "Profile per-layer sensitivity",
        "technical_details": "Use Spearman rho ...",
        "expected_outcomes": "如果发现 divergence 则 X；如果发现 convergence 则 Y。两种结果都 publishable.",
        "methodology_phases": [
            {"phase_number": 1, "name": "Profiling", "description": "...", "expected_compute_hours": 24},
            {"phase_number": 2, "name": "Optimization", "description": "...", "expected_compute_hours": 36},
            {"phase_number": 3, "name": "Cross-validation", "description": "...", "expected_compute_hours": 24},
        ],
        "target_venue": "EMNLP 2026",
        "target_deadline": "2026-05-25",
        "fallback_venue": "AAAI 2027",
        "key_references": paper_ids,
    }


# ---------------------------------------------------------------------------
# _build_proposal 拼装
# ---------------------------------------------------------------------------

class TestBuildProposal:
    def test_assembles_all_fields(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        proposal = _build_proposal(raw, _skeleton(), graph)

        assert proposal.title == "QuantSkip: Do Layers Tolerate ?"
        assert proposal.target_venue == "EMNLP 2026"
        assert len(proposal.methodology_phases) == 3
        assert proposal.total_estimated_hours == 24 + 36 + 24
        assert proposal.skeleton.name == "QuantSkip-test"

    def test_handles_missing_phases(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        del raw["methodology_phases"]
        proposal = _build_proposal(raw, _skeleton(), graph)
        assert proposal.methodology_phases == []
        assert proposal.total_estimated_hours == 0


# ---------------------------------------------------------------------------
# _validate 各种 error code
# ---------------------------------------------------------------------------

class TestValidateAllErrors:
    def test_good_proposal_no_errors(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        proposal = _build_proposal(raw, _skeleton(), graph)
        errors = _validate(proposal, graph, gpu_hours_budget=168)
        assert errors == []
        assert proposal.fits_resource_budget is True

    def test_missing_paper_id(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        raw["key_references"] = ["arxiv:fake.0001", "arxiv:2401.1234"]
        proposal = _build_proposal(raw, _skeleton(), graph)
        errors = _validate(proposal, graph, gpu_hours_budget=168)
        codes = [e[0] for e in errors]
        assert "MISSING_PAPER_ID" in codes

    def test_too_few_refs(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        raw["key_references"] = [graph.papers[0].paper_id]   # 仅 1 个
        proposal = _build_proposal(raw, _skeleton(), graph)
        errors = _validate(proposal, graph, gpu_hours_budget=168)
        assert "TOO_FEW_REFS" in [e[0] for e in errors]

    def test_too_few_phases(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        raw["methodology_phases"] = raw["methodology_phases"][:1]   # 仅 1 个
        proposal = _build_proposal(raw, _skeleton(), graph)
        errors = _validate(proposal, graph, gpu_hours_budget=168)
        assert "TOO_FEW_PHASES" in [e[0] for e in errors]

    def test_over_budget(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        # 把 phase 1 改成 1000 小时
        raw["methodology_phases"][0]["expected_compute_hours"] = 1000
        proposal = _build_proposal(raw, _skeleton(), graph)
        errors = _validate(proposal, graph, gpu_hours_budget=168)
        assert "OVER_BUDGET" in [e[0] for e in errors]
        assert proposal.fits_resource_budget is False

    def test_missing_quant_in_motivation(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        raw["motivation"] = "Existing methods perform poorly. We can do better."
        proposal = _build_proposal(raw, _skeleton(), graph)
        errors = _validate(proposal, graph, gpu_hours_budget=168)
        assert "MISSING_QUANT_IN_MOTIVATION" in [e[0] for e in errors]

    def test_no_dual_outcome_framing(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        raw["expected_outcomes"] = "We expect to achieve better results."
        proposal = _build_proposal(raw, _skeleton(), graph)
        errors = _validate(proposal, graph, gpu_hours_budget=168)
        assert "NO_DUAL_OUTCOME_FRAMING" in [e[0] for e in errors]

    def test_dual_framing_via_english_publishable(self):
        """expected_outcomes 用 'publishable' 关键词也算合格"""
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        raw["expected_outcomes"] = "Whether positive or negative, results are publishable."
        proposal = _build_proposal(raw, _skeleton(), graph)
        errors = _validate(proposal, graph, gpu_hours_budget=168)
        codes = [e[0] for e in errors]
        assert "NO_DUAL_OUTCOME_FRAMING" not in codes


# ---------------------------------------------------------------------------
# elaborate_proposal 端到端（底层工具函数）
# ---------------------------------------------------------------------------

class TestElaboratorNode:
    def test_happy_path_first_attempt(self):
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        fake_resp = MagicMock(content=_json.dumps(raw))
        with patch("darwinian.agents.proposal_elaborator.invoke_with_retry",
                   return_value=fake_resp):
            proposal = elaborate_proposal(_skeleton(), graph, MagicMock())
        assert proposal is not None
        assert proposal.fits_resource_budget is True
        assert len(proposal.key_references) >= 5

    def test_validation_fail_then_retry_success(self):
        """第一次缺 quant，反馈后第二次修正"""
        graph = _graph_with_papers()
        bad = _good_llm_response(graph)
        bad["motivation"] = "Existing fails."   # 没数字
        good = _good_llm_response(graph)

        responses = [
            MagicMock(content=_json.dumps(bad)),
            MagicMock(content=_json.dumps(good)),
        ]
        call_idx = {"i": 0}

        def fake_invoke(llm, messages, **kwargs):
            r = responses[call_idx["i"]]
            call_idx["i"] += 1
            return r

        with patch("darwinian.agents.proposal_elaborator.invoke_with_retry",
                   side_effect=fake_invoke):
            with patch("time.sleep"):
                proposal = elaborate_proposal(_skeleton(), graph, MagicMock())
        assert proposal is not None
        assert proposal.fits_resource_budget is True
        # 两次调用证明经历了反馈重试
        assert call_idx["i"] == 2

    def test_3_failures_returns_last_attempt(self):
        """连续 3 次失败仍返回最后一次（即使有错），调用方决定怎么用"""
        graph = _graph_with_papers()
        bad = _good_llm_response(graph)
        bad["key_references"] = ["arxiv:fake.0001"]   # 永远错

        fake_resp = MagicMock(content=_json.dumps(bad))
        with patch("darwinian.agents.proposal_elaborator.invoke_with_retry",
                   return_value=fake_resp):
            with patch("time.sleep"):
                proposal = elaborate_proposal(_skeleton(), graph, MagicMock())
        # 仍然返回（让调用方判断），不是 None
        assert proposal is not None
        # 故意保留错误的 reference 证明这是失败的产出（不是 None 也不是改对了）
        assert "arxiv:fake.0001" in proposal.key_references
        # 调用方再调 _validate 应该仍能抓出 MISSING_PAPER_ID
        from darwinian.agents.proposal_elaborator import _validate
        residual = _validate(proposal, graph, gpu_hours_budget=168)
        assert any(code == "MISSING_PAPER_ID" for code, _ in residual)

    def test_unparseable_json_skipped(self):
        """LLM 输出无法解析时，三次重试都是垃圾，最终返回 None"""
        fake_resp = MagicMock(content="this is not json")
        graph = _graph_with_papers()
        with patch("darwinian.agents.proposal_elaborator.invoke_with_retry",
                   return_value=fake_resp):
            with patch("time.sleep"):
                proposal = elaborate_proposal(_skeleton(), graph, MagicMock())
        assert proposal is None


# ---------------------------------------------------------------------------
# proposal_elaborator_node (LangGraph wrapper)
# ---------------------------------------------------------------------------

class TestProposalElaboratorGraphNode:
    """验证 graph wrapper 从 state 正确取材料、对每个 branch 调用底层函数、
    把结果写回 state.research_proposals"""

    def _state_with_n_branches(self, n: int):
        graph = _graph_with_papers()
        branches = [
            AbstractionBranch(
                name=f"B{i}",
                description=f"branch {i}",
                algorithm_logic="x",
                math_formulation="y",
                cited_entity_names=["adam", "resnet"],
                solved_limitation_id="L001",
            )
            for i in range(n)
        ]
        h = Hypothesis(core_problem="cp", abstraction_tree=branches)
        return ResearchState(research_direction="x", current_hypothesis=h, concept_graph=graph)

    def test_calls_elaborate_for_each_branch(self):
        """每个 branch 都调一次 elaborate_proposal"""
        state = self._state_with_n_branches(3)
        good_proposal = MagicMock(spec=ResearchProposal)
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal",
                   return_value=good_proposal) as mock_elab:
            out = proposal_elaborator_node(state, MagicMock())
        assert mock_elab.call_count == 3
        assert len(out["research_proposals"]) == 3

    def test_skips_failed_elaborations(self):
        """部分 branch 返 None 不阻塞，已成功的仍写入"""
        state = self._state_with_n_branches(3)
        results = [MagicMock(spec=ResearchProposal), None, MagicMock(spec=ResearchProposal)]
        call_idx = {"i": 0}

        def fake_elab(*args, **kwargs):
            r = results[call_idx["i"]]
            call_idx["i"] += 1
            return r

        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal",
                   side_effect=fake_elab):
            out = proposal_elaborator_node(state, MagicMock())
        # 只有 2 个成功的进入结果
        assert len(out["research_proposals"]) == 2

    def test_no_hypothesis_returns_empty(self):
        """current_hypothesis 为空时安全返空 list"""
        state = ResearchState(research_direction="x")
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal") as mock_elab:
            out = proposal_elaborator_node(state, MagicMock())
        assert mock_elab.call_count == 0
        assert out["research_proposals"] == []

    def test_no_concept_graph_returns_empty(self):
        """concept_graph 为空时安全返空 list（不调底层函数）"""
        h = Hypothesis(
            core_problem="cp",
            abstraction_tree=[AbstractionBranch(
                name="B", description="x", algorithm_logic="x", math_formulation="x",
            )],
        )
        state = ResearchState(research_direction="x", current_hypothesis=h, concept_graph=None)
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal") as mock_elab:
            out = proposal_elaborator_node(state, MagicMock())
        assert mock_elab.call_count == 0
        assert out["research_proposals"] == []

    def test_passes_through_kwargs(self):
        """gpu_hours_budget 和 target_venues 透传给底层函数"""
        state = self._state_with_n_branches(1)
        venues = [{"name": "NeurIPS 2026", "deadline": "2026-05-13"}]
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal",
                   return_value=None) as mock_elab:
            proposal_elaborator_node(state, MagicMock(),
                                      gpu_hours_budget=336, target_venues=venues)
        kw = mock_elab.call_args.kwargs
        assert kw["gpu_hours_budget"] == 336
        assert kw["target_venues"] == venues

    def test_state_can_accept_research_proposals_field(self):
        """ResearchState.research_proposals 字段已加且接受 list[ResearchProposal]"""
        state = ResearchState(research_direction="x")
        # 默认空 list
        assert state.research_proposals == []
        # 能写入
        graph = _graph_with_papers()
        raw = _good_llm_response(graph)
        proposal = _build_proposal(raw, _skeleton(), graph)
        state2 = state.model_copy(update={"research_proposals": [proposal]})
        assert len(state2.research_proposals) == 1
        assert state2.research_proposals[0].title == "QuantSkip: Do Layers Tolerate ?"
