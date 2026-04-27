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


# ===========================================================================
# Phase 1 v3: pack-aware path 测试
# ===========================================================================

from darwinian.agents.proposal_elaborator import (
    elaborate_proposal_from_pack,
    proposal_elaborator_node_v3,
    _build_proposal_v3,
    _validate_v3,
    _build_user_message_v3,
    _render_evidence_by_category,
)
from darwinian.state import (
    ExpectedOutcomes,
    PaperEvidence,
    QuantitativeClaim,
    ResearchConstraints,
    ResearchMaterialPack,
    StructuralHoleHook,
)


# ---------------------------------------------------------------------------
# fixtures for v3
# ---------------------------------------------------------------------------

def _evidence(paper_id, short, venue, category, value, rel="extends"):
    return PaperEvidence(
        paper_id=paper_id, title=f"{short}: full title here",
        short_name=short, venue=venue, year=2024, category=category,
        method_names=[short.lower()], datasets=["MMLU"], metrics=["speedup"],
        quantitative_claims=[QuantitativeClaim(metric_name="speedup",
                                                metric_value=value, setting="Llama-3.1-8B")],
        headline_result=f"{value} speedup", limitations=["small-scale only"],
        relation_to_direction=rel, full_text_used=True,
    )


def _pack(*, forbid_rl: bool = True, n_evidence: int = 4) -> ResearchMaterialPack:
    evidences = [
        _evidence(f"arxiv:240{i}.x", f"Method{i}",
                  "ACL 2024" if i % 2 else "EMNLP 2025",
                  "Layer-skipping methods" if i < 2 else "Quantization methods",
                  f"{1.5 + i * 0.3:.2f}x")
        for i in range(n_evidence)
    ]
    constraints = ResearchConstraints(
        gpu_count=4, gpu_model="RTX PRO 6000 96GB",
        gpu_hours_budget=168.0, wall_clock_days=7,
        forbidden_techniques=["GRPO", "PPO", "DPO", "RLHF"] if forbid_rl else [],
        target_venues=["NeurIPS 2026", "EMNLP 2026"],
    )
    hook = StructuralHoleHook(
        entity_a="quant sensitivity", entity_b="draft acceptance",
        hook_text="No work measures rank correlation between A and B",
        relation_type="divergence",
    )
    return ResearchMaterialPack(
        direction="LLM inference acceleration",
        constraints=constraints,
        paper_evidence=evidences,
        structural_hole_hooks=[hook],
        timeline_signals={"hot_2025_2026": [evidences[0].paper_id]},
    )


def _good_v3_response(pack: ResearchMaterialPack) -> dict:
    paper_ids = [ev.paper_id for ev in pack.paper_evidence[:3]]
    return {
        "title": "QuantSkip: Do A Also Imply B?",
        "elevator_pitch": "200-word pitch",
        "challenges": "**The unstudied gap**: ...",
        "existing_methods": (
            "**Layer-skipping methods**: Method0 (ACL 2024, 1.50x), Method1 (EMNLP 2025, 1.80x).\n"
            "**Quantization methods**: Method2 (ACL 2024, 2.10x), Method3 (EMNLP 2025, 2.40x).\n"
            "**The gap**: No work has compared sensitivity profiles across these two families."
        ),
        "motivation": (
            "Method0 hits 1.50x, Method1 hits 1.80x speedup, Method3 reaches 2.40x. "
            "Combining them is unexplored."
        ),
        "proposed_method": "Profile per-layer dual sensitivity",
        "technical_details": "Spearman rho between vectors",
        "expected_outcomes_structured": {
            "positive_finding": "If Spearman rho < 0.5 then divergence exists, justifying draft-specific metric",
            "negative_finding": "If rho > 0.9 then convergence — accuracy-guided suffices",
            "why_both_publishable": "Both outcomes give actionable guidance",
        },
        "methodology_phases": [
            {"phase_number": 1, "name": "Profile", "description": "...",
             "inputs": ["Llama-3.1-8B"], "outputs": ["sensitivity vec"],
             "expected_compute_hours": 24},
            {"phase_number": 2, "name": "Optimize", "description": "...",
             "inputs": ["vectors"], "outputs": ["Pareto"], "expected_compute_hours": 36},
            {"phase_number": 3, "name": "Validate", "description": "...",
             "inputs": ["Qwen3-8B"], "outputs": ["transfer rate"],
             "expected_compute_hours": 48},
        ],
        "target_venue": "NeurIPS 2026",
        "target_deadline": "2026-05-13",
        "fallback_venue": "EMNLP 2026",
        "key_references": paper_ids,
    }


# ---------------------------------------------------------------------------
# v3: render helpers
# ---------------------------------------------------------------------------

class TestEvidenceRender:
    def test_groups_by_category(self):
        pack = _pack()
        rendered = _render_evidence_by_category(pack.paper_evidence)
        assert "**Layer-skipping methods**" in rendered
        assert "**Quantization methods**" in rendered

    def test_renders_quantitative_claims(self):
        pack = _pack()
        rendered = _render_evidence_by_category(pack.paper_evidence)
        # 第一篇 1.50x speedup 应出现
        assert "1.50x" in rendered or "speedup=1.50x" in rendered

    def test_user_message_includes_constraints_and_hooks(self):
        pack = _pack()
        skel = _skeleton()
        msg = _build_user_message_v3(skel, pack)
        assert "RTX PRO 6000" in msg
        assert "GRPO" in msg     # forbidden 列表
        assert "NeurIPS 2026" in msg
        assert "draft acceptance" in msg   # hook
        assert "hot_2025_2026" in msg     # timeline signal

    def test_user_message_no_hooks_section_when_empty(self):
        pack = _pack()
        pack.structural_hole_hooks = []
        pack.timeline_signals = {}
        msg = _build_user_message_v3(_skeleton(), pack)
        assert "结构洞 hooks" not in msg
        assert "时间线信号" not in msg


# ---------------------------------------------------------------------------
# v3: _build_proposal_v3 装配
# ---------------------------------------------------------------------------

class TestBuildProposalV3:
    def test_metadata_populated(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        assert proposal.status == "draft"
        assert proposal.level == "top-tier"
        assert proposal.seed == pack.direction
        assert proposal.created_at  # ISO 时间戳应有

    def test_outcomes_structured_built(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        so = proposal.expected_outcomes_structured
        assert so is not None
        assert "Spearman rho < 0.5" in so.positive_finding
        assert "convergence" in so.negative_finding

    def test_resource_estimate_from_constraints(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        est = proposal.resource_estimate
        assert est.auto_research["gpu_hours"] == 168
        assert est.auto_research["wall_clock_days"] == 7
        assert "human_hours" in est.human_in_loop
        assert est.manual["wall_clock_days"] == 7 * 5

    def test_key_references_formatted_fallback(self):
        """LLM 没给 key_references_formatted 时，按 PaperEvidence 兜底拼"""
        pack = _pack()
        raw = _good_v3_response(pack)
        # raw 没有 key_references_formatted
        assert "key_references_formatted" not in raw
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        assert len(proposal.key_references_formatted) == 3
        # Method0 在 _pack 里 i=0（偶数）→ "EMNLP 2025"
        assert proposal.key_references_formatted[0] == "Method0: full title here (EMNLP 2025)"


# ---------------------------------------------------------------------------
# v3: _validate_v3 各种 error code
# ---------------------------------------------------------------------------

class TestValidateV3:
    def test_good_no_errors(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        errors = _validate_v3(proposal, pack)
        assert errors == []
        assert proposal.fits_resource_budget is True

    def test_missing_outcomes_struct(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        del raw["expected_outcomes_structured"]
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "MISSING_OUTCOMES_STRUCT" in codes

    def test_outcomes_struct_field_empty(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["expected_outcomes_structured"]["why_both_publishable"] = ""
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "MISSING_OUTCOMES_STRUCT" in codes

    def test_forbidden_technique_flagged(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["proposed_method"] = "We use GRPO + PPO to optimize"
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        errors = _validate_v3(proposal, pack)
        codes = [e[0] for e in errors]
        assert "FORBIDDEN_TECHNIQUE_USED" in codes
        # 详情含被命中的技术
        detail = next(e[1] for e in errors if e[0] == "FORBIDDEN_TECHNIQUE_USED")
        assert "GRPO" in detail and "PPO" in detail

    def test_forbidden_skipped_when_constraint_empty(self):
        pack = _pack(forbid_rl=False)
        raw = _good_v3_response(pack)
        raw["proposed_method"] = "We use GRPO"   # 用了但 constraints 不禁
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "FORBIDDEN_TECHNIQUE_USED" not in codes

    def test_gap_not_declared(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["existing_methods"] = "Layer-skipping methods exist. Quantization methods exist."
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "GAP_NOT_DECLARED" in codes

    def test_missing_paper_id_v3(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["key_references"] = ["arxiv:fake.0001"]
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "MISSING_PAPER_ID" in codes
        assert "TOO_FEW_REFS" in codes  # 只有 1 篇 < 3

    def test_over_budget_v3(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["methodology_phases"][0]["expected_compute_hours"] = 1000
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "OVER_BUDGET" in codes
        assert proposal.fits_resource_budget is False


# ---------------------------------------------------------------------------
# v3: 端到端 elaborate_proposal_from_pack
# ---------------------------------------------------------------------------

class TestElaboratorFromPackEndToEnd:
    def test_happy_path(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        fake_resp = MagicMock(content=_json.dumps(raw))
        with patch("darwinian.agents.proposal_elaborator.invoke_with_retry",
                   return_value=fake_resp):
            proposal = elaborate_proposal_from_pack(_skeleton(), pack, MagicMock())
        assert proposal is not None
        assert proposal.fits_resource_budget is True
        assert proposal.expected_outcomes_structured is not None
        assert proposal.seed == pack.direction
        # 兜底的 references_formatted
        assert len(proposal.key_references_formatted) >= 3

    def test_validation_fail_then_retry(self):
        pack = _pack()
        bad = _good_v3_response(pack)
        del bad["expected_outcomes_structured"]   # 第一次没 outcomes
        good = _good_v3_response(pack)
        responses = [MagicMock(content=_json.dumps(bad)),
                     MagicMock(content=_json.dumps(good))]
        idx = {"i": 0}

        def fake_invoke(llm, messages, **kwargs):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r

        with patch("darwinian.agents.proposal_elaborator.invoke_with_retry",
                   side_effect=fake_invoke):
            with patch("time.sleep"):
                proposal = elaborate_proposal_from_pack(_skeleton(), pack, MagicMock())
        assert idx["i"] == 2    # 重试一次
        assert proposal.expected_outcomes_structured is not None

    def test_unparseable_returns_none(self):
        pack = _pack()
        fake_resp = MagicMock(content="not json at all")
        with patch("darwinian.agents.proposal_elaborator.invoke_with_retry",
                   return_value=fake_resp):
            with patch("time.sleep"):
                proposal = elaborate_proposal_from_pack(_skeleton(), pack, MagicMock())
        assert proposal is None


# ---------------------------------------------------------------------------
# v3 LangGraph wrapper
# ---------------------------------------------------------------------------

class TestNodeV3:
    def _state_with_branches(self, n=2):
        branches = [AbstractionBranch(name=f"B{i}", description="x",
                                      algorithm_logic="x", math_formulation="y")
                    for i in range(n)]
        h = Hypothesis(core_problem="cp", abstraction_tree=branches)
        return ResearchState(research_direction="x", current_hypothesis=h)

    def test_calls_per_branch_when_pack_provided(self):
        state = self._state_with_branches(2)
        pack = _pack()
        good = MagicMock(spec=ResearchProposal)
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal_from_pack",
                   return_value=good) as mock_fn:
            out = proposal_elaborator_node_v3(state, MagicMock(), material_pack=pack)
        assert mock_fn.call_count == 2
        assert len(out["research_proposals"]) == 2

    def test_no_material_pack_returns_empty(self):
        state = self._state_with_branches(1)
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal_from_pack") as mock_fn:
            out = proposal_elaborator_node_v3(state, MagicMock(), material_pack=None)
        assert mock_fn.call_count == 0
        assert out["research_proposals"] == []

    def test_no_hypothesis_returns_empty(self):
        state = ResearchState(research_direction="x")
        pack = _pack()
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal_from_pack") as mock_fn:
            out = proposal_elaborator_node_v3(state, MagicMock(), material_pack=pack)
        assert mock_fn.call_count == 0
        assert out["research_proposals"] == []


# ===========================================================================
# Fix A: 始终从 pack 重建 key_references_formatted（防 LLM 标题幻觉）
# Fix B: WRITING_PHASE_HAS_GPU_HOURS 校验
# ===========================================================================

class TestFixAReferencesFromPack:
    """LLM 给的 key_references_formatted 应被无视，始终用 pack.paper_evidence 重建"""

    def test_llm_given_krf_is_ignored(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        # LLM 故意瞎编标题（模拟 DEL 被瞎编成 "Draft-Enhanced Speculative Decoding"）
        raw["key_references_formatted"] = [
            "Method0: COMPLETELY_HALLUCINATED_TITLE (FAKE_VENUE)",
            "Method1: ANOTHER_FAKE (Wrong 2099)",
        ]
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        # 应使用 pack 的真实 title，不接受 LLM 的瞎编
        for r in proposal.key_references_formatted:
            assert "HALLUCINATED" not in r
            assert "FAKE" not in r
        # 应是 pack.paper_evidence 里的真实 title 拼出来
        assert proposal.key_references_formatted[0] == "Method0: full title here (EMNLP 2025)"

    def test_unknown_paper_id_skipped(self):
        """key_references 含 pack 里没有的 id，渲染时跳过该项（_validate_v3 会另外抓 MISSING_PAPER_ID）"""
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["key_references"] = [pack.paper_evidence[0].paper_id, "arxiv:nonexistent"]
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        assert len(proposal.key_references_formatted) == 1   # 只有真实存在的那一篇被渲


class TestFixBWritingPhaseHasGpuHours:
    """phase name 含 'paper writing' 等关键词且 expected_compute_hours > 0 应触发"""

    def _writing_phase_dict(self, hours: float = 48.0):
        return {
            "phase_number": 4, "name": "Paper Writing and Submission Preparation",
            "description": "Write NeurIPS paper", "inputs": ["results"],
            "outputs": ["PDF"], "expected_compute_hours": hours,
        }

    def test_writing_phase_with_gpu_hours_flagged(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["methodology_phases"].append(self._writing_phase_dict(hours=48.0))
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        errors = _validate_v3(proposal, pack)
        codes = [e[0] for e in errors]
        assert "WRITING_PHASE_HAS_GPU_HOURS" in codes
        # detail 含具体被命中的 phase name + hours
        detail = next(e[1] for e in errors if e[0] == "WRITING_PHASE_HAS_GPU_HOURS")
        assert detail[0][0] == "Paper Writing and Submission Preparation"
        assert detail[0][1] == 48.0

    def test_writing_phase_with_zero_hours_ok(self):
        """expected_compute_hours=0 时不算违反（用户主动归零是接受方案）"""
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["methodology_phases"].append(self._writing_phase_dict(hours=0.0))
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "WRITING_PHASE_HAS_GPU_HOURS" not in codes

    def test_multiple_writing_keywords(self):
        """Submission / Manuscript / Camera-ready 都要抓"""
        pack = _pack()
        for name in ("Manuscript Preparation", "Submission Polish",
                      "Camera Ready Revisions"):
            raw = _good_v3_response(pack)
            raw["methodology_phases"].append({
                "phase_number": 99, "name": name, "description": "x",
                "inputs": [], "outputs": [], "expected_compute_hours": 12.0,
            })
            proposal = _build_proposal_v3(raw, _skeleton(), pack)
            codes = [e[0] for e in _validate_v3(proposal, pack)]
            assert "WRITING_PHASE_HAS_GPU_HOURS" in codes, f"漏抓: {name}"

    def test_phase_name_unrelated_to_writing_passes(self):
        """普通 phase（如 'Profiling'）不被误抓"""
        pack = _pack()
        raw = _good_v3_response(pack)
        # raw 里默认 3 个 phase 都是 'Profile' / 'Optimize' / 'Validate'，不含写作关键词
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "WRITING_PHASE_HAS_GPU_HOURS" not in codes

    def test_feedback_message_contains_offender_names(self):
        from darwinian.agents.proposal_elaborator import _build_feedback_v3
        pack = _pack()
        errors = [("WRITING_PHASE_HAS_GPU_HOURS",
                   [("Paper Writing", 48.0), ("Manuscript Polish", 12.0)])]
        feedback = _build_feedback_v3(errors, pack)
        assert "Paper Writing" in feedback and "48" in feedback
        assert "Manuscript Polish" in feedback and "12" in feedback
        assert "expected_compute_hours 设 0" in feedback or "删掉" in feedback
