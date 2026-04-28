"""
proposal_elaborator 单元测试 (v3 path only — v2 ConceptGraph path 已删除)

不打真实 LLM：mock invoke_with_retry，验证：
- elaborate_proposal_from_pack 端到端
- 8 项硬约束校验各自能抓出错
- 校验失败时带反馈重试
- proposal_elaborator_node_v3 从 state.material_pack 读素材
- Fix A (强制 pack title 防幻觉) + Fix B (写作 phase 不算 GPU)
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.proposal_elaborator import (
    elaborate_proposal_from_pack,
    proposal_elaborator_node_v3,
    _build_proposal_v3,
    _validate_v3,
    _build_user_message_v3,
    _build_feedback_v3,
    _render_evidence_by_category,
)
from darwinian.state import (
    AbstractionBranch,
    ConceptGraph,
    ExpectedOutcomes,
    Hypothesis,
    LimitationRef,
    MethodologyPhase,
    PaperEvidence,
    PaperInfo,
    QuantitativeClaim,
    ResearchConstraints,
    ResearchMaterialPack,
    ResearchProposal,
    ResearchState,
    ResourceEstimate,
    StructuralHoleHook,
)


# ---------------------------------------------------------------------------
# 工具：fixtures
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
    """v3 节点从 state.material_pack 读素材（P0 修复后 LangGraph 兼容签名）"""

    def _state_with_branches(self, n=2, material_pack=None):
        branches = [AbstractionBranch(name=f"B{i}", description="x",
                                      algorithm_logic="x", math_formulation="y")
                    for i in range(n)]
        h = Hypothesis(core_problem="cp", abstraction_tree=branches)
        return ResearchState(
            research_direction="x",
            current_hypothesis=h,
            material_pack=material_pack,
        )

    def test_calls_per_branch_when_pack_in_state(self):
        state = self._state_with_branches(2, material_pack=_pack())
        good = MagicMock(spec=ResearchProposal)
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal_from_pack",
                   return_value=good) as mock_fn:
            out = proposal_elaborator_node_v3(state, MagicMock())
        assert mock_fn.call_count == 2
        assert len(out["research_proposals"]) == 2
        # 验证 pack 是从 state 取的（不是 kwarg）
        assert mock_fn.call_args.kwargs["material_pack"] is state.material_pack

    def test_no_material_pack_in_state_returns_empty(self):
        state = self._state_with_branches(1, material_pack=None)
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal_from_pack") as mock_fn:
            out = proposal_elaborator_node_v3(state, MagicMock())
        assert mock_fn.call_count == 0
        assert out["research_proposals"] == []

    def test_no_hypothesis_returns_empty(self):
        state = ResearchState(research_direction="x", material_pack=_pack())
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal_from_pack") as mock_fn:
            out = proposal_elaborator_node_v3(state, MagicMock())
        assert mock_fn.call_count == 0
        assert out["research_proposals"] == []

    def test_skips_failed_elaborations(self):
        """部分 branch 返 None 不阻塞，已成功的仍写入"""
        state = self._state_with_branches(3, material_pack=_pack())
        results = [MagicMock(spec=ResearchProposal), None, MagicMock(spec=ResearchProposal)]
        idx = {"i": 0}
        def fake(*a, **k):
            r = results[idx["i"]]
            idx["i"] += 1
            return r
        with patch("darwinian.agents.proposal_elaborator.elaborate_proposal_from_pack",
                   side_effect=fake):
            out = proposal_elaborator_node_v3(state, MagicMock())
        assert len(out["research_proposals"]) == 2


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


# ===========================================================================
# Pri-2: Methodology validators (MILESTONE_GATE_MISSING / BUDGET_PHASES_MISMATCH +
# 扩展 WRITING_PHASE_HAS_GPU_HOURS keywords)
# ===========================================================================

from darwinian.agents.proposal_elaborator import (
    _is_hypothesis_validation_phase,
    _has_decision_gate,
)


class TestHypothesisPhaseDetection:
    def test_profiling_phase_detected(self):
        from darwinian.state import MethodologyPhase
        ph = MethodologyPhase(phase_number=1, name="Sensitivity Profiling",
                               description="Profile per-layer entropy")
        assert _is_hypothesis_validation_phase(ph)

    def test_pilot_in_description(self):
        from darwinian.state import MethodologyPhase
        ph = MethodologyPhase(phase_number=1, name="Phase 1",
                               description="Run pilot study to verify correlation")
        assert _is_hypothesis_validation_phase(ph)

    def test_normal_phase_not_detected(self):
        from darwinian.state import MethodologyPhase
        ph = MethodologyPhase(phase_number=1, name="Implementation",
                               description="Build the system")
        assert not _is_hypothesis_validation_phase(ph)


class TestDecisionGateDetection:
    def test_explicit_threshold(self):
        from darwinian.state import MethodologyPhase
        ph = MethodologyPhase(phase_number=1, name="Profiling",
                               description="if r > 0.4 proceed; else pivot")
        assert _has_decision_gate(ph)

    def test_outputs_contain_gate(self):
        from darwinian.state import MethodologyPhase
        ph = MethodologyPhase(phase_number=1, name="Profiling",
                               description="x", outputs=["go/no-go decision"])
        assert _has_decision_gate(ph)

    def test_no_gate_returns_false(self):
        from darwinian.state import MethodologyPhase
        ph = MethodologyPhase(phase_number=1, name="Profiling",
                               description="just measure things")
        assert not _has_decision_gate(ph)


class TestMilestoneGateMissing:
    def test_v9_egat_case_flagged(self):
        """重现 v9: phase 1 是 sensitivity profiling，无 decision gate → 应触发"""
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["methodology_phases"] = [
            {"phase_number": 1, "name": "Entropy Profiling",
             "description": "Profile per-layer entropy correlation",
             "expected_compute_hours": 96},
            {"phase_number": 2, "name": "Halter MLP Training",
             "description": "Train halter on entropy", "expected_compute_hours": 120},
            {"phase_number": 3, "name": "End-to-end Eval",
             "description": "Test", "expected_compute_hours": 60},
        ]
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "MILESTONE_GATE_MISSING" in codes

    def test_phase_with_gate_passes(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["methodology_phases"] = [
            {"phase_number": 1, "name": "Entropy Profiling",
             "description": "Profile correlation. Decision criteria: if Spearman r > 0.4 proceed to Phase 2; else pivot to failure-mode analysis",
             "expected_compute_hours": 96},
            {"phase_number": 2, "name": "Build", "description": "x",
             "expected_compute_hours": 120},
            {"phase_number": 3, "name": "Eval", "description": "y",
             "expected_compute_hours": 60},
        ]
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "MILESTONE_GATE_MISSING" not in codes

    def test_non_hypothesis_phase_not_flagged(self):
        """如果 phase 1 不是 hypothesis validation 就不要求 gate"""
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["methodology_phases"] = [
            {"phase_number": 1, "name": "Implementation",
             "description": "Build system", "expected_compute_hours": 96},
            {"phase_number": 2, "name": "Eval", "description": "x",
             "expected_compute_hours": 60},
            {"phase_number": 3, "name": "Ablation", "description": "y",
             "expected_compute_hours": 50},
        ]
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "MILESTONE_GATE_MISSING" not in codes


class TestBudgetPhasesMismatch:
    def test_v9_644_vs_672_case(self):
        """v9 实测：phase sum=644 但 total=672 → 应触发"""
        pack = _pack()
        raw = _good_v3_response(pack)
        # 强制构造 mismatch: 让 _build_proposal_v3 算出 sum, 然后人工改 total
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        # 改 total 跟 sum 错开
        proposal.total_estimated_hours = proposal.total_estimated_hours + 28
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "BUDGET_PHASES_MISMATCH" in codes

    def test_consistent_passes(self):
        """sum == total 时不触发"""
        pack = _pack()
        raw = _good_v3_response(pack)
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        # 默认 _build_proposal_v3 让 total = sum，应一致
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "BUDGET_PHASES_MISMATCH" not in codes

    def test_within_1h_tolerance_passes(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        proposal.total_estimated_hours = proposal.total_estimated_hours + 0.5
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "BUDGET_PHASES_MISMATCH" not in codes


class TestExpandedWritingKeywords:
    def test_final_analysis_caught(self):
        """新关键词 'final analysis' 也算写作类 phase"""
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["methodology_phases"].append({
            "phase_number": 4, "name": "Final Analysis and Threshold Validation",
            "description": "Wrap up", "expected_compute_hours": 48,
        })
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "WRITING_PHASE_HAS_GPU_HOURS" in codes

    def test_polish_caught(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["methodology_phases"].append({
            "phase_number": 4, "name": "Camera-ready Polish",
            "description": "x", "expected_compute_hours": 12,
        })
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "WRITING_PHASE_HAS_GPU_HOURS" in codes


# ===========================================================================
# Pri-3: VENUE_DEADLINE_INCORRECT
# ===========================================================================

class TestVenueDeadlineValidation:
    def test_v9_neurips_wrong_deadline_caught(self):
        """v9 实测：NeurIPS 2026 写成 2026-09-01"""
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["target_venue"] = "NeurIPS 2026"
        raw["target_deadline"] = "2026-09-01"
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "VENUE_DEADLINE_INCORRECT" in codes

    def test_correct_deadline_passes(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["target_venue"] = "NeurIPS 2026"
        raw["target_deadline"] = "2026-05-13"
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "VENUE_DEADLINE_INCORRECT" not in codes

    def test_unknown_venue_passes(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        raw["target_venue"] = "ICASSP 2027"
        raw["target_deadline"] = "2026-12-99"
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        codes = [e[0] for e in _validate_v3(proposal, pack)]
        assert "VENUE_DEADLINE_INCORRECT" not in codes


# ===========================================================================
# Pri-4: Claim spot-check 集成进 _build_proposal_v3
# ===========================================================================

class TestClaimSpotCheckIntegration:
    def test_unverified_numbers_populated(self):
        """motivation 含 evidence 没有的数字 → unverified_numbers 字段非空"""
        pack = _pack()  # _pack 里默认有 4 个 method 各 1 个 quant claim 如 "1.50x"
        raw = _good_v3_response(pack)
        # 改写 motivation 加一个绝对没在 evidence 里的数字
        raw["motivation"] = (
            "Method0 hits 1.50x speedup; we achieve **42.7% reduction** in latency"
        )
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        # 1.50x 是 evidence 里有的 → 不报
        assert "1.50x" not in proposal.unverified_numbers
        # 42.7% 是 motivation 自加的 → 应被标
        assert any("42.7" in n for n in proposal.unverified_numbers)

    def test_all_grounded_no_unverified(self):
        pack = _pack()
        raw = _good_v3_response(pack)
        # _good_v3_response 默认 motivation 全引 _pack 里的 "1.50x" / "1.80x" / "2.40x"
        proposal = _build_proposal_v3(raw, _skeleton(), pack)
        # _good_v3_response motivation 用 evidence 数字 → 应全干净
        assert proposal.unverified_numbers == [] or all(
            any(n.startswith(prefix) for prefix in ["1.5", "1.8", "2.4"])
            for n in proposal.unverified_numbers
        )

    def test_default_field_empty_list(self):
        from darwinian.state import ResearchProposal as RP
        p = RP(skeleton=_skeleton(), title="t", elevator_pitch="p")
        assert p.unverified_numbers == []
