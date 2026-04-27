"""
状态模型测试

验证 ResearchState 及其子结构的 Pydantic 验证逻辑。
"""

import pytest
from darwinian.state import (
    ResearchState,
    BudgetState,
    FailedRecord,
    Hypothesis,
    AbstractionBranch,
    ExperimentCode,
    ExperimentResult,
    PoisonTestResult,
    PublishMatrix,
    CriticVerdict,
    ExecutionVerdict,
    FinalVerdict,
    # Phase 1 v2 新增
    Entity,
    LimitationRef,
    PaperInfo,
    EntityPair,
    ConceptGraph,
    # Phase 1 v3 新增（seed-schema task）
    StructuralHoleHook,
    ResearchConstraints,
    ExpectedOutcomes,
    ResearchMaterialPack,
    DebateRound,
    DebateResult,
    PaperEvidence,
    QuantitativeClaim,
    ResearchProposal,
)


class TestBudgetState:
    def test_default_not_exhausted(self):
        budget = BudgetState()
        assert not budget.is_exhausted

    def test_exhausted_by_tokens(self):
        budget = BudgetState(remaining_tokens=0)
        assert budget.is_exhausted

    def test_exhausted_by_time(self):
        budget = BudgetState(elapsed_seconds=9999.0)
        assert budget.is_exhausted


class TestPublishMatrix:
    def test_all_false_not_green(self):
        matrix = PublishMatrix()
        assert not matrix.all_green

    def test_all_true_is_green(self):
        matrix = PublishMatrix(
            novelty_passed=True,
            baseline_improved=True,
            robustness_passed=True,
            explainability_generated=True,
        )
        assert matrix.all_green

    def test_partial_not_green(self):
        matrix = PublishMatrix(novelty_passed=True, baseline_improved=True)
        assert not matrix.all_green


class TestHypothesis:
    def make_branch(self, name="TestBranch"):
        return AbstractionBranch(
            name=name,
            description="测试方案",
            algorithm_logic="步骤 1 → 步骤 2",
            math_formulation=r"f(x) = \sigma(Wx + b)",
        )

    def test_valid_hypothesis(self):
        h = Hypothesis(
            core_problem="如何提高模型鲁棒性",
            abstraction_tree=[self.make_branch()],
        )
        assert h.core_problem == "如何提高模型鲁棒性"
        assert len(h.abstraction_tree) == 1

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            Hypothesis(
                core_problem="test",
                abstraction_tree=[self.make_branch()],
                confidence=1.5,  # 超出 [0, 1]
            )

    def test_empty_abstraction_tree_allowed(self):
        # Agent 1 创建 Hypothesis 时 abstraction_tree 为空是合法的（由 Agent 2 填充）
        h = Hypothesis(core_problem="test")
        assert h.abstraction_tree == []


class TestResearchState:
    def test_default_state(self):
        state = ResearchState(research_direction="图神经网络药物发现")
        assert state.research_direction == "图神经网络药物发现"
        assert state.outer_loop_count == 0
        assert len(state.failed_ledger) == 0
        assert state.current_hypothesis is None
        assert not state.publish_matrix.all_green

    def test_state_with_failed_ledger(self):
        record = FailedRecord(
            feature_vector=[0.1, 0.2, 0.3],
            error_summary="数学公式维度不匹配",
            failure_type="MATH_ERROR",
            iteration=1,
        )
        state = ResearchState(
            research_direction="测试方向",
            failed_ledger=[record],
        )
        assert len(state.failed_ledger) == 1
        assert state.failed_ledger[0].failure_type == "MATH_ERROR"


class TestConceptGraph:
    """Phase 1 v2 新增数据结构验证"""

    def test_entity_minimal(self):
        e = Entity(canonical_name="adam", type="method", paper_ids=["p1"])
        assert e.canonical_name == "adam"
        assert e.type == "method"
        assert e.aliases == []

    def test_entity_type_literal_enforced(self):
        with pytest.raises(Exception):
            Entity(canonical_name="x", type="invalid_type", paper_ids=[])

    def test_limitation_ref(self):
        l = LimitationRef(id="a1b2c3d4", text="收敛慢", source_paper_id="p3")
        assert l.id == "a1b2c3d4"
        assert l.source_paper_id == "p3"

    def test_paper_info_defaults(self):
        p = PaperInfo(paper_id="p1")
        assert p.abstract == ""
        assert p.year == 0
        assert p.citation_count == 0
        assert p.source == "semantic_scholar"

    def test_concept_graph_lookup(self):
        g = ConceptGraph(
            entities=[
                Entity(canonical_name="adam", type="method", paper_ids=["p1"]),
                Entity(canonical_name="resnet", type="method", paper_ids=["p2"]),
            ],
            limitations=[
                LimitationRef(id="L_001", text="过拟合", source_paper_id="p1"),
            ],
        )
        assert g.entity_by_name("adam").paper_ids == ["p1"]
        assert g.entity_by_name("nonexistent") is None
        assert g.limitation_by_id("L_001").text == "过拟合"
        assert g.limitation_by_id("nope") is None
        assert g.is_sufficient is False  # 默认 False

    def test_state_accepts_concept_graph(self):
        g = ConceptGraph(is_sufficient=True)
        state = ResearchState(research_direction="test", concept_graph=g)
        assert state.concept_graph is not None
        assert state.concept_graph.is_sufficient is True


class TestAbstractionBranchV2:
    """AbstractionBranch v2 新字段"""

    def test_new_fields_default_empty(self):
        b = AbstractionBranch(
            name="x", description="x", algorithm_logic="x", math_formulation="x",
        )
        assert b.cited_entity_names == []
        assert b.solved_limitation_id == ""
        assert b.existing_combination is False
        assert b.existing_combination_refs == []

    def test_source_domain_now_optional(self):
        # v1 时 source_domain 是 required，v2 改为 optional default=""
        b = AbstractionBranch(
            name="x", description="x", algorithm_logic="x", math_formulation="x",
        )
        assert b.source_domain == ""

    def test_cited_entities_and_limitation(self):
        b = AbstractionBranch(
            name="sparse_attn_mamba",
            description="混合架构",
            algorithm_logic="...",
            math_formulation="...",
            cited_entity_names=["self_attention", "mamba"],
            solved_limitation_id="a1b2c3d4",
        )
        assert "mamba" in b.cited_entity_names
        assert b.solved_limitation_id == "a1b2c3d4"


class TestEnums:
    def test_critic_verdict_values(self):
        assert CriticVerdict.PASS == "PASS"
        assert CriticVerdict.MATH_ERROR == "MATH_ERROR"
        assert CriticVerdict.NOT_NOVEL == "NOT_NOVEL"

    def test_execution_verdict_values(self):
        assert ExecutionVerdict.CODE_ERROR == "code_error"
        assert ExecutionVerdict.INSUFFICIENT == "insufficient"
        assert ExecutionVerdict.SUCCESS == "success"

    def test_final_verdict_values(self):
        assert FinalVerdict.PUBLISH_READY == "publish_ready"
        assert FinalVerdict.ROBUSTNESS_FAIL == "robustness_fail"


# ===========================================================================
# Phase 1 v3 — seed-schema task 新增
# ===========================================================================


class TestStructuralHoleHook:
    def test_minimal(self):
        h = StructuralHoleHook(
            entity_a="quantization sensitivity",
            entity_b="draft acceptance rate",
            hook_text="No work measures rank correlation between A's perplexity-sensitivity and B's draft-quality-sensitivity",
            relation_type="divergence",
        )
        assert h.relation_type == "divergence"
        assert h.score == 0
        assert h.supporting_paper_ids_a == []

    def test_relation_type_enum(self):
        with pytest.raises(Exception):
            StructuralHoleHook(
                entity_a="a", entity_b="b", hook_text="x",
                relation_type="invalid_type",  # 不在枚举里
            )

    def test_with_supporting_papers(self):
        h = StructuralHoleHook(
            entity_a="ramp", entity_b="layerskip",
            hook_text="A 优化 perplexity，B 优化 acceptance rate",
            relation_type="transfer",
            score=12,
            supporting_paper_ids_a=["arxiv:2603.17891"],
            supporting_paper_ids_b=["arxiv:2404.16710"],
        )
        assert h.score == 12
        assert "arxiv:2404.16710" in h.supporting_paper_ids_b


class TestResearchConstraints:
    def test_defaults_match_quantskip_seed(self):
        c = ResearchConstraints()
        # QuantSkip 用例约束反推默认值是否合理
        assert c.gpu_count == 4
        assert c.max_model_params_b == 14.0
        assert c.use_existing_benchmarks_only is True
        assert c.require_human_annotation is False
        assert c.require_no_api_for_main is True
        assert c.forbidden_techniques == []
        assert c.target_venues == []

    def test_quantskip_constraints_roundtrip(self):
        c = ResearchConstraints(
            gpu_count=4,
            gpu_model="RTX PRO 6000 96GB",
            wall_clock_days=7,
            forbidden_techniques=["GRPO", "PPO", "DPO", "RLHF", "RLVR", "RLMT"],
            target_venues=["NeurIPS 2026", "EMNLP 2026"],
        )
        assert "RLVR" in c.forbidden_techniques
        assert c.target_venues[0] == "NeurIPS 2026"


class TestExpectedOutcomes:
    def test_three_fields_required(self):
        out = ExpectedOutcomes(
            positive_finding="如果 draft-sensitivity divergence 显著，则证明需要 draft-specific metric",
            negative_finding="如果 sensitivity 收敛，则验证 accuracy-guided 方法对 spec-decoding 已足够",
            why_both_publishable="两种结果都给社区 actionable guidance",
        )
        assert out.positive_finding
        assert out.negative_finding
        assert out.why_both_publishable

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            ExpectedOutcomes(positive_finding="a", negative_finding="b")  # 缺 why_both_publishable


class TestResearchProposalExpectedOutcomesField:
    """ResearchProposal 同时保留 str 和 structured 字段"""

    def test_legacy_str_still_works(self):
        from darwinian.state import AbstractionBranch
        p = ResearchProposal(
            skeleton=AbstractionBranch(name="x", description="x", algorithm_logic="x", math_formulation="x"),
            title="t", elevator_pitch="e",
            expected_outcomes="自由文本",
        )
        assert p.expected_outcomes == "自由文本"
        assert p.expected_outcomes_structured is None

    def test_structured_optional(self):
        from darwinian.state import AbstractionBranch
        p = ResearchProposal(
            skeleton=AbstractionBranch(name="x", description="x", algorithm_logic="x", math_formulation="x"),
            title="t", elevator_pitch="e",
            expected_outcomes_structured=ExpectedOutcomes(
                positive_finding="X→Y",
                negative_finding="~X→Z",
                why_both_publishable="both ok",
            ),
        )
        assert p.expected_outcomes_structured is not None
        assert p.expected_outcomes_structured.positive_finding == "X→Y"


class TestResearchMaterialPack:
    def test_minimal_only_direction(self):
        mp = ResearchMaterialPack(direction="LLM inference acceleration")
        assert mp.direction == "LLM inference acceleration"
        assert mp.paper_evidence == []
        assert mp.structural_hole_hooks == []
        assert mp.evidence_by_category == {}
        assert isinstance(mp.constraints, ResearchConstraints)

    def test_evidence_by_category_groups(self):
        ev1 = PaperEvidence(
            paper_id="arxiv:2404.16710", title="LayerSkip", short_name="LayerSkip",
            category="Layer-skipping self-speculative methods",
            quantitative_claims=[QuantitativeClaim(metric_name="speedup", metric_value="2.16x")],
            headline_result="2.16x speedup", relation_to_direction="extends",
        )
        ev2 = PaperEvidence(
            paper_id="arxiv:2510.xxxxx", title="DEL", short_name="DEL",
            category="Layer-skipping self-speculative methods",
            quantitative_claims=[QuantitativeClaim(metric_name="speedup", metric_value="2.62x")],
            headline_result="2.62x speedup", relation_to_direction="extends",
        )
        ev3 = PaperEvidence(
            paper_id="arxiv:2603.17891", title="RAMP", short_name="RAMP",
            category="Per-layer mixed-precision quantization",
            quantitative_claims=[QuantitativeClaim(metric_name="PPL", metric_value="5.54")],
            headline_result="5.54 vs 5.60 PPL", relation_to_direction="baseline",
        )
        mp = ResearchMaterialPack(direction="x", paper_evidence=[ev1, ev2, ev3])
        groups = mp.evidence_by_category
        assert len(groups) == 2
        assert len(groups["Layer-skipping self-speculative methods"]) == 2
        assert len(groups["Per-layer mixed-precision quantization"]) == 1

    def test_uncategorized_bucket(self):
        ev = PaperEvidence(
            paper_id="arxiv:x", title="t", short_name="x",
            category="",  # 空 category
            quantitative_claims=[QuantitativeClaim(metric_name="x", metric_value="1")],
            headline_result="1x", relation_to_direction="extends",
        )
        mp = ResearchMaterialPack(direction="x", paper_evidence=[ev])
        assert "uncategorized" in mp.evidence_by_category


class TestDebate:
    def test_round_acceptance_clamped(self):
        with pytest.raises(Exception):
            DebateRound(
                round_number=1, advocate_argument="a", challenger_argument="c",
                judge_assessment="j", estimated_acceptance_rate=1.5,  # > 1
            )

    def test_result_default_not_above_threshold(self):
        r = DebateResult()
        assert r.is_above_threshold is False
        assert r.delta_last_two == float("inf")
        assert r.converged is False

    def test_delta_last_two_after_two_rounds(self):
        r = DebateResult(rounds=[
            DebateRound(round_number=1, advocate_argument="a", challenger_argument="c",
                        judge_assessment="j", estimated_acceptance_rate=0.20),
            DebateRound(round_number=2, advocate_argument="a", challenger_argument="c",
                        judge_assessment="j", estimated_acceptance_rate=0.32),
        ], final_acceptance_rate=0.32)
        assert abs(r.delta_last_two - 0.12) < 1e-9
        assert r.is_above_threshold is True

    def test_threshold_customizable(self):
        r = DebateResult(final_acceptance_rate=0.25, acceptance_threshold=0.20)
        assert r.is_above_threshold is True
