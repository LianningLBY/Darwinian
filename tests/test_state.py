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
