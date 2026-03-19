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
            source_domain="控制论",
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

    def test_empty_abstraction_tree_fails(self):
        with pytest.raises(Exception):
            Hypothesis(
                core_problem="test",
                abstraction_tree=[],  # min_length=1
            )


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
