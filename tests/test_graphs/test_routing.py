"""
图路由逻辑测试

验证 critic_router、execution_router、final_router 的条件路由正确性，
无需真实 LLM 调用（纯状态驱动测试）。
"""

import pytest
from darwinian.state import (
    ResearchState,
    BudgetState,
    CriticVerdict,
    ExecutionVerdict,
    FinalVerdict,
    ExperimentCode,
    ExperimentResult,
    PublishMatrix,
)
from darwinian.graphs.hypothesis_graph import critic_router
from darwinian.graphs.experiment_graph import execution_router, final_router


def make_base_state(**kwargs) -> ResearchState:
    return ResearchState(research_direction="测试方向", **kwargs)


class TestCriticRouter:
    def test_pass_routes_to_end(self):
        state = make_base_state(critic_verdict=CriticVerdict.PASS)
        assert critic_router(state) == "__end__"

    def test_not_novel_routes_to_hypothesis_generator(self):
        state = make_base_state(critic_verdict=CriticVerdict.NOT_NOVEL)
        assert critic_router(state) == "hypothesis_generator"

    def test_math_error_routes_to_bottleneck_miner(self):
        state = make_base_state(critic_verdict=CriticVerdict.MATH_ERROR)
        assert critic_router(state) == "bottleneck_miner"

    def test_loop_limit_forces_end(self):
        state = make_base_state(
            critic_verdict=CriticVerdict.NOT_NOVEL,
            outer_loop_count=99,  # 超出 max_outer_loops=5
            max_outer_loops=5,
        )
        assert critic_router(state) == "__end__"


class TestExecutionRouter:
    def _make_state_with_result(self, verdict: ExecutionVerdict, retry_count: int = 0):
        result = ExperimentResult(execution_verdict=verdict)
        code = ExperimentCode(
            baseline_code="",
            proposed_code="",
            dataset_loader_code="",
            retry_count=retry_count,
        )
        return make_base_state(experiment_result=result, experiment_code=code)

    def test_code_error_routes_to_code_architect(self):
        state = self._make_state_with_result(ExecutionVerdict.CODE_ERROR, retry_count=0)
        assert execution_router(state) == "code_architect"

    def test_code_error_over_limit_routes_to_insufficient(self):
        state = self._make_state_with_result(ExecutionVerdict.CODE_ERROR, retry_count=5)
        assert execution_router(state) == "write_insufficient"

    def test_insufficient_routes_to_write_insufficient(self):
        state = self._make_state_with_result(ExecutionVerdict.INSUFFICIENT)
        assert execution_router(state) == "write_insufficient"

    def test_success_routes_to_poison_generator(self):
        state = self._make_state_with_result(ExecutionVerdict.SUCCESS)
        assert execution_router(state) == "poison_generator"


class TestFinalRouter:
    def test_publish_ready_routes_to_end(self):
        state = make_base_state(final_verdict=FinalVerdict.PUBLISH_READY)
        assert final_router(state) == "__end__"

    def test_robustness_fail_routes_to_write(self):
        state = make_base_state(final_verdict=FinalVerdict.ROBUSTNESS_FAIL)
        assert final_router(state) == "write_robustness_fail"

    def test_none_verdict_routes_to_write(self):
        state = make_base_state(final_verdict=None)
        assert final_router(state) == "write_robustness_fail"
