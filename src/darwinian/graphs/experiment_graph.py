"""
Phase 2: 自动实验与对抗验证子图 (Adversarial Experiment Graph)

节点流程：
  Phase 1 传入
    → dataset_router (确立实验数据)
    → code_architect_node (Agent 4: 生成代码)
    → code_execute (隔离执行)
    → diagnostician_node (Agent 5: 诊断)
    → execution_router (条件路由)
      ├── code_error → code_architect_node (内层循环，最多 5 次)
      ├── insufficient → 写入 failed_ledger → END (终止本轮，通知主图重启 Phase 1)
      └── success → poison_generator_node (Agent 6)
    → code_execute (毒药数据执行)
    → publish_evaluator_node (Agent 7)
    → final_router
      ├── robustness_fail → 写入 failed_ledger → END (打回 Phase 1)
      └── publish_ready → END (生成报告)
"""

from __future__ import annotations

from functools import partial
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from darwinian.state import (
    ResearchState,
    ExecutionVerdict,
    FinalVerdict,
    FailedRecord,
    ExperimentResult,
)
from darwinian.agents.code_architect import code_architect_node
from darwinian.agents.diagnostician import diagnostician_node
from darwinian.agents.poison_generator import poison_generator_node
from darwinian.agents.publish_evaluator import publish_evaluator_node
from darwinian.tools.code_executor import code_execute
from darwinian.tools.dataset_finder import dataset_finder_node
from darwinian.utils.similarity import get_text_embedding


MAX_CODE_RETRIES = 5  # 内层代码修复循环上限


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------

def dataset_router_node(state: ResearchState) -> dict:
    """确认数据集 Schema 已就绪。若没有用户提供的 Schema，填入占位符供后续节点参考。"""
    if not state.dataset_schema:
        return {"dataset_schema": {"type": "unknown", "note": "由 dataset_finder 自动确定"}}
    return {}


def code_execute_node(state: ResearchState) -> dict:
    """工具节点：在 Docker 沙箱中执行完整实验代码（非毒药模式）"""
    if state.experiment_code is None:
        raise ValueError("code_execute_node 调用前必须先运行 code_architect_node")

    result = code_execute(
        experiment_code=state.experiment_code,
        mode="full",
        data_dir=state.user_data_path or None,
    )
    return {"experiment_result": result}


def poison_execute_node(state: ResearchState) -> dict:
    """工具节点：在 Docker 沙箱中执行毒药数据扰动测试"""
    if state.experiment_code is None:
        raise ValueError("poison_execute_node 调用前必须先运行 poison_generator_node")

    result = code_execute(
        experiment_code=state.experiment_code,
        mode="poison",
        data_dir=state.user_data_path or None,
    )

    # 解析扰动后指标，写入 poison_test_result
    if state.poison_test_result is not None:
        proposed_original = state.experiment_result.proposed_metrics if state.experiment_result else {}
        perturbed = result.proposed_metrics  # 毒药模式下也写入 proposed_metrics

        # 计算性能下降比例（取第一个主要指标）
        degradation = _compute_degradation(proposed_original, perturbed)
        updated_poison = state.poison_test_result.model_copy(update={
            "perturbed_metrics": perturbed,
            "degradation_rate": degradation,
        })
        return {
            "experiment_result": result,
            "poison_test_result": updated_poison,
        }

    return {"experiment_result": result}


def write_insufficient_to_ledger(state: ResearchState) -> dict:
    """insufficient 路径：将方法无效记录写入 failed_ledger，准备终止本轮"""
    hypothesis = state.current_hypothesis
    if hypothesis is None:
        return {}

    branch = hypothesis.selected_branch
    text = f"{hypothesis.core_problem} {branch.algorithm_logic if branch else ''}"
    feature_vector = get_text_embedding(text)

    diagnosis = state.experiment_result.diagnosis if state.experiment_result else "方法效果不足基准"
    record = FailedRecord(
        feature_vector=feature_vector,
        error_summary=diagnosis,
        failure_type="INSUFFICIENT",
        iteration=state.outer_loop_count,
        banned_keywords=_extract_method_keywords(branch.name if branch else ""),
    )
    return {"failed_ledger": list(state.failed_ledger) + [record]}


def write_robustness_fail_to_ledger(state: ResearchState) -> dict:
    """robustness_fail 路径：将鲁棒性失败记录写入 failed_ledger"""
    hypothesis = state.current_hypothesis
    if hypothesis is None:
        return {}

    branch = hypothesis.selected_branch
    text = f"{hypothesis.core_problem} {branch.algorithm_logic if branch else ''}"
    feature_vector = get_text_embedding(text)

    degradation = state.poison_test_result.degradation_rate if state.poison_test_result else 0.0
    record = FailedRecord(
        feature_vector=feature_vector,
        error_summary=f"鲁棒性测试失败（性能下降 {degradation:.1%}）",
        failure_type="ROBUSTNESS_FAIL",
        iteration=state.outer_loop_count,
        banned_keywords=_extract_method_keywords(branch.name if branch else ""),
    )
    return {"failed_ledger": list(state.failed_ledger) + [record]}


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------

def execution_router(
    state: ResearchState,
) -> Literal["code_architect", "write_insufficient", "poison_generator"]:
    """执行结果路由"""
    if state.experiment_result is None:
        return "code_architect"

    verdict = state.experiment_result.execution_verdict

    if verdict == ExecutionVerdict.CODE_ERROR:
        # 内层循环：检查重试次数上限
        retry_count = state.experiment_code.retry_count if state.experiment_code else 0
        if retry_count >= MAX_CODE_RETRIES:
            # 超过上限，视为方法无效，终止本轮
            return "write_insufficient"
        return "code_architect"

    elif verdict == ExecutionVerdict.INSUFFICIENT:
        return "write_insufficient"

    elif verdict == ExecutionVerdict.SUCCESS:
        return "poison_generator"

    return "code_architect"


def final_router(state: ResearchState) -> Literal["write_robustness_fail", "__end__"]:
    """最终路由"""
    if state.final_verdict == FinalVerdict.PUBLISH_READY:
        return "__end__"
    else:
        return "write_robustness_fail"


# ---------------------------------------------------------------------------
# 子图构建
# ---------------------------------------------------------------------------

def build_experiment_graph(llm: BaseChatModel) -> StateGraph:
    """
    构建 Phase 2 子图。

    Args:
        llm: 供所有 Agent 使用的 LLM 实例

    Returns:
        编译后的 LangGraph 子图
    """
    graph = StateGraph(ResearchState)

    # 注册节点
    graph.add_node("dataset_router", dataset_router_node)
    graph.add_node("dataset_finder", partial(dataset_finder_node, llm=llm))
    graph.add_node("code_architect", partial(code_architect_node, llm=llm))
    graph.add_node("code_execute", code_execute_node)
    graph.add_node("diagnostician", partial(diagnostician_node, llm=llm))
    graph.add_node("write_insufficient", write_insufficient_to_ledger)
    graph.add_node("poison_generator", partial(poison_generator_node, llm=llm))
    graph.add_node("poison_execute", poison_execute_node)
    graph.add_node("publish_evaluator", partial(publish_evaluator_node, llm=llm))
    graph.add_node("write_robustness_fail", write_robustness_fail_to_ledger)

    # 固定边
    graph.add_edge(START, "dataset_router")
    graph.add_edge("dataset_router", "dataset_finder")
    graph.add_edge("dataset_finder", "code_architect")
    graph.add_edge("code_architect", "code_execute")
    graph.add_edge("code_execute", "diagnostician")
    graph.add_edge("write_insufficient", END)
    graph.add_edge("poison_generator", "poison_execute")
    graph.add_edge("poison_execute", "publish_evaluator")
    graph.add_edge("write_robustness_fail", END)

    # 条件路由边
    graph.add_conditional_edges(
        "diagnostician",
        execution_router,
        {
            "code_architect": "code_architect",
            "write_insufficient": "write_insufficient",
            "poison_generator": "poison_generator",
        },
    )

    graph.add_conditional_edges(
        "publish_evaluator",
        final_router,
        {
            "write_robustness_fail": "write_robustness_fail",
            "__end__": END,
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _compute_degradation(original: dict, perturbed: dict) -> float:
    """计算性能下降比例（取第一个共同指标）"""
    if not original or not perturbed:
        return 0.0

    for key in original:
        if key in perturbed and original[key] != 0:
            orig_val = original[key]
            pert_val = perturbed[key]
            return max(0.0, (orig_val - pert_val) / abs(orig_val))

    return 0.0


def _extract_method_keywords(method_name: str) -> list[str]:
    """从方法名中提取关键词用于 banned_keywords"""
    import re
    words = re.findall(r"[a-z0-9\u4e00-\u9fff]+", method_name.lower())
    # 过滤过短的词
    return [w for w in words if len(w) > 2]
