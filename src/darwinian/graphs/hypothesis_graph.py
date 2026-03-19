"""
Phase 1: 研究方案生成子图 (Hypothesis Generation Graph)

节点流程：
  START
    → preprocess_node (初始化/预处理)
    → bottleneck_miner_node (Agent 1: 瓶颈挖掘)
    → hypothesis_generator_node (Agent 2: 方案合成)
    → theoretical_critic_node (Agent 3: 理论审查)
    → critic_router (条件路由)
      ├── NOT_NOVEL → hypothesis_generator_node (内层重试)
      ├── MATH_ERROR → 写入 failed_ledger → bottleneck_miner_node (大重试)
      └── PASS → END (流转至 Phase 2)
"""

from __future__ import annotations

import time
from functools import partial
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from darwinian.state import (
    ResearchState,
    CriticVerdict,
    FailedRecord,
)
from darwinian.agents.bottleneck_miner import bottleneck_miner_node
from darwinian.agents.hypothesis_generator import (
    hypothesis_generator_node,
    DuplicateHypothesisError,
)
from darwinian.agents.theoretical_critic import theoretical_critic_node
from darwinian.utils.similarity import get_text_embedding


# 最大重试次数常量
MAX_CRITIC_RETRIES = 5   # NOT_NOVEL 触发的内层重试上限
MAX_MINER_RETRIES = 3    # MATH_ERROR 触发的外层重试上限


def preprocess_node(state: ResearchState) -> dict:
    """初始化节点：检查预算、重置本轮计数器"""
    if state.budget_state.is_exhausted:
        raise RuntimeError("预算耗尽，无法启动新一轮研究")

    return {
        "outer_loop_count": state.outer_loop_count + 1,
        "critic_verdict": None,
        "critic_feedback": "",
        "current_hypothesis": None,
    }


def _safe_bottleneck_miner(state: ResearchState, llm: BaseChatModel) -> dict:
    """带重试计数的 Agent 1 包装"""
    return bottleneck_miner_node(state, llm)


def _safe_hypothesis_generator(state: ResearchState, llm: BaseChatModel) -> dict:
    """Agent 2 包装，捕获 DuplicateHypothesisError 并记录到 failed_ledger"""
    try:
        return hypothesis_generator_node(state, llm)
    except DuplicateHypothesisError as e:
        # 将重复方案记录到 failed_ledger，触发重新生成
        hypothesis = state.current_hypothesis
        feature_vector = []
        if hypothesis:
            text = f"{hypothesis.core_problem}"
            feature_vector = get_text_embedding(text)

        record = FailedRecord(
            feature_vector=feature_vector,
            error_summary=f"方案重复（相似度 {e.similarity:.3f}）: {e.matched_record_summary}",
            failure_type="NOT_NOVEL",
            iteration=state.outer_loop_count,
            banned_keywords=[],
        )
        updated_ledger = list(state.failed_ledger) + [record]
        return {
            "failed_ledger": updated_ledger,
            "critic_verdict": CriticVerdict.NOT_NOVEL,
            "critic_feedback": str(e),
        }


def _safe_theoretical_critic(state: ResearchState, llm: BaseChatModel) -> dict:
    return theoretical_critic_node(state, llm)


def critic_router(state: ResearchState) -> Literal["hypothesis_generator", "bottleneck_miner", "__end__"]:
    """
    条件路由函数：根据 critic_verdict 决定下一步。

    - NOT_NOVEL → 退回 Agent 2 重新生成（内层循环）
    - MATH_ERROR → 写入 failed_ledger，退回 Agent 1（大循环）
    - PASS → 结束 Phase 1
    """
    verdict = state.critic_verdict

    # 检查外层循环上限
    if state.outer_loop_count > state.max_outer_loops:
        return "__end__"

    if verdict == CriticVerdict.PASS:
        return "__end__"
    elif verdict == CriticVerdict.NOT_NOVEL:
        return "hypothesis_generator"
    elif verdict == CriticVerdict.MATH_ERROR:
        return "bottleneck_miner"
    else:
        # 兜底：重新生成假设
        return "hypothesis_generator"


def write_math_error_to_ledger(state: ResearchState) -> dict:
    """
    MATH_ERROR 路径上的中间节点：将失败原因写入 failed_ledger，然后流回 Agent 1。
    """
    hypothesis = state.current_hypothesis
    if hypothesis is None:
        return {}

    # 生成语义向量
    text = f"{hypothesis.core_problem} " + " ".join(
        b.algorithm_logic for b in hypothesis.abstraction_tree
    )
    feature_vector = get_text_embedding(text)

    record = FailedRecord(
        feature_vector=feature_vector,
        error_summary=f"数学错误: {state.critic_feedback}",
        failure_type="MATH_ERROR",
        iteration=state.outer_loop_count,
        banned_keywords=[],  # 由 theoretical_critic 返回，此处简化
    )

    return {
        "failed_ledger": list(state.failed_ledger) + [record],
        "current_hypothesis": None,
        "critic_verdict": None,
    }


def build_hypothesis_graph(llm: BaseChatModel) -> StateGraph:
    """
    构建 Phase 1 子图。

    Args:
        llm: 供所有 Agent 使用的 LLM 实例

    Returns:
        编译后的 LangGraph 子图
    """
    graph = StateGraph(ResearchState)

    # 注册节点（使用 partial 绑定 LLM）
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("bottleneck_miner", partial(_safe_bottleneck_miner, llm=llm))
    graph.add_node("hypothesis_generator", partial(_safe_hypothesis_generator, llm=llm))
    graph.add_node("theoretical_critic", partial(_safe_theoretical_critic, llm=llm))
    graph.add_node("write_math_error", write_math_error_to_ledger)

    # 固定边
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "bottleneck_miner")
    graph.add_edge("bottleneck_miner", "hypothesis_generator")
    graph.add_edge("hypothesis_generator", "theoretical_critic")
    graph.add_edge("write_math_error", "bottleneck_miner")

    # 条件路由边
    graph.add_conditional_edges(
        "theoretical_critic",
        critic_router,
        {
            "hypothesis_generator": "hypothesis_generator",
            "bottleneck_miner": "write_math_error",  # 先写 ledger，再回 Agent 1
            "__end__": END,
        },
    )

    return graph.compile()
