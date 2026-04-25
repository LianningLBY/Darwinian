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
from darwinian.agents.proposal_elaborator import proposal_elaborator_node
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
        "hypothesis_retry_count": 0,   # 重置内层重试计数（NOT_NOVEL）
        "miner_retry_count": 0,        # 重置 MATH_ERROR 重试计数
        # 重置实验代码/结果，避免 retry_count 跨外层循环累积
        # 不重置会导致：上轮 retry_count=5 → 新轮第一次 code_error 就被判为 INSUFFICIENT
        "experiment_code": None,
        "experiment_result": None,
    }


def _safe_bottleneck_miner(state: ResearchState, llm: BaseChatModel) -> dict:
    """带重试计数的 Agent 1 包装"""
    return bottleneck_miner_node(state, llm)


def _safe_hypothesis_generator(state: ResearchState, llm: BaseChatModel) -> dict:
    """Agent 2 包装，捕获 DuplicateHypothesisError 并记录到 failed_ledger"""
    try:
        result = hypothesis_generator_node(state, llm)
        # 每次调用 Agent 2 都自增内层重试计数（含初次生成）
        result["hypothesis_retry_count"] = state.hypothesis_retry_count + 1
        return result
    except DuplicateHypothesisError as e:
        # 将重复方案记录到 failed_ledger，触发重新生成
        hypothesis = state.current_hypothesis
        feature_vector = []
        banned_kw: list[str] = []
        if hypothesis:
            text = f"{hypothesis.core_problem}"
            feature_vector = get_text_embedding(text)
            # 从方案名中提取关键词，避免下一轮重复选择同一方向
            import re as _re
            banned_kw = [w for w in _re.findall(r"[a-z\u4e00-\u9fff]{3,}", text.lower())][:5]
            # v2: \u628a\u6bcf\u4e2a branch \u7684 cited_entity_names \u4e5f\u52a0\u8fdb\u53bb\u2014\u2014
            # \u8fd9\u4e9b\u662f\u7cbe\u786e\u672f\u8bed\uff0c\u6bd4\u6b63\u5219\u63d0\u53d6\u7684\u788e\u7247\u6709\u7528\u5f97\u591a
            for branch in hypothesis.abstraction_tree:
                banned_kw.extend(branch.cited_entity_names)
            banned_kw = list(dict.fromkeys(banned_kw))[:15]   # \u53bb\u91cd\u4fdd\u5e8f\uff0c\u6700\u591a 15 \u4e2a

        record = FailedRecord(
            feature_vector=feature_vector,
            error_summary=f"方案重复（相似度 {e.similarity:.3f}）: {e.matched_record_summary}",
            failure_type="NOT_NOVEL",
            iteration=state.outer_loop_count,
            banned_keywords=banned_kw,
        )
        updated_ledger = list(state.failed_ledger) + [record]
        return {
            "failed_ledger": updated_ledger,
            "current_hypothesis": None,   # 必须清空，否则下轮仍拿旧假设判重，形成死循环
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

    if verdict == CriticVerdict.PASS:
        return "__end__"
    elif verdict == CriticVerdict.NOT_NOVEL:
        # 用独立内层计数器限制 NOT_NOVEL 重试，不再误用外层循环计数
        if state.hypothesis_retry_count >= MAX_CRITIC_RETRIES:
            return "__end__"   # 内层耗尽，放弃本轮假设（selected_branch=None → phase1_router 终止）
        return "hypothesis_generator"
    elif verdict == CriticVerdict.MATH_ERROR:
        # 用独立计数器限制 MATH_ERROR 重试，防止 Agent 2 持续返回空树导致无限循环
        if state.miner_retry_count >= MAX_MINER_RETRIES:
            return "__end__"  # MATH_ERROR 耗尽，放弃本轮（selected_branch=None → phase1_router 终止）
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
        banned_keywords=state.last_error_keywords,  # 来自 theoretical_critic 写入 state
    )

    return {
        "failed_ledger": list(state.failed_ledger) + [record],
        "current_hypothesis": None,
        "critic_verdict": None,
        "last_error_keywords": [],  # 清空，避免污染下一轮
        "miner_retry_count": state.miner_retry_count + 1,  # 累计 MATH_ERROR 重试次数
    }


def build_hypothesis_graph(
    llm: BaseChatModel,
    *,
    elaborate_proposals: bool = False,
    gpu_hours_budget: float = 168.0,
    target_venues: list[dict] | None = None,
) -> StateGraph:
    """
    构建 Phase 1 子图。

    Args:
        llm: 供所有 Agent 使用的 LLM 实例
        elaborate_proposals: 若 True，在 hypothesis_generator 之后、theoretical_critic
            之前插入 Agent 2.5 (proposal_elaborator)，把每个 branch 展开成完整
            ResearchProposal。默认 False——展开会显著增加 LLM 调用成本（每 branch 一次），
            主大循环建议关闭，只在 standalone "出 seed" 场景启用。
        gpu_hours_budget: elaborate=True 时透传给 elaborator 校验 phase 总耗时
        target_venues: elaborate=True 时透传给 elaborator 选 deadline

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
    graph.add_edge("write_math_error", "bottleneck_miner")

    # 可选：Agent 2.5 proposal_elaborator
    if elaborate_proposals:
        graph.add_node(
            "proposal_elaborator",
            partial(
                proposal_elaborator_node,
                llm=llm,
                gpu_hours_budget=gpu_hours_budget,
                target_venues=target_venues,
            ),
        )
        graph.add_edge("hypothesis_generator", "proposal_elaborator")
        graph.add_edge("proposal_elaborator", "theoretical_critic")
    else:
        graph.add_edge("hypothesis_generator", "theoretical_critic")

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
