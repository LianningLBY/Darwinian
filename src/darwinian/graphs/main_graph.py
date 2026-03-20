"""
主图入口 (Main Orchestration Graph)

将 Phase 1 和 Phase 2 子图串联，实现完整的大循环控制逻辑：

  START
    → phase1_graph (假设生成子图)
    → phase1_result_router
      ├── phase1_incomplete (预算耗尽或循环超限) → END
      └── phase1_complete → phase2_graph (实验验证子图)
    → phase2_result_router
      ├── phase2_insufficient / robustness_fail → phase1_graph (大循环重启)
      └── publish_ready → save_results → END

用法示例：
    from darwinian.graphs.main_graph import build_main_graph
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-opus-4-6")
    graph = build_main_graph(llm)
    result = graph.invoke({"research_direction": "图神经网络在药物发现中的应用"})
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from darwinian.state import ResearchState, FinalVerdict
from darwinian.graphs.hypothesis_graph import build_hypothesis_graph
from darwinian.graphs.experiment_graph import build_experiment_graph


# ---------------------------------------------------------------------------
# 中间节点
# ---------------------------------------------------------------------------

def phase1_result_router(
    state: ResearchState,
) -> Literal["phase2", "end_budget_exhausted"]:
    """
    Phase 1 结束后的路由：
    - 假设生成成功（有 selected_branch）→ 进入 Phase 2
    - 预算耗尽或循环超限 → 终止
    """
    if state.budget_state.is_exhausted:
        return "end_budget_exhausted"

    if state.outer_loop_count > state.max_outer_loops:
        return "end_budget_exhausted"

    if (
        state.current_hypothesis is not None
        and state.current_hypothesis.selected_branch is not None
    ):
        return "phase2"

    # 记录具体原因，便于诊断（通过 print 在线程中输出，会出现在 stderr）
    if state.current_hypothesis is None:
        print("[phase1_router] current_hypothesis is None → end_budget_exhausted")
    else:
        branches = len(state.current_hypothesis.abstraction_tree)
        selected = state.current_hypothesis.selected_branch
        print(f"[phase1_router] selected_branch={selected} abstraction_tree.len={branches} → end_budget_exhausted")

    return "end_budget_exhausted"


def phase2_result_router(
    state: ResearchState,
) -> Literal["restart_phase1", "end_publish_ready", "end_failed"]:
    """
    Phase 2 结束后的路由：
    - PUBLISH_READY → 输出结果，终止
    - 失败（insufficient / robustness_fail）且未超限 → 重启 Phase 1（大循环）
    - 失败且超限 → 终止
    """
    if state.final_verdict == FinalVerdict.PUBLISH_READY:
        return "end_publish_ready"

    # Phase 2 以某种失败终止（failed_ledger 已被写入）
    if state.outer_loop_count >= state.max_outer_loops:
        return "end_failed"

    return "restart_phase1"


def save_results_node(state: ResearchState) -> dict:
    """
    将最终研究报告写入 results/ 目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / f"report_{timestamp}.md"
    report_path.write_text(state.final_report, encoding="utf-8")

    metrics_path = output_dir / f"metrics_{timestamp}.json"
    metrics = {
        "publish_matrix": state.publish_matrix.model_dump(),
        "baseline_metrics": state.experiment_result.baseline_metrics if state.experiment_result else {},
        "proposed_metrics": state.experiment_result.proposed_metrics if state.experiment_result else {},
        "robustness_metrics": state.robustness_result.perturbed_metrics if state.robustness_result else {},
        "outer_loops": state.outer_loop_count,
        "failed_ledger_size": len(state.failed_ledger),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ 研究完成！报告已保存至 {report_path}")
    return {}


def log_termination_node(state: ResearchState) -> dict:
    """记录提前终止的原因"""
    if state.budget_state.is_exhausted:
        print("⚠️  预算耗尽，研究终止。")
    elif state.outer_loop_count >= state.max_outer_loops:
        print(f"⚠️  已达最大循环次数 ({state.max_outer_loops})，研究终止。")
    else:
        print("⚠️  研究在达到发表标准前终止。")

    if state.failed_ledger:
        print(f"📋 认知账本记录了 {len(state.failed_ledger)} 条失败经验。")
    return {}


# ---------------------------------------------------------------------------
# 主图构建
# ---------------------------------------------------------------------------

def build_main_graph(llm: BaseChatModel) -> StateGraph:
    """
    构建完整的主图（Phase 1 + Phase 2 大循环）。

    Args:
        llm: 所有 Agent 共享的 LLM 实例

    Returns:
        编译后的完整 LangGraph
    """
    # 构建子图
    hypothesis_graph = build_hypothesis_graph(llm)
    experiment_graph = build_experiment_graph(llm)

    graph = StateGraph(ResearchState)

    # 将编译后的子图作为节点注册
    graph.add_node("phase1", hypothesis_graph)
    graph.add_node("phase2", experiment_graph)
    graph.add_node("save_results", save_results_node)
    graph.add_node("log_termination", log_termination_node)

    # 固定边
    graph.add_edge(START, "phase1")
    graph.add_edge("save_results", END)
    graph.add_edge("log_termination", END)

    # 条件路由边：Phase 1 出口
    graph.add_conditional_edges(
        "phase1",
        phase1_result_router,
        {
            "phase2": "phase2",
            "end_budget_exhausted": "log_termination",
        },
    )

    # 条件路由边：Phase 2 出口
    graph.add_conditional_edges(
        "phase2",
        phase2_result_router,
        {
            "restart_phase1": "phase1",    # 大循环回路
            "end_publish_ready": "save_results",
            "end_failed": "log_termination",
        },
    )

    return graph.compile()
