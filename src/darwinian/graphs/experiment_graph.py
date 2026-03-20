"""
Phase 2: 自动实验与鲁棒性验证子图 (Experiment Graph)

节点流程：
  Phase 1 传入
    → dataset_router → dataset_finder
    → code_architect（生成 baseline + proposed + ablation）
    → code_execute（隔离执行 baseline + proposed）
    → diagnostician（诊断结果）
    → execution_router
      ├── code_error → code_architect（内层循环，最多 5 次）
      ├── insufficient → write_insufficient → END
      └── success → ablation_execute（执行消融实验）
    → robustness_generator（选择扰动策略）
    → robustness_execute（执行鲁棒性测试）
    → publish_evaluator（三审稿人模拟 + 终局裁决）
    → final_router
      ├── robustness_fail → write_robustness_fail → END
      └── publish_ready → END
"""

from __future__ import annotations

import json
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
    RobustnessResult,
)
from darwinian.agents.code_architect import code_architect_node
from darwinian.agents.diagnostician import diagnostician_node
from darwinian.agents.poison_generator import poison_generator_node as robustness_generator_node
from darwinian.agents.publish_evaluator import publish_evaluator_node
from darwinian.tools.code_executor import code_execute
from darwinian.tools.dataset_finder import dataset_finder_node
from darwinian.utils.similarity import get_text_embedding


MAX_CODE_RETRIES = 5


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------

def dataset_router_node(state: ResearchState) -> dict:
    if not state.dataset_schema:
        return {"dataset_schema": {"type": "unknown", "note": "由 dataset_finder 自动确定"}}
    return {}


def code_execute_node(state: ResearchState) -> dict:
    """执行完整实验代码（Docker 优先，降级 subprocess）"""
    if state.experiment_code is None:
        raise ValueError("code_execute_node 调用前必须先运行 code_architect_node")

    result = code_execute(
        experiment_code=state.experiment_code,
        mode="full",
        data_dir=state.user_data_path or None,
    )

    if not result.baseline_metrics and not result.proposed_metrics and result.stderr:
        result = result.model_copy(update={"execution_verdict": ExecutionVerdict.CODE_ERROR})

    return {"experiment_result": result}


def ablation_execute_node(state: ResearchState) -> dict:
    """执行消融实验代码，解析各变体指标"""
    if state.experiment_code is None or not state.experiment_code.ablation_code:
        return {"ablation_results": {}}

    # 将 ablation_code 临时写入 dataset_loader_code 位置复用执行逻辑
    ablation_exp_code = state.experiment_code.model_copy(
        update={"baseline_code": state.experiment_code.ablation_code,
                "proposed_code": "# ablation only"}
    )

    # 直接用 subprocess 运行消融代码（不需要 Docker 隔离）
    import tempfile, os, subprocess, sys, json as _json
    ablation_results: dict[str, dict] = {}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            abl_path = os.path.join(tmpdir, "ablation.py")
            with open(abl_path, "w") as f:
                f.write(state.experiment_code.ablation_code)
            proc = subprocess.run(
                [sys.executable, abl_path],
                capture_output=True, text=True, timeout=180, cwd=tmpdir,
            )
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = _json.loads(line)
                    model_name = obj.get("model", "")
                    if model_name.startswith("ablation_"):
                        ablation_results[model_name] = obj.get("metrics", {})
                except Exception:
                    continue
    except Exception:
        pass

    return {"ablation_results": ablation_results}


def robustness_execute_node(state: ResearchState) -> dict:
    """执行鲁棒性测试代码"""
    if state.experiment_code is None:
        raise ValueError("robustness_execute_node 调用前必须先运行 robustness_generator_node")

    result = code_execute(
        experiment_code=state.experiment_code,
        mode="poison",   # 复用 poison 执行路径（读取 robustness_code）
        data_dir=state.user_data_path or None,
    )

    proposed_original = state.experiment_result.proposed_metrics if state.experiment_result else {}
    perturbed = result.proposed_metrics

    if not perturbed and result.stderr:
        degradation = 1.0
    else:
        degradation = _compute_degradation(proposed_original, perturbed)

    base = state.robustness_result
    if base is not None:
        updated = base.model_copy(update={
            "perturbed_metrics": perturbed,
            "degradation_rate": degradation,
        })
    else:
        updated = RobustnessResult(
            perturbation_strategy="unknown",
            perturbed_metrics=perturbed,
            degradation_rate=degradation,
        )

    return {
        "experiment_result": result,
        "robustness_result": updated,
    }


def write_insufficient_to_ledger(state: ResearchState) -> dict:
    hypothesis = state.current_hypothesis
    if hypothesis is None:
        return {}

    branch = hypothesis.selected_branch
    text = f"{hypothesis.core_problem} {branch.algorithm_logic if branch else ''}"
    feature_vector = get_text_embedding(text)

    diagnosis = state.experiment_result.diagnosis if state.experiment_result else "方法效果不足基准"
    banned_kw = state.last_error_keywords or _extract_method_keywords(branch.name if branch else "")
    record = FailedRecord(
        feature_vector=feature_vector,
        error_summary=diagnosis,
        failure_type="INSUFFICIENT",
        iteration=state.outer_loop_count,
        banned_keywords=banned_kw,
    )
    return {
        "failed_ledger": list(state.failed_ledger) + [record],
        "last_error_keywords": [],
    }


def write_robustness_fail_to_ledger(state: ResearchState) -> dict:
    hypothesis = state.current_hypothesis
    if hypothesis is None:
        return {}

    branch = hypothesis.selected_branch
    text = f"{hypothesis.core_problem} {branch.algorithm_logic if branch else ''}"
    feature_vector = get_text_embedding(text)

    degradation = state.robustness_result.degradation_rate if state.robustness_result else 0.0
    banned_kw = state.last_error_keywords or _extract_method_keywords(branch.name if branch else "")
    record = FailedRecord(
        feature_vector=feature_vector,
        error_summary=f"鲁棒性测试失败（性能下降 {degradation:.1%}）",
        failure_type="ROBUSTNESS_FAIL",
        iteration=state.outer_loop_count,
        banned_keywords=banned_kw,
    )
    return {
        "failed_ledger": list(state.failed_ledger) + [record],
        "last_error_keywords": [],
    }


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------

def execution_router(
    state: ResearchState,
) -> Literal["code_architect", "write_insufficient", "ablation_execute"]:
    if state.experiment_result is None:
        return "code_architect"

    verdict = state.experiment_result.execution_verdict

    if verdict == ExecutionVerdict.CODE_ERROR:
        retry_count = state.experiment_code.retry_count if state.experiment_code else 0
        if retry_count >= MAX_CODE_RETRIES:
            return "write_insufficient"
        return "code_architect"
    elif verdict == ExecutionVerdict.INSUFFICIENT:
        return "write_insufficient"
    elif verdict == ExecutionVerdict.SUCCESS:
        return "ablation_execute"

    return "code_architect"


def final_router(state: ResearchState) -> Literal["write_robustness_fail", "__end__"]:
    if state.final_verdict == FinalVerdict.PUBLISH_READY:
        return "__end__"
    return "write_robustness_fail"


# ---------------------------------------------------------------------------
# 子图构建
# ---------------------------------------------------------------------------

def build_experiment_graph(llm: BaseChatModel) -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("dataset_router",       dataset_router_node)
    graph.add_node("dataset_finder",       partial(dataset_finder_node, llm=llm))
    graph.add_node("code_architect",       partial(code_architect_node, llm=llm))
    graph.add_node("code_execute",         code_execute_node)
    graph.add_node("diagnostician",        partial(diagnostician_node, llm=llm))
    graph.add_node("write_insufficient",   write_insufficient_to_ledger)
    graph.add_node("ablation_execute",     ablation_execute_node)
    graph.add_node("robustness_generator", partial(robustness_generator_node, llm=llm))
    graph.add_node("robustness_execute",   robustness_execute_node)
    graph.add_node("publish_evaluator",    partial(publish_evaluator_node, llm=llm))
    graph.add_node("write_robustness_fail", write_robustness_fail_to_ledger)

    # 固定边
    graph.add_edge(START,                "dataset_router")
    graph.add_edge("dataset_router",     "dataset_finder")
    graph.add_edge("dataset_finder",     "code_architect")
    graph.add_edge("code_architect",     "code_execute")
    graph.add_edge("code_execute",       "diagnostician")
    graph.add_edge("write_insufficient", END)
    graph.add_edge("ablation_execute",   "robustness_generator")
    graph.add_edge("robustness_generator", "robustness_execute")
    graph.add_edge("robustness_execute", "publish_evaluator")
    graph.add_edge("write_robustness_fail", END)

    # 条件路由
    graph.add_conditional_edges(
        "diagnostician", execution_router,
        {
            "code_architect":    "code_architect",
            "write_insufficient": "write_insufficient",
            "ablation_execute":  "ablation_execute",
        },
    )
    graph.add_conditional_edges(
        "publish_evaluator", final_router,
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
    if not original or not perturbed:
        return 0.0
    for key in original:
        orig_key = key.replace("_mean", "")
        if orig_key in perturbed and original[key] != 0:
            return max(0.0, (original[key] - perturbed[orig_key]) / abs(original[key]))
        if key in perturbed and original[key] != 0:
            return max(0.0, (original[key] - perturbed[key]) / abs(original[key]))
    return 0.0


def _extract_method_keywords(method_name: str) -> list[str]:
    import re
    words = re.findall(r"[a-z0-9\u4e00-\u9fff]+", method_name.lower())
    return [w for w in words if len(w) > 2]
