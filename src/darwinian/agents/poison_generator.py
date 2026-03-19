"""
Agent 6: 毒药数据生成器 (poison_generator_node)

职责：
- 基于固定扰动策略库（非完全开放生成）选择组合
- 根据方法逻辑弱点选择最有针对性的策略
- 生成扰动代码用于破坏性测试（OOD / Robustness Test）

设计说明：
  按照可行性分析的建议，Agent 6 的职责被缩窄为从固定策略库中「选择+组合」，
  而非完全开放地生成扰动策略，以保证鲁棒性测试的可量化性和可复现性。
"""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState
from darwinian.tools.perturbation_strategies import (
    STRATEGY_REGISTRY,
    PerturbationStrategy,
    generate_perturbation_code,
)


SYSTEM_PROMPT = """你是一个对抗性测试专家。你的任务是针对给定方法的逻辑弱点，从策略库中选择最有针对性的扰动组合。

可用扰动策略库：
{strategy_list}

选择原则：
1. 分析方法的核心假设，选择能打破该假设的策略
2. 至少选择 2 种策略，最多 4 种
3. 优先选择能暴露方法「本质缺陷」的策略，而非随机选择

输出格式（严格 JSON）：
{
  "selected_strategies": ["strategy_name_1", "strategy_name_2", ...],
  "rationale": "选择这些策略的理由，与方法弱点的对应关系",
  "severity": "low" | "medium" | "high"
}

禁止输出 JSON 以外的任何内容。"""


def poison_generator_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 6 节点函数。
    输入: current_hypothesis (selected_branch) + experiment_result (初步成功)
    输出: 生成扰动代码，存入 experiment_code 的 poison_code 字段（临时使用 dataset_loader_code）
    """
    if state.current_hypothesis is None or state.current_hypothesis.selected_branch is None:
        raise ValueError("poison_generator_node 调用前必须有已选定的假设")

    branch = state.current_hypothesis.selected_branch

    # 构建策略列表供 LLM 参考
    strategy_list = "\n".join(
        f"- {name}: {s.description}" for name, s in STRATEGY_REGISTRY.items()
    )

    system_prompt = SYSTEM_PROMPT.format(strategy_list=strategy_list)

    user_message = f"""方法逻辑：
- 名称：{branch.name}
- 算法：{branch.algorithm_logic}
- 数学基础：{branch.math_formulation}
- 来源领域：{branch.source_domain}

初步实验指标（需要破坏性测试）：
{json.dumps(state.experiment_result.proposed_metrics if state.experiment_result else {}, ensure_ascii=False)}

请选择最能暴露该方法弱点的扰动策略组合。"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    raw = json.loads(response.content)
    selected_names: list[str] = raw["selected_strategies"]

    # 验证策略名称合法性
    valid_strategies: list[PerturbationStrategy] = []
    for name in selected_names:
        if name in STRATEGY_REGISTRY:
            valid_strategies.append(STRATEGY_REGISTRY[name])

    if not valid_strategies:
        # 回退到默认策略组合
        valid_strategies = [STRATEGY_REGISTRY["gaussian_noise"], STRATEGY_REGISTRY["label_flip"]]

    # 生成扰动代码
    poison_code = generate_perturbation_code(
        strategies=valid_strategies,
        dataset_schema=state.dataset_schema,
    )

    # 将策略信息存入 poison_test_result（部分初始化）
    from darwinian.state import PoisonTestResult
    poison_result = PoisonTestResult(
        perturbation_strategy=", ".join(s.name for s in valid_strategies),
        perturbed_metrics={},
        degradation_rate=0.0,
    )

    # 暂时将 poison_code 注入 experiment_code（后续由 code_execute 执行）
    updated_code = state.experiment_code.model_copy(
        update={"dataset_loader_code": poison_code}
    ) if state.experiment_code else None

    return {
        "experiment_code": updated_code,
        "poison_test_result": poison_result,
        "messages": [response],
    }
