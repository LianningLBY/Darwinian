"""
Agent 6: 鲁棒性测试生成器 (robustness_generator_node)

职责：
- 分析方法的核心假设，选择最有针对性的扰动策略组合
- 从固定策略库中选择，保证测试可量化、可复现
- 生成鲁棒性测试代码，覆盖：分布偏移、标签噪声、特征缺失、对抗扰动、稀疏数据等维度

（注：函数名保持 poison_generator_node 向后兼容，对外接口不变）
"""

from __future__ import annotations

import json
from darwinian.utils.json_parser import parse_llm_json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState
from darwinian.tools.perturbation_strategies import (
    STRATEGY_REGISTRY,
    PerturbationStrategy,
    generate_perturbation_code,
)


SYSTEM_PROMPT = """你是一个鲁棒性测试专家。你的任务是针对给定方法的核心假设，从策略库中选择最有针对性的扰动组合。

可用扰动策略库：
{strategy_list}

选择原则：
1. 分析方法的核心假设，选择能打破该假设的策略
2. 必须覆盖至少两个不同维度（例如：数据质量 + 分布偏移，或标签噪声 + 对抗扰动）
3. 至少选择 3 种策略，最多 5 种
4. 优先选择能暴露方法「本质脆弱点」的策略

输出格式（严格 JSON）：
{{
  "selected_strategies": ["strategy_name_1", "strategy_name_2", ...],
  "rationale": "选择理由与方法弱点的对应关系",
  "severity": "low" | "medium" | "high",
  "test_dimensions": ["数据质量", "分布偏移", ...]
}}

禁止输出 JSON 以外的任何内容。"""


def poison_generator_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 6 节点函数（兼容旧名称）。
    输入: current_hypothesis + experiment_result（初步成功）
    输出: 生成鲁棒性测试代码，存入 experiment_code.robustness_code
    """
    if state.current_hypothesis is None or state.current_hypothesis.selected_branch is None:
        raise ValueError("robustness_generator_node 调用前必须有已选定的假设")

    branch = state.current_hypothesis.selected_branch

    strategy_list = "\n".join(
        f"- {name}: {s.description}" for name, s in STRATEGY_REGISTRY.items()
    )

    system_prompt = SYSTEM_PROMPT.format(strategy_list=strategy_list)

    user_message = f"""方法逻辑：
- 名称：{branch.name}
- 算法：{branch.algorithm_logic}
- 数学基础：{branch.math_formulation}
- 来源领域：{branch.source_domain}

初步实验指标（需鲁棒性验证）：
{json.dumps(state.experiment_result.proposed_metrics if state.experiment_result else {}, ensure_ascii=False)}

请分析该方法在哪些条件下最可能失效，选择最有针对性的扰动策略。"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    raw = parse_llm_json(response.content)
    selected_names: list[str] = raw.get("selected_strategies", [])

    valid_strategies: list[PerturbationStrategy] = [
        STRATEGY_REGISTRY[name] for name in selected_names if name in STRATEGY_REGISTRY
    ]

    if not valid_strategies:
        # 兜底：覆盖数据质量 + 分布偏移两个维度
        valid_strategies = [
            STRATEGY_REGISTRY["gaussian_noise"],
            STRATEGY_REGISTRY["label_flip"],
            STRATEGY_REGISTRY["ood_distribution_shift"],
        ]

    robustness_code = generate_perturbation_code(
        strategies=valid_strategies,
        dataset_schema=state.dataset_schema,
    )

    from darwinian.state import RobustnessResult
    robustness_result = RobustnessResult(
        perturbation_strategy=", ".join(s.name for s in valid_strategies),
        perturbed_metrics={},
        degradation_rate=0.0,
    )

    updated_code = state.experiment_code.model_copy(
        update={"robustness_code": robustness_code}
    ) if state.experiment_code else None

    return {
        "experiment_code": updated_code,
        "robustness_result": robustness_result,
        "messages": [response],
    }
