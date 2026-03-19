"""
Agent 7: 成果验收员 (publish_evaluator_node)

职责：
- 终局裁判
- 审查包含 Baseline、新方法、毒药数据的完整对比指标
- 更新 publish_matrix（四项全绿才能 END）
- 生成最终研究报告 results.md
"""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState, PublishMatrix, FinalVerdict


SYSTEM_PROMPT = """你是一位严格的学术论文成果验收专家。你的任务是根据完整的实验数据，判断结果是否达到发表水准。

四项评判标准：
1. **新颖性（novelty_passed）**: 方法思路是否具有独创性，不是已有方法的简单变体
2. **基准提升（baseline_improved）**: 提出方法在主要指标上是否显著优于基准方法（提升 > 1%）
3. **鲁棒性（robustness_passed）**: 在毒药数据（扰动数据）下，性能下降比例 < 30%
4. **可解释性（explainability_generated）**: 是否能用直观语言解释为什么该方法有效

输出格式（严格 JSON）：
{
  "novelty_passed": true | false,
  "baseline_improved": true | false,
  "robustness_passed": true | false,
  "explainability_generated": true | false,
  "verdict": "publish_ready" | "robustness_fail",
  "report_markdown": "完整的研究报告 Markdown 文本（仅在 publish_ready 时生成）",
  "rejection_reason": "如果失败，说明最主要的失败原因（用于写入 failed_ledger）",
  "error_keywords": ["失败关键词"]
}

注意：
- 如果 robustness_passed 为 false，verdict 必须为 robustness_fail
- report_markdown 需包含：方法描述、实验设置、结果对比表、结论
禁止输出 JSON 以外的任何内容。"""


def publish_evaluator_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 7 节点函数。
    输入: experiment_result + poison_test_result + current_hypothesis
    输出: 更新 publish_matrix、final_verdict、final_report
    """
    if state.experiment_result is None:
        raise ValueError("publish_evaluator_node 调用前必须有实验结果")

    hypothesis = state.current_hypothesis
    branch = hypothesis.selected_branch if hypothesis else None

    user_message = f"""方法描述：
{f"- 名称：{branch.name}" if branch else "（未知）"}
{f"- 算法：{branch.algorithm_logic}" if branch else ""}

实验结果对比：
基准方法指标：{json.dumps(state.experiment_result.baseline_metrics, ensure_ascii=False)}
提出方法指标：{json.dumps(state.experiment_result.proposed_metrics, ensure_ascii=False)}

鲁棒性测试（毒药数据）：
扰动策略：{state.poison_test_result.perturbation_strategy if state.poison_test_result else "未执行"}
扰动后指标：{json.dumps(state.poison_test_result.perturbed_metrics if state.poison_test_result else {}, ensure_ascii=False)}
性能下降比例：{f"{state.poison_test_result.degradation_rate:.1%}" if state.poison_test_result else "N/A"}

文献新颖性支撑：
{chr(10).join(hypothesis.literature_support) if hypothesis else "无"}

请进行终局裁决。"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw = json.loads(response.content)

    publish_matrix = PublishMatrix(
        novelty_passed=raw["novelty_passed"],
        baseline_improved=raw["baseline_improved"],
        robustness_passed=raw["robustness_passed"],
        explainability_generated=raw["explainability_generated"],
    )
    final_verdict = FinalVerdict(raw["verdict"])

    return {
        "publish_matrix": publish_matrix,
        "final_verdict": final_verdict,
        "final_report": raw.get("report_markdown", ""),
        "messages": [response],
        # 供路由函数写入 failed_ledger
        "_rejection_reason": raw.get("rejection_reason", ""),
        "_error_keywords": raw.get("error_keywords", []),
    }
