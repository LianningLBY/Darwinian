"""
Agent 7: 成果验收员 (publish_evaluator_node)

职责：
- 模拟三位独立审稿人（方法论专家、实验专家、鲁棒性专家）
- 整合消融实验结果，验证各组件贡献
- 综合统计显著性（多 seed 均值/标准差）进行裁决
- 更新 publish_matrix，生成最终研究报告
"""

from __future__ import annotations

import json
from darwinian.utils.json_parser import parse_llm_json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState, PublishMatrix, FinalVerdict
from darwinian.utils.llm_retry import invoke_with_retry


SYSTEM_PROMPT = """你是一个学术论文终审委员会，由三位独立审稿人组成：
- 审稿人 A（方法论专家）：评估新颖性与理论严谨性
- 审稿人 B（实验专家）：评估实验设计、统计显著性与消融实验完整性
- 审稿人 C（鲁棒性专家）：评估鲁棒性测试的覆盖面与结果可信度

评判标准：
1. novelty_passed：方法具有独创性，不是已有方法简单变体
2. baseline_improved：多 seed 均值下，提出方法显著优于基准（均值提升 > 1%，且标准差不重叠）
3. robustness_passed：鲁棒性测试中性能下降 < 30%，且至少覆盖 2 种扰动维度
4. ablation_verified：消融实验证明各关键组件均有正向贡献（去掉任一组件性能下降）
5. explainability_generated：能用直观语言解释为什么该方法有效

输出格式（严格 JSON）：
{
  "reviewer_a": {"score": 1-5, "comment": "方法论评语"},
  "reviewer_b": {"score": 1-5, "comment": "实验评语"},
  "reviewer_c": {"score": 1-5, "comment": "鲁棒性评语"},
  "novelty_passed": true | false,
  "baseline_improved": true | false,
  "robustness_passed": true | false,
  "ablation_verified": true | false,
  "explainability_generated": true | false,
  "verdict": "publish_ready" | "robustness_fail",
  "report_markdown": "完整研究报告（仅 publish_ready 时生成，包含：摘要、方法、实验表格、消融分析、鲁棒性分析、结论）",
  "rejection_reason": "主要拒稿原因（失败时填写）",
  "error_keywords": ["失败关键词"]
}

注意：三位审稿人平均分 < 3 或任一关键项为 false，verdict 必须为 robustness_fail。
禁止输出 JSON 以外的任何内容。"""


def publish_evaluator_node(state: ResearchState, llm: BaseChatModel) -> dict:
    if state.experiment_result is None:
        raise ValueError("publish_evaluator_node 调用前必须有实验结果")

    hypothesis = state.current_hypothesis
    branch = hypothesis.selected_branch if hypothesis else None

    # 整理消融实验摘要
    ablation_summary = "（未执行）"
    if state.ablation_results:
        lines = []
        for variant, metrics in state.ablation_results.items():
            lines.append(f"  - {variant}: {json.dumps(metrics, ensure_ascii=False)}")
        ablation_summary = "\n".join(lines)

    # 整理多 seed 统计
    baseline_m = state.experiment_result.baseline_metrics
    proposed_m = state.experiment_result.proposed_metrics
    stats_note = ""
    for key in set(list(baseline_m.keys()) + list(proposed_m.keys())):
        b_mean = baseline_m.get(f"{key}_mean", baseline_m.get(key, "N/A"))
        b_std  = baseline_m.get(f"{key}_std", "N/A")
        p_mean = proposed_m.get(f"{key}_mean", proposed_m.get(key, "N/A"))
        p_std  = proposed_m.get(f"{key}_std", "N/A")
        stats_note += f"  {key}: baseline={b_mean}±{b_std}, proposed={p_mean}±{p_std}\n"

    robustness = state.robustness_result
    user_message = f"""方法描述：
- 名称：{branch.name if branch else "未知"}
- 算法：{branch.algorithm_logic if branch else ""}

多 seed 实验结果（3 个随机种子均值±标准差）：
{stats_note or json.dumps({"baseline": baseline_m, "proposed": proposed_m}, ensure_ascii=False)}

消融实验（各组件贡献验证）：
{ablation_summary}

鲁棒性测试：
- 扰动策略：{robustness.perturbation_strategy if robustness else "未执行"}
- 扰动后指标：{json.dumps(robustness.perturbed_metrics if robustness else {}, ensure_ascii=False)}
- 性能下降比例：{f"{robustness.degradation_rate:.1%}" if robustness else "N/A"}

文献新颖性支撑：
{chr(10).join(hypothesis.literature_support) if hypothesis else "无"}

请由三位审稿人独立评审并给出终局裁决。"""

    response = invoke_with_retry(llm, [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw = parse_llm_json(response.content)

    # 提取审稿人分数
    peer_scores = {
        "reviewer_a": raw.get("reviewer_a", {}),
        "reviewer_b": raw.get("reviewer_b", {}),
        "reviewer_c": raw.get("reviewer_c", {}),
    }

    publish_matrix = PublishMatrix(
        novelty_passed=raw.get("novelty_passed", False),
        baseline_improved=raw.get("baseline_improved", False),
        robustness_passed=raw.get("robustness_passed", False),
        explainability_generated=raw.get("explainability_generated", False),
    )
    final_verdict = FinalVerdict(raw["verdict"])

    return {
        "publish_matrix": publish_matrix,
        "final_verdict": final_verdict,
        "final_report": raw.get("report_markdown", ""),
        "peer_review_scores": peer_scores,
        "last_error_keywords": raw.get("error_keywords", []),
        "messages": [response],
    }
