"""
Agent 3: 理论审查官 (theoretical_critic_node)

职责：
- 仅审查，不修改方案
- 对数学可行性和新颖性进行审查
- 输出严格的枚举指令: PASS / MATH_ERROR / NOT_NOVEL
"""

from __future__ import annotations

import json
from darwinian.utils.json_parser import parse_llm_json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState, CriticVerdict, AbstractionBranch


SYSTEM_PROMPT = """你是一位严格的理论审查官，专门评估科研方案的数学可行性和新颖性。

审查标准：
1. **数学可行性（MATH_ERROR）**: 方案中的数学公式是否存在明显错误？如维度不匹配、不可微分的操作、违反基本定理等。
2. **新颖性（NOT_NOVEL）**: 方案是否与现有文献中的方法本质相同？注意：改变数据集或超参数不构成新颖性。
3. **通过（PASS）**: 数学上可行且相对新颖，值得实验验证。

输出格式（严格 JSON）：
{
  "verdict": "PASS" | "MATH_ERROR" | "NOT_NOVEL",
  "selected_branch_name": "如果 PASS，填写最优方案分支的 name；否则填 null",
  "feedback": "详细审查意见，MATH_ERROR 时必须指出具体公式错误，NOT_NOVEL 时必须指出对应的已有工作",
  "error_keywords": ["如果失败，提取 2-3 个关键词用于未来过滤"]
}

禁止输出 JSON 以外的任何内容。
重要：你只能输出上述三个值之一，不能输出其他。"""


def theoretical_critic_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 3 节点函数。
    输入: current_hypothesis + 文献检索结果
    输出: 更新 critic_verdict 和 critic_feedback
    """
    if state.current_hypothesis is None:
        raise ValueError("theoretical_critic_node 调用前必须先运行 hypothesis_generator_node")

    hypothesis = state.current_hypothesis
    branches_text = json.dumps(
        [b.model_dump() for b in hypothesis.abstraction_tree],
        ensure_ascii=False,
        indent=2,
    )

    user_message = f"""待审查的研究假设：

核心问题：{hypothesis.core_problem}

解决方案分支：
{branches_text}

文献支撑：
{chr(10).join(hypothesis.literature_support)}

请对上述方案进行严格审查。"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    result = parse_llm_json(response.content)
    verdict = CriticVerdict(result["verdict"])

    # 如果 PASS，选定最优分支
    updated_hypothesis = hypothesis.model_copy()
    if verdict == CriticVerdict.PASS and result.get("selected_branch_name"):
        selected = _find_branch(hypothesis.abstraction_tree, result["selected_branch_name"])
        updated_hypothesis = hypothesis.model_copy(update={"selected_branch": selected})

    return {
        "current_hypothesis": updated_hypothesis,
        "critic_verdict": verdict,
        "critic_feedback": result.get("feedback", ""),
        "messages": [response],
        # 将错误关键词暴露出来，供路由函数写入 failed_ledger
        "_error_keywords": result.get("error_keywords", []),
    }


def _find_branch(branches: list[AbstractionBranch], name: str) -> AbstractionBranch | None:
    for branch in branches:
        if branch.name == name:
            return branch
    return branches[0] if branches else None
