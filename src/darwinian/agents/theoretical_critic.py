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

import re as _re
from darwinian.state import ResearchState, CriticVerdict, AbstractionBranch

# 元投诉特征：core_problem 描述系统/检索状态而非科研问题
_META_COMPLAINT_PATTERNS = [
    r"文献.{0,10}(不匹配|不相关|不可用|检索失败|无法)",
    r"(检索|API|网络).{0,10}(不可用|失败|超时|限流)",
    r"无法.{0,15}(分析|基于这些|进行有效)",
    r"当前.{0,10}(文献|检索|数据).{0,10}(不|无法|缺乏)",
]


SYSTEM_PROMPT = """你是一位严格的理论审查官，专门评估科研方案的数学可行性和新颖性。

审查标准：
1. **数学可行性（MATH_ERROR）**: 方案中的数学公式是否存在明显错误？
   - 维度不匹配、不可微分操作、违反基本定理
   - 算法复杂度不可接受（如 O(n³) 在大规模数据上）
   - 关键假设在目标数据上不成立

2. **新颖性（NOT_NOVEL）**: 方案是否与现有方法本质相同？
   - 改变数据集/超参数/损失函数系数不构成新颖性
   - 简单组合两个已有方法不构成新颖性（除非组合方式本身有创新）
   - 必须指出与哪篇已有工作本质相同（具体论文名称或方法名）
   - 如果不确定是否新颖，倾向于判 NOT_NOVEL 而非 PASS

3. **通过（PASS）**: 数学上可行，且方案在机制上有明确创新点（不仅是参数调整）

输出格式（严格 JSON）：
{
  "verdict": "PASS" | "MATH_ERROR" | "NOT_NOVEL",
  "selected_branch_name": "如果 PASS，填写最优方案分支的 name；否则填 null",
  "feedback": "详细审查意见，MATH_ERROR 时指出具体公式错误，NOT_NOVEL 时指出最相似的已有工作",
  "novelty_concern": "即使 PASS，也要指出方案与已有工作最相似的部分（供后续审查参考）",
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

    # 前置拦截：core_problem 是元投诉（描述检索/系统状态），直接打回 MATH_ERROR
    core = hypothesis.core_problem
    if any(_re.search(p, core) for p in _META_COMPLAINT_PATTERNS):
        return {
            "current_hypothesis": hypothesis,
            "critic_verdict": CriticVerdict.MATH_ERROR,
            "critic_feedback": f"core_problem 描述的是系统状态而非科研问题，需重新生成真实的研究矛盾。原文：{core[:100]}",
            "last_error_keywords": ["文献检索", "不匹配", "系统状态"],
            "messages": [],
        }
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
        "last_error_keywords": result.get("error_keywords", []),
        "messages": [response],
    }


def _find_branch(branches: list[AbstractionBranch], name: str) -> AbstractionBranch | None:
    for branch in branches:
        if branch.name == name:
            return branch
    return branches[0] if branches else None
