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
from darwinian.utils.llm_retry import invoke_with_retry

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
   - cited_entity_names 里出现明显跨界术语（如 ML 方案里引用核物理/SRE/legislative 等无关词）→ 视为 MATH_ERROR

2. **新颖性（NOT_NOVEL）**: 方案是否与现有方法本质相同？
   - **每个 branch 的 `existing_combination` 字段已经基于 cited_entity_names 在
     Semantic Scholar 上做过组合查重**：
     - existing_combination=True 表示该组合的标题/摘要里同时出现所有 cited 术语，
       命中的 paperId 在 existing_combination_refs 里。**这种 branch 强烈倾向 NOT_NOVEL**，
       除非 Advocate 能在 description / algorithm_logic 里明确论证差异化（机制不同、
       约束不同等），否则直接判 NOT_NOVEL
     - existing_combination=False 不等于真新颖（surface match 漏报常见），仍要靠
       概念相似性判断
   - 改变数据集/超参数/损失函数系数不构成新颖性
   - 简单组合两个已有方法不构成新颖性（除非组合方式本身有创新）
   - 必须指出与哪篇已有工作本质相同（具体论文名称或方法名）
   - 如果不确定是否新颖，倾向于判 NOT_NOVEL 而非 PASS

3. **通过（PASS）**: 数学上可行，且方案在机制上有明确创新点（不仅是参数调整）

【你将看到的输入信号】
- 每个 branch 的 cited_entity_names: 该方案明确引用的术语
- 每个 branch 的 solved_limitation_id + 对应缺陷文本: 该方案声称解决的具体问题
- 每个 branch 的 existing_combination + existing_combination_refs: S2 组合查重结果
- ConceptGraph 摘要：术语表 + 缺陷清单（只读，用于交叉验证 cited_entities 是否合理）

输出格式（严格 JSON）：
{
  "verdict": "PASS" | "MATH_ERROR" | "NOT_NOVEL",
  "selected_branch_name": "如果 PASS，填写最优方案分支的 name；否则填 null",
  "feedback": "详细审查意见。MATH_ERROR 时指出具体公式错误或跨界术语；NOT_NOVEL 时指出最相似的已有工作（如 existing_combination_refs 里的 paperId）",
  "novelty_concern": "即使 PASS，也要指出方案与已有工作最相似的部分",
  "error_keywords": ["如果失败，提取 2-3 个关键词用于未来过滤"]
}

禁止输出 JSON 以外的任何内容。
重要：verdict 只能是 PASS / MATH_ERROR / NOT_NOVEL 三个值之一。"""


def theoretical_critic_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 3 节点函数。
    输入: current_hypothesis + 文献检索结果
    输出: 更新 critic_verdict 和 critic_feedback
    """
    if state.current_hypothesis is None:
        raise ValueError("theoretical_critic_node 调用前必须先运行 hypothesis_generator_node")

    hypothesis = state.current_hypothesis

    # 前置拦截①：abstraction_tree 为空，说明 Agent 2 未生成任何方案，直接打回
    if not hypothesis.abstraction_tree:
        return {
            "current_hypothesis": hypothesis,
            "critic_verdict": CriticVerdict.MATH_ERROR,
            "critic_feedback": "方案合成器未生成任何解决方案分支（abstraction_tree 为空），需要重新生成。",
            "last_error_keywords": [],
            "messages": [],
        }

    # 前置拦截②：core_problem 是元投诉（描述检索/系统状态），直接打回 MATH_ERROR
    core = hypothesis.core_problem
    if any(_re.search(p, core) for p in _META_COMPLAINT_PATTERNS):
        return {
            "current_hypothesis": hypothesis,
            "critic_verdict": CriticVerdict.MATH_ERROR,
            "critic_feedback": f"core_problem 描述的是系统状态而非科研问题，需重新生成真实的研究矛盾。原文：{core[:100]}",
            "last_error_keywords": ["文献检索", "不匹配", "系统状态"],
            "messages": [],
        }

    # 前置拦截③：所有分支都被 step 7.5 标 existing_combination=True，省 LLM 调用直接 NOT_NOVEL
    if hypothesis.abstraction_tree and all(b.existing_combination for b in hypothesis.abstraction_tree):
        all_refs = []
        for b in hypothesis.abstraction_tree:
            all_refs.extend(b.existing_combination_refs)
        return {
            "current_hypothesis": hypothesis,
            "critic_verdict": CriticVerdict.NOT_NOVEL,
            "critic_feedback": (
                "所有方案分支的 cited_entity_names 组合都在 S2 中找到了既存工作（"
                f"references: {sorted(set(all_refs))[:5]}）。需重新生成真正未被覆盖的组合。"
            ),
            "last_error_keywords": ["existing combination", "duplicate composition"],
            "messages": [],
        }

    branches_text = _render_branches_for_critic(hypothesis.abstraction_tree, state.concept_graph)

    user_message = f"""待审查的研究假设：

核心问题：{hypothesis.core_problem}

解决方案分支（含 step 7.5 组合查重结果）：
{branches_text}

文献支撑：
{chr(10).join(hypothesis.literature_support)}

请对上述方案进行严格审查。"""

    response = invoke_with_retry(llm, [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    result = parse_llm_json(response.content)
    verdict = CriticVerdict(result["verdict"])

    # 如果 PASS，选定最优分支
    # 注意：LLM 可能返回 null，必须兜底选第一个分支，否则 phase1_result_router 会误判为失败
    updated_hypothesis = hypothesis.model_copy()
    if verdict == CriticVerdict.PASS:
        branch_name = result.get("selected_branch_name")
        selected = _find_branch(hypothesis.abstraction_tree, branch_name) if branch_name else None
        if selected is None and hypothesis.abstraction_tree:
            selected = hypothesis.abstraction_tree[0]
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


def _render_branches_for_critic(branches: list[AbstractionBranch], graph) -> str:
    """
    把 branches 渲染为 critic 可读格式：
    每个分支显示 name/description/algorithm_logic/cited_entities/solved_limitation_text
    + step 7.5 的 existing_combination 标记和命中 paperId
    """
    sections = []
    for i, b in enumerate(branches, 1):
        # 解析 solved_limitation 文本
        limitation_text = "(未指定)"
        if graph and b.solved_limitation_id:
            lim = graph.limitation_by_id(b.solved_limitation_id)
            if lim:
                limitation_text = f"{lim.text}  (来自 paper={lim.source_paper_id})"

        existing_marker = "❗已存在 (NOT_NOVEL 倾向)" if b.existing_combination else "✓ S2 未命中相同组合"
        existing_refs = (
            f"  S2 命中: {b.existing_combination_refs[:3]}" if b.existing_combination else ""
        )

        sections.append(
            f"=== 分支 {i}: {b.name} ===\n"
            f"description: {b.description}\n"
            f"algorithm_logic: {b.algorithm_logic}\n"
            f"math_formulation: {b.math_formulation}\n"
            f"cited_entity_names: {b.cited_entity_names}\n"
            f"solved_limitation: {limitation_text}\n"
            f"existing_combination: {existing_marker}{existing_refs}"
        )
    return "\n\n".join(sections)
