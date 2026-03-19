"""
Agent 1: 瓶颈挖掘机 (bottleneck_miner_node)

职责：
- 调用 Semantic Scholar API 检索相关文献
- 提取文献中的 Limitations 部分
- 结合 failed_ledger 规避已知失败路径
- 输出一个亟待解决的"核心矛盾"
"""

from __future__ import annotations

import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState
from darwinian.tools.semantic_scholar import search_papers_with_limitations


SYSTEM_PROMPT = """你是一个科研瓶颈挖掘专家。你的任务是：
1. 分析给定研究方向的现有文献中的 Limitations
2. 参考「认知账本」中记录的已知失败路径，确保不重蹈覆辙
3. 识别出一个最有价值、最亟待解决的"核心矛盾"

输出格式（严格 JSON）：
{
  "core_problem": "一句话描述核心矛盾",
  "evidence": ["文献 1 标题 - 其 limitation", "文献 2 标题 - 其 limitation"],
  "avoided_paths": ["已规避的失败方向 1", ...],
  "banned_keywords_for_next": ["建议后续 Agent 1 过滤的关键词"]
}

禁止输出 JSON 以外的任何内容。"""


def bottleneck_miner_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 1 节点函数。
    输入: research_direction + failed_ledger
    输出: 更新 current_hypothesis.core_problem
    """
    # 1. 检索文献并提取 Limitations
    papers = search_papers_with_limitations(
        query=state.research_direction,
        banned_keywords=_collect_banned_keywords(state),
        limit=10,
    )

    # 2. 构建 failed_ledger 摘要
    ledger_summary = _format_ledger(state)

    # 3. 调用 LLM
    user_message = f"""研究方向：{state.research_direction}

检索到的文献 Limitations：
{json.dumps(papers, ensure_ascii=False, indent=2)}

认知账本（已知失败路径，必须规避）：
{ledger_summary}

请识别出最值得攻克的核心矛盾。"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    # 4. 解析输出 - 处理可能的 markdown 代码块或空白响应
    content = response.content.strip() if response.content else ""
    
    if not content:
        raise ValueError("LLM 返回了空响应")
    
    # 移除可能的 markdown 代码块标记
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    else:
        # 尝试直接解析
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
    
    result = json.loads(content)

    # 5. 初始化或更新 hypothesis（仅填充 core_problem）
    from darwinian.state import Hypothesis
    new_hypothesis = Hypothesis(
        core_problem=result["core_problem"],
        abstraction_tree=[],
        literature_support=result.get("evidence", []),
    )

    return {
        "current_hypothesis": new_hypothesis,
        "messages": [response],
    }


def _collect_banned_keywords(state: ResearchState) -> list[str]:
    """从所有失败记录中收集禁用关键词"""
    keywords: list[str] = []
    for record in state.failed_ledger:
        keywords.extend(record.banned_keywords)
    return list(set(keywords))


def _format_ledger(state: ResearchState) -> str:
    if not state.failed_ledger:
        return "（暂无失败记录）"
    lines = []
    for i, record in enumerate(state.failed_ledger, 1):
        lines.append(
            f"{i}. [{record.failure_type}] 第 {record.iteration} 轮 — {record.error_summary}"
        )
    return "\n".join(lines)
