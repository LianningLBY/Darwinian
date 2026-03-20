"""
Agent 1: 瓶颈挖掘机 (bottleneck_miner_node)

职责：
- 检索相关文献（最多 20 篇），提取 Limitations
- 分析 Related Work，识别现有方法的共同盲点
- 结合 failed_ledger 规避已知失败路径
- 输出核心矛盾 + Related Work 摘要（供论文写作使用）
"""

from __future__ import annotations

import json
from darwinian.utils.json_parser import parse_llm_json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState
from darwinian.tools.semantic_scholar import search_papers_with_limitations


SYSTEM_PROMPT = """你是一个科研瓶颈挖掘与文献分析专家。你的任务是：
1. 深入分析给定研究方向的现有文献，识别各论文的局限性
2. 归纳现有方法的共同盲点（而非单篇论文的局部问题）
3. 参考「认知账本」中的已知失败路径，确保不重蹈覆辙
4. 识别出一个最有价值、最亟待解决的核心矛盾

输出格式（严格 JSON）：
{
  "core_problem": "一句话描述核心矛盾（需具体，避免泛泛而谈）",
  "evidence": ["论文标题 — 具体 limitation 描述", ...],
  "related_work_summary": "2-3 句话概括现有方法的共同局限，供论文 Related Work 章节参考",
  "research_gap": "现有方法与理想目标之间的具体差距",
  "avoided_paths": ["已规避的失败方向"],
  "banned_keywords_for_next": ["建议后续过滤的关键词"]
}

禁止输出 JSON 以外的任何内容。"""


def bottleneck_miner_node(state: ResearchState, llm: BaseChatModel) -> dict:
    # 检索文献（增加到 20 篇，提高覆盖面）
    papers = search_papers_with_limitations(
        query=state.research_direction,
        banned_keywords=_collect_banned_keywords(state),
        limit=20,
    )

    ledger_summary = _format_ledger(state)

    if papers:
        # 截断每篇摘要，避免 token 过长
        truncated = [
            {**p, "abstract": p.get("abstract", "")[:300]}
            for p in papers
        ]
        literature_section = f"""检索到 {len(papers)} 篇相关文献：
{json.dumps(truncated, ensure_ascii=False, indent=2)}"""
    else:
        literature_section = """文献检索暂时不可用（网络问题或 API 限流）。
请根据你对该研究领域的知识，结合最新发展趋势，直接分析该方向当前面临的主要瓶颈与核心矛盾。
尤其要关注：现有最优方法在哪些场景下仍然失败，以及理论上存在哪些根本性限制。"""

    user_message = f"""研究方向：{state.research_direction}

{literature_section}

认知账本（已知失败路径，必须规避）：
{ledger_summary}

请深入分析现有方法的共同局限，识别最值得攻克的核心矛盾。"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    result = parse_llm_json(response.content)

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
