"""
Agent 1: 瓶颈挖掘机 (bottleneck_miner_node)

Phase 1 v2 职责：
- 调 build_concept_graph 构建 ConceptGraph（分两档检索 → 一跳扩展 → 清洗 → 实体抽取
  → canonical 合并 → 相关性裁剪 → 结构洞）
- 把 ConceptGraph（实体表 + 缺陷 + 结构洞）喂给 LLM，让它在真实数据之上
  识别出核心矛盾 + 写 Related Work 摘要
- 结合 failed_ledger 规避已知失败路径
- 产物：ResearchState.concept_graph + current_hypothesis.core_problem/literature_support
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import ConceptGraph, Hypothesis, ResearchState
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.knowledge_graph import build_concept_graph
from darwinian.utils.llm_retry import invoke_with_retry


SYSTEM_PROMPT = """你是一个科研瓶颈挖掘与文献分析专家。你的任务是：
1. 基于提供的「实体表」和「缺陷清单」——这些都是从真实论文网络中抽取的可追溯信息——识别该方向的核心矛盾
2. 归纳现有方法的**共同盲点**（多篇论文都承认的缺陷优先），而非单篇论文的局部问题
3. 参考「认知账本」中的已知失败路径，确保不重蹈覆辙

输出格式（严格 JSON）：
{
  "core_problem": "一句话描述该领域的科研核心矛盾（必须是科学问题，禁止写'文献不匹配''检索失败'等系统状态）",
  "evidence": ["论文标题 — 具体 limitation 描述", ...],
  "related_work_summary": "2-3 句话概括现有方法的共同局限",
  "research_gap": "现有方法与理想目标之间的具体差距",
  "avoided_paths": ["已规避的失败方向"],
  "banned_keywords_for_next": ["建议后续过滤的关键词"]
}

重要约束：
- core_problem 必须是该研究领域的科学矛盾，不能是对检索结果、系统状态的描述
- 即使实体表为空（概念图数据不足），也要基于领域先验知识给出真实的科研核心矛盾
- 禁止输出任何关于"文献不匹配""无法检索""API 不可用"等内容作为 core_problem
禁止输出 JSON 以外的任何内容。"""


def bottleneck_miner_node(
    state: ResearchState,
    llm: BaseChatModel,
    extractor_llm: BaseChatModel | None = None,
) -> dict:
    """
    Args:
        state: 当前 ResearchState（读 research_direction / failed_ledger）
        llm: 主 LLM（用于 core_problem 的高层归纳）
        extractor_llm: 可选的小模型（用于 batch 实体抽取，Haiku 级别即可）；
                       None 时复用 llm（贵但可用）
    """
    banned_keywords = _collect_banned_keywords(state)
    ledger_summary = _format_ledger(state)

    # Step 1-6.5: 构建 ConceptGraph
    # research_direction 作为 core_problem 的初始代理（后续 LLM 会产出更精确的 core_problem）
    graph = build_concept_graph(
        research_direction=state.research_direction,
        core_problem=state.research_direction,
        llm=extractor_llm or llm,
    )

    # banned_keywords 应用在 entity 层（比 query 层过滤更精细）
    graph = _apply_banned_keywords(graph, banned_keywords)

    # 组装给主 LLM 的 prompt
    user_message = _build_user_message(
        state.research_direction,
        graph,
        ledger_summary,
    )

    response = invoke_with_retry(llm, [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    result = parse_llm_json(response.content)

    new_hypothesis = Hypothesis(
        core_problem=result.get("core_problem", ""),
        abstraction_tree=[],
        literature_support=result.get("evidence", []),
    )

    return {
        "concept_graph": graph,
        "current_hypothesis": new_hypothesis,
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

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


def _apply_banned_keywords(graph: ConceptGraph, banned: list[str]) -> ConceptGraph:
    """
    在 entity 层标记 banned（走和 entity canonical 相同的 normalize 管道）。
    v1 实现：直接过滤掉命中的实体（简单）。v2 可改成保留实体但加 [BANNED] 标签。
    """
    if not banned:
        return graph
    from darwinian.utils.knowledge_graph import _normalize, _word_boundary_contains
    banned_normalized = {_normalize(b) for b in banned if b}

    def _is_banned(name: str) -> bool:
        n = _normalize(name)
        if n in banned_normalized:
            return True
        # word-boundary containment 双向判断
        for b in banned_normalized:
            if b and (_word_boundary_contains(b, n) or _word_boundary_contains(n, b)):
                return True
        return False

    kept = [e for e in graph.entities if not _is_banned(e.canonical_name)]
    return graph.model_copy(update={"entities": kept})


def _build_user_message(
    research_direction: str,
    graph: ConceptGraph,
    ledger_summary: str,
) -> str:
    """组装主 LLM prompt：实体表 + 缺陷清单 + 结构洞提示 + ledger"""
    if not graph.is_sufficient:
        # 降级：数据不足，明确告诉 LLM 基于先验知识作答
        data_section = """⚠️ 概念图数据不足（论文或实体数少于阈值）。请基于你对该领域的先验知识，
直接分析该方向当前面临的主要瓶颈与核心矛盾。"""
    else:
        entities_str = _format_entities(graph)
        limitations_str = _format_limitations(graph)
        pairs_str = _format_novel_pairs(graph)
        data_section = f"""【实体表（从 {len(graph.papers)} 篇论文抽取）】
{entities_str}

【待解缺陷清单】
{limitations_str}

【潜在结构洞（高频但从未共现的术语组合）】
{pairs_str}"""

    return f"""研究方向：{research_direction}

{data_section}

认知账本（已知失败路径，必须规避）：
{ledger_summary}

请基于以上证据，归纳现有方法的共同局限，识别最值得攻克的科学核心矛盾。"""


def _format_entities(graph: ConceptGraph, max_per_type: int = 15) -> str:
    """按 type 分组展示 top entities，每 type 最多 max_per_type 个"""
    by_type: dict[str, list] = {}
    for e in graph.entities:
        by_type.setdefault(e.type, []).append(e)

    lines: list[str] = []
    for etype in ["method", "dataset", "metric", "task_type"]:
        group = by_type.get(etype, [])
        group.sort(key=lambda e: len(e.paper_ids), reverse=True)
        lines.append(f"\n[{etype}] ({len(group)} 个)")
        for e in group[:max_per_type]:
            lines.append(f"  - {e.canonical_name} (出现在 {len(e.paper_ids)} 篇)")
        if len(group) > max_per_type:
            lines.append(f"  ...（还有 {len(group) - max_per_type} 个）")
    return "\n".join(lines)


def _format_limitations(graph: ConceptGraph, max_count: int = 20) -> str:
    if not graph.limitations:
        return "（无）"
    lines = []
    for lim in graph.limitations[:max_count]:
        lines.append(f"  - [{lim.id}] {lim.text}  —  来自 {lim.source_paper_id}")
    if len(graph.limitations) > max_count:
        lines.append(f"  ...（还有 {len(graph.limitations) - max_count} 条）")
    return "\n".join(lines)


def _format_novel_pairs(graph: ConceptGraph) -> str:
    if not graph.novel_pair_hints:
        return "（无）"
    return "\n".join(
        f"  - ({p.entity_a} × {p.entity_b})  score={p.score}"
        for p in graph.novel_pair_hints
    )
