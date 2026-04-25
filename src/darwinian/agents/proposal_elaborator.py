"""
Agent 2.5: Proposal Elaborator (proposal_elaborator_node)

Phase 1 v3 改造的核心：把 Agent 2 产出的 AbstractionBranch 骨架（短描述 +
cited_entity_names）展开成 QuantSkip 风格的 ResearchProposal——含 6 个 section、
phased methodology、target venue、resource estimate、20+ 篇 grounded references。

设计借鉴 HKUDS AI-Researcher 论文（NeurIPS 2025, arxiv:2505.18705）的：
  1. 6-section proposal schema (Challenges/Existing Methods/Motivation/
     Proposed Method/Technical Details/Expected Outcomes)
  2. Divergent-Convergent ideation（生 N 个 angle → 3 标准择优）
  3. 公式↔代码 bidirectional grounding 思想

Darwinian 自有强项：
  1. ConceptGraph 已有 cited_entities + limitations + existing_combination
     给 elaborator 当 grounding，不会编不存在的概念
  2. Phased methodology 模板（HKUDS 论文里没有）+ 资源预算硬校验
  3. Target venue / deadline 计算

Note: 不复制 HKUDS 任何代码（仓库无 LICENSE）。论文中的方法思想是公开学术内容，
可借鉴但实现完全独立。
"""

from __future__ import annotations

import json as _json
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from darwinian.state import (
    AbstractionBranch,
    ConceptGraph,
    MethodologyPhase,
    ResearchProposal,
)
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位资深科研计划撰写专家。你将把一个研究方案骨架展开成顶会风格的完整研究 proposal。

输出必须严格按 JSON schema 包含 6 个 section：
1. challenges: 该方向核心挑战（含具体技术瓶颈）
2. existing_methods: 现有方法分类汇总，每类列代表性工作 + 局限
3. motivation: 为什么现在做。**必须引用 ≥ 3 个 ConceptGraph.papers 里的 quantitative_claims**（具体数字 + 出处）
4. proposed_method: 方法核心思想（先说 high-level intuition）
5. technical_details: 关键算法步骤、公式（用 LaTeX）、超参数
6. expected_outcomes: 预期实验结果 + **必须包含"正反两种结果都可发表"的科学价值论证**

另外需输出：
- title: 含子问题的 catchy 标题，如 "X: Do Y Also Imply Z?"
- elevator_pitch: 200 字左右描述
- methodology_phases: 3-4 个执行阶段，每个含 phase_number, name, description, inputs, outputs, expected_compute_hours
- target_venue + target_deadline (ISO 日期) + fallback_venue
- key_references: ≥ 5 个 paperId（必须从 ConceptGraph.papers 选，不可编造）

【关键硬约束】
- key_references 每个 id 必须在我提供的 ConceptGraph.papers 列表里
- methodology_phases 的 expected_compute_hours 总和必须 ≤ 用户资源预算
- motivation 必须引用 quantitative_claims（不能空说"已有方法效果不佳"）
- expected_outcomes 必须包含"如果发现 X 则证明 Y；如果发现 ~X 则证明 Z"两种 publishable framing

输出严格 JSON，禁止 markdown 包裹。
"""


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------

def proposal_elaborator_node(
    skeleton: AbstractionBranch,
    concept_graph: ConceptGraph,
    llm: BaseChatModel,
    *,
    gpu_hours_budget: float = 168.0,    # 默认 7 天 × 24 小时（4 卡都算上则更多）
    target_venues: list[dict] | None = None,
) -> ResearchProposal | None:
    """
    把骨架展开成完整 ResearchProposal。

    Args:
        skeleton: Agent 2 产出的 branch（含 cited_entity_names + solved_limitation_id）
        concept_graph: bottleneck_miner 产出的 ConceptGraph，提供 grounding
        llm: 主 LLM
        gpu_hours_budget: 资源预算（GPU 小时），用于校验 phases 总耗时
        target_venues: [{"name": "NeurIPS 2026", "deadline": "2026-05-13"}, ...]

    Returns:
        ResearchProposal 或 None（连续校验失败）
    """
    user_message = _build_user_message(skeleton, concept_graph, gpu_hours_budget, target_venues)
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    last_proposal: ResearchProposal | None = None
    for attempt in range(3):
        try:
            response = invoke_with_retry(llm, messages)
            raw = parse_llm_json(response.content)
            proposal = _build_proposal(raw, skeleton, concept_graph)
        except Exception as e:
            print(f"[proposal_elaborator] 第 {attempt+1}/3 次解析/构造失败：{type(e).__name__}: {e}")
            time.sleep(3 * (attempt + 1))
            continue

        # 硬约束校验
        errors = _validate(proposal, concept_graph, gpu_hours_budget)
        if not errors:
            return proposal

        # 失败：带反馈塞回 prompt 重试
        last_proposal = proposal
        feedback = _build_feedback(errors, concept_graph)
        print(f"[proposal_elaborator] 第 {attempt+1}/3 次校验失败：{[e[0] for e in errors]}")
        messages.append(response)
        messages.append(HumanMessage(content=feedback))

    return last_proposal   # 最后一次（可能仍有错），调用方决定怎么用


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def _build_user_message(
    skeleton: AbstractionBranch,
    graph: ConceptGraph,
    gpu_hours_budget: float,
    target_venues: list[dict] | None,
) -> str:
    """组装 user prompt：骨架 + 实体表 + 缺陷 + 量化证据 + 资源约束 + venue 候选"""
    lim = graph.limitation_by_id(skeleton.solved_limitation_id)
    lim_text = lim.text if lim else "(未指定)"

    # 论文引用候选——给 LLM 看完整的 paper_id 池
    papers_block = _render_papers(graph)
    quant_block = _render_quantitative_claims(graph)
    venues_block = _render_venues(target_venues or [])

    return f"""
【骨架方案】
name: {skeleton.name}
description: {skeleton.description}
algorithm_logic: {skeleton.algorithm_logic}
math_formulation: {skeleton.math_formulation}
cited_entity_names: {skeleton.cited_entity_names}
solved_limitation: [{skeleton.solved_limitation_id}] {lim_text}

【可引用的 papers（key_references 必须从这里选）】
{papers_block}

【可引用的具体定量证据（motivation 必须引用 ≥ 3 个）】
{quant_block}

【资源预算】
GPU hours 总预算：{gpu_hours_budget:.0f}
所有 methodology_phases 的 expected_compute_hours 总和必须 ≤ 此预算

【目标 venue 候选】
{venues_block}

请基于以上信息，按 SYSTEM_PROMPT 的 schema 输出完整 ResearchProposal JSON。
"""


def _render_papers(graph: ConceptGraph, max_count: int = 30) -> str:
    """展示 paper 池，供 LLM 选 key_references"""
    if not graph.papers:
        return "（暂无）"
    lines = []
    sorted_papers = sorted(graph.papers, key=lambda p: p.citation_count, reverse=True)
    for p in sorted_papers[:max_count]:
        title = (p.title or "")[:80]
        year = p.year if p.year else "?"
        lines.append(f"  - {p.paper_id}  ({year})  {title}")
    if len(graph.papers) > max_count:
        lines.append(f"  ...（还有 {len(graph.papers) - max_count} 篇未列出，但全部可选）")
    return "\n".join(lines)


def _render_quantitative_claims(graph: ConceptGraph, max_count: int = 20) -> str:
    """
    展示从论文中抽取的具体定量数据。
    Note: ConceptGraph 当前 schema 还没有 quantitative_claims 字段——这一步
    在未来 ConceptGraph 升级后会有真数据；当前先用 limitations 当替代。
    """
    if not graph.limitations:
        return "（暂无具体定量数据）"
    lines = []
    for lim in graph.limitations[:max_count]:
        lines.append(f"  - [{lim.id}] {lim.text[:120]}  (来自 {lim.source_paper_id})")
    return "\n".join(lines)


def _render_venues(venues: list[dict]) -> str:
    if not venues:
        return "  （未指定，由你按 deadline + 方向匹配度自选）"
    return "\n".join(
        f"  - {v.get('name', '?')}  截止日 {v.get('deadline', '?')}"
        for v in venues
    )


# ---------------------------------------------------------------------------
# 解析 LLM 输出 → ResearchProposal
# ---------------------------------------------------------------------------

def _build_proposal(
    raw: dict,
    skeleton: AbstractionBranch,
    graph: ConceptGraph,
) -> ResearchProposal:
    """从 LLM 解析的 dict 组装 ResearchProposal"""
    phases_raw = raw.get("methodology_phases", []) or []
    phases = []
    for i, p in enumerate(phases_raw):
        if not isinstance(p, dict):
            continue
        phases.append(MethodologyPhase(
            phase_number=int(p.get("phase_number", i + 1)),
            name=str(p.get("name", f"Phase {i+1}")),
            description=str(p.get("description", "")),
            inputs=p.get("inputs") or [],
            outputs=p.get("outputs") or [],
            expected_compute_hours=float(p.get("expected_compute_hours", 0.0) or 0.0),
        ))
    total_hours = sum(p.expected_compute_hours for p in phases)

    return ResearchProposal(
        skeleton=skeleton,
        title=str(raw.get("title", "")),
        elevator_pitch=str(raw.get("elevator_pitch", "")),
        challenges=str(raw.get("challenges", "")),
        existing_methods=str(raw.get("existing_methods", "")),
        motivation=str(raw.get("motivation", "")),
        proposed_method=str(raw.get("proposed_method", "")),
        technical_details=str(raw.get("technical_details", "")),
        expected_outcomes=str(raw.get("expected_outcomes", "")),
        methodology_phases=phases,
        total_estimated_hours=total_hours,
        fits_resource_budget=False,    # _validate 后填
        target_venue=str(raw.get("target_venue", "")),
        target_deadline=str(raw.get("target_deadline", "")),
        fallback_venue=str(raw.get("fallback_venue", "")),
        key_references=raw.get("key_references") or [],
    )


# ---------------------------------------------------------------------------
# 硬约束校验
# ---------------------------------------------------------------------------

# error tuple: (code, detail)
# codes: MISSING_PAPER_ID / TOO_FEW_PHASES / OVER_BUDGET / TOO_FEW_REFS /
#        MISSING_QUANT_IN_MOTIVATION / NO_DUAL_OUTCOME_FRAMING

def _validate(
    proposal: ResearchProposal,
    graph: ConceptGraph,
    gpu_hours_budget: float,
) -> list[tuple]:
    errors: list[tuple] = []

    # 1. key_references 每个必须在 ConceptGraph.papers
    valid_paper_ids = {p.paper_id for p in graph.papers}
    invalid_refs = [r for r in proposal.key_references if r not in valid_paper_ids]
    if invalid_refs:
        errors.append(("MISSING_PAPER_ID", invalid_refs))

    # 2. ≥ 5 篇 references
    if len(proposal.key_references) < 5:
        errors.append(("TOO_FEW_REFS", len(proposal.key_references)))

    # 3. ≥ 3 个 phases
    if len(proposal.methodology_phases) < 3:
        errors.append(("TOO_FEW_PHASES", len(proposal.methodology_phases)))

    # 4. 总耗时不能超预算
    if proposal.total_estimated_hours > gpu_hours_budget:
        errors.append(("OVER_BUDGET", (proposal.total_estimated_hours, gpu_hours_budget)))
    else:
        # 通过了就把 fits 标 True
        proposal.fits_resource_budget = True

    # 5. motivation 必须含至少 3 个数字（粗略启发：3 个数字字面量）
    import re as _re
    numbers = _re.findall(r"\d+(?:\.\d+)?\s*(?:x|%|×|MB|GB|hours?)?", proposal.motivation)
    if len(numbers) < 3:
        errors.append(("MISSING_QUANT_IN_MOTIVATION", len(numbers)))

    # 6. expected_outcomes 必须有 dual framing（含正反假设）
    has_dual = (
        ("如果" in proposal.expected_outcomes and "则" in proposal.expected_outcomes)
        or ("if " in proposal.expected_outcomes.lower() and "either" in proposal.expected_outcomes.lower())
        or ("positive" in proposal.expected_outcomes.lower() and "negative" in proposal.expected_outcomes.lower())
        or ("publishable" in proposal.expected_outcomes.lower())
    )
    if not has_dual:
        errors.append(("NO_DUAL_OUTCOME_FRAMING", None))

    return errors


def _build_feedback(errors: list[tuple], graph: ConceptGraph) -> str:
    lines = ["你上次输出未通过校验，请按以下反馈修正后重出："]
    valid_ids_sample = [p.paper_id for p in graph.papers[:10]]

    for code, detail in errors:
        if code == "MISSING_PAPER_ID":
            lines.append(
                f"  ❌ key_references 里这些 id 不在 ConceptGraph.papers: {detail[:5]}\n"
                f"     可选 paper_id 示例（更多见上文 papers 池）: {valid_ids_sample}"
            )
        elif code == "TOO_FEW_REFS":
            lines.append(f"  ❌ key_references 仅 {detail} 个，要求 ≥ 5")
        elif code == "TOO_FEW_PHASES":
            lines.append(f"  ❌ methodology_phases 仅 {detail} 个，要求 ≥ 3")
        elif code == "OVER_BUDGET":
            actual, budget = detail
            lines.append(
                f"  ❌ phases 总耗时 {actual:.0f} 小时超过预算 {budget:.0f} 小时。"
                f"减少 phase 工作量或减少 phase 数量"
            )
        elif code == "MISSING_QUANT_IN_MOTIVATION":
            lines.append(
                f"  ❌ motivation 里只有 {detail} 个数字。要求引用 ≥ 3 个具体定量数据"
                f"（如 '2.16-2.62x speedup'、'5.54 vs 5.60 PPL'）"
            )
        elif code == "NO_DUAL_OUTCOME_FRAMING":
            lines.append(
                f"  ❌ expected_outcomes 缺少正反结果都可发表的论证。"
                f"加一段 '如果实验发现 X 则 Y；如果发现 ~X 则 Z，无论哪种结果都...'"
            )

    lines.append("\n请保持其他字段不变，只修正以上问题。")
    return "\n".join(lines)
