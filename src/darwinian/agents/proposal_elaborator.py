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
import sys
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from darwinian.state import (
    AbstractionBranch,
    ConceptGraph,
    ExpectedOutcomes,
    MethodologyPhase,
    PaperEvidence,
    ResearchConstraints,
    ResearchMaterialPack,
    ResearchProposal,
    ResearchState,
    ResourceEstimate,
    StructuralHoleHook,
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
# 底层工具函数：单 branch 展开
# ---------------------------------------------------------------------------

def elaborate_proposal(
    skeleton: AbstractionBranch,
    concept_graph: ConceptGraph,
    llm: BaseChatModel,
    *,
    gpu_hours_budget: float = 168.0,    # 默认 7 天 × 24 小时（4 卡都算上则更多）
    target_venues: list[dict] | None = None,
) -> ResearchProposal | None:
    """
    把单个骨架展开成完整 ResearchProposal（独立工具函数，不直接接入 LangGraph）。

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
            print(f"[proposal_elaborator] 第 {attempt+1}/3 次解析/构造失败：{type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
            continue

        # 硬约束校验
        errors = _validate(proposal, concept_graph, gpu_hours_budget)
        if not errors:
            return proposal

        # 失败：带反馈塞回 prompt 重试
        last_proposal = proposal
        feedback = _build_feedback(errors, concept_graph)
        print(f"[proposal_elaborator] 第 {attempt+1}/3 次校验失败：{[e[0] for e in errors]}", file=sys.stderr)
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


# ---------------------------------------------------------------------------
# LangGraph 节点 wrapper：从 state 取材料，对每个 branch 调用 elaborate_proposal
# ---------------------------------------------------------------------------

def proposal_elaborator_node(
    state: ResearchState,
    llm: BaseChatModel,
    *,
    gpu_hours_budget: float = 168.0,
    target_venues: list[dict] | None = None,
) -> dict:
    """
    LangGraph 节点版本：

    1. 从 state.current_hypothesis.abstraction_tree 拿所有骨架
    2. 对每个骨架调 elaborate_proposal()
    3. 收集成 list[ResearchProposal] 写回 state.research_proposals

    Args:
        state: ResearchState（必须含 concept_graph + current_hypothesis）
        llm: 主 LLM
        gpu_hours_budget / target_venues: 透传给 elaborate_proposal

    Returns:
        {"research_proposals": [...]}（即使空也返字段，方便 LangGraph merge）

    设计选择：
    - N:N 对齐——每个 branch 各展开一份 proposal，未做"只展开 selected_branch"的优化
      因为 HITL 场景下用户可能想对比所有 N 个 proposal 再挑
    - 即使部分 branch elaborate 失败也不抛异常，已成功的会写入 state
    - state.concept_graph 或 abstraction_tree 为空时安全返空 list
    """
    proposals: list[ResearchProposal] = []

    if state.current_hypothesis is None or not state.current_hypothesis.abstraction_tree:
        return {"research_proposals": proposals}
    if state.concept_graph is None:
        return {"research_proposals": proposals}

    for skeleton in state.current_hypothesis.abstraction_tree:
        proposal = elaborate_proposal(
            skeleton=skeleton,
            concept_graph=state.concept_graph,
            llm=llm,
            gpu_hours_budget=gpu_hours_budget,
            target_venues=target_venues,
        )
        if proposal is not None:
            proposals.append(proposal)

    return {"research_proposals": proposals}


# ===========================================================================
# Phase 1 v3: pack-aware path（接 ResearchMaterialPack 的新接口）
# ===========================================================================
#
# 与老 elaborate_proposal(skeleton, concept_graph, llm) 并存：
#   - 老接口仍 work，下游已写的 LangGraph 节点不破坏
#   - 新接口 elaborate_proposal_from_pack(skeleton, material_pack, llm) 是 v3 真正路径，
#     使用 PaperEvidence 的精确数字 + StructuralHoleHook + ResearchConstraints
#     喂给 LLM 写出 QuantSkip 风格 ResearchProposal
#
# 主要差异：
#   1. prompt 给 LLM 的是按 category 分组的 PaperEvidence + 每篇 quantitative_claims，
#      而不是 ConceptGraph.papers + limitations 半成品
#   2. 要求 LLM 输出 expected_outcomes_structured (positive / negative / why_both)
#      而不是单字符串
#   3. 要求 LLM 输出 key_references_formatted ('Title (Venue)' 列表)
#   4. 校验时强制 forbidden_techniques 不出现在 proposed_method / technical_details
#   5. resource_estimate 三档由 constraints 兜底生成（LLM 可覆盖）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V3 = """你是一位资深科研计划撰写专家。你将基于一份 ResearchMaterialPack（含
一个研究方案骨架 + 已抽好的 PaperEvidence 弹药 + StructuralHoleHook + 资源约束）
展开成 QuantSkip 风格的完整研究 proposal。

输出严格 JSON，含以下顶层字段：

{
  "title": "标题，'X: Do Y Also Imply Z?' 的子问题式 catchy 形式",
  "elevator_pitch": "200 字左右描述，必须含具体数字（来自 PaperEvidence）",
  "challenges": "核心问题段，含'**The unstudied gap**: ...'明确指出 gap",
  "existing_methods": "**研究现状**：按 category 分组列举（'**Category**: paper1 (venue, headline_result), paper2 (...)'），最后一段必须是'**The gap**: ...'",
  "motivation": "Why Now，必须引用 ≥ 3 个 PaperEvidence.quantitative_claims（具体数字 + 出处），并含一段机制级解释（如'A 是 local distributional 性质，B 是 global functional 性质'）",
  "proposed_method": "方法核心思想（先 high-level intuition，再说为什么能填上 gap）",
  "technical_details": "关键算法步骤、公式（用 LaTeX）、超参数",
  "expected_outcomes_structured": {
    "positive_finding": "如果实验得到 X 则证明 Y。必须含可观测信号阈值",
    "negative_finding": "如果实验得到 ~X 则证明 Z。同样可发表",
    "why_both_publishable": "为什么两种结果对社区都有 actionable guidance"
  },
  "methodology_phases": [
    {"phase_number": 1, "name": "...", "description": "...",
     "inputs": [...], "outputs": [...], "expected_compute_hours": 24.0},
    ...
  ],
  "target_venue": "主目标 venue，从 constraints.target_venues 选",
  "target_deadline": "ISO 日期",
  "fallback_venue": "备选 venue",
  "key_references": ["paper_id1", ...],   // 必须从 PaperEvidence.paper_id 选
  "key_references_formatted": ["LayerSkip: Enabling Early Exit ... (ACL 2024)", ...]
}

【关键硬约束】
1. key_references 每个 id 必须在 ResearchMaterialPack.paper_evidence 列表里
2. methodology_phases 的 expected_compute_hours 总和必须 ≤ constraints.gpu_hours_budget
3. motivation 必须引用 ≥ 3 个 quantitative_claims（含具体数字 + 出处）
4. expected_outcomes_structured 三个字段全部非空
5. proposed_method 和 technical_details 不得出现 constraints.forbidden_techniques 中的任何技术名
6. existing_methods 末尾必须有'**The gap**: ...'段，明确说明 gap

输出严格 JSON，禁止 markdown 包裹。
"""


def elaborate_proposal_from_pack(
    skeleton: AbstractionBranch,
    material_pack: ResearchMaterialPack,
    llm: BaseChatModel,
    *,
    max_retries: int = 2,
) -> ResearchProposal | None:
    """
    v3 路径：接 ResearchMaterialPack（不是 ConceptGraph），喂更丰富的素材给 LLM
    生成 QuantSkip 风格 ResearchProposal。

    Args:
        skeleton: AbstractionBranch 骨架
        material_pack: Phase A 调研产出（含 paper_evidence + structural_hole_hooks
                       + constraints + concept_graph 可选）
        llm: 主 LLM
        max_retries: 校验失败带反馈重试次数（默认 2，加上首次共 3 次尝试）

    Returns:
        ResearchProposal 或 None（连续解析失败）
    """
    user_message = _build_user_message_v3(skeleton, material_pack)
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT_V3),
        HumanMessage(content=user_message),
    ]

    last_proposal: ResearchProposal | None = None
    for attempt in range(max_retries + 1):
        try:
            response = invoke_with_retry(llm, messages)
            raw = parse_llm_json(response.content)
            proposal = _build_proposal_v3(raw, skeleton, material_pack)
        except Exception as e:
            print(f"[elaborator_v3] 第 {attempt+1}/{max_retries+1} 次解析失败: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
            continue

        errors = _validate_v3(proposal, material_pack)
        if not errors:
            return proposal

        last_proposal = proposal
        if attempt < max_retries:
            feedback = _build_feedback_v3(errors, material_pack)
            print(f"[elaborator_v3] 校验失败 ({attempt+1}/{max_retries+1}): {[e[0] for e in errors]}", file=sys.stderr)
            messages.append(response)
            messages.append(HumanMessage(content=feedback))

    return last_proposal


# ---------------------------------------------------------------------------
# v3 prompt 构造
# ---------------------------------------------------------------------------

def _build_user_message_v3(
    skeleton: AbstractionBranch,
    pack: ResearchMaterialPack,
) -> str:
    constraints = pack.constraints
    parts = [
        f"【骨架方案】\nname: {skeleton.name}\ndescription: {skeleton.description}\n"
        f"algorithm_logic: {skeleton.algorithm_logic}\nmath_formulation: {skeleton.math_formulation}\n"
        f"cited_entity_names: {skeleton.cited_entity_names}\n",
        "\n【研究方向（用户输入原文）】\n" + pack.direction,
        "\n\n【约束条件】\n" + _render_constraints(constraints),
        "\n\n【可引用的 PaperEvidence（按 category 分组，含 quantitative_claims）】\n"
        + _render_evidence_by_category(pack.paper_evidence),
    ]
    if pack.structural_hole_hooks:
        parts.append("\n\n【结构洞 hooks（建议在 motivation/why_now 引用）】\n"
                     + _render_hooks(pack.structural_hole_hooks))
    if pack.timeline_signals:
        parts.append("\n\n【时间线信号（给 Why Now 提供时间感）】\n"
                     + _render_timeline(pack.timeline_signals))

    parts.append(
        "\n\n请按 SYSTEM_PROMPT_V3 的 JSON schema 输出完整 ResearchProposal。"
        "key_references 必须全部来自上述 PaperEvidence.paper_id。"
        "expected_outcomes_structured 三个字段全部填。"
        f"methodology_phases 总耗时 ≤ {constraints.gpu_hours_budget:.0f} 小时。"
    )
    return "".join(parts)


def _render_constraints(c: ResearchConstraints) -> str:
    forbidden = ", ".join(c.forbidden_techniques) if c.forbidden_techniques else "(无)"
    venues = ", ".join(c.target_venues) if c.target_venues else "(自选)"
    return (
        f"  - GPU: {c.gpu_count}× {c.gpu_model or 'unspecified'}, "
        f"预算 {c.gpu_hours_budget:.0f} 小时\n"
        f"  - 时长: {c.wall_clock_days} 天\n"
        f"  - 模型规模上限: {c.max_model_params_b}B 参数\n"
        f"  - 必须用现成 benchmark: {c.use_existing_benchmarks_only}\n"
        f"  - 禁用人工标注: {not c.require_human_annotation}\n"
        f"  - 禁用技术: {forbidden}\n"
        f"  - 主实验禁用闭源 API: {c.require_no_api_for_main}\n"
        f"  - 候选 venues: {venues}"
    )


def _render_evidence_by_category(
    evidence_list: list[PaperEvidence],
    max_per_category: int = 6,
) -> str:
    if not evidence_list:
        return "（暂无）"
    groups: dict[str, list[PaperEvidence]] = {}
    for ev in evidence_list:
        cat = ev.category or "uncategorized"
        groups.setdefault(cat, []).append(ev)

    lines = []
    for cat, evs in groups.items():
        lines.append(f"\n**{cat}**:")
        for ev in evs[:max_per_category]:
            claims = "; ".join(
                f"{c.metric_name}={c.metric_value}@{c.setting or '?'}"
                for c in ev.quantitative_claims[:3]
            ) or ev.headline_result or "(无定量数据)"
            lines.append(
                f"  - [{ev.paper_id}] {ev.short_name} ({ev.venue}) — {claims}\n"
                f"    relation_to_direction: {ev.relation_to_direction}; "
                f"limitations: {'; '.join(ev.limitations[:2]) or '(未填)'}"
            )
        if len(evs) > max_per_category:
            lines.append(f"  ...（同 category 还有 {len(evs) - max_per_category} 篇）")
    return "\n".join(lines)


def _render_hooks(hooks: list[StructuralHoleHook]) -> str:
    lines = []
    for h in hooks:
        lines.append(
            f"  - [{h.relation_type}] {h.entity_a} × {h.entity_b}\n"
            f"    {h.hook_text}"
        )
    return "\n".join(lines)


def _render_timeline(signals: dict[str, list[str]]) -> str:
    lines = []
    for bucket, ids in signals.items():
        lines.append(f"  - {bucket}: {', '.join(ids[:5])}"
                     + (f" (+{len(ids)-5} more)" if len(ids) > 5 else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v3 解析 LLM 输出 → ResearchProposal
# ---------------------------------------------------------------------------

def _build_proposal_v3(
    raw: dict,
    skeleton: AbstractionBranch,
    pack: ResearchMaterialPack,
) -> ResearchProposal:
    """从 v3 LLM 输出装配 ResearchProposal，含 metadata + structured outcomes + resource est"""
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

    # Structured outcomes
    outcomes_struct = None
    so_raw = raw.get("expected_outcomes_structured")
    if isinstance(so_raw, dict):
        try:
            outcomes_struct = ExpectedOutcomes(
                positive_finding=str(so_raw.get("positive_finding", "")),
                negative_finding=str(so_raw.get("negative_finding", "")),
                why_both_publishable=str(so_raw.get("why_both_publishable", "")),
            )
        except Exception:
            outcomes_struct = None

    # Resource estimate：constraints 兜底，LLM 输出可覆盖（如果以后 schema 加上）
    constraints = pack.constraints
    resource_est = ResourceEstimate(
        auto_research={
            "gpu_hours": int(constraints.gpu_hours_budget),
            "wall_clock_days": constraints.wall_clock_days,
        },
        human_in_loop={
            "gpu_hours": int(constraints.gpu_hours_budget),
            "wall_clock_days": constraints.wall_clock_days,
            "human_hours": constraints.wall_clock_days * 2,
        },
        manual={
            "gpu_hours": int(constraints.gpu_hours_budget),
            "wall_clock_days": constraints.wall_clock_days * 5,
            "human_hours": constraints.wall_clock_days * 40,
        },
    )

    # key_references_formatted：LLM 给就用，否则从 PaperEvidence 兜底拼
    key_refs = raw.get("key_references") or []
    krf_raw = raw.get("key_references_formatted") or []
    if not krf_raw:
        evidence_by_id = {ev.paper_id: ev for ev in pack.paper_evidence}
        krf_raw = [
            f"{evidence_by_id[r].title} ({evidence_by_id[r].venue})"
            for r in key_refs if r in evidence_by_id
        ]

    return ResearchProposal(
        skeleton=skeleton,
        status="draft",
        level="top-tier",
        seed=pack.direction,
        created_at=_now_iso(),
        title=str(raw.get("title", "")),
        elevator_pitch=str(raw.get("elevator_pitch", "")),
        challenges=str(raw.get("challenges", "")),
        existing_methods=str(raw.get("existing_methods", "")),
        motivation=str(raw.get("motivation", "")),
        proposed_method=str(raw.get("proposed_method", "")),
        technical_details=str(raw.get("technical_details", "")),
        expected_outcomes=str(raw.get("expected_outcomes", "")),    # 兼容字段
        expected_outcomes_structured=outcomes_struct,
        methodology_phases=phases,
        total_estimated_hours=total_hours,
        fits_resource_budget=False,    # _validate_v3 后填
        target_venue=str(raw.get("target_venue", "")),
        target_deadline=str(raw.get("target_deadline", "")),
        fallback_venue=str(raw.get("fallback_venue", "")),
        key_references=key_refs,
        key_references_formatted=krf_raw,
        resource_estimate=resource_est,
    )


def _now_iso() -> str:
    """ISO 时间戳，独立成函数方便测试 mock"""
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# v3 硬约束校验
# ---------------------------------------------------------------------------

# 新增 error code:
#   MISSING_OUTCOMES_STRUCT: expected_outcomes_structured 为空或字段空
#   FORBIDDEN_TECHNIQUE_USED: proposed_method/technical_details 含禁用技术
#   GAP_NOT_DECLARED: existing_methods 没有 '**The gap**' 段
# 复用老 error code: MISSING_PAPER_ID / TOO_FEW_REFS / TOO_FEW_PHASES /
#                    OVER_BUDGET / MISSING_QUANT_IN_MOTIVATION


def _validate_v3(
    proposal: ResearchProposal,
    pack: ResearchMaterialPack,
) -> list[tuple]:
    import re as _re
    errors: list[tuple] = []
    constraints = pack.constraints

    # 1. key_references 必须在 paper_evidence 里
    valid_ids = {ev.paper_id for ev in pack.paper_evidence}
    invalid_refs = [r for r in proposal.key_references if r not in valid_ids]
    if invalid_refs:
        errors.append(("MISSING_PAPER_ID", invalid_refs))

    # 2. ≥ 3 篇 references（v3 比 v2 宽松，因为素材本身就少）
    if len(proposal.key_references) < 3:
        errors.append(("TOO_FEW_REFS", len(proposal.key_references)))

    # 3. ≥ 3 个 phases
    if len(proposal.methodology_phases) < 3:
        errors.append(("TOO_FEW_PHASES", len(proposal.methodology_phases)))

    # 4. 总耗时 ≤ 预算
    if proposal.total_estimated_hours > constraints.gpu_hours_budget:
        errors.append(("OVER_BUDGET",
                       (proposal.total_estimated_hours, constraints.gpu_hours_budget)))
    else:
        proposal.fits_resource_budget = True

    # 5. motivation 含 ≥ 3 个数字
    numbers = _re.findall(r"\d+(?:\.\d+)?\s*(?:x|%|×|MB|GB|hours?)?", proposal.motivation)
    if len(numbers) < 3:
        errors.append(("MISSING_QUANT_IN_MOTIVATION", len(numbers)))

    # 6. expected_outcomes_structured 必须存在且三字段非空
    so = proposal.expected_outcomes_structured
    if so is None or not so.positive_finding.strip() or not so.negative_finding.strip() \
            or not so.why_both_publishable.strip():
        errors.append(("MISSING_OUTCOMES_STRUCT", None))

    # 7. forbidden_techniques 不得出现在 proposed_method / technical_details
    if constraints.forbidden_techniques:
        scope = (proposal.proposed_method + "\n" + proposal.technical_details).upper()
        hits = [t for t in constraints.forbidden_techniques if t.upper() in scope]
        if hits:
            errors.append(("FORBIDDEN_TECHNIQUE_USED", hits))

    # 8. existing_methods 必须含 '**The gap**' 段
    if "**The gap**" not in proposal.existing_methods:
        errors.append(("GAP_NOT_DECLARED", None))

    return errors


def _build_feedback_v3(errors: list[tuple], pack: ResearchMaterialPack) -> str:
    lines = ["你上次输出未通过校验，请按以下反馈修正后重出："]
    valid_ids_sample = [ev.paper_id for ev in pack.paper_evidence[:5]]

    for code, detail in errors:
        if code == "MISSING_PAPER_ID":
            lines.append(
                f"  ❌ key_references 这些 id 不在 PaperEvidence: {detail[:5]}\n"
                f"     可选 paper_id: {valid_ids_sample}"
            )
        elif code == "TOO_FEW_REFS":
            lines.append(f"  ❌ key_references 仅 {detail} 个，要求 ≥ 3")
        elif code == "TOO_FEW_PHASES":
            lines.append(f"  ❌ methodology_phases 仅 {detail} 个，要求 ≥ 3")
        elif code == "OVER_BUDGET":
            actual, budget = detail
            lines.append(
                f"  ❌ phases 总耗时 {actual:.0f} > 预算 {budget:.0f} 小时；减 phase 工作量"
            )
        elif code == "MISSING_QUANT_IN_MOTIVATION":
            lines.append(
                f"  ❌ motivation 仅 {detail} 个数字。从 PaperEvidence.quantitative_claims "
                "中引用 ≥ 3 个具体数字（如 '2.16-2.62x speedup'、'5.54 vs 5.60 PPL'）"
            )
        elif code == "MISSING_OUTCOMES_STRUCT":
            lines.append(
                "  ❌ expected_outcomes_structured 缺失或字段空。"
                "必须提供 positive_finding / negative_finding / why_both_publishable 三字段"
            )
        elif code == "FORBIDDEN_TECHNIQUE_USED":
            lines.append(
                f"  ❌ proposed_method / technical_details 出现了禁用技术: {detail}\n"
                "     从这些禁用技术中**完全移除**，改用其他方法"
            )
        elif code == "GAP_NOT_DECLARED":
            lines.append(
                "  ❌ existing_methods 末尾必须有 '**The gap**: ...' 段，"
                "明确说明现有方法都没做到的事是什么"
            )

    lines.append("\n请保持其他正确字段不变，只修正以上问题。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v3 LangGraph 节点 wrapper：state.material_pack 优先，回退到 concept_graph
# ---------------------------------------------------------------------------

def proposal_elaborator_node_v3(
    state: ResearchState,
    llm: BaseChatModel,
    *,
    material_pack: ResearchMaterialPack | None = None,
) -> dict:
    """
    v3 节点：material_pack 显式传入或从 state 取，对每个 branch 调
    elaborate_proposal_from_pack()。

    Args:
        state: ResearchState
        llm: 主 LLM
        material_pack: 显式传入。若为 None，本节点直接返空（state 暂未承载该字段）

    Returns:
        {"research_proposals": [...]}
    """
    proposals: list[ResearchProposal] = []
    if material_pack is None:
        return {"research_proposals": proposals}
    if state.current_hypothesis is None or not state.current_hypothesis.abstraction_tree:
        return {"research_proposals": proposals}

    for skeleton in state.current_hypothesis.abstraction_tree:
        proposal = elaborate_proposal_from_pack(
            skeleton=skeleton, material_pack=material_pack, llm=llm,
        )
        if proposal is not None:
            proposals.append(proposal)
    return {"research_proposals": proposals}
