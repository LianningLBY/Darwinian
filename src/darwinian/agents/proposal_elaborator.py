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


# ===========================================================================
# Phase 1 v3: pack-aware path（唯一路径，v2 ConceptGraph-only 路径已删除）
# ===========================================================================
#
# 入口：elaborate_proposal_from_pack(skeleton, material_pack, llm)
#   - 使用 PaperEvidence 的精确数字（quantitative_claims）作弹药
#   - 使用 StructuralHoleHook + ResearchConstraints 给 motivation 充实素材
#   - 输出 expected_outcomes_structured (positive / negative / why_both)
#   - 输出 key_references_formatted（'Title (Venue)' 列表）
#   - 校验强制 forbidden_techniques 不出现在 proposed_method / technical_details
#   - resource_estimate 三档由 constraints 兜底（LLM 可覆盖）
#
# 历史：v2 路径 (elaborate_proposal + ConceptGraph) 在 P0 cleanup 中删除——
#   _render_quantitative_claims 用 limitations 假装定量数据，导致 LLM 编造数字。
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
7. **methodology_phases 只放消耗 GPU 的实验阶段**。写论文 / 投稿 / 准备 supplementary
   等纯人力工作**不要**作为单独 phase（它们不消耗 GPU 小时）。phase name 含
   'paper writing' / 'submission' / 'manuscript' 字样的，要么删掉要么 expected_compute_hours=0

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

    # key_references_formatted：**始终从 PaperEvidence 重建，丢弃 LLM 输出**
    # 防 LLM 幻觉论文标题（如把 DEL "Dynamic Exit Layer" 瞎编成 "Draft-Enhanced
    # Speculative Decoding"）。pack.paper_evidence 有真实 title + venue，唯一可信源
    key_refs = raw.get("key_references") or []
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
#   WRITING_PHASE_HAS_GPU_HOURS: phase 含 paper writing/submission/manuscript
#                                 字样且 expected_compute_hours > 0
# 复用老 error code: MISSING_PAPER_ID / TOO_FEW_REFS / TOO_FEW_PHASES /
#                    OVER_BUDGET / MISSING_QUANT_IN_MOTIVATION


_WRITING_PHASE_KEYWORDS = (
    "paper writing", "writing", "submission", "manuscript",
    "supplementary", "camera-ready", "camera ready",
)


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

    # 9. 写作类 phase 不得算 GPU 小时（防 LLM 把 "Paper Writing" 当成 48h GPU 任务）
    writing_phase_violations = []
    for ph in proposal.methodology_phases:
        if ph.expected_compute_hours <= 0:
            continue
        name_lower = ph.name.lower()
        if any(kw in name_lower for kw in _WRITING_PHASE_KEYWORDS):
            writing_phase_violations.append((ph.name, ph.expected_compute_hours))
    if writing_phase_violations:
        errors.append(("WRITING_PHASE_HAS_GPU_HOURS", writing_phase_violations))

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
        elif code == "WRITING_PHASE_HAS_GPU_HOURS":
            offenders = ", ".join(f"'{n}' ({h:.0f}h)" for n, h in detail)
            lines.append(
                f"  ❌ 这些 phase 是写作类工作不消耗 GPU，但 expected_compute_hours > 0: "
                f"{offenders}\n"
                "     要么删掉该 phase（推荐），要么 expected_compute_hours 设 0。"
                "写论文 / 投稿 / 准备 supplementary 不算在 methodology_phases 里"
            )

    lines.append("\n请保持其他正确字段不变，只修正以上问题。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v3 LangGraph 节点 wrapper：从 state.material_pack 读素材
# ---------------------------------------------------------------------------

def proposal_elaborator_node_v3(
    state: ResearchState,
    llm: BaseChatModel,
) -> dict:
    """
    LangGraph 节点：对 state.current_hypothesis.abstraction_tree 中每个 branch 调用
    elaborate_proposal_from_pack(), 把 material_pack 从 state.material_pack 取。

    LangGraph 节点签名只接受 (state, llm) —— P0 修复（2026-04-27）将 material_pack
    从 kwarg 改为从 state 读取。phase_a_orchestrator 应在 state 中填充 material_pack
    后再走到此节点。

    Args:
        state: ResearchState（必须含 material_pack 和 current_hypothesis.abstraction_tree）
        llm: 主 LLM

    Returns:
        {"research_proposals": [...]}（即使空也返字段，方便 LangGraph merge）
    """
    proposals: list[ResearchProposal] = []
    if state.material_pack is None:
        return {"research_proposals": proposals}
    if state.current_hypothesis is None or not state.current_hypothesis.abstraction_tree:
        return {"research_proposals": proposals}

    for skeleton in state.current_hypothesis.abstraction_tree:
        proposal = elaborate_proposal_from_pack(
            skeleton=skeleton, material_pack=state.material_pack, llm=llm,
        )
        if proposal is not None:
            proposals.append(proposal)
    return {"research_proposals": proposals}
