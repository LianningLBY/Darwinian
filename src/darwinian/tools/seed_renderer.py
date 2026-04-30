"""
Seed Renderer：把 ResearchProposal 渲染成 QuantSkip 风格的 seed.md。

设计目标：
  - 纯函数，不依赖 LLM；输入 ResearchProposal + 可选 ResearchMaterialPack，输出 markdown
  - 严格对齐 QuantSkip seed 模板的 section 顺序与 metadata 块
  - 渲染时优先用 expected_outcomes_structured（如有），fallback 到 expected_outcomes 自由文本
  - 渲染缺字段时用占位符 '(待补)' 而非崩溃，方便人眼快速定位 elaborator 漏的地方
"""

from __future__ import annotations

from darwinian.state import (
    DebateResult,
    FeasibilityChallenge,
    MechanismAlignment,
    NoveltyAssessment,
    ResearchMaterialPack,
    ResearchProposal,
)


def render_proposal(
    proposal: ResearchProposal,
    *,
    material_pack: ResearchMaterialPack | None = None,
) -> str:
    """
    把 ResearchProposal 渲染成 QuantSkip 风格 markdown。

    Args:
        proposal: 必填，elaborator 产出
        material_pack: 可选；提供时用 pack.direction 作为"种子"原文，pack.constraints
                       做资源预估的兜底

    Returns:
        完整 markdown 字符串（含 metadata 块、各 section、引用列表）
    """
    parts: list[str] = []

    # ---- 标题 + metadata 块 ----
    title = proposal.title or "(待补标题)"
    parts.append(f"# {title}\n\n")

    # R10-Pri-2: relevance warning banner（Phase A 真相关论文不足时立即提醒）
    if material_pack is not None and material_pack.relevance_warning:
        parts.append(
            "> ⚠️ **Phase A Relevance Warning**: "
            + material_pack.relevance_warning
            + "\n\n"
        )

    seed_text = proposal.seed or (material_pack.direction if material_pack else "")
    parts.append(_render_metadata_block(proposal, seed_text))

    # ---- 描述（elevator pitch）----
    parts.append("\n\n## 描述\n")
    parts.append(proposal.elevator_pitch or "(待补)")

    # ---- 核心问题 ----
    parts.append("\n\n## 核心问题\n")
    parts.append(proposal.challenges or "(待补)")

    # ---- 机遇 / Why Now ----
    parts.append("\n\n## 机遇 / Why Now\n")
    parts.append(proposal.motivation or "(待补)")

    # ---- 方法思路 ----
    parts.append("\n\n## 方法思路\n")
    parts.append(_render_methodology(proposal))

    # ---- 研究现状 ----
    parts.append("\n\n## 研究现状\n")
    parts.append(proposal.existing_methods or "(待补)")

    # ---- 目标 Venue ----
    parts.append("\n\n## 目标 Venue\n")
    parts.append(_render_venues(proposal))

    # ---- 关键参考 ----
    parts.append("\n\n## 关键参考\n")
    parts.append(_render_references(proposal))

    # ---- 资源预估 ----
    parts.append("\n\n## 资源预估\n")
    parts.append(_render_resource_estimate(proposal))

    # ---- 新颖性评估（SciMON boost 后填）----
    if proposal.novelty_assessment is not None:
        parts.append("\n\n## 新颖性评估\n")
        parts.append(_render_novelty(proposal.novelty_assessment))

    # ---- Feasibility Challenger 攻击结果 (R9c) ----
    if proposal.feasibility_challenge is not None:
        parts.append(_render_feasibility_challenge(proposal.feasibility_challenge))

    # ---- Mechanism Alignment Checker 结果 (R11) ----
    # 仅 cross-domain 类比的 proposal 才渲染（避免 not_applicable 的空 section）
    if proposal.mechanism_alignment is not None and proposal.mechanism_alignment.is_cross_domain:
        parts.append(_render_mechanism_alignment(proposal.mechanism_alignment))

    # ---- Proposal Debater 结果 (R19) ----
    if proposal.debate_result is not None and proposal.debate_result.rounds:
        parts.append(_render_debate(proposal.debate_result))

    # ---- Spot-check audit hint (Pri-4) ----
    if proposal.unverified_numbers:
        parts.append("\n\n## ⚠️ Audit: 未在 paper_evidence 找到出处的数字\n")
        parts.append(
            "以下数字出现在 motivation 段但 paper_evidence.quantitative_claims 里没出现，"
            "可能是 LLM 派生计算（合法）或编造（需修）。**请人工核对**：\n\n"
        )
        for n in proposal.unverified_numbers:
            parts.append(f"- `{n}`\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# 子渲染器
# ---------------------------------------------------------------------------

def _render_metadata_block(proposal: ResearchProposal, seed_text: str) -> str:
    """
    QuantSkip 顶部的 metadata 块：
        **状态**: 待审阅
        **级别**: top-tier
        **种子**: ...
        **创建时间**: ISO timestamp
    """
    # markdown 硬换行 = 行末两个空格 + 换行
    status_zh = _translate_status(proposal.status)
    seed_display = seed_text.strip() if seed_text else "(待补)"
    lines = [
        f"**状态**: {status_zh}  ",
        f"**级别**: {proposal.level or '(待补)'}  ",
        f"**种子**: {seed_display}  ",
        f"**创建时间**: {proposal.created_at or '(待补)'}",
    ]
    return "\n".join(lines)


_STATUS_MAP = {
    "draft": "待审阅",
    "under_review": "审稿中",
    "approved": "已立项",
    "rejected": "已驳回",
}


def _translate_status(s: str) -> str:
    return _STATUS_MAP.get(s, s or "(待补)")


def _render_methodology(proposal: ResearchProposal) -> str:
    """
    优先按 phases 渲染；如果没 phases 退回 proposed_method + technical_details 自由文本。
    """
    if proposal.methodology_phases:
        out = [proposal.proposed_method or ""]
        if out[0]:
            out.append("\n\n")
        for phase in proposal.methodology_phases:
            inputs = ", ".join(phase.inputs) if phase.inputs else "—"
            outputs = ", ".join(phase.outputs) if phase.outputs else "—"
            out.append(
                f"({phase.phase_number}) **{phase.name}**: {phase.description}\n"
                f"  - inputs: {inputs}\n"
                f"  - outputs: {outputs}\n"
                f"  - expected_compute_hours: {phase.expected_compute_hours:.1f}\n\n"
            )
        if proposal.technical_details:
            out.append(f"\n**Technical details**:\n{proposal.technical_details}")
        # ---- expected outcomes 拼到最后 ----
        out.append(f"\n\n**Expected outcomes**:\n{_render_outcomes(proposal)}")
        return "".join(out).rstrip()

    # fallback：无 phases 时，至少把 proposed_method + technical_details 渲出来
    fallback = []
    if proposal.proposed_method:
        fallback.append(proposal.proposed_method)
    if proposal.technical_details:
        fallback.append(f"\n\n**Technical details**:\n{proposal.technical_details}")
    fallback.append(f"\n\n**Expected outcomes**:\n{_render_outcomes(proposal)}")
    return "".join(fallback) or "(待补)"


def _render_outcomes(proposal: ResearchProposal) -> str:
    """
    优先用 expected_outcomes_structured（v3 schema）；fallback 到老 str 字段。
    structured 渲染时显式分 positive / negative / why-both 三段。
    """
    structured = proposal.expected_outcomes_structured
    if structured is not None:
        return (
            f"- **正向发现**: {structured.positive_finding}\n"
            f"- **反向发现**: {structured.negative_finding}\n"
            f"- **为什么两种结果都可发表**: {structured.why_both_publishable}"
        )
    return proposal.expected_outcomes or "(待补)"


def _render_venues(proposal: ResearchProposal) -> str:
    lines = []
    if proposal.target_venue:
        deadline = f" (截止日 {proposal.target_deadline})" if proposal.target_deadline else ""
        lines.append(f"- {proposal.target_venue}{deadline} — primary target")
    if proposal.fallback_venue:
        lines.append(f"- {proposal.fallback_venue} — fallback if timeline slips")
    return "\n".join(lines) if lines else "(待补)"


def _render_references(proposal: ResearchProposal) -> str:
    """
    优先用 key_references_formatted（QuantSkip 风格的 'Name: Title (Venue Year)'），
    fallback 到 key_references 裸 paperId 列表。
    """
    if proposal.key_references_formatted:
        return "\n".join(f"- {r}" for r in proposal.key_references_formatted)
    if proposal.key_references:
        return "\n".join(f"- {r}" for r in proposal.key_references)
    return "(待补)"


def _render_resource_estimate(proposal: ResearchProposal) -> str:
    est = proposal.resource_estimate
    lines = []
    if est.auto_research:
        lines.append(f"- auto_research: {_format_dict(est.auto_research)}")
    if est.human_in_loop:
        lines.append(f"- human_in_loop: {_format_dict(est.human_in_loop)}")
    if est.manual:
        lines.append(f"- manual: {_format_dict(est.manual)}")
    if not lines:
        lines.append("(待补)")
    return "\n".join(lines)


def _format_dict(d: dict) -> str:
    """把 {gpu_hours: 24, usd_cost: 50, wall_clock_days: 3} 渲成可读字符串"""
    if not d:
        return "(待补)"
    return ", ".join(f"{k}={v}" for k, v in d.items())


def _render_novelty(na: NoveltyAssessment) -> str:
    """渲染 NoveltyAssessment 为 markdown 子段"""
    lines = [
        f"- **overlap_level**: `{na.overlap_level}` (novelty_score={na.novelty_score:.2f})",
    ]
    if na.closest_work_title:
        ref = na.closest_work_title
        if na.closest_work_paper_id:
            ref = f"{ref} ({na.closest_work_paper_id})"
        lines.append(f"- **closest prior work**: {ref}")
    if na.overlap_summary:
        lines.append(f"- **overlap**: {na.overlap_summary}")
    if na.differentiation_gap:
        lines.append(f"- **differentiation gap**: {na.differentiation_gap}")
    return "\n".join(lines)


def _render_feasibility_challenge(fc: FeasibilityChallenge) -> str:
    """
    渲染 FeasibilityChallenger 输出为 '⚠️ Feasibility Risks' section。

    severity 用 emoji 区分以便扫读：🟢 low / 🟡 medium / 🔴 high
    verdict 总评放最上面，risks 按 severity 降序列出。
    """
    sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    verdict_icon = {"go": "🟢", "go_with_mitigations": "🟡", "rework": "🔴"}
    parts = ["\n\n## ⚠️ Feasibility Risks (R9c adversarial pass)\n\n"]
    parts.append(
        f"**Verdict**: {verdict_icon.get(fc.overall_verdict, '')} `{fc.overall_verdict}`\n\n"
    )
    if fc.summary:
        parts.append(f"**Summary**: {fc.summary}\n\n")
    if not fc.risks:
        parts.append("_(no risks identified — 罕见，建议人工再 review 一遍)_\n")
        return "".join(parts)
    for r in fc.risks:
        icon = sev_icon.get(r.severity, "")
        parts.append(
            f"- {icon} **[{r.severity.upper()} / {r.category}]** {r.description}\n"
        )
        if r.mitigation:
            parts.append(f"  - *mitigation*: {r.mitigation}\n")
    return "".join(parts)


def _render_mechanism_alignment(ma: MechanismAlignment) -> str:
    """
    渲染 R11 Mechanism Alignment Checker 输出。

    使用 emoji 让 PI 一眼看到 verdict：
    🟢 aligned / 🟡 loose_analogy / 🔴 hand_waved
    各维度也用 emoji：✅ aligned / 🟡 loose / ❌ broken
    """
    overall_icon = {
        "aligned": "🟢",
        "loose_analogy": "🟡",
        "hand_waved": "🔴",
        "not_applicable": "⚪",
    }
    dim_icon = {"aligned": "✅", "loose": "🟡", "broken": "❌"}
    dim_label = {
        "formal_correspondence": "Formal correspondence",
        "assumption_correspondence": "Assumption correspondence",
        "metric_correspondence": "Metric correspondence",
        "invariant_correspondence": "Invariant correspondence",
        "scaling_correspondence": "Scaling correspondence",
    }
    parts = ["\n\n## ⚠️ Mechanism Alignment (R11 cross-domain check)\n\n"]
    parts.append(
        f"**Verdict**: {overall_icon.get(ma.overall_verdict, '')} `{ma.overall_verdict}`\n\n"
    )
    if ma.source_domain or ma.target_domain:
        parts.append(
            f"**Cross-domain mapping**: `{ma.source_domain}` → `{ma.target_domain}`\n\n"
        )
    if ma.recommendation:
        parts.append(f"**Recommendation**: {ma.recommendation}\n\n")
    if not ma.dimensions:
        parts.append("_(no dimensional critique reported)_\n")
        return "".join(parts)
    parts.append("**Dimensional breakdown**:\n\n")
    for d in ma.dimensions:
        icon = dim_icon.get(d.verdict, "")
        label = dim_label.get(d.dimension, d.dimension)
        parts.append(f"- {icon} **{label}** [{d.verdict}]: {d.explanation}\n")
    return "".join(parts)


def _render_debate(dr: DebateResult) -> str:
    """
    渲染 R19/R20 三方辩论结果。

    R20: 三档 verdict (go/conditional_go/no_go) + Main / D&B Track 分别 rate
    + 沿用 R19 多轮 advocate/challenger/judge 论点展示
    """
    verdict_icon = {"go": "🟢", "conditional_go": "🟡", "no_go": "🔴"}.get(
        dr.final_verdict, "⚪"
    )
    parts = ["\n\n## ⚖️ Proposal Debate (R19/R20 advocate vs challenger vs judge)\n\n"]
    parts.append(
        f"**Verdict**: {verdict_icon} `{dr.final_verdict}` "
        f"({'converged' if dr.converged else 'not converged'})\n\n"
    )
    parts.append(
        f"**NeurIPS Main Track 中稿率**: `{dr.final_acceptance_rate_main:.2f}` | "
        f"**D&B Track 中稿率**: `{dr.final_acceptance_rate_db:.2f}`\n\n"
    )
    parts.append(f"**Rounds**: {len(dr.rounds)}\n\n")
    parts.append("**Main Track rate trajectory**: " + " → ".join(
        f"{r.acceptance_rate_main:.2f}" for r in dr.rounds
    ) + "\n\n")

    for r in dr.rounds:
        parts.append(
            f"### Round {r.round_number} "
            f"(main={r.acceptance_rate_main:.2f} / db={r.acceptance_rate_db:.2f})\n\n"
        )
        parts.append("**🟦 Advocate**:\n\n")
        parts.append(f"> {r.advocate_argument}\n\n")
        parts.append("**🟥 Challenger**:\n\n")
        parts.append(f"> {r.challenger_argument}\n\n")
        parts.append("**⚖️ Judge**:\n\n")
        parts.append(f"> {r.judge_assessment}\n\n")
        if r.revisions_proposed:
            parts.append("**Judge 修订建议**:\n\n")
            for rev in r.revisions_proposed:
                parts.append(f"- {rev}\n")
            parts.append("\n")
    return "".join(parts)


def render_tournament_overview(
    tournament,
    proposals: list,
) -> str:
    """
    把 TournamentResult 渲染成"## 锦标赛全候选概览"段。

    Args:
        tournament: TournamentResult
        proposals: 全部 candidate proposals（list[ResearchProposal]）
    """
    if not tournament or not tournament.elo_rankings:
        return ""

    title_to_proposal = {(p.title or f"proposal_{i}"): p
                         for i, p in enumerate(proposals)}

    lines = ["## 锦标赛全候选概览\n"]
    lines.append(
        f"本次 Phase C 锦标赛跑了 {len(tournament.matches)} 场 pairwise "
        f"对决，按 Elo 排序如下：\n"
    )

    for i, ranking in enumerate(tournament.elo_rankings):
        pid = ranking["proposal_id"]
        p = title_to_proposal.get(pid)
        is_disq = pid in tournament.disqualified_ids
        is_top = pid in tournament.top_k_ids[:2]

        badges = []
        if is_top and not is_disq:
            badges.append("🏆 top-2")
        if is_disq:
            badges.append("⚠️ 撞车")
        badge_str = " ".join(badges)

        title_short = (p.title if p else pid)[:80]
        lines.append(
            f"\n### {i+1}. {badge_str} (Elo={ranking['elo']:.1f}, "
            f"W{ranking['wins']}/L{ranking['losses']}/T{ranking['ties']})"
        )
        lines.append(f"\n**{title_short}**")
        if p and p.elevator_pitch:
            lines.append(f"\n{p.elevator_pitch[:300]}")
        if p and p.novelty_assessment:
            na = p.novelty_assessment
            lines.append(
                f"\n*novelty: `{na.overlap_level}` "
                f"({na.novelty_score:.2f}); "
                f"closest: {na.closest_work_title[:60] or 'n/a'}*"
            )

    if tournament.disqualified_ids:
        lines.append(
            f"\n\n**说明**: ⚠️ 撞车 = SciMON novelty boost 未收敛 "
            f"(overlap_level=substantial+)，建议丢弃或重新设计。"
        )
    return "\n".join(lines)
