"""
Feasibility Challenger (R9c) — adversarial pass，专门攻击 top-1 proposal 的可行性弱点。

设计动机：
- HindSight 2026 / Si et al. 实测 LLM 单 judge pass 系统性低估 feasibility（倾向给"听
  起来 novel"的 idea 高分，忽略 phase 时间不够 / 依赖未成熟工具 / 数据获取阻塞）
- Tournament 的 feasibility 维度是 pairwise 相对排序，绝对的可行性"洞"还得专门 attack
- Co-Scientist / DeepReview 都引入 adversarial reviewer 角色，单独跑 feasibility 攻击

成本：仅在 top-1 proposal 跑 1 次 LLM call（~6s, 0.001$），non-blocking 输出 risks 列表，
渲染到 seed.md 的 "⚠️ Feasibility Risks" section。

为什么 non-blocking：
- LLM 可能误报（说 "scope too broad" 但实际 phase 写得很细）
- 强行 block 会让 pipeline 经常卡在 "rework" 死循环
- PI 看 risks 列表 + mitigation 后人工决定 go/no-go
"""

from __future__ import annotations

import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import (
    FeasibilityChallenge,
    FeasibilityRisk,
    ResearchConstraints,
    ResearchProposal,
)
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry


_VALID_CATEGORIES = {
    "budget", "dependency", "data", "timeline", "scope", "evaluation", "other",
}
_VALID_SEVERITIES = {"low", "medium", "high"}
_VALID_VERDICTS = {"go", "go_with_mitigations", "rework"}


_CHALLENGER_PROMPT = """你是一名严苛的科研可行性 reviewer（adversarial reviewer），
专门挑 proposal 的可行性漏洞。你的目标是在执行前发现"看起来合理但跑不通"的 trap。

【攻击维度】
- budget: GPU-hours / wall-clock 是否真够？phase compute_hours 加起来 vs 总预算 budget
  超出 / 是否假设了 batch size 太大塞不下 GPU memory？
- dependency: 是否依赖未成熟的工具/库？（如 'requires SGLang fast-path' 但 SGLang 在该
  scenario 下是否稳定？是否要自己实现 CUDA kernel？）
- data: 数据获取是否阻塞？是否要 human annotation（被 constraints 禁了）？是否有 license/
  access 问题？是否假设了某 dataset 存在但实际不公开？
- timeline: phase 时间分配是否过紧？是否给 debug 留时间？是否假设 LLM call 永远成功无 retry？
- scope: 方法是否过于宽泛？三个 contribution 写一篇 paper 摆不下？是否未对齐 venue 长度限制？
- evaluation: 提到的 metric 是否真存在？baseline 是否可获取？是否要重跑 baseline 还是用论文报告的数？
- other: 上述未覆盖的可行性 risk

【严重性定义】
- low: warning。可上线但执行时要监控（如"timeline 紧张需要严格 stand-up"）
- medium: 不修就出问题。需要 mitigation 后再执行（如"phase 2 200h 但 budget 只剩 168，
  砍 ablation 数即可"）
- high: 阻断性。不修就跑不通（如"依赖 dataset X 但 X 实际不公开，必须换数据源"）

【输出要求】
- 严格 JSON，无 markdown code fence
- description 必须含具体出处（引用 phase 编号 / 方法 sub-component），不要空泛
- mitigation 必须 actionable（有具体改法），不能是 "需要更多研究"
- overall_verdict:
    * go = 0 个 high + ≤2 个 medium
    * go_with_mitigations = 0 个 high + ≥3 medium 或 ≥1 个不易解决的 medium
    * rework = ≥1 个 high

【输出 schema】
{
  "risks": [
    {
      "category": "budget" | "dependency" | "data" | "timeline" | "scope" | "evaluation" | "other",
      "severity": "low" | "medium" | "high",
      "description": "≤80 词具体描述，含出处",
      "mitigation": "actionable 修补建议，可以为空字符串"
    },
    ...
  ],
  "overall_verdict": "go" | "go_with_mitigations" | "rework",
  "summary": "30-80 词总评，给 PI 看的执行建议"
}

不要输出空 risks 列表 — 任何 proposal 都能找到至少 1 条值得标的 low/medium 风险。
"""


def _format_constraints(c: ResearchConstraints) -> str:
    """把 constraints 渲染成 attacker 能看的简表"""
    return (
        f"- gpu: {c.gpu_count}× {c.gpu_model}\n"
        f"- gpu_hours_budget: {c.gpu_hours_budget}\n"
        f"- wall_clock_days: {c.wall_clock_days}\n"
        f"- max_model_params_b: {c.max_model_params_b}\n"
        f"- forbidden_techniques: {c.forbidden_techniques}\n"
        f"- require_human_annotation_allowed: {c.require_human_annotation}\n"
        f"- require_no_api_for_main: {c.require_no_api_for_main}\n"
        f"- target_venues: {c.target_venues}\n"
    )


def _format_phases(proposal: ResearchProposal) -> str:
    """phase 列表（attacker 主要从这里挑 budget/timeline 漏洞）"""
    if not proposal.methodology_phases:
        return "(no phases)"
    lines = []
    for ph in proposal.methodology_phases:
        lines.append(
            f"  Phase {ph.phase_number}: {ph.name} | "
            f"{ph.expected_compute_hours}h | "
            f"description: {ph.description[:160]}"
        )
    lines.append(
        f"  total_estimated_hours: {proposal.total_estimated_hours} / "
        f"fits_budget: {proposal.fits_resource_budget}"
    )
    return "\n".join(lines)


def challenge_feasibility(
    proposal: ResearchProposal,
    constraints: ResearchConstraints,
    llm: BaseChatModel,
) -> FeasibilityChallenge | None:
    """
    对 single proposal 做 adversarial feasibility attack。

    Args:
        proposal: 通常是 tournament top-1
        constraints: 跟 elaborator 看到的同一份（attacker 要算 budget 必须知道）
        llm: 推荐 cheap-mid（attacker prompt 短，不需要顶配）

    Returns:
        FeasibilityChallenge 或 None（LLM 失败）
    """
    user_msg = (
        f"【Constraints】\n{_format_constraints(constraints)}\n\n"
        f"【Proposal title】\n{proposal.title}\n\n"
        f"【Motivation (摘)】\n{proposal.motivation[:600]}\n\n"
        f"【Proposed method (摘)】\n{proposal.proposed_method[:600]}\n\n"
        f"【Technical details (摘)】\n{proposal.technical_details[:600]}\n\n"
        f"【Phases】\n{_format_phases(proposal)}\n\n"
        f"【Target venue】\n{proposal.target_venue} (deadline: {proposal.target_deadline})\n\n"
        "请按 SYSTEM_PROMPT 输出 JSON，至少 1 条 risk。"
    )
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_CHALLENGER_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(
            f"[feasibility_challenger] 失败: {type(e).__name__}: {str(e)[:120]}",
            file=sys.stderr,
        )
        return None

    risks = _parse_risks(raw.get("risks") or [])
    verdict = str(raw.get("overall_verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        verdict = _derive_verdict(risks)
    summary = str(raw.get("summary", "")).strip()[:300]

    return FeasibilityChallenge(
        risks=risks,
        overall_verdict=verdict,
        summary=summary,
    )


def _parse_risks(raw_risks: list) -> list[FeasibilityRisk]:
    """parse + 清洗 LLM 给的 risks 列表，无效条目丢弃，按 severity 降序排"""
    out: list[FeasibilityRisk] = []
    for r in raw_risks:
        if not isinstance(r, dict):
            continue
        cat = str(r.get("category", "")).strip().lower()
        sev = str(r.get("severity", "")).strip().lower()
        desc = str(r.get("description", "")).strip()
        if cat not in _VALID_CATEGORIES or sev not in _VALID_SEVERITIES or not desc:
            continue
        out.append(FeasibilityRisk(
            category=cat,
            severity=sev,
            description=desc[:400],
            mitigation=str(r.get("mitigation", "")).strip()[:300],
        ))
    sev_order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda r: sev_order[r.severity])
    return out


def _derive_verdict(risks: list[FeasibilityRisk]) -> str:
    """LLM 没给合法 verdict 时按 risks 自动推断"""
    n_high = sum(1 for r in risks if r.severity == "high")
    n_med = sum(1 for r in risks if r.severity == "medium")
    if n_high >= 1:
        return "rework"
    if n_med >= 3:
        return "go_with_mitigations"
    return "go"
