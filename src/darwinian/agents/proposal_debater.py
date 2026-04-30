"""
Proposal Debater (R19) — Co-Scientist 风格的 Advocate / Challenger / Judge 三方辩论。

设计动机：
v3-v7 LIVE 实测 R9c (Feasibility) + R11 (Mechanism) + tournament Elo 都偏向"找问题"
（adversarial critique），缺少正方系统论证 idea **为什么值得做**。Co-Scientist 的
debate loop 是这个 gap 的标准答案：
- Advocate: 论证 idea 中稿可能性 + actionable 优势
- Challenger: 攻击 idea 弱点（跟 R9c 不同：R9c 攻工程，Challenger 攻 idea 本身）
- Judge: 综合两边给中稿率估计 + 修订建议

为什么需要：用户明确要求"seed 也要正反面去辩论是否可行"。

成本：3 LLM call / round × N round。默认 N=2（advocate-challenger-judge × 2 轮）≈ ¥0.3。
仅 top-1 跑。

收敛条件：
- final_acceptance_rate ≥ acceptance_threshold (默认 0.30)
- 最近两轮 |delta| < convergence_delta (默认 0.05)
- 任一条件满足 → converged=True 提前停止
- 否则跑满 max_rounds

Schema 早就在 state.py 定义了（DebateRound / DebateResult），这次实现而已。
"""

from __future__ import annotations

import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import (
    DebateResult,
    DebateRound,
    ResearchProposal,
)
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ADVOCATE_PROMPT = """你是科研 idea 的**正方辩护人**。给定一份 ResearchProposal，
你要论证为什么这个 idea 值得做、能中稿。

【任务】
1. 提炼 3-5 个最强 selling points（含具体数字 / 引用 / mechanism 说明）
2. 反驳常见质疑（"已有 prior work" / "scope 太宽" / "数据集风险"）
3. 估计中稿可能性（针对 target_venue，如 NeurIPS/EMNLP）

【输出严格 JSON】
{
  "argument": "300-500 词的辩护陈述，含 selling points + 反驳常见质疑 + 中稿估计",
  "estimated_acceptance_rate": 0.25,
  "key_strengths": ["strength 1 (≤30 词)", "strength 2", ...]
}

【❗】
- 不要无脑吹捧。如果 idea 真的弱，acceptance_rate 应低于 0.20
- argument 必须引用 proposal 里的具体 phase / metric / mechanism
- 不要写 <think>，直接 JSON 开头
"""


_CHALLENGER_PROMPT = """你是科研 idea 的**反方挑战者**。给定一份 ResearchProposal
和正方的 advocate_argument，你要攻击 idea 本身（不是工程可行性 — R9c 已做那个）。

【攻击维度】
1. **Novelty 真的成立吗** — 提名 2-3 篇可能撞车的真实 prior work（直接给名字 / 方向）
2. **Methodology 严谨度** — 实验设计 / metric / baseline 选择有没有漏洞
3. **Contribution 真的 publishable** — 即使技术 OK，故事是不是足够 interesting
4. **Generalization** — 即使在 specific dataset 上 work，对社区有没有普遍价值

不要重复 R9c 攻击的工程问题（budget / timeline / data license）。

【输出严格 JSON】
{
  "argument": "300-500 词的反驳陈述，含具体可能撞车的 prior work + methodology 漏洞",
  "weaknesses": ["weakness 1 (≤40 词)", "weakness 2", ...],
  "potential_collisions": ["可能撞车的具体论文名 / 方向 (≤30 词)", ...]
}

【❗】
- 不要无脑否定。如果 idea 真的 novel，weaknesses 可以只有 1-2 条
- 引用具体的 method 名 / dataset 名 / 论文名（没必要的话不要瞎编）
- 不要写 <think>，直接 JSON 开头
"""


_JUDGE_PROMPT = """你是中立科研 reviewer。已读完 advocate 论证 + challenger 反驳，
请综合裁决。

【任务】
1. 列出哪些 challenger 的论点站得住、哪些站不住（指名）
2. 估计 revised 中稿率（advocate 的 rate × 你的判断 challenger 命中程度）
3. 给 advocate 下一轮修订建议（actionable，非"加更多实验"这种空泛）

【输出严格 JSON】
{
  "assessment": "300-500 词裁决说明，逐条评 challenger 论点 + 综合判断",
  "estimated_acceptance_rate": 0.25,
  "revisions_proposed": [
    "actionable 修订点 1 (≤50 词)",
    "actionable 修订点 2",
    ...
  ]
}

【❗】
- estimated_acceptance_rate 应该比 advocate 给的更**保守**（中位数偏低），
  因为 reviewer 看 weakness 更敏感
- revisions_proposed 必须 actionable（如"换 ISCXVPN2016 数据集"），不要"加 more rigor"
- 不要写 <think>，直接 JSON 开头
"""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def debate_proposal(
    proposal: ResearchProposal,
    llm: BaseChatModel,
    *,
    max_rounds: int = 2,
    acceptance_threshold: float = 0.30,
    convergence_delta: float = 0.05,
) -> DebateResult:
    """
    对 proposal 跑 max_rounds 轮 advocate-challenger-judge 辩论。

    Args:
        proposal: 通常是 tournament top-1
        llm: 推荐 cheap-mid（每轮 3 call，N=2 共 6 call ≈ ¥0.3）
        max_rounds: 最多几轮（默认 2，实测足够看出 idea 强弱）
        acceptance_threshold: 立项门槛（默认 0.30）
        convergence_delta: 判收敛的相邻轮 delta（默认 0.05）

    Returns:
        DebateResult，含每轮 DebateRound + 最终 acceptance_rate + converged 标志
    """
    rounds: list[DebateRound] = []
    last_revisions: list[str] = []

    for round_idx in range(1, max_rounds + 1):
        print(f"[debate] Round {round_idx}/{max_rounds} 启动", file=sys.stderr)

        # 1. Advocate
        advocate_out = _run_advocate(proposal, llm, last_revisions)
        if advocate_out is None:
            print(f"[debate] Round {round_idx} advocate 失败，提前终止", file=sys.stderr)
            break

        # 2. Challenger
        challenger_out = _run_challenger(proposal, advocate_out["argument"], llm)
        if challenger_out is None:
            print(f"[debate] Round {round_idx} challenger 失败，提前终止", file=sys.stderr)
            break

        # 3. Judge
        judge_out = _run_judge(
            proposal, advocate_out["argument"], challenger_out["argument"], llm,
        )
        if judge_out is None:
            print(f"[debate] Round {round_idx} judge 失败，提前终止", file=sys.stderr)
            break

        rd = DebateRound(
            round_number=round_idx,
            advocate_argument=advocate_out["argument"][:1500],
            challenger_argument=challenger_out["argument"][:1500],
            judge_assessment=judge_out["assessment"][:1500],
            estimated_acceptance_rate=judge_out["acceptance_rate"],
            revisions_proposed=judge_out["revisions"],
        )
        rounds.append(rd)
        print(
            f"[debate] Round {round_idx} 完成: rate={rd.estimated_acceptance_rate:.2f}, "
            f"{len(rd.revisions_proposed)} 个修订建议",
            file=sys.stderr,
        )
        last_revisions = rd.revisions_proposed

        # 收敛检测
        if len(rounds) >= 2:
            delta = abs(
                rounds[-1].estimated_acceptance_rate
                - rounds[-2].estimated_acceptance_rate
            )
            if (
                rounds[-1].estimated_acceptance_rate >= acceptance_threshold
                and delta < convergence_delta
            ):
                print(
                    f"[debate] 收敛: rate={rounds[-1].estimated_acceptance_rate:.2f} "
                    f"≥ {acceptance_threshold}, delta={delta:.3f} < {convergence_delta}",
                    file=sys.stderr,
                )
                break

    if not rounds:
        return DebateResult(
            rounds=[],
            final_acceptance_rate=0.0,
            acceptance_threshold=acceptance_threshold,
            convergence_delta=convergence_delta,
            converged=False,
        )

    final_rate = rounds[-1].estimated_acceptance_rate
    converged = final_rate >= acceptance_threshold and (
        len(rounds) < 2
        or abs(rounds[-1].estimated_acceptance_rate
               - rounds[-2].estimated_acceptance_rate) < convergence_delta
    )
    return DebateResult(
        rounds=rounds,
        final_acceptance_rate=final_rate,
        acceptance_threshold=acceptance_threshold,
        convergence_delta=convergence_delta,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Helpers — 单个 LLM call
# ---------------------------------------------------------------------------


def _format_proposal_for_debate(p: ResearchProposal) -> str:
    """通用 proposal 摘要给三方都看"""
    phase_lines = [
        f"  P{ph.phase_number} ({ph.expected_compute_hours}h): {ph.name} — "
        f"{ph.description[:120]}"
        for ph in p.methodology_phases
    ]
    return (
        f"title: {p.title}\n"
        f"target venue: {p.target_venue}\n"
        f"motivation (摘): {p.motivation[:600]}\n"
        f"proposed_method (摘): {p.proposed_method[:600]}\n"
        f"phases (total {p.total_estimated_hours}h):\n"
        + ("\n".join(phase_lines) if phase_lines else "  (no phases)")
    )


def _run_advocate(
    proposal: ResearchProposal,
    llm: BaseChatModel,
    last_revisions: list[str],
) -> dict | None:
    user_msg = (
        f"【Proposal】\n{_format_proposal_for_debate(proposal)}\n\n"
        + (
            f"【上一轮 Judge 修订建议（采纳后再辩护）】\n"
            + "\n".join(f"- {r}" for r in last_revisions)
            + "\n\n"
            if last_revisions else ""
        )
        + "请按 SYSTEM_PROMPT 输出 JSON，论证此 idea 值得做。"
    )
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_ADVOCATE_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(f"[debate] advocate 失败: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None
    arg = str(raw.get("argument", "")).strip()
    if not arg:
        return None
    rate = float(raw.get("estimated_acceptance_rate", 0.0) or 0.0)
    return {
        "argument": arg,
        "rate": max(0.0, min(1.0, rate)),
        "strengths": [str(s).strip()[:200] for s in (raw.get("key_strengths") or [])],
    }


def _run_challenger(
    proposal: ResearchProposal,
    advocate_argument: str,
    llm: BaseChatModel,
) -> dict | None:
    user_msg = (
        f"【Proposal】\n{_format_proposal_for_debate(proposal)}\n\n"
        f"【正方 advocate 论证】\n{advocate_argument[:1500]}\n\n"
        "请按 SYSTEM_PROMPT 输出 JSON，攻击 idea 本身（非工程可行性）。"
    )
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_CHALLENGER_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(f"[debate] challenger 失败: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None
    arg = str(raw.get("argument", "")).strip()
    if not arg:
        return None
    return {
        "argument": arg,
        "weaknesses": [str(s).strip()[:200] for s in (raw.get("weaknesses") or [])],
        "collisions": [
            str(s).strip()[:200] for s in (raw.get("potential_collisions") or [])
        ],
    }


def _run_judge(
    proposal: ResearchProposal,
    advocate_argument: str,
    challenger_argument: str,
    llm: BaseChatModel,
) -> dict | None:
    user_msg = (
        f"【Proposal】\n{_format_proposal_for_debate(proposal)}\n\n"
        f"【正方 advocate 论证】\n{advocate_argument[:1500]}\n\n"
        f"【反方 challenger 反驳】\n{challenger_argument[:1500]}\n\n"
        "请按 SYSTEM_PROMPT 输出 JSON，综合裁决并给修订建议。"
    )
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_JUDGE_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(f"[debate] judge 失败: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None
    assessment = str(raw.get("assessment", "")).strip()
    if not assessment:
        return None
    rate = float(raw.get("estimated_acceptance_rate", 0.0) or 0.0)
    return {
        "assessment": assessment,
        "acceptance_rate": max(0.0, min(1.0, rate)),
        "revisions": [
            str(r).strip()[:300] for r in (raw.get("revisions_proposed") or [])
            if str(r).strip()
        ],
    }
