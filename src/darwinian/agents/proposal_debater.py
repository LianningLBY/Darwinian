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

_ADVOCATE_PROMPT = """你是科研 idea 的**正方辩护人**。给定一份 ResearchProposal 和 portfolio
资源约束，你要论证为什么这个 idea 值得做、能中稿，**同时不能盲目吹捧**。

【6 维度评审框架（必须每条都简短论证，参考 jaywen 外部对抗模板）】
1. **Motivation 强度**：方向问题真实吗？需求 well-documented 吗？
2. **Methodological Novelty**：方法/视角真新吗？跟 prior work 区分明确吗？
3. **Expected Impact**：实验成功后社区会被影响多少？
4. **Statistical Robustness**：实验设计 sample size / baseline / null control 站得住吗？
5. **Execution Feasibility (under portfolio constraints)**：能在
   ≤200 GPU-hours / API ≤$200 / 不要大量人工标注 的硬约束下跑完吗？
6. **中稿概率**：分 NeurIPS Main Track / D&B Track 分别估计

【任务】
1. 提炼 3-5 个最强 selling points（含具体数字 / 引用 / mechanism 说明）
2. 反驳常见质疑（"已有 prior work" / "scope 太宽" / "数据集风险"）
3. 给出 Main Track 和 D&B Track 中稿率（不能虚高 — 大部分 idea 即使强也只 15-25%）

【输出严格 JSON】
{
  "argument": "400-600 词的辩护陈述，必须按 6 维度逐条简短论证 + 反驳常见质疑",
  "acceptance_rate_main": 0.15,
  "acceptance_rate_db": 0.30,
  "estimated_acceptance_rate": 0.15,
  "key_strengths": ["strength 1 (≤30 词)", "strength 2", ...]
}

【❗ Anti-cheerleading】
- 不要无脑吹捧。如果 idea 真的弱（如 evaluation paper 投 Main Track）main_rate 应 < 0.10
- 主攻 Main Track，但坦诚承认 — evaluation/benchmark paper 通常 D&B Track 更合适
- argument 必须引用 proposal 里的具体 phase / metric / mechanism
- estimated_acceptance_rate = acceptance_rate_main（pipeline 主指标）
- 不要写 <think>，直接 JSON 开头
"""


_CHALLENGER_PROMPT = """你是科研 idea 的**反方挑战者**（devil's advocate）。给定一份
ResearchProposal 和正方 advocate_argument，你要攻击 idea 本身。

【辩论规则】
- **每轮先承认对方说得对的地方**（避免无意义拉锯）
- 然后逐点反驳，具体到技术细节，不要泛泛而谈
- **不要重复 R9c 攻击的工程问题**（budget / timeline / data license）— 那是另一个 agent 的事

【5 维度攻击框架（参考 jaywen 外部对抗模板）】
1. **Motivation 真问题吗** — 是不是 advocate 编出来的需求？社区真在乎吗？
2. **Novelty 真成立吗** — 提名 2-3 篇可能撞车的真实 prior work（具名）
3. **Methodology 严谨度** — 实验设计 / metric / baseline / null control / sample size 漏洞
4. **Statistical Robustness** — power analysis 站得住吗？n 够大吗？多重比较 correction 做了吗？
5. **Contribution 真的 publishable** — 故事 fundamental 吗？还是 incremental empirical？
   - 这是 method paper 还是 evaluation/benchmark paper？投错 venue (Main Track vs D&B) 会 desk reject

【输出严格 JSON】
{
  "concessions": ["承认 advocate 说得对的 1-2 点 (≤40 词)"],
  "argument": "400-600 词的反驳陈述，含具体可能撞车的 prior work + methodology 漏洞 + venue mismatch",
  "weaknesses": ["weakness 1 (≤40 词)", "weakness 2", ...],
  "potential_collisions": ["可能撞车的具体论文名 / 方向 (≤30 词)", ...],
  "venue_mismatch_risk": "如果 idea 是 evaluation paper 投 Main Track，明确指出"
}

【❗】
- 不要无脑否定。如果 idea 真的 novel，weaknesses 可以只 1-2 条
- 引用具体的 method 名 / dataset 名 / 论文名（不知道就不要瞎编）
- 必须 concessions ≥1 条（防无脑攻击拉锯）
- 不要写 <think>，直接 JSON 开头
"""


_JUDGE_PROMPT = """你是中立科研 reviewer。已读完 advocate 论证 + challenger 反驳，
请综合裁决并给出 Go / No-Go / Conditional Go 决策。

【任务】
1. 列出哪些 challenger 的论点站得住、哪些站不住（指名）
2. 估计 revised 中稿率（分 Main Track / D&B Track）— 必须比 advocate 保守
3. 给三档决策 + actionable 修订建议

【三档决策（参考 jaywen 外部对抗模板的 Go/No-Go/Conditional Go）】
- **go**: 立项进 portfolio，可执行（rate_main ≥ 0.15 且无致命漏洞）
- **conditional_go**: 按 revisions_proposed 修补后立项（rate_main ≥ 0.10 但有
  非致命问题，如 venue 错 / scope 砍 / dataset 换）
- **no_go**: 不进 portfolio（rate_main < 0.10 或有致命漏洞如 idea 本身立不住 /
  资源 3x 超支 / methodology 伪相关）

【输出严格 JSON】
{
  "assessment": "400-600 词裁决说明，逐条评 challenger 论点 + 综合判断",
  "verdict": "go" | "conditional_go" | "no_go",
  "acceptance_rate_main": 0.10,
  "acceptance_rate_db": 0.25,
  "estimated_acceptance_rate": 0.10,
  "revisions_proposed": [
    "actionable 修订点 1 (≤50 词)",
    "actionable 修订点 2",
    ...
  ]
}

【❗】
- rate_main 应比 advocate 给的更**保守**（reviewer 对 weakness 更敏感）
- estimated_acceptance_rate = acceptance_rate_main（pipeline 主指标）
- revisions 必须 actionable：✅"换 ISCXVPN2016"，❌"加 more rigor"
- venue 错（evaluation paper 投 Main）→ 默认 conditional_go (revisions=换 D&B)
  + main rate 低 db rate 高
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
    last_verdict: str = "no_go"   # R20: 跟踪最后一轮 verdict

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
            acceptance_rate_main=judge_out.get("rate_main", judge_out["acceptance_rate"]),
            acceptance_rate_db=judge_out.get("rate_db", 0.0),
            revisions_proposed=judge_out["revisions"],
        )
        rounds.append(rd)
        print(
            f"[debate] Round {round_idx} 完成: main={rd.acceptance_rate_main:.2f} / "
            f"db={rd.acceptance_rate_db:.2f}, verdict={judge_out.get('verdict', '?')}, "
            f"{len(rd.revisions_proposed)} 个修订建议",
            file=sys.stderr,
        )
        last_revisions = rd.revisions_proposed
        last_verdict = judge_out.get("verdict", "no_go")

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
            final_verdict="no_go",
            final_acceptance_rate_main=0.0,
            final_acceptance_rate_db=0.0,
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
        final_verdict=last_verdict if last_verdict in {"go", "no_go", "conditional_go"} else "no_go",
        final_acceptance_rate_main=rounds[-1].acceptance_rate_main,
        final_acceptance_rate_db=rounds[-1].acceptance_rate_db,
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
    rate_main = float(raw.get("acceptance_rate_main", rate) or rate)
    rate_db = float(raw.get("acceptance_rate_db", 0.0) or 0.0)
    verdict_raw = str(raw.get("verdict", "")).strip().lower()
    verdict = verdict_raw if verdict_raw in {"go", "no_go", "conditional_go"} else (
        # 兜底：根据 rate 推断
        "go" if rate_main >= 0.15
        else "conditional_go" if rate_main >= 0.10
        else "no_go"
    )
    return {
        "assessment": assessment,
        "verdict": verdict,
        "rate_main": max(0.0, min(1.0, rate_main)),
        "rate_db": max(0.0, min(1.0, rate_db)),
        "acceptance_rate": max(0.0, min(1.0, rate)),
        "revisions": [
            str(r).strip()[:300] for r in (raw.get("revisions_proposed") or [])
            if str(r).strip()
        ],
    }
