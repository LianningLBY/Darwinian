"""
Proposal Tournament — Multi-idea Tournament Selection (借鉴 Co-Scientist Elo 模式)。

设计动机：当前 pipeline 一次只生 1 个 idea，撞车（identical with prior work）就完了，
没备选。而且单 idea 易被 elaborator 偏向带跑（v9 EAT 方向漂成 EGAT）。

本模块解决：
  1. multi_elaborate: 一次生 N=5 个 candidate proposal，每个 anchor 不同 phenomenon
  2. run_tournament: C(N,2) pairwise LLM compare → Elo 排名
  3. top-K 输出，撞车的（novelty boost 未收敛）标 disqualified

参考：DeepMind Co-Scientist (Gemini 2.0) 的 Elo 锦标赛设计。
LLM pairwise comparison 比绝对打分校准更好（HindSight 2026 指出 LLM 绝对打分
系统性偏向"听起来 novel"的 idea）。

成本：1 次 multi_elaborate (5 LLM call) + 10 次 pairwise (10 LLM call) ≈ 6 min
"""

from __future__ import annotations

import sys
import itertools

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import (
    AbstractionBranch,
    Phenomenon,
    ResearchMaterialPack,
    ResearchProposal,
    TournamentMatch,
    TournamentResult,
)
from darwinian.agents.proposal_elaborator import elaborate_proposal_from_pack
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry


_VALID_WINNERS = {"a", "b", "tie"}
_DEFAULT_ELO_START = 1200.0
_DEFAULT_ELO_K = 32.0


# ===========================================================================
# multi_elaborate: 5 个候选 proposal，每个 anchor 不同 phenomenon
# ===========================================================================

def multi_elaborate(
    skeleton: AbstractionBranch,
    pack: ResearchMaterialPack,
    llm: BaseChatModel,
    *,
    n_candidates: int = 5,
) -> list[ResearchProposal]:
    """
    生 N 个 candidate proposal。每个 anchor 不同的 phenomenon（如果 phenomena 数量
    不够 N，则降级用 hook anchor，再不够用同 pack 跑），保证 idea 多样性。

    返回：list[ResearchProposal]，长度 ≤ n_candidates。失败的不计入。
    """
    proposals: list[ResearchProposal] = []
    anchors = _select_anchors(pack, n_candidates)
    print(f"[tournament] multi_elaborate 用 {len(anchors)} 个不同 anchor 跑 elaborator",
          file=sys.stderr)

    for i, anchor_pack in enumerate(anchors):
        print(f"[tournament] candidate {i+1}/{len(anchors)} 启动 elaborator",
              file=sys.stderr)
        proposal = elaborate_proposal_from_pack(
            skeleton=skeleton, material_pack=anchor_pack, llm=llm,
        )
        if proposal is not None:
            proposals.append(proposal)
        else:
            print(f"[tournament] candidate {i+1} elaborate 失败，跳过", file=sys.stderr)
    print(f"[tournament] multi_elaborate 完成: {len(proposals)}/{len(anchors)} 成功",
          file=sys.stderr)
    return proposals


def _select_anchors(
    pack: ResearchMaterialPack,
    n_candidates: int,
) -> list[ResearchMaterialPack]:
    """
    给每个 candidate 一份不同的 anchor pack：选一个 phenomenon / hook 作主线，
    其他素材保留。返回 N 份变体 pack。

    策略：
    - 优先用 phenomenon 当 anchor（比 hook 更深的 idea 信号）
    - phenomenon 不够 → hook 补
    - 都不够 → 复用 pack（多样性靠 LLM 温度）
    """
    n_phenomena = len(pack.phenomena)
    n_hooks = len(pack.structural_hole_hooks)
    anchor_packs: list[ResearchMaterialPack] = []

    for i in range(n_candidates):
        if i < n_phenomena:
            # 把这个 phenomenon 提到第一位（elaborator prompt 默认按顺序看）
            ph_anchor = pack.phenomena[i]
            other_phenomena = [p for j, p in enumerate(pack.phenomena) if j != i]
            anchor_packs.append(pack.model_copy(update={
                "phenomena": [ph_anchor] + other_phenomena,
            }))
        elif i < n_phenomena + n_hooks:
            # 把这个 hook 提到第一位
            h_idx = i - n_phenomena
            h_anchor = pack.structural_hole_hooks[h_idx]
            other_hooks = [h for j, h in enumerate(pack.structural_hole_hooks) if j != h_idx]
            anchor_packs.append(pack.model_copy(update={
                "structural_hole_hooks": [h_anchor] + other_hooks,
            }))
        else:
            # 素材不够，复用 pack（多样性靠 LLM 采样温度）
            anchor_packs.append(pack)

    return anchor_packs


# ===========================================================================
# pairwise_compare: 单场 LLM 比较
# ===========================================================================

_PAIRWISE_PROMPT = """你是科研评审专家。给定两份研究 proposal A 和 B（同一研究方向），
你要按三个维度比较哪个更强，并综合判定 winner。

输出严格 JSON：
{
  "winner": "a" | "b" | "tie",
  "rubric_scores": {
    "novelty": "a" | "b" | "tie",
    "feasibility": "a" | "b" | "tie",
    "impact": "a" | "b" | "tie"
  },
  "judge_reasoning": "一句话综合理由（30-80 词），含三维度分别比较"
}

【三维度定义】
- novelty: 跟已知 prior work 的差异度。重点看 motivation 是否揭示真正的 gap，
  方法是否避开已被做过的组合。
- feasibility: 给定 4×RTX PRO 6000 / 7 天预算，能否真跑通。看 phase 设计、
  方法成熟度、是否依赖未成熟工具。
- impact: 假设实验成功，对社区贡献多大。看是 incremental improvement 还是
  揭示深层 mechanism。

【tie 慎用】只有真分不出高下时才标 tie，避免懒判。
"""


def pairwise_compare(
    proposal_a: ResearchProposal,
    proposal_b: ResearchProposal,
    llm: BaseChatModel,
) -> TournamentMatch | None:
    """
    单场 LLM 比较 a vs b。失败返 None（外层不影响其他对决）。

    proposal_id 用 title（可能不唯一但实战够用）。
    """
    a_id = proposal_a.title or "(no_title_a)"
    b_id = proposal_b.title or "(no_title_b)"
    user_msg = (
        f"【proposal A】\n"
        f"title: {proposal_a.title}\n"
        f"motivation (摘): {proposal_a.motivation[:500]}\n"
        f"proposed_method (摘): {proposal_a.proposed_method[:500]}\n\n"
        f"【proposal B】\n"
        f"title: {proposal_b.title}\n"
        f"motivation (摘): {proposal_b.motivation[:500]}\n"
        f"proposed_method (摘): {proposal_b.proposed_method[:500]}\n\n"
        "请按 SYSTEM_PROMPT 输出 JSON。"
    )
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_PAIRWISE_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(f"[tournament] pairwise_compare 失败 {a_id[:30]} vs {b_id[:30]}: "
              f"{type(e).__name__}", file=sys.stderr)
        return None

    winner = str(raw.get("winner", "")).strip().lower()
    if winner not in _VALID_WINNERS:
        return None

    rubric = raw.get("rubric_scores") or {}
    rubric_clean = {
        k: str(rubric.get(k, "")).strip().lower()
        for k in ("novelty", "feasibility", "impact")
        if str(rubric.get(k, "")).strip().lower() in _VALID_WINNERS
    }

    return TournamentMatch(
        proposal_a_id=a_id,
        proposal_b_id=b_id,
        winner=winner,
        judge_reasoning=str(raw.get("judge_reasoning", "")).strip()[:300],
        rubric_scores=rubric_clean,
    )


# ===========================================================================
# Elo 排名
# ===========================================================================

def _expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score: prob(a wins) given ratings"""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update_elo(
    ratings: dict[str, float],
    a_id: str, b_id: str, winner: str,
    k: float = _DEFAULT_ELO_K,
) -> None:
    """In-place 更新 ratings dict。winner: 'a' / 'b' / 'tie'"""
    ra, rb = ratings[a_id], ratings[b_id]
    ea = _expected_score(ra, rb)
    eb = _expected_score(rb, ra)
    if winner == "a":
        score_a, score_b = 1.0, 0.0
    elif winner == "b":
        score_a, score_b = 0.0, 1.0
    else:   # tie
        score_a, score_b = 0.5, 0.5
    ratings[a_id] = ra + k * (score_a - ea)
    ratings[b_id] = rb + k * (score_b - eb)


def compute_elo_rankings(
    proposal_ids: list[str],
    matches: list[TournamentMatch],
) -> list[dict]:
    """
    根据 matches 算 Elo + 累计 win/loss/tie。

    Returns:
        list of dict: [{"proposal_id": ..., "elo": ..., "wins": ..., "losses": ..., "ties": ...}]
        按 elo 降序
    """
    ratings = {pid: _DEFAULT_ELO_START for pid in proposal_ids}
    stats = {pid: {"wins": 0, "losses": 0, "ties": 0} for pid in proposal_ids}

    for m in matches:
        if m.proposal_a_id not in ratings or m.proposal_b_id not in ratings:
            continue   # 防御：跨锦标赛串味
        _update_elo(ratings, m.proposal_a_id, m.proposal_b_id, m.winner)
        if m.winner == "a":
            stats[m.proposal_a_id]["wins"] += 1
            stats[m.proposal_b_id]["losses"] += 1
        elif m.winner == "b":
            stats[m.proposal_b_id]["wins"] += 1
            stats[m.proposal_a_id]["losses"] += 1
        else:
            stats[m.proposal_a_id]["ties"] += 1
            stats[m.proposal_b_id]["ties"] += 1

    rankings = []
    for pid in proposal_ids:
        rankings.append({
            "proposal_id": pid,
            "elo": round(ratings[pid], 1),
            **stats[pid],
        })
    rankings.sort(key=lambda r: r["elo"], reverse=True)
    return rankings


# ===========================================================================
# run_tournament: 主入口
# ===========================================================================

def run_tournament(
    proposals: list[ResearchProposal],
    llm: BaseChatModel,
    *,
    top_k: int = 2,
) -> TournamentResult:
    """
    主入口。对 N 个 proposal 跑 C(N,2) 次 pairwise → Elo 排名 → 取 top-K。

    Args:
        proposals: list[ResearchProposal]
        llm: 评审 LLM（推荐 cheap-mid 级）
        top_k: 取 top 几个（默认 2）

    Returns:
        TournamentResult，含 matches / elo_rankings / top_k_ids。
        disqualified_ids 由调用方根据 novelty assessment 单独填充（本函数不知道）。
    """
    if len(proposals) < 2:
        # 1 个 proposal 不需要锦标赛，直接当 top
        return TournamentResult(
            matches=[],
            elo_rankings=[{
                "proposal_id": (proposals[0].title or "single") if proposals else "",
                "elo": _DEFAULT_ELO_START,
                "wins": 0, "losses": 0, "ties": 0,
            }] if proposals else [],
            top_k_ids=[(proposals[0].title or "single")] if proposals else [],
        )

    proposal_ids = [(p.title or f"proposal_{i}") for i, p in enumerate(proposals)]
    # 处理 title 重复
    seen = set()
    for i, pid in enumerate(proposal_ids):
        if pid in seen:
            proposal_ids[i] = f"{pid}_{i}"
        seen.add(proposal_ids[i])
    id_to_proposal = dict(zip(proposal_ids, proposals))

    # C(N,2) 配对
    pairs = list(itertools.combinations(proposal_ids, 2))
    print(f"[tournament] 准备跑 {len(pairs)} 场 pairwise (C({len(proposal_ids)},2))",
          file=sys.stderr)

    matches: list[TournamentMatch] = []
    for i, (a_id, b_id) in enumerate(pairs):
        print(f"[tournament] match {i+1}/{len(pairs)}: "
              f"'{a_id[:30]}' vs '{b_id[:30]}'", file=sys.stderr)
        m = pairwise_compare(id_to_proposal[a_id], id_to_proposal[b_id], llm)
        if m is None:
            print(f"[tournament] match {i+1} 失败，跳过", file=sys.stderr)
            continue
        # 强制 id 一致（防 title 含特殊字符走样）
        m.proposal_a_id = a_id
        m.proposal_b_id = b_id
        matches.append(m)

    rankings = compute_elo_rankings(proposal_ids, matches)
    top_k_ids = [r["proposal_id"] for r in rankings[:top_k]]

    print(f"[tournament] 完成: top_{top_k} = {top_k_ids}", file=sys.stderr)
    return TournamentResult(
        matches=matches,
        elo_rankings=rankings,
        top_k_ids=top_k_ids,
    )
