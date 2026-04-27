"""
Novelty Booster — SciMON-style 新颖性迭代提升。

设计动机：v7 elaborator 出来的 idea 有时跟 prior work 高度重叠（如 SpecMQuant
已经做过 quantized draft + acceptance rate 测量），但 pipeline 没机制发现。
本工具关闭这个洞：每个 proposal 出来后调一次 novelty boost loop——
搜最相似 prior work → LLM 评估 overlap → 若 substantial 则 refine 突出差异 →
循环 max_rounds 轮。

成本：每轮 ~2 次 S2 search + 2 次 LLM 调用 ≈ 30s。3 轮 < 2 分钟。

参考：SciMON (ACL 2024) 的 iterative novelty boosting loop。
"""

from __future__ import annotations

import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import (
    NoveltyAssessment,
    NoveltyBoostResult,
    ResearchProposal,
)
from darwinian.tools.semantic_scholar import search_papers
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.knowledge_graph import _dedup_papers_by_id
from darwinian.utils.llm_retry import invoke_with_retry


_VALID_OVERLAP_LEVELS = {"none", "partial", "substantial", "identical"}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUERY_EXTRACT_PROMPT = """你是 academic search 专家。给定一份研究 proposal，输出 1-2 条
最能命中"该 proposal 的潜在 prior work"的 S2 搜索 query。

输出严格 JSON：{"queries": ["query1", "query2"]}

【约束】
1. 每条 query 3-7 个英文词，含 proposal 的核心方法名 + 评估指标
2. 不要太宽（避免命中泛论文），也不要太窄（避免命中不到论文）
3. 优先用 proposal 提到的具体技术 / 数据集
4. 输出 1-2 条即可
"""


_OVERLAP_ASSESS_PROMPT = """你是科研新颖性评审专家。给定一份"我们的 proposal"和几篇 prior work
（已通过相似度搜索召回），你要判断 ours 与 prior work 中**最相似那一篇**的重叠程度。

输出严格 JSON：
{
  "closest_paper_id": "...",
  "closest_title": "...",
  "overlap_level": "none" | "partial" | "substantial" | "identical",
  "overlap_summary": "一句话说明 ours 跟最相似那篇哪些内容重叠",
  "differentiation_gap": "ours 还需要怎么改才能跟它真正区分开（非空，给后续 refinement 用）",
  "novelty_score": 0.0-1.0
}

【overlap_level 判断标准】
- none: prior work 跟 ours 方向完全不同（如 ours 做 LLM inference, prior 做 vision）
- partial: 有交集但 contribution 角度不同（ours 测 X 指标，prior 测 Y 指标）
- substantial: prior 已做过类似的事，ours 仅在数据集 / 模型规模上有别
- identical: 本质同一个 idea，ours 几乎是 prior 的复现

【novelty_score 对照】
- 0.9-1.0: none
- 0.6-0.9: partial（可发表）
- 0.3-0.6: substantial（需大改）
- 0.0-0.3: identical（应丢弃或重新设计）
"""


_REFINE_PROMPT = """你是 proposal 修订专家。基于 prior work overlap 反馈，重写 motivation 和
proposed_method 两段，让 ours 跟 closest_work 真正区分开。

输出严格 JSON：{"motivation": "...", "proposed_method": "..."}

【修订要求】
1. motivation 段必须**显式提到 closest_work**（"While X et al. showed Y..."）
2. proposed_method 段开头必须有"**Key differentiation**: ..."一段，明确说明
   ours 做的事 prior 没做
3. 不要删除原有的具体数字引用，要在保留之上突出差异
4. 长度跟原版相当（不要膨胀）
"""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def boost_novelty(
    proposal: ResearchProposal,
    direction: str,
    llm: BaseChatModel,
    *,
    max_rounds: int = 3,
    s2_limit: int = 5,
    convergence_levels: set = None,
) -> tuple[ResearchProposal, NoveltyBoostResult]:
    """
    SciMON-style novelty boost：搜最相似 prior work → LLM 判 overlap →
    substantial+ 则 refine → 循环 max_rounds 轮。

    Args:
        proposal: 待评估/提升的 proposal
        direction: 研究方向原文
        llm: LLM（推荐 cheap-mid 级，如 Haiku）
        max_rounds: 最多迭代轮数（默认 3）
        s2_limit: 每条 query 召回 paper 数（默认 5）
        convergence_levels: 收敛条件——overlap_level 落入此集合就停。默认
                           {none, partial}（partial 也算够用）

    Returns:
        (修订后的 proposal, NoveltyBoostResult)
        即使 max_rounds 后未收敛，仍返回最后一轮的 proposal——调用方决定是否丢弃
    """
    if convergence_levels is None:
        convergence_levels = {"none", "partial"}

    revisions_log: list[str] = []
    last_assessment: NoveltyAssessment | None = None
    current = proposal

    for round_idx in range(max_rounds):
        # Step 1: 抽 S2 search query
        queries = _extract_queries(current, llm)
        print(f"[novelty_booster] round {round_idx+1}: queries = {queries}",
              file=sys.stderr)

        # Step 2: S2 召回最相似 prior work
        candidates = _search_prior_work(queries, s2_limit)
        if not candidates:
            print(f"[novelty_booster] round {round_idx+1}: 未召回任何 prior work，"
                  "跳过 (overlap_level=none)", file=sys.stderr)
            last_assessment = NoveltyAssessment(
                overlap_level="none", novelty_score=0.95,
                differentiation_gap="(无 prior work 召回)",
            )
            break

        # Step 3: LLM 评估 overlap
        last_assessment = _assess_overlap(current, candidates, direction, llm)
        if last_assessment is None:
            print(f"[novelty_booster] round {round_idx+1}: assessment 失败，"
                  "保守视为 partial 后退出", file=sys.stderr)
            last_assessment = NoveltyAssessment(
                overlap_level="partial", novelty_score=0.6,
                differentiation_gap="(LLM 评估失败，保守标 partial)",
            )
            break

        print(f"[novelty_booster] round {round_idx+1}: overlap_level="
              f"{last_assessment.overlap_level}, novelty_score="
              f"{last_assessment.novelty_score:.2f}", file=sys.stderr)

        # Step 4: 收敛判断
        if last_assessment.overlap_level in convergence_levels:
            break

        # Step 5: refine（如果还有 round 余量）
        if round_idx < max_rounds - 1:
            refined = _refine_for_novelty(current, last_assessment, llm)
            if refined is not None:
                revisions_log.append(
                    f"round {round_idx+1}: refined motivation+proposed_method "
                    f"(closest={last_assessment.closest_work_title[:40]})"
                )
                current = refined
            else:
                revisions_log.append(
                    f"round {round_idx+1}: refine 失败，停止迭代"
                )
                break

    converged = (
        last_assessment is not None
        and last_assessment.overlap_level in convergence_levels
    )

    # 把 assessment 写回 proposal
    current.novelty_assessment = last_assessment

    return current, NoveltyBoostResult(
        rounds_taken=round_idx + 1,
        final_assessment=last_assessment,
        converged=converged,
        revisions_log=revisions_log,
    )


# ---------------------------------------------------------------------------
# Step 1: extract queries
# ---------------------------------------------------------------------------

def _extract_queries(proposal: ResearchProposal, llm: BaseChatModel) -> list[str]:
    """让 LLM 抽 1-2 条精确 S2 search query。失败时回退到 title 当 query。"""
    summary = (
        f"title: {proposal.title}\n"
        f"proposed_method: {proposal.proposed_method[:600]}\n"
        f"technical_details: {proposal.technical_details[:400]}"
    )
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_QUERY_EXTRACT_PROMPT),
            HumanMessage(content=f"研究方案：\n{summary}\n\n请输出 1-2 条搜索 query。"),
        ])
        raw = parse_llm_json(response.content)
        queries = raw.get("queries") or []
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if cleaned:
            return cleaned[:2]
    except Exception as e:
        print(f"[novelty_booster] _extract_queries 失败: {type(e).__name__}",
              file=sys.stderr)
    # fallback：用 title
    return [proposal.title[:80]] if proposal.title else []


# ---------------------------------------------------------------------------
# Step 2: search prior work
# ---------------------------------------------------------------------------

def _search_prior_work(queries: list[str], limit_per_query: int) -> list[dict]:
    """对每条 query 调 S2 search，合并去重。S2 失败时跳过该 query 不阻塞。"""
    pool: list[dict] = []
    for q in queries:
        if not q:
            continue
        try:
            results = search_papers(q, limit=limit_per_query) or []
            pool.extend(results)
        except Exception as e:
            print(f"[novelty_booster] search '{q[:40]}' 失败: {type(e).__name__}",
                  file=sys.stderr)
    return _dedup_papers_by_id(pool)[:8]   # 总最多 8 篇喂给 LLM


# ---------------------------------------------------------------------------
# Step 3: assess overlap
# ---------------------------------------------------------------------------

def _assess_overlap(
    proposal: ResearchProposal,
    candidates: list[dict],
    direction: str,
    llm: BaseChatModel,
) -> NoveltyAssessment | None:
    """让 LLM 把 ours 跟召回的 prior work 对比，输出 NoveltyAssessment。失败返 None。"""
    ours_summary = (
        f"title: {proposal.title}\n"
        f"motivation (摘): {proposal.motivation[:400]}\n"
        f"proposed_method (摘): {proposal.proposed_method[:600]}"
    )
    candidates_block = _render_candidates(candidates)
    user_msg = (
        f"【方向】{direction}\n\n"
        f"【我们的 proposal】\n{ours_summary}\n\n"
        f"【召回的 prior work（按相似度排）】\n{candidates_block}\n\n"
        "请按 SYSTEM_PROMPT 输出 JSON。"
    )

    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_OVERLAP_ASSESS_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(f"[novelty_booster] _assess_overlap 解析失败: {type(e).__name__}",
              file=sys.stderr)
        return None

    overlap_level = str(raw.get("overlap_level", "")).strip().lower()
    if overlap_level not in _VALID_OVERLAP_LEVELS:
        print(f"[novelty_booster] overlap_level='{overlap_level}' 非法",
              file=sys.stderr)
        return None

    try:
        novelty_score = float(raw.get("novelty_score", 0.0))
    except (TypeError, ValueError):
        novelty_score = 0.0
    novelty_score = max(0.0, min(1.0, novelty_score))   # clamp

    return NoveltyAssessment(
        overlap_level=overlap_level,
        closest_work_paper_id=str(raw.get("closest_paper_id", "")),
        closest_work_title=str(raw.get("closest_title", "")),
        overlap_summary=str(raw.get("overlap_summary", "")),
        differentiation_gap=str(raw.get("differentiation_gap", "")),
        novelty_score=novelty_score,
    )


def _render_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return "（无召回）"
    lines = []
    for p in candidates[:8]:
        pid = p.get("paperId", "?")
        title = (p.get("title") or "")[:80]
        abstract = (p.get("abstract") or "")[:200]
        year = p.get("year", "?")
        lines.append(f"  - [{pid}] ({year}) {title}\n    abstract: {abstract}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 5: refine for novelty
# ---------------------------------------------------------------------------

def _refine_for_novelty(
    proposal: ResearchProposal,
    assessment: NoveltyAssessment,
    llm: BaseChatModel,
) -> ResearchProposal | None:
    """根据 assessment 反馈，让 LLM 重写 motivation + proposed_method 两段。"""
    user_msg = (
        f"【原 motivation】\n{proposal.motivation}\n\n"
        f"【原 proposed_method】\n{proposal.proposed_method}\n\n"
        f"【overlap 反馈】\n"
        f"closest_work: {assessment.closest_work_title}\n"
        f"overlap: {assessment.overlap_summary}\n"
        f"differentiation_gap: {assessment.differentiation_gap}\n\n"
        "请按 SYSTEM_PROMPT 输出修订后的 JSON。"
    )

    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_REFINE_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(f"[novelty_booster] _refine 失败: {type(e).__name__}", file=sys.stderr)
        return None

    new_motivation = str(raw.get("motivation", "")).strip()
    new_method = str(raw.get("proposed_method", "")).strip()
    if not new_motivation or not new_method:
        return None

    # 用 model_copy 不破坏其他字段
    return proposal.model_copy(update={
        "motivation": new_motivation,
        "proposed_method": new_method,
    })
