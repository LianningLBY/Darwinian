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

_QUERY_EXTRACT_PROMPT = """你是 academic search 专家。给定一份研究 proposal，输出 **3 条不同**的
S2 搜索 query 命中其潜在 prior work，**必须覆盖三种术语角度**。

输出严格 JSON：{"queries": ["q_current", "q_classic", "q_methodname"]}

【三条 query 必须分别用不同术语角度】

1. **q_current**：用 proposal 自身的术语（current / 时髦词）
   适合命中近 1-2 年同方向工作。
   例：proposal 写 "thinking tokens" → query 用 "thinking tokens"

2. **q_classic**：**paraphrase 用经典/foundational 术语**
   重点在覆盖 SciMON 的关键 blind spot——proposal 的现代术语往往是
   过去 1-2 年才流行，但同思想的 foundational 论文用更老术语描述。
   常见映射：
     thinking tokens / latent reasoning   → adaptive computation / pondering / halting probability / dynamic depth
     speculative decoding draft           → assisted generation / lookahead / parallel decoding
     KV cache eviction                    → attention sparsity / token pruning / memory compression
     mixture of experts routing           → conditional computation / gating
     chain-of-thought                     → scratchpad / step-by-step reasoning / multi-hop
     diffusion language model             → non-autoregressive generation / iterative refinement
   query 必须**显式包含至少 1 个 classic 术语**，禁用 proposal 自己的现代术语。
   例：proposal 是 "entropy thinking tokens" → query 用 "halting probability adaptive computation reasoning"

3. **q_methodname**：用 proposal 提到的**最近似的具体方法名**（含 acronym）
   适合命中"被 cite 但 proposal 没强调"的相邻工作。
   例：proposal 提到自己叫 "EATT"，但 thinking 类的相邻方法有 PonderNet / ACT / Quiet-STaR / STaR / HALT-CoT。
   query 用 "PonderNet ACT Quiet-STaR thinking" 这种点名串。

【通用约束】
- 每条 query 3-7 个英文词
- 不要太宽（避免命中泛论文），也不要太窄（避免命中不到论文）
- 必须输出 3 条（少于 3 视为不完整）
"""


_OVERLAP_ASSESS_PROMPT = """你是**严格的**科研新颖性评审专家。给定一份"我们的 proposal"和几篇 prior work
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

【关键：默认从严，下列情况自动升级为 substantial+】
1. ours 的 motivation 或 method 段**直接提到 closest_work**（用名字、acronym、
   或 'X et al. showed Y' 形式引用）→ 至少 substantial（说明你自己都觉得它相关）
2. ours 跟 closest_work **共享核心机制 + 同类评估指标**（如都用 entropy 当 halting
   signal + 都测 acceptance rate）→ substantial 起步
3. ours 的 differentiation 仅是 "static threshold → learned threshold" / "fixed →
   adaptive" / "rule-based → ML-based" 这种 engineering improvement → substantial
4. ours 跟 closest_work 在**同一具体方法名**下（如都是 self-speculative decoding
   + 都用 quantized draft）→ identical 起步

【overlap_level 判断标准】（保守 → 激进）
- identical: 本质同一个 idea，ours 几乎是 prior 的复现，或者只是 incremental 替换
  threshold/optimizer/dataset
- substantial: prior 已做过类似的核心 mechanism，ours 仅在 scale / variant / 数据
  集上有别。**审稿人会觉得 incremental**
- partial: 有交集但 contribution 角度本质不同（ours 测 X 指标 + prior 测 Y 指标，
  且 X 跟 Y 不能互相 derive）。**可发表但 motivation 必须 explicit differentiate**
- none: prior 跟 ours 方向完全不同（如 ours 做 LLM inference, prior 做 vision）

【novelty_score 对照】（注意：宁低勿高，不要给 incremental 工作高分）
- 0.85-1.0: none（罕见，需真正新方向）
- 0.55-0.85: partial（可发表，需明确 differentiation）
- 0.25-0.55: substantial（需大改或 reframe）
- 0.0-0.25: identical（应丢弃或重新设计）

【最终：先给 score 再选 level】
- 先给一个 0-1 的 score，然后**严格按上面 score 区间反推 overlap_level**
- 不要先选 level 再凑 score——LLM 系统性偏向给 partial 0.7 这种"听起来安全"的中间值
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

def _extract_queries(
    proposal: ResearchProposal,
    llm: BaseChatModel,
    *,
    max_queries: int = 3,
) -> list[str]:
    """
    让 LLM 抽 3 条 S2 search query (current / classic / methodname)。
    失败时回退到 title 当单条 query。

    设计动机：v8 实测发现 EATT 召回 0 prior work，但实际 HALT-CoT / PonderNet /
    Quiet-STaR 都是高度相关——SciMON 的 query 用了 "thinking tokens" 等近 1-2 年
    流行词，漏了用经典术语 (halting / pondering / adaptive computation) 描述
    的 foundational 工作。强制三角度查询关闭这个 blind spot。
    """
    summary = (
        f"title: {proposal.title}\n"
        f"proposed_method: {proposal.proposed_method[:600]}\n"
        f"technical_details: {proposal.technical_details[:400]}"
    )
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_QUERY_EXTRACT_PROMPT),
            HumanMessage(content=f"研究方案：\n{summary}\n\n请按 SYSTEM_PROMPT 输出 "
                                  "3 条 query (current/classic/methodname 三角度)。"),
        ])
        raw = parse_llm_json(response.content)
        queries = raw.get("queries") or []
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if cleaned:
            return cleaned[:max_queries]
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

    closest_title = str(raw.get("closest_title", ""))
    closest_paper_id = str(raw.get("closest_paper_id", ""))

    # 防御性升级 1: 如果 ours.motivation/method 直接提到 closest_work
    # （LLM 自己都引用，说明跟它有强关联），自动升级 partial → substantial
    overlap_level, novelty_score = _auto_upgrade_if_self_cited(
        proposal, closest_title, overlap_level, novelty_score,
    )

    # 防御性升级 2: 如果 score < 0.55 但 LLM 标了 partial（违反 score-level 一致性），
    # 强制升 substantial 的判定。反之亦然下调。
    overlap_level, novelty_score = _enforce_score_level_consistency(
        overlap_level, novelty_score,
    )

    return NoveltyAssessment(
        overlap_level=overlap_level,
        closest_work_paper_id=closest_paper_id,
        closest_work_title=closest_title,
        overlap_summary=str(raw.get("overlap_summary", "")),
        differentiation_gap=str(raw.get("differentiation_gap", "")),
        novelty_score=novelty_score,
    )


def _auto_upgrade_if_self_cited(
    proposal: ResearchProposal,
    closest_title: str,
    overlap_level: str,
    novelty_score: float,
) -> tuple[str, float]:
    """
    如果 ours.motivation/method 段直接提到 closest_work（用 acronym 或 title 关键词），
    自动升级 overlap_level + 拉低 novelty_score。
    防 LLM 给"自己都引了的相关工作"宽松判定。
    """
    if not closest_title or overlap_level in ("substantial", "identical"):
        return overlap_level, novelty_score
    # 提取 closest_title 的关键词（首词或 acronym）
    title_keywords = _extract_title_keywords(closest_title)
    proposal_text = (proposal.motivation + "\n" + proposal.proposed_method).lower()
    for kw in title_keywords:
        if kw and kw.lower() in proposal_text:
            print(
                f"[novelty_booster] 自动升级: ours 自己引用了 closest_work "
                f"'{kw}' → overlap_level partial → substantial, "
                f"score {novelty_score:.2f} → {min(novelty_score, 0.5):.2f}",
                file=sys.stderr,
            )
            return "substantial", min(novelty_score, 0.5)
    return overlap_level, novelty_score


def _extract_title_keywords(title: str) -> list[str]:
    """
    从论文标题抽 1-3 个能用来 substring 匹配的关键词：
    - 首个 ALL-CAPS 单词（如 'PonderNet:' → 'PonderNet'）
    - 冒号前的 acronym（如 'EAGLE: Speculative Sampling' → 'EAGLE'）
    """
    import re
    keywords = []
    # 冒号前的 word
    if ":" in title:
        before_colon = title.split(":", 1)[0].strip()
        if 2 <= len(before_colon) <= 30:
            keywords.append(before_colon)
    # 标题前 1-2 个 CamelCase / ALL-CAPS 词
    for m in re.finditer(r"\b([A-Z][a-zA-Z\-]{2,}|[A-Z]{2,})", title):
        if m.group(1) not in keywords and len(keywords) < 3:
            keywords.append(m.group(1))
    return keywords


def _enforce_score_level_consistency(
    overlap_level: str, novelty_score: float,
) -> tuple[str, float]:
    """
    score 跟 level 必须一致（按 prompt 里的 score 区间反推 level）：
    - 0.0-0.25 → identical
    - 0.25-0.55 → substantial
    - 0.55-0.85 → partial
    - 0.85-1.0 → none
    LLM 经常 score=0.7 但 level=partial（一致），或 score=0.4 但 level=partial
    （不一致 → 修正 substantial）。本函数以 score 为准强制对齐 level。
    """
    if novelty_score < 0.25:
        target = "identical"
    elif novelty_score < 0.55:
        target = "substantial"
    elif novelty_score < 0.85:
        target = "partial"
    else:
        target = "none"
    if target != overlap_level:
        print(
            f"[novelty_booster] 一致性修正: score {novelty_score:.2f} 对应 "
            f"'{target}' 但 LLM 标 '{overlap_level}' → 改 '{target}'",
            file=sys.stderr,
        )
    return target, novelty_score


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
