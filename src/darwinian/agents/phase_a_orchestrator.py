"""
Phase A Orchestrator — 编排"读"阶段，从研究方向自动产出 ResearchMaterialPack。

设计目标：把现有 4 个工具串成单一入口，给 elaborator 喂完美素材包：

  build_concept_graph         (S2 搜索 + 一跳引用扩展 + 实体抽取 + 结构洞)
       ↓
  resolve_arxiv_ids           (对 top-K paper 查 externalIds.ArXiv)
       ↓
  fetch_arxiv_latex + render_for_llm  (拉全文 LaTeX 节选)
       ↓
  batch_extract_evidence      (paper_evidence_extractor 深抽取五元组)
       ↓
  组装 ResearchMaterialPack（含 paper_evidence + concept_graph + timeline_signals）

未实现（留给后续）：
  - structural_hole_hooks 的 hook_text 自动生成（需新写 hook_writer LLM 调用）
  - prior_failures 接 state.failed_ledger
"""

from __future__ import annotations

import sys
from typing import Callable

from langchain_core.language_models import BaseChatModel

from darwinian.state import (
    PaperEvidence,
    PaperInfo,
    ResearchConstraints,
    ResearchMaterialPack,
)
from darwinian.tools.arxiv_latex_fetcher import fetch_arxiv_latex, render_for_llm
from darwinian.tools.paper_evidence_extractor import batch_extract_evidence
from darwinian.tools.semantic_scholar import get_paper_details
from darwinian.utils.knowledge_graph import build_concept_graph


def build_research_material_pack(
    direction: str,
    constraints: ResearchConstraints,
    extractor_llm: BaseChatModel,
    evidence_llm: BaseChatModel,
    *,
    top_k_evidence: int = 12,
    backend: str | None = None,
) -> ResearchMaterialPack:
    """
    Phase A 主入口：研究方向 + 约束 → 完整 ResearchMaterialPack。

    Args:
        direction: 用户输入的研究方向原文
        constraints: 资源/合规约束（已结构化）
        extractor_llm: 用于浅实体抽取（推荐 Haiku 级小模型，便宜）
        evidence_llm:  用于 paper_evidence_extractor 深抽取（推荐 Opus / GPT-4 级）
        top_k_evidence: 深抽取的论文数量上限（默认 12，按 citationCount 取头）
        backend: build_concept_graph 的检索后端 ("s2" / "arxiv" / None=env)

    Returns:
        完整 ResearchMaterialPack，可直接喂给 elaborate_proposal_from_pack
    """
    # ---- Step 1: 文献发现 + 实体表 + 结构洞候选 ----
    print(f"[phase_a] Step 1/4: build_concept_graph(direction={direction[:60]!r})", file=sys.stderr)
    graph = build_concept_graph(
        research_direction=direction,
        core_problem=direction,
        llm=extractor_llm,
        backend=backend,
    )
    print(f"[phase_a] graph: {len(graph.papers)} papers, "
          f"{len(graph.entities)} entities, {len(graph.limitations)} limitations, "
          f"{len(graph.novel_pair_hints)} novel pairs", file=sys.stderr)

    # ---- Step 2: 选 top-K paper 做深抽取 ----
    # 排序键：(entity_hits, citation_count) 降序
    # entity_hits = 这篇论文贡献了多少个不同的 entity（method/dataset/metric）
    # 设计动机：纯按 citation 排会让 GPT-3 / Llama 2 / PyTorch 等基础工作压过
    #   方向相关的小众论文（LayerSkip / DEL / QSpec），因为基础论文引用量比方向
    #   论文高一个数量级。entity_hits 反映"实体抽取认为多相关"，是更准的方向
    #   匹配信号。citation_count 作为同分时的 tiebreaker 保留。
    top_papers = _select_top_papers_by_relevance(graph, top_k_evidence)
    print(f"[phase_a] Step 2/4: 选 top-{len(top_papers)} 篇做深抽取 "
          f"（按 entity hits + citation 排序）", file=sys.stderr)

    # ---- Step 3: 解析 arxiv_id（用于全文拉取）----
    arxiv_id_by_paperid = _resolve_arxiv_ids(top_papers)
    n_with_arxiv = sum(1 for v in arxiv_id_by_paperid.values() if v)
    print(f"[phase_a] Step 3/4: {n_with_arxiv}/{len(top_papers)} 篇可拿 arxiv 全文", file=sys.stderr)

    # ---- Step 4: 深抽取 PaperEvidence ----
    full_text_provider = _make_full_text_provider(arxiv_id_by_paperid)
    papers_for_extraction = [
        {
            "paper_id": _format_evidence_id(p, arxiv_id_by_paperid),
            "title": p.title,
            "abstract": p.abstract,
        }
        for p in top_papers
    ]
    print(f"[phase_a] Step 4/4: batch_extract_evidence(n={len(papers_for_extraction)})",
          file=sys.stderr)
    paper_evidence = batch_extract_evidence(
        papers_for_extraction,
        direction=direction,
        llm=evidence_llm,
        full_text_provider=full_text_provider,
    )
    print(f"[phase_a] 深抽取完成: {len(paper_evidence)} 篇成功", file=sys.stderr)

    # ---- Step 5: timeline_signals ----
    timeline = _bucket_by_year(top_papers, arxiv_id_by_paperid)

    return ResearchMaterialPack(
        direction=direction,
        constraints=constraints,
        paper_evidence=paper_evidence,
        concept_graph=graph,
        structural_hole_hooks=[],   # 留给后续 hook_writer 实现
        timeline_signals=timeline,
        prior_failures=[],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_top_papers_by_relevance(
    graph: "ConceptGraph",
    top_k: int,
) -> list[PaperInfo]:
    """
    按 (entity_hits, citation_count) 降序选 top-K paper。

    entity_hits[paper_id] = 这篇 paper 出现在多少个不同 entity 的 paper_ids 列表里。
    数值越大说明 entity 抽取认为它跟方向越相关（贡献了越多方法/数据集/指标实体）。

    与单纯按 citation 排的差异：
    - 基础工作论文（GPT-3/Llama 2/PyTorch）entity 抽取通常只 hit 1-2 个泛词
      （"transformer" / "language model"），entity_hits 低
    - 方向相关论文（LayerSkip / DEL / QSpec）会 hit 多个具体方法/指标
      （"speculative decoding" / "draft model" / "acceptance rate" / "early exit"），
      entity_hits 显著高
    - 同 entity_hits 的用 citation_count tiebreak（保留原有信号但作为次要）
    """
    from collections import Counter
    from darwinian.state import ConceptGraph
    entity_hits: Counter[str] = Counter()
    for e in graph.entities:
        for pid in e.paper_ids:
            entity_hits[pid] += 1

    return sorted(
        graph.papers,
        key=lambda p: (entity_hits.get(p.paper_id, 0), p.citation_count or 0),
        reverse=True,
    )[:top_k]


def _resolve_arxiv_ids(papers: list[PaperInfo]) -> dict[str, str]:
    """
    对每个 PaperInfo.paper_id 查 externalIds.ArXiv。
    返回 dict[s2_paper_id -> arxiv_id]，找不到的 value=""

    Note: build_concept_graph 走 S2 path 时 paper_id 是 S2 paperId（hex 串），
    需要再调一次 get_paper_details 才能拿 arxiv_id。所有调用走 S2 缓存，
    重复跑同一方向时几乎 0 成本。
    """
    result: dict[str, str] = {}
    for p in papers:
        if not p.paper_id:
            continue
        # 已经是 arxiv 格式（如 "2404.16710"）就直接用
        if _looks_like_arxiv_id(p.paper_id):
            result[p.paper_id] = p.paper_id
            continue
        try:
            detail = get_paper_details(p.paper_id, fields="externalIds")
            if detail:
                arxiv_id = (detail.get("externalIds") or {}).get("ArXiv", "") or ""
                result[p.paper_id] = arxiv_id
            else:
                result[p.paper_id] = ""
        except Exception as e:
            print(f"[phase_a] resolve_arxiv_ids 跳过 {p.paper_id}: {type(e).__name__}",
                  file=sys.stderr)
            result[p.paper_id] = ""
    return result


def _looks_like_arxiv_id(s: str) -> bool:
    """形如 '2404.16710' 或 '2404.16710v2' 的视为已经是 arxiv id"""
    import re
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", s))


def _format_evidence_id(p: PaperInfo, arxiv_map: dict[str, str]) -> str:
    """
    构造 PaperEvidence.paper_id 字段：
    - 有 arxiv_id 优先用 'arxiv:<id>'（让下游引用更可读）
    - 没有就回退 's2:<paperId>'
    """
    arxiv_id = arxiv_map.get(p.paper_id, "")
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return f"s2:{p.paper_id}"


def _make_full_text_provider(
    arxiv_map: dict[str, str],
) -> Callable[[str], str]:
    """
    返回一个 callable(evidence_paper_id) -> rendered_full_text。
    供 batch_extract_evidence 内部调用——按 paper_id 拉全文。

    Note: evidence_paper_id 形如 'arxiv:2404.16710' 或 's2:abc...'。
    s2 形式的我们也尝试回查 arxiv_map 一次（万一 PaperInfo 有别名）。
    arxiv 形式直接拉。失败返空字符串 → batch_extract_evidence 自动降级 abstract-only。
    """
    # 把 arxiv_map（s2_paperid -> arxiv_id）反向也建一份索引
    rev = {f"arxiv:{v}": v for v in arxiv_map.values() if v}
    rev.update({f"s2:{k}": v for k, v in arxiv_map.items() if v})

    def provider(evidence_paper_id: str) -> str:
        arxiv_id = rev.get(evidence_paper_id, "")
        # evidence_paper_id 可能是 'arxiv:2404.16710'，尝试直接解析
        if not arxiv_id and evidence_paper_id.startswith("arxiv:"):
            arxiv_id = evidence_paper_id.split(":", 1)[1]
        if not arxiv_id:
            return ""
        try:
            src = fetch_arxiv_latex(arxiv_id)
            if src and src.has_full_text:
                return render_for_llm(src)
        except Exception as e:
            print(f"[phase_a] full_text_provider 跳过 {arxiv_id}: {type(e).__name__}",
                  file=sys.stderr)
        return ""

    return provider


def _bucket_by_year(
    papers: list[PaperInfo],
    arxiv_map: dict[str, str],
) -> dict[str, list[str]]:
    """
    分时间桶，给 elaborator 在 Why Now section 引用时间感。

    桶定义：
      - foundational_pre_2024: < 2024 年（基础工作）
      - hot_2025_2026:         2025-2026 年（最近热门）
    """
    buckets: dict[str, list[str]] = {
        "foundational_pre_2024": [],
        "hot_2025_2026": [],
    }
    for p in papers:
        evid_id = _format_evidence_id(p, arxiv_map)
        year = p.year or 0
        if year < 2024 and year > 0:
            buckets["foundational_pre_2024"].append(evid_id)
        elif year >= 2025:
            buckets["hot_2025_2026"].append(evid_id)
    # 删空桶
    return {k: v for k, v in buckets.items() if v}
