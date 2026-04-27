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
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import (
    PaperEvidence,
    PaperInfo,
    ResearchConstraints,
    ResearchMaterialPack,
)
from darwinian.agents.hook_writer import write_structural_hole_hooks
from darwinian.tools.arxiv_latex_fetcher import fetch_arxiv_latex, render_for_llm
from darwinian.tools.paper_evidence_extractor import batch_extract_evidence
from darwinian.tools.semantic_scholar import (
    GRAPH_FIELDS,
    get_citations,
    get_paper_details,
    get_references,
    search_papers,
)
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry
from darwinian.utils.knowledge_graph import build_concept_graph, _dedup_papers_by_id
from darwinian.utils.similarity import compute_cosine_similarity, get_text_embedding


# ===========================================================================
# Scheme X: build_seed_pool —— LLM 列方向 seed → S2 verify → 一跳 → rerank
# 替换 S2 keyword search 路径，从根上解决 candidate pool 缺方向论文的问题
# ===========================================================================

_LLM_SEED_PROMPT = """你是科研助理。给定一个研究方向，请列出该方向**最重要**的 12-18 篇论文 seed
（必须是真实存在的、已发表/arxiv 可查的论文），每篇含 arxiv_id + 完整 title + 一句话理由。

输出严格 JSON：
{
  "seed_papers": [
    {"arxiv_id": "2404.16710", "title": "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", "reason": "..."},
    ...
  ]
}

【关键约束】
1. arxiv_id 形如 'YYMM.NNNNN'（如 '2404.16710'），不带 v1/v2 后缀，不带 'arxiv:' 前缀
2. title 写**完整官方标题**（含子标题），用于 fallback verify
3. 优先选近 2 年（2024-2026）方向核心工作，配少量 2-3 年内基础工作
4. 覆盖该方向不同子赛道（如 speculative decoding 含 LayerSkip / DEL / Medusa / EAGLE / QSpec / SpecInfer / Sequoia 等）
5. 不要列基础工作如 GPT-3 / Llama 2 / PyTorch（除非它们就是方向本身）
6. 列 12-18 篇，宁缺勿滥

输出严格 JSON，不要 markdown 包裹。
"""


def build_seed_pool(
    direction: str,
    llm: BaseChatModel,
    *,
    n_seeds: int = 15,
    refs_per_seed: int = 8,
    cits_per_seed: int = 8,
    final_pool_size: int = 50,
) -> list[dict]:
    """
    Scheme X 主入口。返回 list of S2-style paper dict（含 paperId/title/abstract/year/
    citationCount/externalIds），可直接喂给 build_concept_graph(seed_pool=...)。

    流程：
      1. LLM 列 N 个 seed (arxiv_id + title + reason)
      2. 对每个 seed：先 S2 verify by arxiv_id；失败则用 title fuzzy match 回捞
      3. 对验证通过的 seeds：调 get_references / get_citations 一跳扩展
      4. 按 (is_seed, direction relevance, citation) 重排，取 top-K
    """
    # Step 1: LLM 列 seed 候选
    candidates = _llm_list_seed_papers(direction, llm, n=n_seeds)
    print(f"[seed_pool] LLM 列出 {len(candidates)} 个 seed 候选", file=sys.stderr)

    # Step 2: verify + recover
    seeds: list[dict] = []
    for cand in candidates:
        paper = _verify_and_recover_seed(cand)
        if paper:
            seeds.append(paper)
    print(f"[seed_pool] verify 通过 {len(seeds)}/{len(candidates)} 个 seed", file=sys.stderr)

    if not seeds:
        return []

    # Step 3: 一跳扩展
    expanded = _expand_seeds_one_hop(seeds, refs_per_seed, cits_per_seed)
    print(f"[seed_pool] 一跳扩展后候选池 {len(expanded)} 篇", file=sys.stderr)

    # Step 4: rerank
    seed_ids = {s.get("paperId", "") for s in seeds if s.get("paperId")}
    ranked = _rerank_by_direction_relevance(expanded, direction, seed_ids)
    return ranked[:final_pool_size]


def _llm_list_seed_papers(direction: str, llm: BaseChatModel, *, n: int = 15) -> list[dict]:
    """让 LLM 列 N 个方向 seed，返回 list of {arxiv_id, title, reason}"""
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_LLM_SEED_PROMPT),
            HumanMessage(content=f"研究方向：{direction}\n\n请列出 {n} 篇该方向核心 seed 论文。"),
        ])
        raw = parse_llm_json(response.content)
        candidates = raw.get("seed_papers") or []
        return [c for c in candidates if isinstance(c, dict) and c.get("arxiv_id")]
    except Exception as e:
        print(f"[seed_pool] _llm_list_seed_papers 失败: {type(e).__name__}: {e}",
              file=sys.stderr)
        return []


def _verify_and_recover_seed(cand: dict) -> dict | None:
    """
    机制 1: arxiv_id verify + title fuzzy fallback。
    LLM 经常把 arxiv_id 末几位记错（2404.16710 → 2404.16701）。
    直接 verify 失败 → 用 title 搜索 S2 回捞。
    """
    arxiv_id = (cand.get("arxiv_id") or "").strip().lstrip("arxiv:").lstrip("ArXiv:")
    title = (cand.get("title") or "").strip()

    # 1. 用 arxiv_id 直查
    if arxiv_id:
        try:
            detail = get_paper_details(f"ArXiv:{arxiv_id}", fields=GRAPH_FIELDS)
            if detail:
                return detail
        except Exception:
            pass

    # 2. fallback: 用 title 搜 S2
    if title:
        try:
            results = search_papers(title, limit=3)
            for r in results:
                if _title_similarity(r.get("title", "") or "", title) > 0.85:
                    return r
        except Exception:
            pass

    return None


def _title_similarity(a: str, b: str) -> float:
    """归一化 token Jaccard，简单但对论文标题足够"""
    import re
    if not a or not b:
        return 0.0
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _expand_seeds_one_hop(
    seeds: list[dict],
    refs_per_seed: int,
    cits_per_seed: int,
) -> list[dict]:
    """
    对每个 seed 取 references + citations。比 expand_one_hop 简单：
    - 不需要复杂的累加去重（直接 dedup_papers_by_id）
    - 控制每个 seed 的 refs/cits 数量在 8-10 内（防 S2 限流）
    """
    pool = list(seeds)   # seeds 自己也进 pool
    for s in seeds:
        pid = s.get("paperId", "")
        if not pid:
            continue
        try:
            pool.extend(get_references(pid, limit=refs_per_seed) or [])
        except Exception as e:
            print(f"[seed_pool] get_references({pid}) 失败: {type(e).__name__}",
                  file=sys.stderr)
        try:
            pool.extend(get_citations(pid, limit=cits_per_seed) or [])
        except Exception as e:
            print(f"[seed_pool] get_citations({pid}) 失败: {type(e).__name__}",
                  file=sys.stderr)
    return _dedup_papers_by_id(pool)


def _rerank_by_direction_relevance(
    papers: list[dict],
    direction: str,
    seed_ids: set,
) -> list[dict]:
    """
    机制 2: rerank 按 (is_seed, TF-IDF sim to direction, citation) 降序。
    避免 filter_and_rank 的纯 citation 排序把方向论文（cit 低）甩到末尾。
    """
    direction_emb = get_text_embedding(direction)

    def sort_key(p: dict) -> tuple:
        pid = p.get("paperId") or ""
        is_seed = pid in seed_ids
        text = ((p.get("title") or "") + " " + (p.get("abstract") or "")).strip()
        sim = (
            compute_cosine_similarity(direction_emb, get_text_embedding(text))
            if text else 0.0
        )
        cit = p.get("citationCount") or 0
        return (is_seed, sim, cit)

    return sorted(papers, key=sort_key, reverse=True)


# ===========================================================================
# Phase A 主入口
# ===========================================================================


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
    # ---- Step 0: build_seed_pool (Scheme X) —— 用 LLM 知识列方向 seed ----
    # 替换 S2 keyword search 路径。S2 关键词搜索对 "LLM inference" 这种宽 query
    # 会命中所有 LLM 大论文（GPT-3 / Llama 2 / PyTorch），把方向相关的小众论文
    # （LayerSkip / DEL / QSpec）压在 100+ 名外。LLM 列 seed 直接精确命中方向。
    print(f"[phase_a] Step 0/4: build_seed_pool（LLM 列方向 seed + 一跳扩展 + rerank）",
          file=sys.stderr)
    seed_pool = build_seed_pool(direction, extractor_llm)
    print(f"[phase_a] seed_pool: {len(seed_pool)} 篇候选", file=sys.stderr)

    # ---- Step 1: 实体表 + 结构洞（用 seed_pool 跳过 build_concept_graph 的搜索层）----
    print(f"[phase_a] Step 1/4: build_concept_graph(seed_pool=<{len(seed_pool)} papers>)",
          file=sys.stderr)
    graph = build_concept_graph(
        research_direction=direction,
        core_problem=direction,
        llm=extractor_llm,
        backend=backend,
        seed_pool=seed_pool,
    )
    print(f"[phase_a] graph: {len(graph.papers)} papers, "
          f"{len(graph.entities)} entities, {len(graph.limitations)} limitations, "
          f"{len(graph.novel_pair_hints)} novel pairs", file=sys.stderr)

    # ---- Step 1.5: hook_writer 把 EntityPair 升级为 StructuralHoleHook ----
    # elaborator prompt 里【结构洞 hooks】section 之前永远空（structural_hole_hooks=[]），
    # 现在用 LLM 把 top-K novel pair 升级为带叙事 + relation_type 的 hook
    print(f"[phase_a] Step 1.5/4: hook_writer 升级 top-{min(5, len(graph.novel_pair_hints))} novel pairs",
          file=sys.stderr)
    structural_hole_hooks = write_structural_hole_hooks(
        graph.novel_pair_hints,
        graph.entities,
        graph.papers,
        direction,
        extractor_llm,
        max_hooks=5,
    )
    print(f"[phase_a] hooks: 生成 {len(structural_hole_hooks)} 条 StructuralHoleHook",
          file=sys.stderr)

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
        structural_hole_hooks=structural_hole_hooks,
        timeline_signals=timeline,
        prior_failures=[],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPAND_QUERIES_SYSTEM_PROMPT = """你是 academic search query 专家。给定一个研究方向描述，
你要拆出 3-5 个**具体且互补的**学术搜索关键词组，覆盖该方向的核心子赛道和方法名。

输出严格 JSON：
{"queries": ["query1", "query2", ...]}

【关键约束】
1. 每条 query 是 3-8 个英文词的关键词组（不是一句话）
2. 优先使用方向里出现的具体方法名 / 数据集 / 现象，而不是宽词
   - 不好: "LLM efficient inference"  → S2 会命中所有 LLM 论文
   - 好:   "self-speculative decoding draft model"
3. 每条 query 应能命中 8-30 篇方向相关论文（不要太宽不要太窄）
4. 多条 query 之间应覆盖不同子赛道，不要互相重复
5. 输出 3-5 条，不超过 5 条

示例（方向='LLM inference acceleration speculative decoding quantization'）：
{"queries": [
  "self-speculative decoding draft model",
  "speculative decoding tree attention LLM",
  "Medusa EAGLE LayerSkip inference",
  "mixed precision quantization LLM",
  "early exit transformer inference"
]}

输出严格 JSON，不要 markdown 包裹。
"""


def _expand_search_queries(
    direction: str,
    llm: BaseChatModel,
    *,
    max_queries: int = 5,
) -> list[str]:
    """
    用便宜 LLM 把宽研究方向拆成 3-5 条具体子查询。

    LLM 失败时返空列表（让 build_concept_graph 退化为单查询行为，0 回归风险）。
    """
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_EXPAND_QUERIES_SYSTEM_PROMPT),
            HumanMessage(content=f"研究方向：{direction}\n\n请输出 3-5 条具体子查询。"),
        ])
        raw = parse_llm_json(response.content)
        queries = raw.get("queries") or []
        # 清洗：strip + 非空 + 截到 max_queries
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return cleaned[:max_queries]
    except Exception as e:
        print(f"[phase_a] _expand_search_queries 失败: {type(e).__name__}: {e}",
              file=sys.stderr)
        return []


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
      - hot_2024_2026:         2024-2026 年（近期热门）
        Note: 2024 是 LLM acceleration 等多个赛道的爆发年（LayerSkip / EAGLE-2 /
        Medusa-2 等），不能像之前 _bucket_by_year v1 那样把 2024 漏掉
    """
    buckets: dict[str, list[str]] = {
        "foundational_pre_2024": [],
        "hot_2024_2026": [],
    }
    for p in papers:
        evid_id = _format_evidence_id(p, arxiv_map)
        year = p.year or 0
        if 0 < year < 2024:
            buckets["foundational_pre_2024"].append(evid_id)
        elif year >= 2024:
            buckets["hot_2024_2026"].append(evid_id)
    # 删空桶
    return {k: v for k, v in buckets.items() if v}
