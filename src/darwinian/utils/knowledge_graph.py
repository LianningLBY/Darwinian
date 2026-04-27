"""
Phase 1 v2 核心模块：从论文网络构建 ConceptGraph。

11 步流水线中这个模块承担 step 2 / 3 / 4 / 5 / 6 / 6.5：
  - expand_one_hop: 对每篇种子论文做一跳引用图扩展
  - filter_and_rank: 清洗、去重、剪枝候选池到 top K
  - batch_extract_entities: 小模型 batch 抽四元组 + limitations
  - canonicalize_merge: 别名合并
  - rank_relevance_top_k: TF-IDF 相关性 + 频次兜底（commit 5）
  - find_novel_pairs: 共现矩阵找结构洞（commit 5）
"""

from __future__ import annotations

import hashlib
import json as _json
import os
import re
import string
from itertools import combinations
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import ConceptGraph, Entity, EntityPair, LimitationRef, PaperInfo
from darwinian.tools import arxiv_search
from darwinian.tools import semantic_scholar as ss
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry
from darwinian.utils.similarity import compute_cosine_similarity, get_text_embedding


def _dedup_papers_by_id(papers: list[dict]) -> list[dict]:
    """按 paperId 去重，保留首次出现的版本（保留 search 的相关性顺序）"""
    seen: set = set()
    out: list[dict] = []
    for p in papers:
        pid = p.get("paperId") or p.get("paper_id") or ""
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Step 2: 一跳引用图扩展
# ---------------------------------------------------------------------------

def expand_one_hop(
    seeds: list[dict],
    max_refs_per_paper: int = 30,
    max_cits_per_paper: int = 30,
) -> list[dict]:
    """
    对每篇种子论文向两个方向各走一步：
      - references: 这篇引用了谁（历史脉络）
      - citations: 谁引用了这篇（后续演进）

    Args:
        seeds: search_papers_two_tiered() 的输出
        max_refs_per_paper / max_cits_per_paper: 每篇种子各方向最多捞多少邻居

    Returns:
        候选池 = 种子 + 所有扩展结果（未去重，由 filter_and_rank 处理）。
        失败的 S2 请求会返空列表降级，不影响其他种子。
    """
    pool: list[dict] = list(seeds)
    for seed in seeds:
        pid = seed.get("paperId")
        if not pid:
            continue
        refs = ss.get_references(pid, limit=max_refs_per_paper)
        cits = ss.get_citations(pid, limit=max_cits_per_paper)
        pool.extend(refs)
        pool.extend(cits)
    return pool


# ---------------------------------------------------------------------------
# Step 3: 清洗、去重、剪枝
# ---------------------------------------------------------------------------

def filter_and_rank(
    candidates: Iterable[dict],
    min_abstract_len: int = 100,
    top_k: int = 60,
) -> list[dict]:
    """
    三步处理候选池：
      1. 按 paperId 去重（无 id 的条目丢弃，无法追溯）
      2. 过滤 abstract 字符数 < min_abstract_len 的论文（空或过短会污染下游实体抽取）
      3. 按 citationCount 降序取 top_k

    Args:
        candidates: expand_one_hop() 的输出
        min_abstract_len: abstract 最小字符数，默认 100
        top_k: 剪枝后保留数量

    Returns:
        干净的 paper dict 列表，字段与 S2 原始 response 一致（paperId/title/abstract/year/citationCount）。
    """
    # 去重：同 paperId 只留第一次出现
    seen: dict[str, dict] = {}
    for p in candidates:
        if not isinstance(p, dict):
            continue
        pid = p.get("paperId")
        if not pid or pid in seen:
            continue
        seen[pid] = p

    # 过滤短 abstract
    filtered = [
        p for p in seen.values()
        if len((p.get("abstract") or "")) >= min_abstract_len
    ]

    # 按引用数降序排序（缺失 citationCount 当作 0）
    filtered.sort(
        key=lambda p: p.get("citationCount") or 0,
        reverse=True,
    )

    return filtered[:top_k]


# ---------------------------------------------------------------------------
# Step 4: 批量实体抽取（小模型）
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """你是一位结构化信息抽取助手。请从下列论文的标题和摘要中，为每篇论文抽取以下字段：

- paper_id: 保持输入给你的原始值，不要改写
- method: 论文中使用/提出的**具体方法名**列表。规则：用最短通用英文名、全小写、去连字符/下划线。如 "Adam Optimizer" → "adam"
- dataset: 论文使用/评估的数据集名称列表（如 "imagenet", "coco"）
- metric: 论文使用的评估指标列表（如 "top-1 accuracy", "bleu"）
- task_type: 任务类型，从 ["classification", "regression", "generation", "retrieval", "detection", "segmentation", "reinforcement_learning", "language_modeling", "other"] 选一个
- limitations: 论文自己承认的缺陷/局限列表，每条一句话中文或英文

请严格返回 JSON 对象，格式：
{
  "papers": [
    {"paper_id": "...", "method": [...], "dataset": [...], "metric": [...], "task_type": "...", "limitations": [...]},
    ...
  ]
}

不要添加任何说明文字或 markdown。
"""


def _chunk(seq: list, n: int) -> Iterable[list]:
    """把列表切成大小 n 的批"""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _build_extract_user_prompt(batch: list[dict]) -> str:
    """把一批 paper 序列化为 LLM 可读的紧凑 JSON（只保留抽取需要的字段）"""
    compact = [
        {
            "paper_id": p.get("paperId", ""),
            "title": (p.get("title") or "")[:300],
            "abstract": (p.get("abstract") or "")[:1500],
        }
        for p in batch
    ]
    return "请处理以下论文：\n" + _json.dumps(compact, ensure_ascii=False)


def batch_extract_entities(
    papers: list[dict],
    llm: Any,
    batch_size: int = 8,
) -> list[dict]:
    """
    批量抽取实体五元组。每 batch_size 篇调一次 LLM，减少请求数。

    Args:
        papers: filter_and_rank() 的输出
        llm: LangChain 兼容 ChatModel（推荐用 Haiku）
        batch_size: 每批论文数

    Returns:
        list of {paper_id, method[], dataset[], metric[], task_type, limitations[]}。
        某个 batch 解析失败会被跳过，不影响其他 batch。
    """
    all_extractions: list[dict] = []
    for batch in _chunk(papers, batch_size):
        messages = [
            SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=_build_extract_user_prompt(batch)),
        ]
        try:
            response = invoke_with_retry(llm, messages)
            parsed = parse_llm_json(response.content)
        except Exception as e:
            print(f"[knowledge_graph] batch extract 失败，跳过：{type(e).__name__}")
            continue

        papers_field = parsed.get("papers") if isinstance(parsed, dict) else None
        if not isinstance(papers_field, list):
            continue
        for item in papers_field:
            if isinstance(item, dict) and item.get("paper_id"):
                all_extractions.append(item)
    return all_extractions


# ---------------------------------------------------------------------------
# Step 5: 别名归一化 + 全局合并
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _normalize(name: str) -> str:
    """
    实体名规范化：小写 + 去标点 + 多空白压成单空格 + 首尾去空。
    这是 precise match 用的 key，原始写法保留在 aliases 里。
    """
    if not isinstance(name, str):
        return ""
    s = name.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", s).strip()


def _word_boundary_contains(short: str, long: str) -> bool:
    """
    短串是否作为完整词出现在长串中。
    例: "adam" 在 "adam optimizer" 里 → True；"bert" 在 "bertopic" → False。
    """
    if not short or not long or short == long:
        return False
    pattern = r"\b" + re.escape(short) + r"\b"
    return re.search(pattern, long) is not None


def _limitation_id(text: str, paper_id: str) -> str:
    """LimitationRef 的稳定 id：md5(text+paper_id)[:8]"""
    return hashlib.md5(f"{text}|{paper_id}".encode("utf-8")).hexdigest()[:8]


VALID_TYPES = {"method", "dataset", "metric"}


# LLM 常见的"抽不到"占位短语，用于 limitation 过滤
_INVALID_LIMITATION_PATTERNS = [
    r"does not explicitly",
    r"does not state",
    r"no (explicit|specific|clear) limitation",
    r"no limitations? (are |is )?(mentioned|stated|given)",
    r"not (?:explicitly |specifically )?(?:mentioned|stated|discussed|given)",
    r"the abstract (?:does not|doesn't)",
    r"^n/?a\b",
    r"^none\b",
    r"unknown",
    r"not (?:available|provided|applicable)",
]
_INVALID_LIMITATION_RE = re.compile(
    "|".join(_INVALID_LIMITATION_PATTERNS), re.IGNORECASE
)


def _is_valid_limitation(text: str) -> bool:
    """
    判断 LLM 抽出的 limitation 是否有效内容，还是"没抽到"的占位语。

    规则：
      - 长度过短（< 15 字符）视为无效
      - 命中 _INVALID_LIMITATION_PATTERNS 任一模式视为无效
      - 其他视为有效
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < 15:
        return False
    if _INVALID_LIMITATION_RE.search(stripped):
        return False
    return True


def canonicalize_merge(
    papers: list[dict],
    raw_extractions: list[dict],
) -> tuple[list[Entity], list[LimitationRef], list[PaperInfo]]:
    """
    把 batch_extract_entities 的原始输出合并为 ConceptGraph 的三个列表。

    流程：
      1. 按 (type, normalized_name) 精确分组，合并 paper_ids 和原始 aliases
         （仅 method / dataset / metric；task_type 不是 Entity，只写入 PaperInfo）
      2. 同类型内按长度升序，做 word-boundary substring 合并（短 ⊂ 长 → 短并入长）
      3. limitations 过滤无效占位语（"does not state"、"N/A" 等），
         用 md5 生成稳定 id，同文本+paper_id 去重
      4. PaperInfo 从 S2 原始数据 + 抽取出的 task_type 组装

    Args:
        papers: filter_and_rank 的输出（包含 S2 元数据）
        raw_extractions: batch_extract_entities 的输出

    Returns:
        (entities, limitations, paper_infos)
    """
    # --- 建索引：paper_id -> S2 原始数据
    paper_by_id: dict[str, dict] = {p.get("paperId"): p for p in papers if p.get("paperId")}

    # --- 第 1 步：按 (type, normalized) 分组
    # key = (type, normalized_name)，value = {"canonical": 最长原始名, "aliases": set, "paper_ids": set}
    buckets: dict[tuple[str, str], dict] = {}

    def _add(e_type: str, raw_name: str, paper_id: str):
        norm = _normalize(raw_name)
        if not norm or e_type not in VALID_TYPES:
            return
        key = (e_type, norm)
        bucket = buckets.setdefault(key, {"canonical": raw_name, "aliases": set(), "paper_ids": set()})
        # canonical 取最短的原始名（符合 prompt 要求"最短通用名"）
        if len(raw_name) < len(bucket["canonical"]):
            bucket["aliases"].add(bucket["canonical"])
            bucket["canonical"] = raw_name
        elif raw_name != bucket["canonical"]:
            bucket["aliases"].add(raw_name)
        bucket["paper_ids"].add(paper_id)

    limitations: list[LimitationRef] = []
    seen_lim_ids: set[str] = set()
    extracted_task_types: dict[str, str] = {}  # paper_id → task_type

    for item in raw_extractions:
        pid = item.get("paper_id", "")
        if not pid:
            continue
        for m in item.get("method") or []:
            _add("method", str(m), pid)
        for d in item.get("dataset") or []:
            _add("dataset", str(d), pid)
        for met in item.get("metric") or []:
            _add("metric", str(met), pid)
        tt = item.get("task_type")
        if isinstance(tt, str) and tt.strip():
            # task_type 只写入 PaperInfo.task_type，不进入实体表
            extracted_task_types[pid] = tt.strip().lower()
        # limitations（过滤"没抽到"占位语）
        for lim in item.get("limitations") or []:
            if not _is_valid_limitation(lim):
                continue
            text = lim.strip()
            lid = _limitation_id(text, pid)
            if lid in seen_lim_ids:
                continue
            seen_lim_ids.add(lid)
            limitations.append(LimitationRef(id=lid, text=text, source_paper_id=pid))

    # 初始 Entity 列表（精确合并后）
    entities: list[Entity] = []
    for (etype, norm), b in buckets.items():
        entities.append(Entity(
            canonical_name=_normalize(b["canonical"]),   # 再 normalize 一次确保 canonical 规范
            aliases=sorted(b["aliases"]),
            type=etype,
            paper_ids=sorted(b["paper_ids"]),
        ))

    # --- 第 2 步：同类型内 word-boundary containment 合并
    entities = _merge_containment(entities)

    # --- 第 3 步：PaperInfo 组装
    paper_infos: list[PaperInfo] = []
    for pid, p in paper_by_id.items():
        paper_infos.append(PaperInfo(
            paper_id=pid,
            title=p.get("title") or "",
            abstract=p.get("abstract") or "",
            year=p.get("year") or 0,
            citation_count=p.get("citationCount") or 0,
            task_type=extracted_task_types.get(pid, ""),
            source="semantic_scholar",
        ))

    return entities, limitations, paper_infos


def _merge_containment(entities: list[Entity]) -> list[Entity]:
    """
    同类型内按 canonical_name 长度升序遍历。
    若 short 作为完整词出现在 long 里（且类型相同），把 long 并入 short——
    保留**短**名字做 canonical（符合 batch 抽取 prompt "用最短通用英文名"约定），
    long 的原 canonical 进 short.aliases。被并入的 long 从结果中移除。
    """
    by_type: dict[str, list[Entity]] = {}
    for e in entities:
        by_type.setdefault(e.type, []).append(e)

    merged: list[Entity] = []
    for etype, group in by_type.items():
        # 按长度升序，短的先被考察。对每个 short 尝试找一个能合并它的 long
        group.sort(key=lambda e: len(e.canonical_name))
        alive = list(group)
        i = 0
        while i < len(alive):
            short = alive[i]
            target_long_idx = -1
            for j in range(i + 1, len(alive)):
                long = alive[j]
                if _word_boundary_contains(short.canonical_name, long.canonical_name):
                    target_long_idx = j
                    break
            if target_long_idx >= 0:
                long = alive[target_long_idx]
                # long 的 aliases + canonical 并入 short
                combined_aliases = set(short.aliases) | set(long.aliases) | {long.canonical_name}
                combined_paper_ids = set(short.paper_ids) | set(long.paper_ids)
                alive[i] = Entity(
                    canonical_name=short.canonical_name,    # 保留**短**名做 canonical
                    aliases=sorted(combined_aliases - {short.canonical_name}),
                    type=short.type,
                    paper_ids=sorted(combined_paper_ids),
                )
                # 移除 long
                alive.pop(target_long_idx)
                # i 不动，继续在合并后的 short 上看有没有更多 long 能并
                continue
            i += 1
        merged.extend(alive)
    return merged


# ---------------------------------------------------------------------------
# Step 6: 实体相关性裁剪（TF-IDF + 频次兜底）
# ---------------------------------------------------------------------------

def rank_relevance_top_k(
    entities: list[Entity],
    core_problem: str,
    top_by_relevance: int = 60,
    top_by_popularity: int = 20,
) -> list[Entity]:
    """
    从全体实体中挑出 ~70 个最该给 Agent 2 看的，避免 prompt 爆炸：
      - top_by_relevance 个按 core_problem 的 TF-IDF 余弦相似度排序
      - top_by_popularity 个按 paper_ids 数量降序（防止跨域冷门实体被 TF-IDF 误杀）
      - 最终去重合并

    Args:
        entities: canonicalize_merge 的输出（全体实体）
        core_problem: Agent 1 产出的核心问题文本
        top_by_relevance: 按相关性挑多少个
        top_by_popularity: 按热度兜底多少个

    Returns:
        精选实体列表（长度 ≤ top_by_relevance + top_by_popularity）
    """
    if not entities:
        return []

    core_vec = get_text_embedding(core_problem or "")

    def _entity_text(e: Entity) -> str:
        return " ".join([e.canonical_name, *e.aliases])

    scored: list[tuple[float, Entity]] = []
    for e in entities:
        emb = get_text_embedding(_entity_text(e))
        scored.append((compute_cosine_similarity(core_vec, emb), e))

    scored.sort(key=lambda x: x[0], reverse=True)
    by_relevance = [e for _, e in scored[:top_by_relevance]]

    by_popularity = sorted(entities, key=lambda e: len(e.paper_ids), reverse=True)[:top_by_popularity]

    # 合并去重（按 (type, canonical_name) 唯一键）
    seen: set[tuple[str, str]] = set()
    result: list[Entity] = []
    for e in by_relevance + by_popularity:
        key = (e.type, e.canonical_name)
        if key in seen:
            continue
        seen.add(key)
        result.append(e)
    return result


# ---------------------------------------------------------------------------
# Step 6.5: 共现矩阵找结构洞
# ---------------------------------------------------------------------------

def find_novel_pairs(
    entities: list[Entity],
    max_pairs: int = 10,
    min_papers_each: int = 3,
) -> list[EntityPair]:
    """
    在全体实体中找"高频但从未共现"的 pair——潜在的结构洞（Sakana/ResearchAgent 思路的轻量版）。

    规则：
      - 两端 entity 的 paper_ids 不得有交集（从未共现）
      - 两端各自的 paper_ids 数量 ≥ min_papers_each（两端都成熟）
      - score = min(len(a.paper_ids), len(b.paper_ids))  # 偏向两端都成熟的组合
      - 按 score 降序取 top max_pairs

    Note: 只在相同或不同 type 间都允许配对（method × dataset 也可以是结构洞）。
    """
    if not entities:
        return []

    candidates: list[tuple[int, Entity, Entity]] = []
    for a, b in combinations(entities, 2):
        if len(a.paper_ids) < min_papers_each or len(b.paper_ids) < min_papers_each:
            continue
        if set(a.paper_ids) & set(b.paper_ids):
            continue   # 已经共现过
        score = min(len(a.paper_ids), len(b.paper_ids))
        candidates.append((score, a, b))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [
        EntityPair(entity_a=a.canonical_name, entity_b=b.canonical_name, score=s)
        for s, a, b in candidates[:max_pairs]
    ]


# ---------------------------------------------------------------------------
# 充分性判断
# ---------------------------------------------------------------------------

MIN_ENTITIES_FOR_HARD_CONSTRAINT = 20
MIN_PAPERS_FOR_HARD_CONSTRAINT = 10


def is_graph_sufficient(entities: list[Entity], papers: list[PaperInfo]) -> bool:
    """
    数据是否足够支撑 Agent 2 的硬约束。
    不足时 hypothesis_generator 降级走老 prompt（不强制 cited_entity_names）。
    """
    return len(entities) >= MIN_ENTITIES_FOR_HARD_CONSTRAINT and len(papers) >= MIN_PAPERS_FOR_HARD_CONSTRAINT


# ---------------------------------------------------------------------------
# 顶层编排：一把梭构建 ConceptGraph
# ---------------------------------------------------------------------------

def build_concept_graph(
    research_direction: str,
    core_problem: str,
    llm: Any,
    *,
    classic_limit: int = 20,
    recent_limit: int = 20,
    max_refs_per_paper: int = 30,
    max_cits_per_paper: int = 30,
    top_k_papers: int = 60,
    batch_size: int = 8,
    top_by_relevance: int = 60,
    top_by_popularity: int = 20,
    max_novel_pairs: int = 10,
    backend: str | None = None,
    extra_queries: list[str] | None = None,
    seed_pool: list[dict] | None = None,
) -> ConceptGraph:
    """
    从 research_direction 开始，走完 step 1–6.5 的完整管道，产出 ConceptGraph。

    每一步失败都降级（不抛异常），最终 is_sufficient 反映数据是否够硬约束。

    Args:
        backend: 文献源。None 时读 DARWINIAN_SEARCH_BACKEND 环境变量，默认 "s2"。
                 - "s2": Semantic Scholar，支持 citation graph 一跳扩展（需要 S2_API_KEY 较稳）
                 - "arxiv": arxiv.org，公开免费无限流，但无 citation graph（跳过 step 2）
        extra_queries: 额外的搜索关键词列表（query expansion）。
                 用途：宽 research_direction 在 S2 关键词搜索下命中泛论文 ——
                 通过几条更具体的子查询（如 'self-speculative decoding draft model'）
                 把方向论文喂进 candidate pool。每条独立调 search_papers_two_tiered，
                 结果按 paperId 去重后并入 seeds。None 或空列表时无影响。
        seed_pool: Scheme X 路径 —— 已经预先准备好的候选论文池（list of S2-style dicts），
                 含 paperId/title/abstract/year/citationCount/externalIds。
                 提供时**完全跳过 search + filter_and_rank**，直接用这批 paper 走
                 entity 抽取 + 结构洞 pipeline。用于 build_seed_pool 这种用 LLM 知识
                 列 seed 后做一跳扩展再 rerank 的高质量候选场景。
                 注意：seed_pool 提供时 backend / extra_queries 被忽略。
    """
    if backend is None:
        backend = os.environ.get("DARWINIAN_SEARCH_BACKEND", "s2").lower()

    # Scheme X 路径：seed_pool 已预先准备好，跳过 search + filter_and_rank
    if seed_pool is not None:
        top_papers = seed_pool[:top_k_papers]
        # 跳过 candidate-level 操作，直接进 entity 抽取
        return _build_graph_from_papers(
            top_papers, core_problem, llm,
            batch_size=batch_size,
            top_by_relevance=top_by_relevance,
            top_by_popularity=top_by_popularity,
            max_novel_pairs=max_novel_pairs,
        )

    # Step 1 + Step 2（视 backend 而定）
    if backend == "arxiv":
        # arxiv 没有 citation 端点 → 分两档搜索直接当候选池
        candidates = arxiv_search.search_papers_arxiv_two_tiered(
            research_direction,
            classic_limit=classic_limit,
            recent_limit=recent_limit,
        )
        # arxiv 模式也接受 extra_queries：每条单独搜并合并去重
        if extra_queries:
            for q in extra_queries:
                if not q or not q.strip():
                    continue
                more = arxiv_search.search_papers_arxiv_two_tiered(
                    q, classic_limit=classic_limit // 2, recent_limit=recent_limit // 2,
                )
                candidates = _dedup_papers_by_id(candidates + more)
    else:
        # 默认 S2：分两档 + 一跳扩展
        seeds = ss.search_papers_two_tiered(
            research_direction,
            classic_limit=classic_limit,
            recent_limit=recent_limit,
        )
        # 额外子查询：每条独立搜然后合并 seeds（去重后再 expand_one_hop）
        if extra_queries:
            for q in extra_queries:
                if not q or not q.strip():
                    continue
                more = ss.search_papers_two_tiered(
                    q, classic_limit=classic_limit // 2, recent_limit=recent_limit // 2,
                )
                seeds = _dedup_papers_by_id(seeds + more)
        candidates = expand_one_hop(seeds, max_refs_per_paper, max_cits_per_paper)

    # Step 3: 清洗 + 剪枝
    top_papers = filter_and_rank(candidates, top_k=top_k_papers)
    # Step 4-6.5 抽取实体 + 结构洞
    return _build_graph_from_papers(
        top_papers, core_problem, llm,
        batch_size=batch_size,
        top_by_relevance=top_by_relevance,
        top_by_popularity=top_by_popularity,
        max_novel_pairs=max_novel_pairs,
        # arxiv 模式无 citation graph，候选池较稀疏，结构洞阈值降到 2
        pair_min_papers=2 if backend == "arxiv" else 3,
    )


def _build_graph_from_papers(
    top_papers: list[dict],
    core_problem: str,
    llm: Any,
    *,
    batch_size: int = 8,
    top_by_relevance: int = 60,
    top_by_popularity: int = 20,
    max_novel_pairs: int = 10,
    pair_min_papers: int = 3,
) -> ConceptGraph:
    """
    从已经准备好的 paper 列表构建 ConceptGraph（Steps 4-6.5）。

    供 build_concept_graph 内部复用，也供 Scheme X 路径直接复用——
    seed_pool 跳过 search + filter_and_rank 后从这里继续。
    """
    raw = batch_extract_entities(top_papers, llm, batch_size=batch_size)
    entities, limitations, paper_infos = canonicalize_merge(top_papers, raw)
    pruned = rank_relevance_top_k(entities, core_problem, top_by_relevance, top_by_popularity)
    novel_pairs = find_novel_pairs(
        pruned, max_pairs=max_novel_pairs, min_papers_each=pair_min_papers,
    )
    sufficient = is_graph_sufficient(pruned, paper_infos)
    return ConceptGraph(
        papers=paper_infos,
        entities=pruned,
        limitations=limitations,
        novel_pair_hints=novel_pairs,
        is_sufficient=sufficient,
    )
