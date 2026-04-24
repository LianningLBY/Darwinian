"""
Phase 1 v2 核心模块：从论文网络构建 ConceptGraph。

11 步流水线中这个模块承担 step 2 / 3 / 4 / 5 / 6 / 6.5：
  - expand_one_hop: 对每篇种子论文做一跳引用图扩展
  - filter_and_rank: 清洗、去重、剪枝候选池到 top K
  - batch_extract_entities: 小模型 batch 抽四元组 + limitations（commit 4）
  - canonicalize_merge: 别名合并（commit 4）
  - rank_relevance_top_k: TF-IDF 相关性 + 频次兜底（commit 5）
  - find_novel_pairs: 共现矩阵找结构洞（commit 5）
"""

from __future__ import annotations

from typing import Iterable

from darwinian.tools import semantic_scholar as ss


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
