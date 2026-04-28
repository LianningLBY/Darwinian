"""
Semantic Scholar API 工具

负责：
- 按研究方向检索文献（支持分两档：经典 + 近三年）
- 提取每篇文献的 Limitations 段落（legacy 正则路径，v2 由 knowledge_graph 升级）
- 引用图遍历：/paper/{id}/references 和 /paper/{id}/citations
- 文件级 pickle 缓存（7 天 TTL），避免开发期重复请求
- SEMANTIC_SCHOLAR_API_KEY 环境变量自动提速
- 根据 banned_keywords 过滤不需要的方向
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import httpx


SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = "title,abstract,year,authors,tldr,externalIds"
# 引用图扩展默认 fields，带 citationCount 供剪枝
GRAPH_FIELDS = "paperId,title,abstract,year,citationCount,externalIds"

CACHE_DIR = Path(os.environ.get("DARWINIAN_S2_CACHE_DIR", ".cache/s2"))
CACHE_TTL_SECONDS = int(os.environ.get("DARWINIAN_S2_CACHE_TTL", 7 * 24 * 3600))

# S2 introductory rate plan: 1 req/s cumulative across all endpoints.
# 默认 1.1s 留 10% buffer，DARWINIAN_S2_MIN_INTERVAL 可覆盖（如有更高 plan）
_MIN_INTERVAL_SECONDS = float(os.environ.get("DARWINIAN_S2_MIN_INTERVAL", "1.1"))
_LAST_REQUEST_TIME: float = 0.0
# Pri-6: 429 改指数退避 3 轮 (5s / 10s / 20s)，比原来 1 轮稳得多
_429_BACKOFF_SCHEDULE = [5.0, 10.0, 20.0]

# Pri-6: in-memory LRU 加速 session 内重复（disk pickle 慢）
_INMEM_CACHE: dict = {}
_INMEM_CACHE_MAX = int(os.environ.get("DARWINIAN_S2_INMEM_CACHE_MAX", "5000"))

# Pri-6: 统计计数器（任意时刻调 print_s2_stats() 看）
_S2_STATS = {
    "inmem_hits": 0,
    "disk_hits": 0,
    "http_calls": 0,
    "http_429s": 0,
    "http_failures": 0,
}


def _api_key() -> str | None:
    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY")


def _respect_rate_limit() -> None:
    """阻塞到与上次 S2 请求间隔 >= _MIN_INTERVAL_SECONDS。"""
    global _LAST_REQUEST_TIME
    elapsed = time.time() - _LAST_REQUEST_TIME
    if elapsed < _MIN_INTERVAL_SECONDS:
        time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
    _LAST_REQUEST_TIME = time.time()


def _headers() -> dict[str, str]:
    key = _api_key()
    return {"x-api-key": key} if key else {}


def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
    """为缓存生成稳定 key：endpoint + 排序后 params 的 md5 摘要"""
    canon = endpoint + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.md5(canon.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Any | None:
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > CACHE_TTL_SECONDS:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_set(key: str, value: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f)
    except Exception:
        pass  # 缓存失败不能影响主流程


def _s2_get(endpoint: str, params: dict[str, Any], *, use_cache: bool = True) -> dict | None:
    """
    统一 GET 封装：in-mem cache → disk cache → 限流 → 请求（含指数退避）→ 回写两层 cache。

    限流：每次实际 HTTP 请求前确保距上次 S2 请求 >= _MIN_INTERVAL_SECONDS（默认 1.1s）。
    缓存命中不计入限流。

    429 处理 (Pri-6): 指数退避 3 轮 (5s / 10s / 20s)，比原来 1 轮稳。
    任何 HTTP 失败（含全部 429 重试用完）返 None，调用方决定降级。
    """
    cache_key = _cache_key(endpoint, params) if use_cache else None

    # Layer 1: in-memory LRU
    if use_cache and cache_key in _INMEM_CACHE:
        _S2_STATS["inmem_hits"] += 1
        return _INMEM_CACHE[cache_key]

    # Layer 2: disk pickle
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            _S2_STATS["disk_hits"] += 1
            _inmem_set(cache_key, cached)   # promote 到 in-mem
            return cached

    # Layer 3: 网络请求（含指数退避）
    data = None
    for attempt, backoff in enumerate(_429_BACKOFF_SCHEDULE):
        _respect_rate_limit()
        _S2_STATS["http_calls"] += 1
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}{endpoint}",
                    params=params,
                    headers=_headers(),
                )
            if resp.status_code == 429:
                _S2_STATS["http_429s"] += 1
                if attempt < len(_429_BACKOFF_SCHEDULE) - 1:
                    print(f"[s2] 429 命中 (round {attempt+1}/{len(_429_BACKOFF_SCHEDULE)})，"
                          f"{backoff}s 后重试...")
                    time.sleep(backoff)
                    continue
                else:
                    print(f"[s2] 429 重试 {len(_429_BACKOFF_SCHEDULE)} 次仍失败，放弃")
                    _S2_STATS["http_failures"] += 1
                    return None
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception:
            _S2_STATS["http_failures"] += 1
            return None

    if data is None:
        return None
    if use_cache:
        _cache_set(cache_key, data)
        _inmem_set(cache_key, data)
    return data


def _inmem_set(key: str, value: Any) -> None:
    """Pri-6: LRU 简单实现：超 max 时丢一半最老的（dict 保留插入顺序）"""
    _INMEM_CACHE[key] = value
    if len(_INMEM_CACHE) > _INMEM_CACHE_MAX:
        # 删最老的一半
        n_drop = _INMEM_CACHE_MAX // 2
        keys_to_drop = list(_INMEM_CACHE.keys())[:n_drop]
        for k in keys_to_drop:
            del _INMEM_CACHE[k]


def get_s2_stats() -> dict:
    """返回 session 累计 S2 调用统计 (Pri-6)"""
    total_lookups = (
        _S2_STATS["inmem_hits"] + _S2_STATS["disk_hits"] + _S2_STATS["http_calls"]
    )
    return {
        **_S2_STATS,
        "total_lookups": total_lookups,
        "cache_hit_rate": (
            (_S2_STATS["inmem_hits"] + _S2_STATS["disk_hits"]) / total_lookups
            if total_lookups else 0.0
        ),
    }


def reset_s2_stats() -> None:
    """重置统计（测试用）"""
    for k in _S2_STATS:
        _S2_STATS[k] = 0


def clear_inmem_cache() -> None:
    """清空 in-memory cache（测试用）"""
    _INMEM_CACHE.clear()


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def search_papers(
    query: str,
    limit: int = 20,
    year: str | None = None,
    fields: str = GRAPH_FIELDS,
) -> list[dict]:
    """
    基础搜索接口（Phase 1 v2 使用）。

    Args:
        query: 搜索关键词
        limit: 返回上限（S2 API 允许 1-100）
        year: 年份过滤，S2 支持 "2023-2026"、"2023:" 等语法
        fields: 逗号分隔的字段列表
    """
    params: dict[str, Any] = {"query": query, "limit": min(limit, 100), "fields": fields}
    if year:
        params["year"] = year
    data = _s2_get("/paper/search", params)
    if not data:
        return []
    return data.get("data", []) or []


def search_papers_two_tiered(
    query: str,
    classic_limit: int = 20,
    recent_limit: int = 20,
    recent_year: str = "2023-2026",
) -> list[dict]:
    """
    分两档检索：经典（不限年份）+ 近三年。按 paperId 去重，优先保留经典档条目。

    Returns: 合并去重后的 List[dict]。
    """
    classics = search_papers(query, limit=classic_limit)
    recent = search_papers(query, limit=recent_limit, year=recent_year)
    merged: dict[str, dict] = {}
    for p in classics + recent:  # classics 先进，后来同 id 的 recent 不覆盖
        pid = p.get("paperId")
        if pid and pid not in merged:
            merged[pid] = p
    return list(merged.values())


def get_references(paper_id: str, limit: int = 30, fields: str = GRAPH_FIELDS) -> list[dict]:
    """
    获取某篇论文**引用的**论文列表（即 references，往前翻：这篇引了谁）。

    S2 响应结构：{"data": [{"citedPaper": {...}}, ...]}，此函数解包后直接返回 paper 字典列表。
    论文无引用时 S2 会返 {"data": null} 而非空数组，已用 `or []` 防御。
    失败返回空列表。
    """
    data = _s2_get(
        f"/paper/{paper_id}/references",
        {"limit": min(limit, 100), "fields": fields},
    )
    if not data:
        return []
    items = data.get("data") or []   # 防 {"data": null}
    return [it.get("citedPaper") for it in items if isinstance(it, dict) and it.get("citedPaper")]


def get_citations(paper_id: str, limit: int = 30, fields: str = GRAPH_FIELDS) -> list[dict]:
    """
    获取**引用这篇论文的**论文列表（即 citations，往后翻：谁引了这篇）。

    S2 响应结构：{"data": [{"citingPaper": {...}}, ...]}，论文无 citation 时返
    {"data": null}，已用 `or []` 防御。失败返回空列表。
    """
    data = _s2_get(
        f"/paper/{paper_id}/citations",
        {"limit": min(limit, 100), "fields": fields},
    )
    if not data:
        return []
    items = data.get("data") or []   # 防 {"data": null}
    return [it.get("citingPaper") for it in items if isinstance(it, dict) and it.get("citingPaper")]


def get_paper_details(paper_id: str, fields: str = GRAPH_FIELDS) -> dict | None:
    """查单篇论文详情（带缓存）。"""
    return _s2_get(f"/paper/{paper_id}", {"fields": fields})


def get_paper_by_doi(doi: str, fields: str = GRAPH_FIELDS) -> dict | None:
    """
    通过 DOI 查 S2 论文详情。

    S2 支持 paper_id 多种前缀：DOI、ARXIV、CorpusId 等。本函数封装 DOI 路径。
    Endpoint: /paper/DOI:{doi}

    Args:
        doi: 不带任何前缀的纯 DOI（如 "10.18653/v1/2024.acl-main.123"）
        fields: 逗号分隔字段

    Returns:
        paper dict 或 None（DOI 不存在 / 网络失败）
    """
    if not doi or not doi.strip():
        return None
    return _s2_get(f"/paper/DOI:{doi.strip()}", {"fields": fields})


def batch_search(
    queries: list[str],
    limit_per_query: int = 10,
    year: str | None = None,
    fields: str = GRAPH_FIELDS,
) -> list[list[dict]]:
    """
    批量搜索：对每个 query 调一次 search_papers，返回每个 query 的结果列表。

    注意：这不是 S2 的"batch"端点（那个是按 paperId 批量），是多次 search 的循环。
    每次调用之间有 _MIN_INTERVAL_SECONDS 节流，防触发 429。

    Args:
        queries: 多个查询关键词
        limit_per_query: 每个 query 返回上限
        year: 年份过滤，应用于所有 query
        fields: 字段

    Returns:
        list[list[dict]] —— 与 queries 一一对应。失败的 query 返空 list 不抛错。
    """
    return [search_papers(q, limit=limit_per_query, year=year, fields=fields) for q in queries]


def get_papers_batch(
    paper_ids: list[str],
    fields: str = GRAPH_FIELDS,
) -> list[dict]:
    """
    S2 真正的 batch 接口：/paper/batch，单次 POST 拿多个 paper 详情，比循环 N 次省限速。

    Args:
        paper_ids: paper_id 列表（最多 500 个，超过 S2 会拒）
        fields: 字段

    Returns:
        与 paper_ids 一一对应的 paper dict 列表（不存在的 id 对应位置为空 dict）

    Note: S2 文档明确支持 paperIds 多种格式：原生 paperId、DOI:、ARXIV:、CorpusId: 等。
    """
    if not paper_ids:
        return []

    # 缓存命中检查（单 batch 整体缓存）
    cache_k = _cache_key("/paper/batch", {"ids": ",".join(sorted(paper_ids)), "fields": fields})
    cached = _cache_get(cache_k)
    if cached is not None:
        return cached

    # Pri-6: 用同样的指数退避 schedule
    result = None
    for attempt, backoff in enumerate(_429_BACKOFF_SCHEDULE):
        _respect_rate_limit()
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/batch",
                    params={"fields": fields},
                    json={"ids": paper_ids[:500]},
                    headers=_headers(),
                )
            if resp.status_code == 429:
                _S2_STATS["http_429s"] += 1
                if attempt < len(_429_BACKOFF_SCHEDULE) - 1:
                    print(f"[s2 batch] 429 命中 (round {attempt+1}/"
                          f"{len(_429_BACKOFF_SCHEDULE)})，{backoff}s 后重试...")
                    time.sleep(backoff)
                    continue
                else:
                    return []
            resp.raise_for_status()
            result = resp.json() or []
            break
        except Exception:
            return []
    if result is None:
        return []

    # 防 list 里夹 None
    cleaned = [item if isinstance(item, dict) else {} for item in result]
    _cache_set(cache_k, cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Legacy: v1 用的高级搜索（保留供 bottleneck_miner 现有代码兼容）
# ---------------------------------------------------------------------------

def search_papers_with_limitations(
    query: str,
    banned_keywords: list[str] | None = None,
    limit: int = 10,
    api_key: str | None = None,
) -> list[dict]:
    """
    [legacy] v1 检索接口：搜索 + 用正则抽 limitation + banned 过滤。

    Phase 1 v2 由 knowledge_graph 模块用结构化 LLM 抽取替代。此函数保留兼容性。
    """
    # api_key 参数保留向后兼容，实际用环境变量统一
    params = {
        "query": query,
        "limit": min(limit * 2, 50),
        "fields": DEFAULT_FIELDS,
    }
    data = _s2_get("/paper/search", params)
    if not data:
        return []

    papers = data.get("data", []) or []
    results = []
    for paper in papers:
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""

        if banned_keywords and _contains_banned_keyword(title + abstract, banned_keywords):
            continue

        limitations = _extract_limitations(abstract)
        results.append({
            "title": title,
            "year": paper.get("year"),
            "abstract": abstract[:500],
            "limitations": limitations,
            "paper_id": paper.get("paperId", ""),
        })
        if len(results) >= limit:
            break
    return results


def _extract_limitations(text: str) -> str:
    """[legacy] 从摘要中抽 limitation 句段（v2 由 LLM 结构化替代）。"""
    if not text:
        return ""

    patterns = [
        r"(?i)(limitation[s]?|drawback[s]?|shortcoming[s]?)[^.]*\.",
        r"(?i)however[,\s][^.]*\.",
        r"(?i)(challenge[s]?|difficult)[^.]*\.",
        r"(?i)(fail[s]?|cannot|unable)[^.]*\.",
    ]

    found: list[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, text[:2000])
        if matches:
            found.extend(matches[:2])

    if found:
        return " ".join(str(m) for m in found[:3])

    sentences = text.split(". ")
    return ". ".join(sentences[-2:]) if len(sentences) >= 2 else text[-200:]


def _contains_banned_keyword(text: str, banned_keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in banned_keywords)
