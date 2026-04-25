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
# 429 时的退避：sleep 指定秒后重试一次，再失败就放弃
_RETRY_AFTER_429_SECONDS = 5.0


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
    统一 GET 封装：缓存 → 限流 → 请求 → 回写缓存。

    限流：每次实际 HTTP 请求前确保距上次 S2 请求 >= _MIN_INTERVAL_SECONDS（默认 1.1s）。
    缓存命中不计入限流。

    429 处理：检测到 429 后 sleep _RETRY_AFTER_429_SECONDS（默认 5s）重试一次，再失败放弃。

    失败返回 None（调用方决定降级）。
    """
    if use_cache:
        cached = _cache_get(_cache_key(endpoint, params))
        if cached is not None:
            return cached

    for attempt in range(2):  # 最多 1 次 429 重试
        _respect_rate_limit()
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}{endpoint}",
                    params=params,
                    headers=_headers(),
                )
            if resp.status_code == 429 and attempt == 0:
                # 限流碰撞，退避后再试一次
                print(f"[s2] 429 命中，{_RETRY_AFTER_429_SECONDS}s 后重试...")
                time.sleep(_RETRY_AFTER_429_SECONDS)
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None
        break
    else:
        return None  # 两次都 429

    if use_cache:
        _cache_set(_cache_key(endpoint, params), data)
    return data


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
