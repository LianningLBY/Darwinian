"""
arxiv.org 搜索工具（备选文献源，无需 API key）

arxiv 的搜索 API（https://export.arxiv.org/api/query）返回 Atom XML：
- 公开免费，匿名可用
- 建议请求间隔 3s，文档：https://info.arxiv.org/help/api/user-manual.html
- 只提供搜索，没有 /references 或 /citations（无 citation graph）

本模块返回与 semantic_scholar.search_papers 结构兼容的字典列表：
  {"paperId": "arxiv:2401.12345", "title": ..., "abstract": ...,
   "year": 2024, "citationCount": 0}

paperId 加 "arxiv:" 前缀以区分 S2 id。citationCount 固定为 0（arxiv 无此字段），
下游 filter_and_rank 的 citation 排序会退化为"按 arxiv 返回顺序"。
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

from darwinian.tools.semantic_scholar import _cache_get, _cache_key, _cache_set


ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

# arxiv 官方建议单源请求间隔 ≥ 3s
_LAST_REQUEST_TIME: float = 0.0
_MIN_INTERVAL = float(os.environ.get("DARWINIAN_ARXIV_MIN_INTERVAL", "3.0"))


def _respect_rate_limit() -> None:
    global _LAST_REQUEST_TIME
    elapsed = time.time() - _LAST_REQUEST_TIME
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_REQUEST_TIME = time.time()


def _extract_arxiv_id(entry_id: str) -> str:
    """从 Atom <id> 抽纯 arxiv id，如 'http://arxiv.org/abs/2401.12345v1' → '2401.12345'"""
    if not entry_id:
        return ""
    # 去掉 URL 前缀
    if "/abs/" in entry_id:
        entry_id = entry_id.rsplit("/abs/", 1)[1]
    # 去掉 vN 版本后缀
    if "v" in entry_id:
        parts = entry_id.rsplit("v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            entry_id = parts[0]
    return entry_id


def _parse_entry(entry: ET.Element) -> dict[str, Any] | None:
    """把单条 Atom <entry> 转成 S2 兼容的 paper 字典"""
    def _text(tag: str) -> str:
        el = entry.find(f"atom:{tag}", ARXIV_NS)
        return (el.text or "").strip() if el is not None and el.text else ""

    arxiv_id = _extract_arxiv_id(_text("id"))
    if not arxiv_id:
        return None

    title = " ".join(_text("title").split())  # 压缩空白
    abstract = " ".join(_text("summary").split())
    published = _text("published")
    year = 0
    if published and len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])

    return {
        "paperId": f"arxiv:{arxiv_id}",
        "title": title,
        "abstract": abstract,
        "year": year,
        "citationCount": 0,   # arxiv 无此字段
        "source": "arxiv",
    }


def search_papers_arxiv(
    query: str,
    limit: int = 20,
    sort_by: str = "relevance",
) -> list[dict]:
    """
    arxiv 搜索，返回与 S2 structure 兼容的 paper 字典列表。

    Args:
        query: 搜索关键词（会被 all: 前缀包起）
        limit: 返回上限
        sort_by: "relevance" | "lastUpdatedDate" | "submittedDate"

    Returns:
        list of {paperId, title, abstract, year, citationCount=0, source="arxiv"}
        失败返空列表。
    """
    if not query:
        return []

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(limit, 100),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    # 复用 S2 模块的缓存机制（目录和 TTL 都共享）
    cache_k = _cache_key("/arxiv/query", params)
    cached = _cache_get(cache_k)
    if cached is not None:
        return cached

    _respect_rate_limit()
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(ARXIV_API_BASE, params=params)
            resp.raise_for_status()
            xml_text = resp.text
    except Exception:
        return []

    # 解析 Atom XML
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    entries = root.findall("atom:entry", ARXIV_NS)
    papers = []
    for e in entries:
        parsed = _parse_entry(e)
        if parsed:
            papers.append(parsed)

    _cache_set(cache_k, papers)
    return papers


def search_papers_arxiv_two_tiered(
    query: str,
    classic_limit: int = 20,
    recent_limit: int = 20,
) -> list[dict]:
    """
    arxiv 版的"分两档"：一档按 relevance 不限年份，一档按 submittedDate 拿最新。
    按 paperId 去重，优先保留 relevance 档。
    """
    classics = search_papers_arxiv(query, limit=classic_limit, sort_by="relevance")
    recent = search_papers_arxiv(query, limit=recent_limit, sort_by="submittedDate")
    merged: dict[str, dict] = {}
    for p in classics + recent:
        pid = p.get("paperId")
        if pid and pid not in merged:
            merged[pid] = p
    return list(merged.values())
