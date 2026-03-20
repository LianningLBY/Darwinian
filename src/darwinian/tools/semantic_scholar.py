"""
Semantic Scholar API 工具

负责：
- 按研究方向检索文献
- 提取每篇文献的 Limitations 段落
- 根据 banned_keywords 过滤不需要的方向
"""

from __future__ import annotations

import re
import httpx


SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = "title,abstract,year,authors,tldr,externalIds"


def search_papers_with_limitations(
    query: str,
    banned_keywords: list[str] | None = None,
    limit: int = 10,
    api_key: str | None = None,
) -> list[dict]:
    """
    检索相关文献并提取 Limitations 信息。

    Args:
        query: 搜索查询词
        banned_keywords: 需要过滤掉的关键词（来自 failed_ledger）
        limit: 返回文献数量上限
        api_key: Semantic Scholar API Key（可选，无则使用公开接口）

    Returns:
        文献列表，每项包含 title、abstract、year、limitations（提取的局限性段落）
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": min(limit * 2, 50),  # 多拉一些，过滤后再截断
        "fields": DEFAULT_FIELDS,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        # 网络/HTTP 错误时返回空列表，由调用方决定降级策略
        return []

    papers = data.get("data", [])

    results = []
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "") or ""

        # 过滤含禁用关键词的文献
        if banned_keywords and _contains_banned_keyword(title + abstract, banned_keywords):
            continue

        limitations = _extract_limitations(abstract)
        results.append({
            "title": title,
            "year": paper.get("year"),
            "abstract": abstract[:500],  # 截断避免 token 爆炸
            "limitations": limitations,
            "paper_id": paper.get("paperId", ""),
        })

        if len(results) >= limit:
            break

    return results


def _extract_limitations(text: str) -> str:
    """
    从摘要或正文中提取 Limitations 相关内容。
    优先匹配明确的 limitation 关键词段落。
    """
    if not text:
        return ""

    # 尝试匹配 "limitation", "drawback", "however", "challenge" 等关键句
    patterns = [
        r"(?i)(limitation[s]?|drawback[s]?|shortcoming[s]?)[^.]*\.",
        r"(?i)however[,\s][^.]*\.",
        r"(?i)(challenge[s]?|difficult)[^.]*\.",
        r"(?i)(fail[s]?|cannot|unable)[^.]*\.",
    ]

    found = []
    for pattern in patterns:
        matches = re.findall(pattern, text[:2000])
        if matches:
            found.extend(matches[:2])

    if found:
        return " ".join(str(m) for m in found[:3])

    # 若无明确 limitation 句，返回摘要最后两句（通常是结论/局限）
    sentences = text.split(". ")
    return ". ".join(sentences[-2:]) if len(sentences) >= 2 else text[-200:]


def _contains_banned_keyword(text: str, banned_keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in banned_keywords)
