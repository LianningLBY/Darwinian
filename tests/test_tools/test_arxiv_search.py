"""
arxiv_search 测试

不打真实 arxiv API：mock httpx，验证：
- Atom XML 解析正确性
- arxiv id 抽取（去掉 /abs/ 前缀和 vN 后缀）
- 分两档去重
- 缓存命中
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from darwinian.tools import arxiv_search as ax


# ---------------------------------------------------------------------------
# _extract_arxiv_id
# ---------------------------------------------------------------------------

class TestExtractArxivId:
    def test_full_url(self):
        assert ax._extract_arxiv_id("http://arxiv.org/abs/2401.12345v1") == "2401.12345"

    def test_no_version(self):
        assert ax._extract_arxiv_id("http://arxiv.org/abs/2401.12345") == "2401.12345"

    def test_old_style(self):
        assert ax._extract_arxiv_id("http://arxiv.org/abs/cs/0607055v2") == "cs/0607055"

    def test_empty(self):
        assert ax._extract_arxiv_id("") == ""


# ---------------------------------------------------------------------------
# _parse_entry
# ---------------------------------------------------------------------------

class TestParseEntry:
    def test_parses_full_entry(self):
        import xml.etree.ElementTree as ET
        xml = """<entry xmlns="http://www.w3.org/2005/Atom">
            <id>http://arxiv.org/abs/2401.12345v1</id>
            <title>A Study of Transformers</title>
            <summary>We present a long abstract about transformers and their efficiency.</summary>
            <published>2024-01-15T00:00:00Z</published>
        </entry>"""
        entry = ET.fromstring(xml)
        parsed = ax._parse_entry(entry)
        assert parsed["paperId"] == "arxiv:2401.12345"
        assert parsed["title"] == "A Study of Transformers"
        assert "transformers" in parsed["abstract"]
        assert parsed["year"] == 2024
        assert parsed["citationCount"] == 0
        assert parsed["source"] == "arxiv"

    def test_whitespace_compressed(self):
        import xml.etree.ElementTree as ET
        xml = """<entry xmlns="http://www.w3.org/2005/Atom">
            <id>http://arxiv.org/abs/2401.00001v1</id>
            <title>Multi

              line  title
            </title>
            <summary>x</summary>
            <published>2024-01-01T00:00:00Z</published>
        </entry>"""
        entry = ET.fromstring(xml)
        parsed = ax._parse_entry(entry)
        assert parsed["title"] == "Multi line title"

    def test_missing_id_returns_none(self):
        import xml.etree.ElementTree as ET
        xml = """<entry xmlns="http://www.w3.org/2005/Atom">
            <title>X</title>
        </entry>"""
        entry = ET.fromstring(xml)
        assert ax._parse_entry(entry) is None


# ---------------------------------------------------------------------------
# search_papers_arxiv — 用 mock 的 httpx 响应
# ---------------------------------------------------------------------------

def _fake_feed(n=2):
    entries_xml = "".join([
        f"""<entry>
            <id>http://arxiv.org/abs/2401.{str(i).zfill(5)}v1</id>
            <title>Paper {i}</title>
            <summary>Abstract for paper {i} about transformers.</summary>
            <published>2024-01-{str(i+1).zfill(2)}T00:00:00Z</published>
        </entry>"""
        for i in range(n)
    ])
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    {entries_xml}
</feed>"""


class TestSearchPapersArxiv:
    def test_returns_parsed_papers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ax, "_MIN_INTERVAL", 0.0)  # 测试跳过 sleep
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        fake_response = MagicMock()
        fake_response.text = _fake_feed(n=3)
        fake_response.raise_for_status = MagicMock()

        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.get = MagicMock(return_value=fake_response)

        with patch("httpx.Client", return_value=fake_client):
            papers = ax.search_papers_arxiv("transformer", limit=10)

        assert len(papers) == 3
        assert all(p["paperId"].startswith("arxiv:") for p in papers)
        assert all(p["source"] == "arxiv" for p in papers)

    def test_empty_query_returns_empty(self):
        assert ax.search_papers_arxiv("") == []

    def test_http_error_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ax, "_MIN_INTERVAL", 0.0)
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        with patch("httpx.Client", side_effect=Exception("network boom")):
            assert ax.search_papers_arxiv("x") == []

    def test_malformed_xml_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ax, "_MIN_INTERVAL", 0.0)
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        fake_response = MagicMock()
        fake_response.text = "not xml at all"
        fake_response.raise_for_status = MagicMock()
        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.get = MagicMock(return_value=fake_response)
        with patch("httpx.Client", return_value=fake_client):
            assert ax.search_papers_arxiv("x") == []

    def test_cache_hit_skips_http(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ax, "_MIN_INTERVAL", 0.0)
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        fake_response = MagicMock()
        fake_response.text = _fake_feed(n=2)
        fake_response.raise_for_status = MagicMock()
        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.get = MagicMock(return_value=fake_response)

        with patch("httpx.Client", return_value=fake_client) as MockCli:
            ax.search_papers_arxiv("same_query", limit=5)
            ax.search_papers_arxiv("same_query", limit=5)   # 第二次应命中缓存
        # 只应发一次 HTTP
        assert MockCli.call_count == 1


# ---------------------------------------------------------------------------
# search_papers_arxiv_two_tiered
# ---------------------------------------------------------------------------

class TestTwoTieredArxiv:
    def test_dedupe_by_paper_id(self):
        classics = [{"paperId": "arxiv:1", "title": "c1"}, {"paperId": "arxiv:2", "title": "c2"}]
        recent = [{"paperId": "arxiv:2", "title": "r2-dup"}, {"paperId": "arxiv:3", "title": "r3"}]
        with patch.object(ax, "search_papers_arxiv", side_effect=[classics, recent]):
            merged = ax.search_papers_arxiv_two_tiered("q")
        ids = sorted(p["paperId"] for p in merged)
        assert ids == ["arxiv:1", "arxiv:2", "arxiv:3"]
        # 经典档 arxiv:2 被保留
        two = next(p for p in merged if p["paperId"] == "arxiv:2")
        assert two["title"] == "c2"
