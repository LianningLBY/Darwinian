"""
semantic_scholar 工具测试

不打真实 S2 API：mock httpx 和/或 _s2_get 验证：
- 响应结构解包（references/citations 的 citedPaper/citingPaper 包装）
- 分两档检索的去重逻辑
- 缓存键稳定性
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from darwinian.tools import semantic_scholar as ss


# ---------------------------------------------------------------------------
# 缓存键
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_same_inputs_same_key(self):
        k1 = ss._cache_key("/paper/x/references", {"limit": 30, "fields": "title"})
        k2 = ss._cache_key("/paper/x/references", {"limit": 30, "fields": "title"})
        assert k1 == k2

    def test_param_order_irrelevant(self):
        k1 = ss._cache_key("/p", {"a": 1, "b": 2})
        k2 = ss._cache_key("/p", {"b": 2, "a": 1})
        assert k1 == k2

    def test_different_params_different_key(self):
        k1 = ss._cache_key("/p", {"limit": 30})
        k2 = ss._cache_key("/p", {"limit": 31})
        assert k1 != k2


# ---------------------------------------------------------------------------
# get_references / get_citations 的 wrap 解包
# ---------------------------------------------------------------------------

class TestReferencesUnwrap:
    def test_references_unwraps_cited_paper(self):
        fake_resp = {
            "data": [
                {"citedPaper": {"paperId": "p1", "title": "A"}},
                {"citedPaper": {"paperId": "p2", "title": "B"}},
                {"citedPaper": None},  # 有时 S2 返回 null
            ]
        }
        with patch.object(ss, "_s2_get", return_value=fake_resp):
            refs = ss.get_references("root_id")
        assert len(refs) == 2
        assert refs[0]["paperId"] == "p1"

    def test_references_empty_on_api_fail(self):
        with patch.object(ss, "_s2_get", return_value=None):
            assert ss.get_references("x") == []

    def test_citations_unwraps_citing_paper(self):
        fake_resp = {"data": [{"citingPaper": {"paperId": "p3"}}]}
        with patch.object(ss, "_s2_get", return_value=fake_resp):
            cits = ss.get_citations("root_id")
        assert cits == [{"paperId": "p3"}]

    def test_references_handles_null_data(self):
        """S2 论文无引用时返 {"data": null}——之前会炸 'NoneType not iterable'"""
        with patch.object(ss, "_s2_get", return_value={"data": None}):
            assert ss.get_references("paper_with_no_refs") == []

    def test_citations_handles_null_data(self):
        """同上，无 citation 的论文 S2 返 {"data": null}"""
        with patch.object(ss, "_s2_get", return_value={"data": None}):
            assert ss.get_citations("brand_new_paper") == []

    def test_references_skips_non_dict_items(self):
        """防御：data 数组里偶发 None 或非 dict 条目"""
        fake = {"data": [
            {"citedPaper": {"paperId": "p1"}},
            None,
            "garbage",
            {"citedPaper": None},
            {"citedPaper": {"paperId": "p2"}},
        ]}
        with patch.object(ss, "_s2_get", return_value=fake):
            refs = ss.get_references("x")
        assert [r["paperId"] for r in refs] == ["p1", "p2"]


# ---------------------------------------------------------------------------
# 分两档检索
# ---------------------------------------------------------------------------

class TestTwoTieredSearch:
    def test_dedupe_by_paper_id_preferring_classics(self):
        classics = [
            {"paperId": "A", "title": "classic-A", "year": 2018},
            {"paperId": "B", "title": "classic-B", "year": 2020},
        ]
        recent = [
            {"paperId": "B", "title": "recent-B-dup", "year": 2024},  # 同 id，应该保经典档
            {"paperId": "C", "title": "recent-C", "year": 2025},
        ]
        with patch.object(ss, "search_papers", side_effect=[classics, recent]):
            merged = ss.search_papers_two_tiered("transformer")
        ids = [p["paperId"] for p in merged]
        assert set(ids) == {"A", "B", "C"}
        # 经典档的 B 被保留
        b = next(p for p in merged if p["paperId"] == "B")
        assert b["title"] == "classic-B"

    def test_empty_on_both_fail(self):
        with patch.object(ss, "search_papers", return_value=[]):
            assert ss.search_papers_two_tiered("x") == []


# ---------------------------------------------------------------------------
# 缓存命中逻辑
# ---------------------------------------------------------------------------

class TestCacheRoundtrip:
    def test_set_and_get_within_ttl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        ss._cache_set("k1", {"hello": "world"})
        assert ss._cache_get("k1") == {"hello": "world"}

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        assert ss._cache_get("not_existent") is None

    def test_expired_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "CACHE_TTL_SECONDS", 0)  # 立即过期
        ss._cache_set("k2", "value")
        # TTL=0 意味着任何已写入的文件 mtime 的 age > 0 即过期
        import time
        time.sleep(0.01)
        assert ss._cache_get("k2") is None


# ---------------------------------------------------------------------------
# api_key 环境变量
# ---------------------------------------------------------------------------

class TestApiKey:
    def test_headers_without_key(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        assert ss._headers() == {}

    def test_headers_with_key(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "k_xyz")
        assert ss._headers() == {"x-api-key": "k_xyz"}


# ---------------------------------------------------------------------------
# Rate limiter + 429 重试
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_respects_min_interval(self, monkeypatch):
        """连续两次调用必须间隔 >= _MIN_INTERVAL_SECONDS"""
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.05)
        monkeypatch.setattr(ss, "_LAST_REQUEST_TIME", 0.0)
        import time as _t
        ss._respect_rate_limit()
        t1 = _t.time()
        ss._respect_rate_limit()
        t2 = _t.time()
        assert (t2 - t1) >= 0.04   # 留点余量

    def test_first_call_no_wait(self, monkeypatch):
        """首次调用（_LAST_REQUEST_TIME=0）不应阻塞"""
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 100.0)
        monkeypatch.setattr(ss, "_LAST_REQUEST_TIME", 0.0)
        import time as _t
        t0 = _t.time()
        ss._respect_rate_limit()
        assert (_t.time() - t0) < 0.01


class TestS2Get429Retry:
    def _mock_client_with_responses(self, responses):
        """构造一个 httpx.Client mock，依次返回 responses 列表里的 Response"""
        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.get = MagicMock(side_effect=responses)
        return fake_client

    def _resp(self, status, json_body=None):
        """模拟 httpx Response：所有 4xx/5xx 都会让 raise_for_status() 抛（与真 httpx 行为一致）"""
        r = MagicMock()
        r.status_code = status
        r.json = MagicMock(return_value=json_body or {})
        if status >= 400:
            r.raise_for_status = MagicMock(side_effect=Exception(f"HTTP {status}"))
        else:
            r.raise_for_status = MagicMock()
        return r

    def test_success_first_try(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(ss, "_RETRY_AFTER_429_SECONDS", 0.0)
        client = self._mock_client_with_responses([self._resp(200, {"data": [1]})])
        with patch("httpx.Client", return_value=client):
            data = ss._s2_get("/x", {"q": "test"})
        assert data == {"data": [1]}
        assert client.get.call_count == 1

    def test_429_then_success_retries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(ss, "_RETRY_AFTER_429_SECONDS", 0.0)
        client = self._mock_client_with_responses([
            self._resp(429),
            self._resp(200, {"ok": True}),
        ])
        with patch("httpx.Client", return_value=client):
            data = ss._s2_get("/x", {"q": "y"})
        assert data == {"ok": True}
        assert client.get.call_count == 2

    def test_429_twice_gives_up(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(ss, "_RETRY_AFTER_429_SECONDS", 0.0)
        client = self._mock_client_with_responses([
            self._resp(429),
            self._resp(429),
        ])
        with patch("httpx.Client", return_value=client):
            data = ss._s2_get("/x", {"q": "z"})
        assert data is None
        # 两次 attempt 都打了
        assert client.get.call_count == 2

    def test_cache_hit_skips_rate_limit_and_http(self, tmp_path, monkeypatch):
        """缓存命中时既不应打 HTTP 也不应等限流"""
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 100.0)
        # 预存一份缓存
        ss._cache_set(ss._cache_key("/cached", {"q": "z"}), {"hit": True})
        with patch("httpx.Client") as mc:
            data = ss._s2_get("/cached", {"q": "z"})
        assert data == {"hit": True}
        assert mc.call_count == 0  # 没打 HTTP


# ---------------------------------------------------------------------------
# Tool 4: get_paper_by_doi / batch_search / get_papers_batch
# ---------------------------------------------------------------------------

class TestGetPaperByDoi:
    def test_passes_doi_prefix(self):
        """DOI 应被 /paper/DOI:{doi} 路径引用"""
        with patch.object(ss, "_s2_get", return_value={"paperId": "abc"}) as mock_get:
            paper = ss.get_paper_by_doi("10.18653/v1/2024.acl-main.123")
        assert paper == {"paperId": "abc"}
        endpoint = mock_get.call_args.args[0]
        assert endpoint == "/paper/DOI:10.18653/v1/2024.acl-main.123"

    def test_strips_whitespace(self):
        with patch.object(ss, "_s2_get", return_value={"x": 1}) as mock_get:
            ss.get_paper_by_doi("  10.18653/v1/y  ")
        assert mock_get.call_args.args[0] == "/paper/DOI:10.18653/v1/y"

    def test_empty_returns_none_no_call(self):
        with patch.object(ss, "_s2_get") as mock_get:
            assert ss.get_paper_by_doi("") is None
            assert ss.get_paper_by_doi("   ") is None
        assert mock_get.call_count == 0


class TestBatchSearch:
    def test_one_call_per_query(self):
        with patch.object(ss, "search_papers", side_effect=[[{"id": 1}], [{"id": 2}]]) as mock:
            out = ss.batch_search(["query A", "query B"])
        assert mock.call_count == 2
        assert out == [[{"id": 1}], [{"id": 2}]]

    def test_year_param_propagates(self):
        with patch.object(ss, "search_papers", return_value=[]) as mock:
            ss.batch_search(["x"], year="2023-2026")
        assert mock.call_args.kwargs["year"] == "2023-2026"

    def test_empty_queries_returns_empty(self):
        with patch.object(ss, "search_papers") as mock:
            assert ss.batch_search([]) == []
        assert mock.call_count == 0


class TestGetPapersBatch:
    def _resp(self, status, json_body=None):
        r = MagicMock()
        r.status_code = status
        r.json = MagicMock(return_value=json_body if json_body is not None else [])
        if status >= 400:
            r.raise_for_status = MagicMock(side_effect=Exception(f"HTTP {status}"))
        else:
            r.raise_for_status = MagicMock()
        return r

    def _client(self, response):
        c = MagicMock()
        c.__enter__ = MagicMock(return_value=c)
        c.__exit__ = MagicMock(return_value=False)
        c.post = MagicMock(return_value=response)
        return c

    def test_empty_ids_returns_empty(self):
        with patch("httpx.Client") as mc:
            assert ss.get_papers_batch([]) == []
        assert mc.call_count == 0

    def test_success_returns_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.0)
        client = self._client(self._resp(200, [{"paperId": "p1"}, {"paperId": "p2"}]))
        with patch("httpx.Client", return_value=client):
            out = ss.get_papers_batch(["p1", "p2"])
        assert len(out) == 2
        assert out[0]["paperId"] == "p1"

    def test_replaces_none_with_empty_dict(self, tmp_path, monkeypatch):
        """S2 batch 偶尔返 None（id 不存在），转成空 dict 占位"""
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.0)
        client = self._client(self._resp(200, [{"paperId": "p1"}, None, {"paperId": "p3"}]))
        with patch("httpx.Client", return_value=client):
            out = ss.get_papers_batch(["p1", "bad", "p3"])
        assert out == [{"paperId": "p1"}, {}, {"paperId": "p3"}]

    def test_cache_hit_skips_http(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.0)
        cached = [{"paperId": "p1"}]
        cache_k = ss._cache_key("/paper/batch", {"ids": "p1", "fields": ss.GRAPH_FIELDS})
        ss._cache_set(cache_k, cached)
        with patch("httpx.Client") as mc:
            out = ss.get_papers_batch(["p1"])
        assert out == cached
        assert mc.call_count == 0

    def test_429_then_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(ss, "_RETRY_AFTER_429_SECONDS", 0.0)
        responses = [self._resp(429), self._resp(200, [{"paperId": "p1"}])]
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.post = MagicMock(side_effect=responses)
        with patch("httpx.Client", return_value=client):
            out = ss.get_papers_batch(["p1"])
        assert out == [{"paperId": "p1"}]

    def test_truncates_above_500(self, tmp_path, monkeypatch):
        """S2 限制 ≤ 500 paperIds，超过会被静默截断"""
        monkeypatch.setattr(ss, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ss, "_MIN_INTERVAL_SECONDS", 0.0)
        ids = [f"p{i}" for i in range(600)]
        client = self._client(self._resp(200, [{"paperId": f"p{i}"} for i in range(500)]))
        with patch("httpx.Client", return_value=client):
            ss.get_papers_batch(ids)
        sent = client.post.call_args.kwargs["json"]
        assert len(sent["ids"]) == 500
