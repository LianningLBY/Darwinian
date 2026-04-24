"""
knowledge_graph 模块测试

不打真实 S2 API：mock semantic_scholar.get_references / get_citations。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from darwinian.utils import knowledge_graph as kg


# ---------------------------------------------------------------------------
# expand_one_hop
# ---------------------------------------------------------------------------

class TestExpandOneHop:
    def test_includes_seeds(self):
        seeds = [{"paperId": "A", "title": "seed-A"}]
        with patch("darwinian.tools.semantic_scholar.get_references", return_value=[]):
            with patch("darwinian.tools.semantic_scholar.get_citations", return_value=[]):
                pool = kg.expand_one_hop(seeds)
        assert pool == seeds

    def test_adds_refs_and_citations(self):
        seeds = [{"paperId": "A", "title": "seed-A"}]
        refs = [{"paperId": "R1"}, {"paperId": "R2"}]
        cits = [{"paperId": "C1"}]
        with patch("darwinian.tools.semantic_scholar.get_references", return_value=refs):
            with patch("darwinian.tools.semantic_scholar.get_citations", return_value=cits):
                pool = kg.expand_one_hop(seeds)
        ids = [p.get("paperId") for p in pool]
        assert "A" in ids
        assert "R1" in ids and "R2" in ids
        assert "C1" in ids

    def test_skips_seeds_without_paper_id(self):
        seeds = [{"title": "no_id"}, {"paperId": "B"}]
        with patch("darwinian.tools.semantic_scholar.get_references", return_value=[]) as mock_refs:
            with patch("darwinian.tools.semantic_scholar.get_citations", return_value=[]):
                kg.expand_one_hop(seeds)
        # 只对有 paperId 的种子调 API
        assert mock_refs.call_count == 1


# ---------------------------------------------------------------------------
# filter_and_rank
# ---------------------------------------------------------------------------

class TestFilterAndRank:
    def test_dedupe_by_paper_id(self):
        candidates = [
            {"paperId": "A", "abstract": "x" * 200, "citationCount": 10},
            {"paperId": "A", "abstract": "y" * 200, "citationCount": 5},  # 重复
            {"paperId": "B", "abstract": "z" * 200, "citationCount": 3},
        ]
        out = kg.filter_and_rank(candidates)
        ids = [p["paperId"] for p in out]
        assert ids.count("A") == 1
        assert "B" in ids

    def test_filters_short_abstract(self):
        candidates = [
            {"paperId": "A", "abstract": "short", "citationCount": 100},  # < 100
            {"paperId": "B", "abstract": "x" * 150, "citationCount": 1},
        ]
        out = kg.filter_and_rank(candidates)
        assert len(out) == 1
        assert out[0]["paperId"] == "B"

    def test_filters_none_abstract(self):
        candidates = [
            {"paperId": "A", "abstract": None, "citationCount": 100},
            {"paperId": "B", "abstract": "x" * 150, "citationCount": 1},
        ]
        out = kg.filter_and_rank(candidates)
        assert len(out) == 1 and out[0]["paperId"] == "B"

    def test_sorts_by_citation_desc(self):
        candidates = [
            {"paperId": "low", "abstract": "x" * 200, "citationCount": 1},
            {"paperId": "high", "abstract": "x" * 200, "citationCount": 100},
            {"paperId": "mid", "abstract": "x" * 200, "citationCount": 10},
        ]
        out = kg.filter_and_rank(candidates)
        assert [p["paperId"] for p in out] == ["high", "mid", "low"]

    def test_top_k_cutoff(self):
        candidates = [
            {"paperId": f"p{i}", "abstract": "x" * 200, "citationCount": i}
            for i in range(100)
        ]
        out = kg.filter_and_rank(candidates, top_k=10)
        assert len(out) == 10
        assert out[0]["paperId"] == "p99"  # 最大

    def test_missing_paper_id_dropped(self):
        candidates = [
            {"abstract": "x" * 200, "citationCount": 50},  # 无 id
            {"paperId": "B", "abstract": "x" * 200, "citationCount": 1},
        ]
        out = kg.filter_and_rank(candidates)
        assert len(out) == 1 and out[0]["paperId"] == "B"

    def test_missing_citation_count_treated_as_zero(self):
        candidates = [
            {"paperId": "A", "abstract": "x" * 200},  # 无 citationCount
            {"paperId": "B", "abstract": "x" * 200, "citationCount": 5},
        ]
        out = kg.filter_and_rank(candidates)
        assert out[0]["paperId"] == "B"  # B 排在前

    def test_non_dict_input_ignored(self):
        # 防御：S2 偶尔返回非 dict 的异常项
        candidates = [
            None,
            "not_a_dict",
            {"paperId": "A", "abstract": "x" * 200, "citationCount": 1},
        ]
        out = kg.filter_and_rank(candidates)
        assert len(out) == 1
