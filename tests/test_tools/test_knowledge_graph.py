"""
knowledge_graph 模块测试

不打真实 S2 API / LLM：mock semantic_scholar 和注入 fake LLM。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# _normalize + _word_boundary_contains
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lower_and_trim(self):
        assert kg._normalize("Adam Optimizer ") == "adam optimizer"

    def test_punctuation_stripped(self):
        assert kg._normalize("self-attention") == "self attention"

    def test_underscore_stripped(self):
        assert kg._normalize("multi_head_attention") == "multi head attention"

    def test_empty_returns_empty(self):
        assert kg._normalize("") == ""
        assert kg._normalize(None) == ""


class TestWordBoundaryContains:
    def test_adam_in_adam_optimizer(self):
        assert kg._word_boundary_contains("adam", "adam optimizer") is True

    def test_bert_not_in_bertopic(self):
        assert kg._word_boundary_contains("bert", "bertopic") is False

    def test_gan_not_in_organ(self):
        assert kg._word_boundary_contains("gan", "organ segmentation") is False

    def test_identical_returns_false(self):
        # 相同字符串不算 contain（避免自己合并自己）
        assert kg._word_boundary_contains("adam", "adam") is False

    def test_empty_strings(self):
        assert kg._word_boundary_contains("", "adam") is False
        assert kg._word_boundary_contains("adam", "") is False


# ---------------------------------------------------------------------------
# _limitation_id
# ---------------------------------------------------------------------------

class TestLimitationId:
    def test_stable_hash(self):
        a = kg._limitation_id("收敛慢", "p1")
        b = kg._limitation_id("收敛慢", "p1")
        assert a == b and len(a) == 8

    def test_different_paper_different_id(self):
        a = kg._limitation_id("收敛慢", "p1")
        b = kg._limitation_id("收敛慢", "p2")
        assert a != b


# ---------------------------------------------------------------------------
# batch_extract_entities（用 fake LLM）
# ---------------------------------------------------------------------------

class TestBatchExtractEntities:
    def _fake_llm_returning(self, json_str):
        """构造一个 mock llm，invoke 返回带 content 的 object"""
        fake = MagicMock()
        fake.invoke = MagicMock(return_value=MagicMock(content=json_str))
        return fake

    def test_single_batch_parsed(self):
        llm = self._fake_llm_returning('{"papers":[{"paper_id":"p1","method":["adam"],"dataset":["imagenet"],"metric":["top-1"],"task_type":"classification","limitations":["slow"]}]}')
        papers = [{"paperId": "p1", "title": "t", "abstract": "x" * 200}]
        out = kg.batch_extract_entities(papers, llm, batch_size=8)
        assert len(out) == 1
        assert out[0]["paper_id"] == "p1"
        assert out[0]["method"] == ["adam"]

    def test_multiple_batches(self):
        llm = self._fake_llm_returning('{"papers":[{"paper_id":"x","method":[],"dataset":[],"metric":[],"task_type":"other","limitations":[]}]}')
        papers = [{"paperId": f"p{i}"} for i in range(20)]
        kg.batch_extract_entities(papers, llm, batch_size=8)
        # 20 篇 / 8 = 3 批
        assert llm.invoke.call_count == 3

    def test_malformed_json_skipped(self):
        llm = self._fake_llm_returning("not json at all")
        papers = [{"paperId": "p1", "abstract": "x" * 200}]
        out = kg.batch_extract_entities(papers, llm)
        assert out == []

    def test_entries_without_paper_id_dropped(self):
        llm = self._fake_llm_returning('{"papers":[{"method":["x"]},{"paper_id":"p2","method":["y"]}]}')
        papers = [{"paperId": "p2"}]
        out = kg.batch_extract_entities(papers, llm)
        assert len(out) == 1
        assert out[0]["paper_id"] == "p2"


# ---------------------------------------------------------------------------
# canonicalize_merge
# ---------------------------------------------------------------------------

class TestCanonicalizeMerge:
    def _papers(self):
        return [
            {"paperId": "p1", "title": "T1", "abstract": "A1", "year": 2023, "citationCount": 10},
            {"paperId": "p2", "title": "T2", "abstract": "A2", "year": 2024, "citationCount": 5},
        ]

    def test_exact_duplicate_merged(self):
        raw = [
            {"paper_id": "p1", "method": ["Adam"], "dataset": [], "metric": [], "task_type": "classification", "limitations": []},
            {"paper_id": "p2", "method": ["adam"], "dataset": [], "metric": [], "task_type": "classification", "limitations": []},
        ]
        entities, _, _ = kg.canonicalize_merge(self._papers(), raw)
        adam_es = [e for e in entities if e.canonical_name == "adam" and e.type == "method"]
        assert len(adam_es) == 1
        assert sorted(adam_es[0].paper_ids) == ["p1", "p2"]

    def test_word_boundary_containment_merged(self):
        raw = [
            {"paper_id": "p1", "method": ["adam"], "dataset": [], "metric": [], "task_type": "o", "limitations": []},
            {"paper_id": "p2", "method": ["adam optimizer"], "dataset": [], "metric": [], "task_type": "o", "limitations": []},
        ]
        entities, _, _ = kg.canonicalize_merge(self._papers(), raw)
        methods = [e for e in entities if e.type == "method"]
        assert len(methods) == 1
        # canonical 保留较长的（是长串的 canonical_name）
        assert methods[0].canonical_name == "adam optimizer"
        assert "adam" in methods[0].aliases
        assert set(methods[0].paper_ids) == {"p1", "p2"}

    def test_bertopic_not_merged_with_bert(self):
        raw = [
            {"paper_id": "p1", "method": ["bert"], "dataset": [], "metric": [], "task_type": "o", "limitations": []},
            {"paper_id": "p2", "method": ["bertopic"], "dataset": [], "metric": [], "task_type": "o", "limitations": []},
        ]
        entities, _, _ = kg.canonicalize_merge(self._papers(), raw)
        methods = [e for e in entities if e.type == "method"]
        # 不应合并
        canonicals = {e.canonical_name for e in methods}
        assert "bert" in canonicals and "bertopic" in canonicals

    def test_different_types_not_merged(self):
        # 同名字但不同 type 不应合并
        raw = [
            {"paper_id": "p1", "method": ["transformer"], "dataset": ["transformer"], "metric": [], "task_type": "o", "limitations": []},
        ]
        entities, _, _ = kg.canonicalize_merge(self._papers(), raw)
        transformers = [e for e in entities if e.canonical_name == "transformer"]
        assert len(transformers) == 2
        assert {e.type for e in transformers} == {"method", "dataset"}

    def test_limitations_extracted_with_stable_id(self):
        raw = [
            {"paper_id": "p1", "method": [], "dataset": [], "metric": [], "task_type": "o", "limitations": ["收敛慢"]},
        ]
        _, limitations, _ = kg.canonicalize_merge(self._papers(), raw)
        assert len(limitations) == 1
        assert limitations[0].text == "收敛慢"
        assert limitations[0].source_paper_id == "p1"
        assert len(limitations[0].id) == 8

    def test_duplicate_limitation_deduped(self):
        raw = [
            {"paper_id": "p1", "method": [], "dataset": [], "metric": [], "task_type": "o", "limitations": ["a", "a"]},
        ]
        _, limitations, _ = kg.canonicalize_merge(self._papers(), raw)
        assert len(limitations) == 1

    def test_paper_info_assembled(self):
        raw = [
            {"paper_id": "p1", "method": [], "dataset": [], "metric": [], "task_type": "classification", "limitations": []},
        ]
        _, _, paper_infos = kg.canonicalize_merge(self._papers(), raw)
        p1 = next(p for p in paper_infos if p.paper_id == "p1")
        assert p1.title == "T1"
        assert p1.task_type == "classification"
        assert p1.citation_count == 10

    def test_empty_extractions_no_entities(self):
        entities, lims, paper_infos = kg.canonicalize_merge(self._papers(), [])
        assert entities == []
        assert lims == []
        # paper_infos 依然基于输入 papers 构造
        assert len(paper_infos) == 2

    def test_invalid_type_skipped(self):
        # 防御：LLM 可能乱写 task_type 导致某些条目，但 _add 只接受 VALID_TYPES
        # 这里间接测试：task_type 写入 Entity.type="task_type"，值作为 canonical_name
        raw = [
            {"paper_id": "p1", "method": [], "dataset": [], "metric": [], "task_type": "my_weird_type", "limitations": []},
        ]
        entities, _, _ = kg.canonicalize_merge(self._papers(), raw)
        task_entities = [e for e in entities if e.type == "task_type"]
        assert len(task_entities) == 1
        assert task_entities[0].canonical_name == "my weird type"
