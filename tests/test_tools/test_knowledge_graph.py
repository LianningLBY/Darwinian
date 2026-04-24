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


class TestIsValidLimitation:
    """过滤 LLM '抽不到' 占位语"""

    def test_valid_content(self):
        assert kg._is_valid_limitation("Slow convergence on large batch sizes.") is True
        assert kg._is_valid_limitation("长序列推理时内存消耗爆炸式增长。") is True

    def test_too_short(self):
        assert kg._is_valid_limitation("too slow") is False
        assert kg._is_valid_limitation("") is False
        assert kg._is_valid_limitation("x" * 10) is False

    def test_not_string(self):
        assert kg._is_valid_limitation(None) is False
        assert kg._is_valid_limitation(123) is False

    def test_placeholder_phrases(self):
        cases = [
            "The abstract does not explicitly state limitations.",
            "does not state any limitations",
            "No explicit limitation is mentioned.",
            "No limitations are mentioned in the abstract.",
            "Not explicitly discussed in the abstract.",
            "The abstract doesn't mention limitations clearly.",
            "N/A",
            "None",
            "unknown limitations of the method",
            "not available in abstract",
        ]
        for text in cases:
            assert kg._is_valid_limitation(text) is False, f"应视为无效：{text!r}"

    def test_case_insensitive(self):
        assert kg._is_valid_limitation("THE ABSTRACT DOES NOT EXPLICITLY STATE LIMITATIONS") is False
        assert kg._is_valid_limitation("No Limitations Mentioned") is False


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
            {"paper_id": "p1", "method": [], "dataset": [], "metric": [], "task_type": "o",
             "limitations": ["Convergence is slow on large batch sizes."]},
        ]
        _, limitations, _ = kg.canonicalize_merge(self._papers(), raw)
        assert len(limitations) == 1
        assert limitations[0].text == "Convergence is slow on large batch sizes."
        assert limitations[0].source_paper_id == "p1"
        assert len(limitations[0].id) == 8

    def test_duplicate_limitation_deduped(self):
        raw = [
            {"paper_id": "p1", "method": [], "dataset": [], "metric": [], "task_type": "o",
             "limitations": [
                 "Convergence is slow on large batch sizes.",
                 "Convergence is slow on large batch sizes.",
             ]},
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

    def test_task_type_not_in_entities(self):
        """task_type 不应作为 Entity 进入实体表（只写入 PaperInfo.task_type）"""
        raw = [
            {"paper_id": "p1", "method": ["adam"], "dataset": [], "metric": [],
             "task_type": "classification", "limitations": []},
        ]
        entities, _, paper_infos = kg.canonicalize_merge(self._papers(), raw)
        # entities 里只有 method，没有 task_type 类型
        assert all(e.type in {"method", "dataset", "metric"} for e in entities)
        assert all(e.type != "task_type" for e in entities)
        # 但 paper_infos 里该 paper 的 task_type 填上了
        p1 = next(p for p in paper_infos if p.paper_id == "p1")
        assert p1.task_type == "classification"

    def test_invalid_limitation_filtered(self):
        """LLM 说"没抽到"的占位语不应进 limitations 列表"""
        raw = [
            {"paper_id": "p1", "method": [], "dataset": [], "metric": [], "task_type": "o",
             "limitations": [
                 "The abstract does not explicitly state limitations.",
                 "does not state any limitations",
                 "N/A",
                 "None",
                 "Slow convergence on large batch sizes.",   # 这一条是真的
             ]},
        ]
        _, lims, _ = kg.canonicalize_merge(self._papers(), raw)
        texts = [l.text for l in lims]
        assert len(lims) == 1
        assert texts[0] == "Slow convergence on large batch sizes."

    def test_short_limitation_filtered(self):
        """过短的 limitation（< 15 字符）过滤"""
        raw = [
            {"paper_id": "p1", "method": [], "dataset": [], "metric": [], "task_type": "o",
             "limitations": ["too slow", "This is a reasonably detailed limitation statement."]},
        ]
        _, lims, _ = kg.canonicalize_merge(self._papers(), raw)
        assert len(lims) == 1
        assert "detailed" in lims[0].text


# ---------------------------------------------------------------------------
# rank_relevance_top_k
# ---------------------------------------------------------------------------

class TestRankRelevance:
    def _make(self, name, paper_ids=None, etype="method"):
        from darwinian.state import Entity
        return Entity(canonical_name=name, type=etype, paper_ids=paper_ids or ["p1"])

    def test_empty_returns_empty(self):
        assert kg.rank_relevance_top_k([], "some problem") == []

    def test_top_k_cutoff(self):
        entities = [self._make(f"entity_{i}") for i in range(100)]
        out = kg.rank_relevance_top_k(entities, "test", top_by_relevance=5, top_by_popularity=0)
        assert len(out) <= 5

    def test_popularity_fallback(self):
        # 热门实体应该通过 popularity 路径保留，即使与 core_problem 文本不相关
        popular = self._make("xyzzy_irrelevant_word", paper_ids=[f"p{i}" for i in range(50)])
        unpopular = self._make("quantum_compute", paper_ids=["p_only"])
        entities = [popular, unpopular]
        out = kg.rank_relevance_top_k(entities, "machine learning",
                                       top_by_relevance=1, top_by_popularity=1)
        names = {e.canonical_name for e in out}
        assert "xyzzy_irrelevant_word" in names  # popularity 路径兜底

    def test_dedupe_when_both_paths_pick_same(self):
        popular = self._make("adam", paper_ids=[f"p{i}" for i in range(30)])
        entities = [popular]
        out = kg.rank_relevance_top_k(entities, "adam momentum",
                                       top_by_relevance=1, top_by_popularity=1)
        assert len(out) == 1

    def test_different_types_not_deduped(self):
        from darwinian.state import Entity
        m = Entity(canonical_name="transformer", type="method", paper_ids=["p1", "p2", "p3"])
        d = Entity(canonical_name="transformer", type="dataset", paper_ids=["p1", "p2", "p3"])
        out = kg.rank_relevance_top_k([m, d], "transformer", 10, 10)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# find_novel_pairs
# ---------------------------------------------------------------------------

class TestFindNovelPairs:
    def _make(self, name, paper_ids):
        from darwinian.state import Entity
        return Entity(canonical_name=name, type="method", paper_ids=paper_ids)

    def test_empty_returns_empty(self):
        assert kg.find_novel_pairs([]) == []

    def test_both_mature_no_co_occurrence_pair_found(self):
        a = self._make("mamba", ["p1", "p2", "p3"])
        b = self._make("flash_attention", ["p4", "p5", "p6"])
        pairs = kg.find_novel_pairs([a, b], min_papers_each=3)
        assert len(pairs) == 1
        assert {pairs[0].entity_a, pairs[0].entity_b} == {"mamba", "flash_attention"}
        # score = min(3, 3) = 3
        assert pairs[0].score == 3

    def test_co_occurring_pair_excluded(self):
        a = self._make("adam", ["p1", "p2", "p3"])
        b = self._make("resnet", ["p3", "p4", "p5"])  # 共现于 p3
        pairs = kg.find_novel_pairs([a, b], min_papers_each=3)
        assert pairs == []

    def test_below_min_papers_excluded(self):
        a = self._make("a", ["p1"])   # 仅 1 篇
        b = self._make("b", ["p2", "p3", "p4"])
        pairs = kg.find_novel_pairs([a, b], min_papers_each=3)
        assert pairs == []

    def test_score_uses_min_not_sum(self):
        a = self._make("popular", ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"])
        b = self._make("mature", ["p20", "p21", "p22"])
        pairs = kg.find_novel_pairs([a, b], min_papers_each=3)
        assert pairs[0].score == 3  # min(10, 3)，不是 sum

    def test_top_sorted_by_score(self):
        a = self._make("a", [f"x{i}" for i in range(10)])
        b = self._make("b", [f"y{i}" for i in range(5)])
        c = self._make("c", [f"z{i}" for i in range(3)])
        pairs = kg.find_novel_pairs([a, b, c], max_pairs=10, min_papers_each=3)
        # a-b: min=5, a-c: min=3, b-c: min=3
        assert pairs[0].score == 5  # a-b 第一


# ---------------------------------------------------------------------------
# is_graph_sufficient
# ---------------------------------------------------------------------------

class TestIsGraphSufficient:
    def test_meets_both_thresholds(self):
        from darwinian.state import Entity, PaperInfo
        entities = [Entity(canonical_name=f"e{i}", type="method", paper_ids=["p1"])
                    for i in range(20)]
        papers = [PaperInfo(paper_id=f"p{i}") for i in range(10)]
        assert kg.is_graph_sufficient(entities, papers) is True

    def test_fails_entities_threshold(self):
        from darwinian.state import Entity, PaperInfo
        entities = [Entity(canonical_name="e", type="method", paper_ids=["p1"])]
        papers = [PaperInfo(paper_id=f"p{i}") for i in range(10)]
        assert kg.is_graph_sufficient(entities, papers) is False

    def test_fails_papers_threshold(self):
        from darwinian.state import Entity, PaperInfo
        entities = [Entity(canonical_name=f"e{i}", type="method", paper_ids=["p1"])
                    for i in range(20)]
        papers = [PaperInfo(paper_id="p1")]
        assert kg.is_graph_sufficient(entities, papers) is False


# ---------------------------------------------------------------------------
# build_concept_graph orchestration（完全 mock）
# ---------------------------------------------------------------------------

class TestBuildConceptGraph:
    def test_pipeline_end_to_end_mocked(self):
        """不打真实 API/LLM，mock 掉 step 1/2/4，验证整条管道能拼出 ConceptGraph"""
        fake_seeds = [{"paperId": "p1", "title": "T1", "abstract": "x" * 200, "citationCount": 50}]
        fake_candidates_after_hop = fake_seeds  # expand_one_hop 返回原样
        fake_llm = MagicMock()
        fake_llm.invoke = MagicMock(return_value=MagicMock(content=(
            '{"papers":[{"paper_id":"p1","method":["adam"],"dataset":["imagenet"],'
            '"metric":[],"task_type":"classification","limitations":["slow convergence"]}]}'
        )))

        with patch("darwinian.tools.semantic_scholar.search_papers_two_tiered",
                   return_value=fake_seeds):
            with patch("darwinian.utils.knowledge_graph.expand_one_hop",
                       return_value=fake_candidates_after_hop):
                graph = kg.build_concept_graph(
                    research_direction="test",
                    core_problem="how to train fast",
                    llm=fake_llm,
                )

        assert len(graph.papers) == 1 and graph.papers[0].paper_id == "p1"
        assert any(e.canonical_name == "adam" for e in graph.entities)
        assert len(graph.limitations) == 1
        # 数据太少：is_sufficient 应为 False
        assert graph.is_sufficient is False

    def test_arxiv_backend_skips_s2_and_expansion(self):
        """backend=arxiv 时应走 arxiv 源，完全绕过 S2 + expand_one_hop"""
        fake_arxiv_papers = [
            {"paperId": "arxiv:2401.00001", "title": "T", "abstract": "x" * 200,
             "year": 2024, "citationCount": 0, "source": "arxiv"},
        ]
        fake_llm = MagicMock()
        fake_llm.invoke = MagicMock(return_value=MagicMock(content=(
            '{"papers":[{"paper_id":"arxiv:2401.00001","method":["adam"],'
            '"dataset":[],"metric":[],"task_type":"classification","limitations":[]}]}'
        )))

        with patch("darwinian.tools.arxiv_search.search_papers_arxiv_two_tiered",
                   return_value=fake_arxiv_papers) as mock_arxiv:
            with patch("darwinian.tools.semantic_scholar.search_papers_two_tiered") as mock_s2:
                with patch("darwinian.utils.knowledge_graph.expand_one_hop") as mock_expand:
                    graph = kg.build_concept_graph(
                        research_direction="test",
                        core_problem="test",
                        llm=fake_llm,
                        backend="arxiv",
                    )
        # arxiv 源被调用，S2 + expand 都没调用
        assert mock_arxiv.call_count == 1
        assert mock_s2.call_count == 0
        assert mock_expand.call_count == 0
        assert graph.papers[0].paper_id == "arxiv:2401.00001"

    def test_env_var_sets_backend(self, monkeypatch):
        """DARWINIAN_SEARCH_BACKEND=arxiv 应该触发 arxiv 分支"""
        monkeypatch.setenv("DARWINIAN_SEARCH_BACKEND", "arxiv")
        fake_llm = MagicMock()
        fake_llm.invoke = MagicMock(return_value=MagicMock(
            content='{"papers":[]}'
        ))
        with patch("darwinian.tools.arxiv_search.search_papers_arxiv_two_tiered",
                   return_value=[]) as mock_arxiv:
            with patch("darwinian.tools.semantic_scholar.search_papers_two_tiered") as mock_s2:
                kg.build_concept_graph("x", "x", llm=fake_llm)  # 不传 backend
        assert mock_arxiv.call_count == 1
        assert mock_s2.call_count == 0
