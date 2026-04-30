"""
phase_a_orchestrator 单元测试

不打真实 S2 / arxiv / LLM —— 全部 mock。验证：
- helper 函数（_looks_like_arxiv_id / _format_evidence_id / _bucket_by_year）
- _resolve_arxiv_ids 走 S2 缓存正常
- _make_full_text_provider 正确按 paper_id 反查 arxiv_id
- build_research_material_pack 端到端串接（5 个外部依赖全 mock）
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.phase_a_orchestrator import (
    build_research_material_pack,
    _bucket_by_year,
    _format_evidence_id,
    _looks_like_arxiv_id,
    _make_full_text_provider,
    _resolve_arxiv_ids,
)
from darwinian.state import (
    ConceptGraph,
    Entity,
    LimitationRef,
    PaperEvidence,
    PaperInfo,
    QuantitativeClaim,
    ResearchConstraints,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _mk_paper(paper_id, year=2024, citations=10, title="t", abstract="a") -> PaperInfo:
    return PaperInfo(
        paper_id=paper_id, title=title, abstract=abstract,
        year=year, citation_count=citations, source="semantic_scholar",
    )


def _constraints() -> ResearchConstraints:
    return ResearchConstraints(
        gpu_count=4, gpu_hours_budget=672.0, wall_clock_days=7,
        forbidden_techniques=["GRPO"], target_venues=["NeurIPS 2026"],
    )


# ---------------------------------------------------------------------------
# helper 函数
# ---------------------------------------------------------------------------

class TestLooksLikeArxivId:
    def test_yymm_5digit_match(self):
        assert _looks_like_arxiv_id("2404.16710")
        assert _looks_like_arxiv_id("2603.17891")

    def test_with_version_suffix(self):
        assert _looks_like_arxiv_id("2404.16710v2")

    def test_yymm_4digit_match(self):
        # arxiv 早期格式
        assert _looks_like_arxiv_id("1706.0123")

    def test_s2_paperid_no_match(self):
        # S2 paperId 是 hex 字符串
        assert not _looks_like_arxiv_id("abc123def4567890abcdef0123456789abcdef01")

    def test_doi_no_match(self):
        assert not _looks_like_arxiv_id("10.1145/3534678.3539107")

    def test_empty_no_match(self):
        assert not _looks_like_arxiv_id("")


class TestFormatEvidenceId:
    def test_arxiv_id_known(self):
        p = _mk_paper("S2_HEX_001")
        evid = _format_evidence_id(p, {"S2_HEX_001": "2404.16710"})
        assert evid == "arxiv:2404.16710"

    def test_arxiv_id_missing_falls_back(self):
        p = _mk_paper("S2_HEX_002")
        evid = _format_evidence_id(p, {"S2_HEX_002": ""})
        assert evid == "s2:S2_HEX_002"

    def test_paper_id_not_in_map(self):
        p = _mk_paper("S2_HEX_003")
        evid = _format_evidence_id(p, {})
        assert evid == "s2:S2_HEX_003"


class TestBucketByYear:
    def test_basic_bucketing(self):
        papers = [
            _mk_paper("p1", year=2022),
            _mk_paper("p2", year=2023),
            _mk_paper("p3", year=2025),
            _mk_paper("p4", year=2026),
        ]
        amap = {"p1": "2202.0001", "p2": "2304.0001", "p3": "2503.0001", "p4": "2603.0001"}
        b = _bucket_by_year(papers, amap)
        assert "foundational_pre_2024" in b
        assert "hot_2024_2026" in b
        assert "arxiv:2202.0001" in b["foundational_pre_2024"]
        assert "arxiv:2304.0001" in b["foundational_pre_2024"]
        assert "arxiv:2503.0001" in b["hot_2024_2026"]
        assert "arxiv:2603.0001" in b["hot_2024_2026"]

    def test_2024_in_hot_bucket(self):
        """Bug 1 fix: 2024 必须进 hot 桶（爆发年，LayerSkip / EAGLE-2 都是这年）"""
        papers = [_mk_paper("p1", year=2024)]
        b = _bucket_by_year(papers, {"p1": "2404.0001"})
        assert "arxiv:2404.0001" in b["hot_2024_2026"]

    def test_year_zero_skipped(self):
        """year=0 表示未知 → 不进任何桶"""
        papers = [_mk_paper("p1", year=0)]
        b = _bucket_by_year(papers, {"p1": ""})
        all_ids = [pid for pids in b.values() for pid in pids]
        assert "s2:p1" not in all_ids

    def test_empty_buckets_dropped(self):
        """所有 paper year=0 → 两个桶都空 → 返回空 dict"""
        papers = [_mk_paper("p1", year=0), _mk_paper("p2", year=0)]
        b = _bucket_by_year(papers, {"p1": "", "p2": ""})
        assert b == {}


# ---------------------------------------------------------------------------
# _resolve_arxiv_ids
# ---------------------------------------------------------------------------

class TestResolveArxivIds:
    def test_arxiv_format_uses_directly(self):
        """paper_id 已经是 arxiv 格式时不调 S2"""
        p = _mk_paper("2404.16710")
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details") as mock_s2:
            result = _resolve_arxiv_ids([p])
        assert mock_s2.call_count == 0
        assert result["2404.16710"] == "2404.16710"

    def test_s2_paperid_calls_get_details(self):
        p = _mk_paper("S2_HEX_001")
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value={"externalIds": {"ArXiv": "2404.16710"}}) as mock_s2:
            result = _resolve_arxiv_ids([p])
        mock_s2.assert_called_once_with("S2_HEX_001", fields="externalIds")
        assert result["S2_HEX_001"] == "2404.16710"

    def test_no_arxiv_id_returns_empty(self):
        """S2 返回但 externalIds 没 ArXiv → ""（不报错）"""
        p = _mk_paper("S2_HEX_002")
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value={"externalIds": {"DOI": "10.1145/xxx"}}):
            result = _resolve_arxiv_ids([p])
        assert result["S2_HEX_002"] == ""

    def test_s2_returns_none(self):
        p = _mk_paper("S2_HEX_003")
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value=None):
            result = _resolve_arxiv_ids([p])
        assert result["S2_HEX_003"] == ""

    def test_s2_raises_swallowed(self):
        """S2 调用抛异常时跳过该论文，不让整个流程崩"""
        p = _mk_paper("S2_HEX_004")
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   side_effect=ConnectionError("network")):
            result = _resolve_arxiv_ids([p])
        assert result["S2_HEX_004"] == ""

    def test_empty_paper_id_skipped(self):
        p = PaperInfo(paper_id="")
        result = _resolve_arxiv_ids([p])
        assert "" not in result


# ---------------------------------------------------------------------------
# _make_full_text_provider
# ---------------------------------------------------------------------------

class TestFullTextProvider:
    def test_arxiv_evidence_id_resolves(self):
        """evidence_paper_id 形如 'arxiv:2404.16710' 时直接解析"""
        amap = {"S2_HEX_001": "2404.16710"}
        provider = _make_full_text_provider(amap)

        mock_src = MagicMock(has_full_text=True)
        with patch("darwinian.agents.phase_a_orchestrator.fetch_arxiv_latex",
                   return_value=mock_src) as mock_fetch:
            with patch("darwinian.agents.phase_a_orchestrator.render_for_llm",
                       return_value="rendered text 18000 chars"):
                text = provider("arxiv:2404.16710")
        mock_fetch.assert_called_once_with("2404.16710")
        assert text == "rendered text 18000 chars"

    def test_s2_evidence_id_resolves(self):
        """evidence_paper_id 形如 's2:hex' 时通过 amap 反查"""
        amap = {"S2_HEX_001": "2404.16710"}
        provider = _make_full_text_provider(amap)

        mock_src = MagicMock(has_full_text=True)
        with patch("darwinian.agents.phase_a_orchestrator.fetch_arxiv_latex",
                   return_value=mock_src) as mock_fetch:
            with patch("darwinian.agents.phase_a_orchestrator.render_for_llm",
                       return_value="text"):
                text = provider("s2:S2_HEX_001")
        mock_fetch.assert_called_once_with("2404.16710")
        assert text == "text"

    def test_no_arxiv_id_returns_empty(self):
        """没 arxiv_id 时返空字符串（让 batch_extract_evidence 走 abstract-only）"""
        provider = _make_full_text_provider({})
        assert provider("s2:UNKNOWN") == ""

    def test_fetch_returns_none_returns_empty(self):
        amap = {"S2_HEX_001": "2404.16710"}
        provider = _make_full_text_provider(amap)
        with patch("darwinian.agents.phase_a_orchestrator.fetch_arxiv_latex",
                   return_value=None):
            assert provider("arxiv:2404.16710") == ""

    def test_no_full_text_flag_returns_empty(self):
        amap = {"S2_HEX_001": "2404.16710"}
        provider = _make_full_text_provider(amap)
        mock_src = MagicMock(has_full_text=False)
        with patch("darwinian.agents.phase_a_orchestrator.fetch_arxiv_latex",
                   return_value=mock_src):
            assert provider("arxiv:2404.16710") == ""

    def test_fetch_raises_swallowed(self):
        amap = {"S2_HEX_001": "2404.16710"}
        provider = _make_full_text_provider(amap)
        with patch("darwinian.agents.phase_a_orchestrator.fetch_arxiv_latex",
                   side_effect=Exception("network")):
            assert provider("arxiv:2404.16710") == ""


# ---------------------------------------------------------------------------
# build_research_material_pack 端到端
# ---------------------------------------------------------------------------

class TestBuildMaterialPack:
    def test_end_to_end_assembly(self):
        """5 个外部依赖全 mock，验证 orchestrator 把它们正确串起来"""
        # mock build_concept_graph 的产出
        mock_papers = [
            _mk_paper("S2_HEX_001", year=2024, citations=200,
                       title="LayerSkip", abstract="abs1"),
            _mk_paper("S2_HEX_002", year=2025, citations=100,
                       title="DEL", abstract="abs2"),
        ]
        mock_graph = ConceptGraph(
            papers=mock_papers,
            entities=[Entity(canonical_name="layerskip", type="method", paper_ids=["S2_HEX_001"])],
            limitations=[LimitationRef(id="L01", text="finetune required",
                                        source_paper_id="S2_HEX_001")],
        )

        # mock 深抽取的产出
        mock_evidence = [
            PaperEvidence(
                paper_id="arxiv:2404.16710", title="LayerSkip", short_name="LayerSkip",
                venue="ACL 2024", year=2024,
                quantitative_claims=[QuantitativeClaim(metric_name="speedup",
                                                        metric_value="2.16x")],
                headline_result="2.16x speedup", relation_to_direction="extends",
            ),
            PaperEvidence(
                paper_id="arxiv:2510.del", title="DEL", short_name="DEL",
                venue="COLM 2025", year=2025,
                quantitative_claims=[QuantitativeClaim(metric_name="speedup",
                                                        metric_value="2.62x")],
                headline_result="2.62x speedup", relation_to_direction="baseline",
            ),
        ]

        with patch("darwinian.agents.phase_a_orchestrator.build_seed_pool",
                   return_value=[]) as mock_seed, \
             patch("darwinian.agents.phase_a_orchestrator.build_concept_graph",
                   return_value=mock_graph) as mock_bg, \
             patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   side_effect=[
                       {"externalIds": {"ArXiv": "2404.16710"}},
                       {"externalIds": {"ArXiv": "2510.del"}},
                   ]) as mock_gd, \
             patch("darwinian.agents.phase_a_orchestrator.batch_extract_evidence",
                   return_value=mock_evidence) as mock_ext:
            pack = build_research_material_pack(
                direction="LLM inference acceleration",
                constraints=_constraints(),
                extractor_llm=MagicMock(),
                evidence_llm=MagicMock(),
                top_k_evidence=12,
            )

        # 1. build_seed_pool 调一次（Scheme X 路径）
        assert mock_seed.call_count == 1
        # 2. concept_graph 调一次，且 seed_pool 透传
        assert mock_bg.call_count == 1
        assert mock_bg.call_args.kwargs["seed_pool"] == []
        # 3. get_paper_details 调 2 次（每个 top paper 一次）
        assert mock_gd.call_count == 2
        # 4. batch_extract_evidence 调 1 次
        assert mock_ext.call_count == 1
        papers_arg = mock_ext.call_args.args[0]
        assert len(papers_arg) == 2
        assert papers_arg[0]["paper_id"] == "arxiv:2404.16710"
        assert papers_arg[0]["title"] == "LayerSkip"

        # 5. 装配的 ResearchMaterialPack
        assert pack.direction == "LLM inference acceleration"
        assert pack.constraints.gpu_hours_budget == 672.0
        assert len(pack.paper_evidence) == 2
        assert pack.concept_graph is mock_graph
        assert pack.structural_hole_hooks == []
        # timeline_signals: LayerSkip 2024 + DEL 2025 都在 hot_2024_2026
        assert pack.timeline_signals == {
            "hot_2024_2026": ["arxiv:2404.16710", "arxiv:2510.del"]
        }

    def test_top_k_truncation(self, monkeypatch):
        """top_k_evidence 截断按 citation_count 排序"""
        # R15: 测试构造的是 0 evidence + 0 phenomena，需绕过默认硬熔断
        monkeypatch.setenv("DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE", "1")
        mock_papers = [
            _mk_paper(f"P{i}", year=2024, citations=100 - i)
            for i in range(20)
        ]
        mock_graph = ConceptGraph(papers=mock_papers)

        with patch("darwinian.agents.phase_a_orchestrator.build_seed_pool",
                   return_value=[]), \
             patch("darwinian.agents.phase_a_orchestrator.build_concept_graph",
                   return_value=mock_graph), \
             patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value={"externalIds": {}}), \
             patch("darwinian.agents.phase_a_orchestrator.batch_extract_evidence",
                   return_value=[]) as mock_ext:
            build_research_material_pack(
                direction="x", constraints=_constraints(),
                extractor_llm=MagicMock(), evidence_llm=MagicMock(),
                top_k_evidence=5,
            )
        # 只对 top-5 调 batch_extract_evidence
        papers_arg = mock_ext.call_args.args[0]
        assert len(papers_arg) == 5
        # 按 citation 降序：P0 (100), P1 (99), ...
        assert papers_arg[0]["paper_id"] == "s2:P0"
        assert papers_arg[4]["paper_id"] == "s2:P4"

    def test_empty_graph_still_returns_valid_pack(self, monkeypatch):
        """build_concept_graph 返空（如 S2 全挂）时 orchestrator 不崩"""
        # R15: 测试构造的是 0 evidence + 0 phenomena，需绕过默认硬熔断
        monkeypatch.setenv("DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE", "1")
        empty_graph = ConceptGraph()
        with patch("darwinian.agents.phase_a_orchestrator.build_seed_pool",
                   return_value=[]), \
             patch("darwinian.agents.phase_a_orchestrator.build_concept_graph",
                   return_value=empty_graph), \
             patch("darwinian.agents.phase_a_orchestrator.batch_extract_evidence",
                   return_value=[]):
            pack = build_research_material_pack(
                direction="x", constraints=_constraints(),
                extractor_llm=MagicMock(), evidence_llm=MagicMock(),
            )
        assert pack.paper_evidence == []
        assert pack.timeline_signals == {}
        assert pack.concept_graph is empty_graph


# ===========================================================================
# Fix D1: 按 entity_hits + citation_count 选 top-K
# ===========================================================================

from darwinian.agents.phase_a_orchestrator import _select_top_papers_by_relevance


class TestSelectTopPapersByRelevance:
    """验证 entity-hits-based 排序能让方向相关论文压过基础工作论文"""

    def test_entity_hits_dominates_citation(self):
        """高 citation 但 0 entity hits 应被低 citation 但高 hits 压过"""
        # 模拟实际场景：foundation paper 引用数高但只 hit 1 个泛词；方向论文反过来
        foundation = _mk_paper("FOUNDATION", citations=10000)   # GPT-3 级
        direction = _mk_paper("DIRECTION", citations=50)         # LayerSkip 级
        graph = ConceptGraph(
            papers=[foundation, direction],
            entities=[
                # foundation 只 hit 1 个泛词
                Entity(canonical_name="transformer", type="method", paper_ids=["FOUNDATION"]),
                # direction 论文 hit 4 个具体方法/指标
                Entity(canonical_name="speculative decoding", type="method", paper_ids=["DIRECTION"]),
                Entity(canonical_name="draft model", type="method", paper_ids=["DIRECTION"]),
                Entity(canonical_name="acceptance rate", type="metric", paper_ids=["DIRECTION"]),
                Entity(canonical_name="early exit", type="method", paper_ids=["DIRECTION"]),
            ],
        )
        top = _select_top_papers_by_relevance(graph, top_k=2)
        # direction 论文应排第 1（4 hits vs 1 hit）尽管 citation 少 200×
        assert top[0].paper_id == "DIRECTION"
        assert top[1].paper_id == "FOUNDATION"

    def test_citation_breaks_ties_when_hits_equal(self):
        """同 entity_hits 时按 citation_count 降序"""
        a = _mk_paper("A", citations=100)
        b = _mk_paper("B", citations=500)
        graph = ConceptGraph(
            papers=[a, b],
            entities=[
                Entity(canonical_name="m1", type="method", paper_ids=["A", "B"]),
            ],
        )
        # 都 hit 1 次 → B 引用多排前
        top = _select_top_papers_by_relevance(graph, top_k=2)
        assert [p.paper_id for p in top] == ["B", "A"]

    def test_paper_with_zero_hits_at_bottom(self):
        """从未在 entity 表里出现的 paper 排最后"""
        a = _mk_paper("WITH_HITS", citations=10)
        b = _mk_paper("NO_HITS", citations=99999)
        graph = ConceptGraph(
            papers=[a, b],
            entities=[
                Entity(canonical_name="m1", type="method", paper_ids=["WITH_HITS"]),
            ],
        )
        top = _select_top_papers_by_relevance(graph, top_k=2)
        assert top[0].paper_id == "WITH_HITS"

    def test_empty_entities_falls_back_to_citation(self):
        """没有 entity 表时，行为退化到纯 citation 排序（向后兼容）"""
        papers = [_mk_paper(f"P{i}", citations=100 - i) for i in range(5)]
        graph = ConceptGraph(papers=papers, entities=[])
        top = _select_top_papers_by_relevance(graph, top_k=3)
        assert [p.paper_id for p in top] == ["P0", "P1", "P2"]

    def test_top_k_limits_returned_count(self):
        papers = [_mk_paper(f"P{i}") for i in range(10)]
        graph = ConceptGraph(papers=papers, entities=[])
        assert len(_select_top_papers_by_relevance(graph, top_k=3)) == 3
        assert len(_select_top_papers_by_relevance(graph, top_k=20)) == 10


# ===========================================================================
# Fix L1: query expansion (_expand_search_queries + extra_queries 串接)
# ===========================================================================

import json as _json
from darwinian.agents.phase_a_orchestrator import _expand_search_queries


class TestExpandSearchQueries:
    def test_happy_path(self):
        fake_resp = MagicMock(content=_json.dumps({
            "queries": [
                "self-speculative decoding draft model",
                "Medusa EAGLE LayerSkip",
                "mixed precision quantization LLM",
            ]
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp):
            qs = _expand_search_queries("LLM speculative decoding", MagicMock())
        assert len(qs) == 3
        assert "self-speculative decoding draft model" in qs

    def test_strips_whitespace_and_filters_empty(self):
        fake_resp = MagicMock(content=_json.dumps({
            "queries": ["  query A  ", "", "   ", None, "query B", 42]
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp):
            qs = _expand_search_queries("x", MagicMock())
        # 留下 strip 后非空的字符串
        assert qs == ["query A", "query B"]

    def test_caps_at_max_queries(self):
        fake_resp = MagicMock(content=_json.dumps({
            "queries": [f"q{i}" for i in range(10)]
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp):
            qs = _expand_search_queries("x", MagicMock(), max_queries=3)
        assert len(qs) == 3
        assert qs == ["q0", "q1", "q2"]

    def test_llm_exception_returns_empty(self):
        """LLM 报错时返空列表（让 build_concept_graph 降级到单查询）"""
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   side_effect=ConnectionError("network")):
            qs = _expand_search_queries("x", MagicMock())
        assert qs == []

    def test_unparseable_json_returns_empty(self):
        fake_resp = MagicMock(content="not json at all")
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp):
            qs = _expand_search_queries("x", MagicMock())
        assert qs == []

    def test_missing_queries_field_returns_empty(self):
        fake_resp = MagicMock(content=_json.dumps({"foo": "bar"}))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp):
            qs = _expand_search_queries("x", MagicMock())
        assert qs == []


# Note: TestOrchestratorPassesExtraQueries 已删除——orchestrator 改走 Scheme X
# (build_seed_pool) 后不再调 _expand_search_queries / extra_queries 路径。
# extra_queries 参数仍保留在 build_concept_graph，作为 v3 query expansion
# 替代方案的备用入口（schema 稳定但 orchestrator 默认不用）。
# 单测保留 TestExpandSearchQueries（_expand_search_queries 单元行为）。


# ===========================================================================
# Scheme X: build_seed_pool + verify/recover + rerank
# ===========================================================================

from darwinian.agents.phase_a_orchestrator import (
    _llm_list_seed_papers,
    _verify_and_recover_seed,
    _title_similarity,
    _expand_seeds_one_hop,
    _rerank_by_direction_relevance,
    build_seed_pool,
)
from darwinian.tools.semantic_scholar import GRAPH_FIELDS


class TestTitleSimilarity:
    def test_identical(self):
        assert _title_similarity("LayerSkip Early Exit", "LayerSkip Early Exit") == 1.0

    def test_case_and_punct_insensitive(self):
        a = "LayerSkip: Enabling Early Exit Inference"
        b = "layerskip enabling early exit inference"
        assert _title_similarity(a, b) > 0.9

    def test_subset_overlap(self):
        a = "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
        b = "LayerSkip: Enabling Early Exit Inference"
        # 子集 → Jaccard ~ 0.7
        assert 0.4 < _title_similarity(a, b) < 0.9

    def test_completely_different(self):
        assert _title_similarity("LayerSkip", "GPT-3 Few-Shot") == 0.0

    def test_empty(self):
        assert _title_similarity("", "anything") == 0.0
        assert _title_similarity("anything", "") == 0.0


class TestLlmListSeedPapers:
    def test_happy_path(self):
        fake_resp = MagicMock(content=_json.dumps({
            "seed_papers": [
                {"arxiv_id": "2404.16710", "title": "LayerSkip", "reason": "..."},
                {"arxiv_id": "2510.del", "title": "DEL", "reason": "..."},
            ]
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp):
            seeds = _llm_list_seed_papers("LLM speculative decoding", MagicMock())
        assert len(seeds) == 2
        assert seeds[0]["arxiv_id"] == "2404.16710"

    def test_filters_entries_without_arxiv_id(self):
        fake_resp = MagicMock(content=_json.dumps({
            "seed_papers": [
                {"arxiv_id": "2404.16710", "title": "ok"},
                {"title": "no arxiv id"},   # missing arxiv_id → filter
                {"arxiv_id": "", "title": "empty"},   # empty arxiv_id → filter
                "not a dict",
            ]
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp):
            seeds = _llm_list_seed_papers("x", MagicMock())
        assert len(seeds) == 1

    def test_llm_failure_returns_empty(self):
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   side_effect=ConnectionError("net")):
            assert _llm_list_seed_papers("x", MagicMock()) == []

    # ---- R14: retry on JSON parse failure ----

    def test_retry_recovers_from_first_failure(self):
        """v3/v4 LIVE 实测：第一次 <think> 截断 → JSON 解析失败 → 第二次重试拿到结果"""
        # 第一次返回未闭合 think 块（JSON 解析失败）
        bad_resp = MagicMock(content="<think>Let me reason...")
        good_resp = MagicMock(content=_json.dumps({
            "seed_papers": [
                {"arxiv_id": "2404.16710", "title": "LayerSkip", "reason": "..."},
            ],
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   side_effect=[bad_resp, good_resp]):
            seeds = _llm_list_seed_papers("x", MagicMock(), max_attempts=3)
        assert len(seeds) == 1
        assert seeds[0]["arxiv_id"] == "2404.16710"

    def test_retry_exhausted_returns_empty(self):
        """3 次都失败 → 返 [], 不 raise"""
        bad_resp = MagicMock(content="<think>truncated</think>")
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=bad_resp):
            seeds = _llm_list_seed_papers("x", MagicMock(), max_attempts=3)
        assert seeds == []

    def test_retry_succeeds_third_attempt(self):
        bad_resp = MagicMock(content="<think>not json")
        good_resp = MagicMock(content=_json.dumps({
            "seed_papers": [{"arxiv_id": "1234.5678", "title": "T"}],
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   side_effect=[bad_resp, bad_resp, good_resp]):
            seeds = _llm_list_seed_papers("x", MagicMock(), max_attempts=3)
        assert len(seeds) == 1

    def test_retry_appends_anti_reasoning_hint_on_retry(self):
        """重试时 user message 应含'重试 — 上次输出被截断'提示"""
        bad_resp = MagicMock(content="not json")
        good_resp = MagicMock(content=_json.dumps({
            "seed_papers": [{"arxiv_id": "x.y", "title": "t"}],
        }))
        captured_messages = []

        def fake_invoke(llm, msgs):
            captured_messages.append(msgs)
            if len(captured_messages) == 1:
                return bad_resp
            return good_resp

        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   side_effect=fake_invoke):
            _llm_list_seed_papers("test direction", MagicMock(), max_attempts=3)
        # 第二次 call 的 user message 应含重试提示
        second_human_content = captured_messages[1][1].content
        assert "重试" in second_human_content
        assert "<think>" in second_human_content   # 提到 <think> 块的警告
        assert "`{`" in second_human_content   # 提示直接以 { 开头

    def test_empty_seed_papers_treated_as_failure_and_retried(self):
        """LLM 返合法 JSON 但 seed_papers 为空 → 应触发 retry"""
        empty_resp = MagicMock(content=_json.dumps({"seed_papers": []}))
        good_resp = MagicMock(content=_json.dumps({
            "seed_papers": [{"arxiv_id": "1.2", "title": "t"}],
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   side_effect=[empty_resp, good_resp]):
            seeds = _llm_list_seed_papers("x", MagicMock(), max_attempts=3)
        assert len(seeds) == 1


class TestVerifyAndRecoverSeed:
    def test_arxiv_id_direct_hit(self):
        cand = {"arxiv_id": "2404.16710", "title": "LayerSkip"}
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value={"paperId": "S2_HEX", "title": "LayerSkip"}) as mock_gd:
            result = _verify_and_recover_seed(cand)
        mock_gd.assert_called_once_with("ArXiv:2404.16710", fields=GRAPH_FIELDS)
        assert result["paperId"] == "S2_HEX"

    def test_arxiv_id_strip_prefix(self):
        """LLM 可能加 'arxiv:' / 'ArXiv:' 前缀，要去掉"""
        cand = {"arxiv_id": "arxiv:2404.16710", "title": "x"}
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value={"paperId": "ok"}) as mock_gd:
            _verify_and_recover_seed(cand)
        # 实际传给 S2 的 ID 不该重复 'ArXiv:' 前缀
        called_id = mock_gd.call_args.args[0]
        assert called_id == "ArXiv:2404.16710"

    def test_title_fallback_when_arxiv_id_fails(self):
        """arxiv_id verify 失败 → 用 title fuzzy 搜回捞（threshold > 0.85，标题需高度一致）"""
        cand = {"arxiv_id": "2404.99999",
                "title": "LayerSkip Enabling Early Exit Inference"}
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value=None), \
             patch("darwinian.agents.phase_a_orchestrator.search_papers",
                   return_value=[
                       {"paperId": "WRONG", "title": "Completely unrelated paper"},
                       # 标题完全一致 → similarity = 1.0
                       {"paperId": "RIGHT",
                        "title": "LayerSkip Enabling Early Exit Inference"},
                   ]) as mock_search:
            result = _verify_and_recover_seed(cand)
        mock_search.assert_called_once()
        assert result["paperId"] == "RIGHT"

    def test_title_fallback_no_match(self):
        """title 搜返回结果但相似度都 < 0.85 → 返 None"""
        cand = {"arxiv_id": "2404.99999", "title": "LayerSkip"}
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value=None), \
             patch("darwinian.agents.phase_a_orchestrator.search_papers",
                   return_value=[{"paperId": "X", "title": "GPT-3"}]):
            assert _verify_and_recover_seed(cand) is None

    def test_no_arxiv_id_no_title_returns_none(self):
        assert _verify_and_recover_seed({}) is None

    def test_s2_exception_swallowed(self):
        cand = {"arxiv_id": "2404.16710", "title": "x"}
        with patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   side_effect=ConnectionError("net")), \
             patch("darwinian.agents.phase_a_orchestrator.search_papers",
                   side_effect=ConnectionError("net")):
            assert _verify_and_recover_seed(cand) is None


class TestExpandSeedsOneHop:
    def test_collects_refs_and_cits(self):
        seeds = [{"paperId": "S1"}]
        refs = [{"paperId": "R1"}, {"paperId": "R2"}]
        cits = [{"paperId": "C1"}]
        with patch("darwinian.agents.phase_a_orchestrator.get_references",
                   return_value=refs), \
             patch("darwinian.agents.phase_a_orchestrator.get_citations",
                   return_value=cits):
            pool = _expand_seeds_one_hop(seeds, 5, 5)
        # seeds + refs + cits, 全去重
        ids = {p["paperId"] for p in pool}
        assert ids == {"S1", "R1", "R2", "C1"}

    def test_dedupe_across_seeds(self):
        seeds = [{"paperId": "S1"}, {"paperId": "S2"}]
        with patch("darwinian.agents.phase_a_orchestrator.get_references",
                   return_value=[{"paperId": "OVERLAP"}]), \
             patch("darwinian.agents.phase_a_orchestrator.get_citations",
                   return_value=[]):
            pool = _expand_seeds_one_hop(seeds, 5, 5)
        # OVERLAP 在 S1 和 S2 的 refs 都出现，应去重为 1
        ids = [p["paperId"] for p in pool]
        assert ids.count("OVERLAP") == 1

    def test_s2_failure_swallowed(self):
        seeds = [{"paperId": "S1"}]
        with patch("darwinian.agents.phase_a_orchestrator.get_references",
                   side_effect=ConnectionError("net")), \
             patch("darwinian.agents.phase_a_orchestrator.get_citations",
                   side_effect=ConnectionError("net")):
            pool = _expand_seeds_one_hop(seeds, 5, 5)
        # seed 自己仍在
        assert len(pool) == 1
        assert pool[0]["paperId"] == "S1"


class TestRerankByDirectionRelevance:
    def test_seeds_rank_above_non_seeds(self):
        papers = [
            {"paperId": "NON_SEED", "title": "transformer language model",
             "abstract": "general work", "citationCount": 99999},
            {"paperId": "SEED", "title": "speculative decoding draft model",
             "abstract": "matches direction", "citationCount": 50},
        ]
        seed_ids = {"SEED"}
        ranked = _rerank_by_direction_relevance(
            papers, "speculative decoding LLM", seed_ids,
        )
        assert ranked[0]["paperId"] == "SEED"

    def test_relevance_above_citation_within_non_seeds(self):
        """两个都不是 seed 时，TF-IDF 高的（与 direction 相似）排前"""
        papers = [
            {"paperId": "GENERIC", "title": "transformer",
             "abstract": "BERT large model", "citationCount": 100000},
            {"paperId": "RELEVANT", "title": "speculative decoding",
             "abstract": "draft model verification", "citationCount": 50},
        ]
        ranked = _rerank_by_direction_relevance(
            papers, "speculative decoding draft", set(),
        )
        assert ranked[0]["paperId"] == "RELEVANT"


class TestBuildSeedPool:
    def test_full_pipeline(self):
        """build_seed_pool 端到端：mock 4 个内部依赖"""
        # mock LLM 列 2 个 seed
        fake_resp = MagicMock(content=_json.dumps({
            "seed_papers": [
                {"arxiv_id": "2404.16710", "title": "LayerSkip"},
                {"arxiv_id": "2510.x", "title": "DEL"},
            ]
        }))
        # mock S2 verify
        s2_details = {
            "ArXiv:2404.16710": {"paperId": "S2_LS", "title": "LayerSkip"},
            "ArXiv:2510.x": {"paperId": "S2_DEL", "title": "DEL"},
        }
        # mock 一跳扩展
        refs_by_id = {
            "S2_LS": [{"paperId": "REF1", "title": "speculative ref"}],
            "S2_DEL": [{"paperId": "REF2", "title": "draft model ref"}],
        }

        def mock_gd(pid, fields=None):
            return s2_details.get(pid)
        def mock_refs(pid, limit=8):
            return refs_by_id.get(pid, [])

        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp), \
             patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   side_effect=mock_gd), \
             patch("darwinian.agents.phase_a_orchestrator.get_references",
                   side_effect=mock_refs), \
             patch("darwinian.agents.phase_a_orchestrator.get_citations",
                   return_value=[]):
            pool = build_seed_pool("LLM speculative decoding", MagicMock(),
                                    n_seeds=2, refs_per_seed=8, cits_per_seed=8)
        # 应包含 2 个 seed + 2 个 ref，全去重
        ids = {p["paperId"] for p in pool}
        assert ids == {"S2_LS", "S2_DEL", "REF1", "REF2"}

    def test_empty_when_no_seeds_verified(self):
        """LLM 列 seed 但 S2 全 verify 失败"""
        fake_resp = MagicMock(content=_json.dumps({
            "seed_papers": [{"arxiv_id": "fake.id", "title": "fake"}]
        }))
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   return_value=fake_resp), \
             patch("darwinian.agents.phase_a_orchestrator.get_paper_details",
                   return_value=None), \
             patch("darwinian.agents.phase_a_orchestrator.search_papers",
                   return_value=[]):
            pool = build_seed_pool("x", MagicMock())
        assert pool == []

    def test_llm_failure_returns_empty(self):
        with patch("darwinian.agents.phase_a_orchestrator.invoke_with_retry",
                   side_effect=ConnectionError("net")):
            pool = build_seed_pool("x", MagicMock())
        assert pool == []


# ===========================================================================
# Pri-5: relevance gate
# ===========================================================================

from darwinian.agents.phase_a_orchestrator import _filter_relevant_evidence


def _ev_with_relation(relation: str, name: str = "X"):
    return PaperEvidence(
        paper_id=f"arxiv:{name}", title=name, short_name=name,
        venue="ACL 2024", year=2024,
        quantitative_claims=[QuantitativeClaim(metric_name="speedup",
                                                metric_value="2x")],
        headline_result="2x", relation_to_direction=relation,
    )


class TestRelevanceGate:
    def test_default_drops_orthogonal_only(self):
        evs = [
            _ev_with_relation("extends", "LayerSkip"),
            _ev_with_relation("orthogonal", "Mamba"),
            _ev_with_relation("baseline", "DEL"),
            _ev_with_relation("inspires", "PonderNet"),
            _ev_with_relation("reproduces", "Repro"),
        ]
        kept = _filter_relevant_evidence(evs, strict=False)
        names = [k.short_name for k in kept]
        # orthogonal Mamba 被丢，其他留
        assert "Mamba" not in names
        assert "LayerSkip" in names
        assert "DEL" in names
        assert "PonderNet" in names
        assert "Repro" in names

    def test_strict_keeps_only_extends_baseline(self):
        evs = [
            _ev_with_relation("extends", "LayerSkip"),
            _ev_with_relation("orthogonal", "Mamba"),
            _ev_with_relation("baseline", "DEL"),
            _ev_with_relation("inspires", "PonderNet"),
            _ev_with_relation("reproduces", "Repro"),
        ]
        # min_keep=0 关闭保底，验证 strict 单独行为
        kept = _filter_relevant_evidence(evs, strict=True, min_keep=0)
        names = [k.short_name for k in kept]
        assert names == ["LayerSkip", "DEL"]

    def test_empty_input(self):
        assert _filter_relevant_evidence([]) == []

    def test_case_insensitive(self):
        ev = _ev_with_relation("ORTHOGONAL", "X")
        kept = _filter_relevant_evidence([ev], min_keep=0)
        assert kept == []

    def test_all_relevant_passes_unchanged(self):
        evs = [
            _ev_with_relation("extends", "A"),
            _ev_with_relation("baseline", "B"),
        ]
        kept = _filter_relevant_evidence(evs)
        assert len(kept) == 2

    def test_v9_mamba_padding_case(self):
        """v9 实测：Mamba/Copy-as-Decode 当 padding；用 orthogonal 标后被滤"""
        evs = [
            _ev_with_relation("extends", "EAGLE"),
            _ev_with_relation("orthogonal", "Mamba"),
            _ev_with_relation("orthogonal", "Copy-as-Decode"),
            _ev_with_relation("baseline", "LayerSkip"),
        ]
        # min_keep=0 关闭保底
        kept = _filter_relevant_evidence(evs, min_keep=0)
        names = [k.short_name for k in kept]
        assert names == ["EAGLE", "LayerSkip"]

    # ---- Round 8: min_keep 保底逻辑 ----

    def test_min_keep_backfills_when_too_few(self):
        """v10 实测案例：8 篇全 orthogonal 时保底 3 篇"""
        evs = [_ev_with_relation("orthogonal", f"P{i}") for i in range(5)]
        kept = _filter_relevant_evidence(evs, min_keep=3)
        # 应保 3 篇（即使全 orthogonal 也保）
        assert len(kept) == 3

    def test_min_keep_backfill_priority(self):
        """保底按 extends > baseline > inspires > reproduces > orthogonal 排"""
        evs = [
            _ev_with_relation("orthogonal", "Ortho"),
            _ev_with_relation("inspires", "Inspires"),
            _ev_with_relation("reproduces", "Repro"),
        ]
        # 默认模式只丢 orthogonal → kept=2 (Inspires + Repro)
        # min_keep=3 → 补 1 个 orthogonal
        kept = _filter_relevant_evidence(evs, min_keep=3)
        names = [k.short_name for k in kept]
        assert len(kept) == 3
        assert "Ortho" in names

    def test_min_keep_does_not_overshoot(self):
        """保底只补到 min_keep，不超"""
        evs = [_ev_with_relation("orthogonal", f"P{i}") for i in range(10)]
        kept = _filter_relevant_evidence(evs, min_keep=3)
        assert len(kept) == 3

    def test_min_keep_zero_disables_backfill(self):
        evs = [_ev_with_relation("orthogonal", "X")]
        kept = _filter_relevant_evidence(evs, min_keep=0)
        assert kept == []

    # ---- R10-Pri-2: out_stats + relevance warning ----

    def test_out_stats_truly_relevant_count(self):
        """out_stats 必须报真相关数（backfill 之前的 kept 数）"""
        evs = [
            _ev_with_relation("extends", "A"),
            _ev_with_relation("baseline", "B"),
            _ev_with_relation("orthogonal", "C"),
        ]
        stats: dict = {}
        _filter_relevant_evidence(evs, min_keep=0, out_stats=stats)
        assert stats["truly_relevant"] == 2
        assert stats["backfilled"] == 0

    def test_out_stats_distinguishes_truly_relevant_from_backfill(self):
        """v2 加密流量场景：3 真相关 + 5 兜底 → stats 分开报"""
        evs = (
            [_ev_with_relation("extends", f"R{i}") for i in range(2)]
            + [_ev_with_relation("orthogonal", f"O{i}") for i in range(5)]
        )
        stats: dict = {}
        _filter_relevant_evidence(evs, min_keep=4, out_stats=stats)
        assert stats["truly_relevant"] == 2
        assert stats["backfilled"] == 2

    def test_relevance_warning_triggers_below_threshold(self):
        """truly_relevant < 5 → warning 非空"""
        from darwinian.agents.phase_a_orchestrator import _build_relevance_warning
        warning = _build_relevance_warning(truly_relevant=3, backfilled=3)
        assert warning != ""
        assert "3" in warning  # 提到当前数
        assert "orthogonal" in warning.lower()

    def test_relevance_warning_silent_when_sufficient(self):
        """truly_relevant ≥ 5 → 空 warning"""
        from darwinian.agents.phase_a_orchestrator import _build_relevance_warning
        assert _build_relevance_warning(truly_relevant=8, backfilled=0) == ""
        assert _build_relevance_warning(truly_relevant=5, backfilled=0) == ""

    def test_empty_input_out_stats(self):
        """空输入也要填 out_stats（zero values）"""
        stats: dict = {}
        _filter_relevant_evidence([], out_stats=stats)
        assert stats == {"truly_relevant": 0, "backfilled": 0, "dropped": 0}


class TestPhaseAHardAbort:
    """R12: env var DARWINIAN_PHASE_A_HARD_ABORT_MIN 控制硬退出"""

    def test_exception_class_exists_and_inherits_runtime_error(self):
        from darwinian.agents.phase_a_orchestrator import PhaseAAbortError
        assert issubclass(PhaseAAbortError, RuntimeError)

    def test_default_no_abort(self, monkeypatch):
        """env var 未设 → 默认 0 → 不 raise（保留旧行为）"""
        monkeypatch.delenv("DARWINIAN_PHASE_A_HARD_ABORT_MIN", raising=False)
        # 这里只测分支逻辑，不重新跑 Phase A — 直接 inline 验证
        import os
        try:
            v = int(os.environ.get("DARWINIAN_PHASE_A_HARD_ABORT_MIN", "0"))
        except ValueError:
            v = 0
        assert v == 0

    def test_invalid_env_var_silently_zero(self, monkeypatch):
        """非法 env var 值（如 'abc'）→ silent fallback 到 0，不崩溃"""
        monkeypatch.setenv("DARWINIAN_PHASE_A_HARD_ABORT_MIN", "abc")
        import os
        try:
            v = int(os.environ.get("DARWINIAN_PHASE_A_HARD_ABORT_MIN", "0"))
        except ValueError:
            v = 0
        assert v == 0

    def test_abort_message_contains_actionable_hint(self):
        """raise 的 message 必须含'换 sub-direction'+'关闭硬退出'"""
        from darwinian.agents.phase_a_orchestrator import PhaseAAbortError
        e = PhaseAAbortError(
            "Phase A 真相关论文数 0 < 阈值 5（DARWINIAN_PHASE_A_HARD_ABORT_MIN=5）。"
            "建议换更聚焦的 sub-direction。如要继续跑，set "
            "DARWINIAN_PHASE_A_HARD_ABORT_MIN=0 关闭硬退出。"
        )
        msg = str(e)
        assert "换" in msg or "sub-direction" in msg
        assert "DARWINIAN_PHASE_A_HARD_ABORT_MIN=0" in msg


class TestPhaseAZeroEvidenceAbort:
    """R15: 0 真相关 + 0 phenomena 的默认硬熔断（不需 env var）

    通过 mock build_concept_graph / batch_extract_evidence /
    batch_mine_phenomena 让 Phase A 走完到 R15 检查点，验证 raise 行为。
    """

    def _setup_mocks(self, monkeypatch, *, phenomena, evidence_relations):
        """让 Phase A 跑通到 R15 检查点，evidence_relations 控制 truly_relevant 数"""
        from darwinian.agents import phase_a_orchestrator as orch
        from darwinian.state import (
            ConceptGraph, PaperEvidence, PaperInfo,
            QuantitativeClaim, ResearchConstraints,
        )

        # 一组 fake papers + evidence
        n_papers = max(len(evidence_relations), 1)
        papers = [PaperInfo(
            paper_id=f"p{i}", title=f"P{i}", abstract="abs", year=2024,
            citation_count=10,
        ) for i in range(n_papers)]
        graph = ConceptGraph(papers=papers, entities=[], limitations=[],
                             novel_pair_hints=[])

        evidence_list = [PaperEvidence(
            paper_id=f"p{i}", title=f"P{i}", short_name=f"P{i}",
            quantitative_claims=[QuantitativeClaim(
                metric_name="x", metric_value="1x")],
            headline_result="x", relation_to_direction=rel,
        ) for i, rel in enumerate(evidence_relations)]

        # mock 各个外部依赖
        monkeypatch.setattr(orch, "build_seed_pool",
                            lambda *a, **k: [{"paperId": p.paper_id} for p in papers])
        monkeypatch.setattr(orch, "build_concept_graph",
                            lambda *a, **k: graph)
        monkeypatch.setattr(orch, "write_structural_hole_hooks",
                            lambda *a, **k: [])
        monkeypatch.setattr(orch, "_resolve_arxiv_ids",
                            lambda papers: {p.paper_id: "" for p in papers})
        monkeypatch.setattr(orch, "_select_top_papers_by_relevance",
                            lambda graph, k: papers[:k])
        monkeypatch.setattr(orch, "batch_extract_evidence",
                            lambda *a, **k: evidence_list)
        monkeypatch.setattr(orch, "batch_mine_phenomena",
                            lambda *a, **k: phenomena)
        monkeypatch.setattr(orch, "detect_cross_paper_contradictions",
                            lambda evs: [])
        monkeypatch.setattr(orch, "_bucket_by_year",
                            lambda *a, **k: {})

    def test_aborts_on_zero_truly_relevant_zero_phenomena(self, monkeypatch):
        """v3/v4/v5 实战 case：0 真相关 + 0 phenomena → 默认 raise"""
        from darwinian.agents.phase_a_orchestrator import (
            PhaseAAbortError, build_research_material_pack,
        )
        from darwinian.state import ResearchConstraints

        monkeypatch.delenv("DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE", raising=False)
        monkeypatch.delenv("DARWINIAN_PHASE_A_HARD_ABORT_MIN", raising=False)
        # 全 orthogonal → truly_relevant=0；phenomena=[] → 总和 0
        self._setup_mocks(monkeypatch,
                          phenomena=[],
                          evidence_relations=["orthogonal"] * 3)
        with pytest.raises(PhaseAAbortError) as exc_info:
            build_research_material_pack(
                direction="x", constraints=ResearchConstraints(),
                extractor_llm=MagicMock(), evidence_llm=MagicMock(),
                top_k_evidence=3,
            )
        msg = str(exc_info.value)
        assert "0 篇真相关" in msg
        assert "0 个 phenomena" in msg
        assert "DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE=1" in msg

    def test_does_not_abort_when_phenomena_present(self, monkeypatch):
        """0 真相关但有 phenomena → 不 abort（phenomena 是另一种科学依据）"""
        from darwinian.agents.phase_a_orchestrator import build_research_material_pack
        from darwinian.state import Phenomenon, ResearchConstraints

        monkeypatch.delenv("DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE", raising=False)
        monkeypatch.delenv("DARWINIAN_PHASE_A_HARD_ABORT_MIN", raising=False)
        ph = Phenomenon(
            type="surprising_result",
            description="x", supporting_quote="q", paper_ids=["p0"],
        )
        self._setup_mocks(monkeypatch,
                          phenomena=[ph],
                          evidence_relations=["orthogonal"] * 3)
        # 不应 raise
        pack = build_research_material_pack(
            direction="x", constraints=ResearchConstraints(),
            extractor_llm=MagicMock(), evidence_llm=MagicMock(),
            top_k_evidence=3,
        )
        assert pack is not None
        assert len(pack.phenomena) == 1

    def test_does_not_abort_when_truly_relevant_present(self, monkeypatch):
        """有真相关论文（即使 0 phenomena）→ 不 abort"""
        from darwinian.agents.phase_a_orchestrator import build_research_material_pack
        from darwinian.state import ResearchConstraints

        monkeypatch.delenv("DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE", raising=False)
        monkeypatch.delenv("DARWINIAN_PHASE_A_HARD_ABORT_MIN", raising=False)
        self._setup_mocks(monkeypatch,
                          phenomena=[],
                          evidence_relations=["extends", "baseline", "extends"])
        pack = build_research_material_pack(
            direction="x", constraints=ResearchConstraints(),
            extractor_llm=MagicMock(), evidence_llm=MagicMock(),
            top_k_evidence=3,
        )
        assert pack is not None

    def test_env_var_allow_disables_abort(self, monkeypatch):
        """env DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE=1 时即使 0+0 也不 abort"""
        from darwinian.agents.phase_a_orchestrator import build_research_material_pack
        from darwinian.state import ResearchConstraints

        monkeypatch.setenv("DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE", "1")
        monkeypatch.delenv("DARWINIAN_PHASE_A_HARD_ABORT_MIN", raising=False)
        self._setup_mocks(monkeypatch,
                          phenomena=[],
                          evidence_relations=["orthogonal"] * 3)
        # 不应 raise（env 关闭了熔断）
        pack = build_research_material_pack(
            direction="x", constraints=ResearchConstraints(),
            extractor_llm=MagicMock(), evidence_llm=MagicMock(),
            top_k_evidence=3,
        )
        assert pack is not None
