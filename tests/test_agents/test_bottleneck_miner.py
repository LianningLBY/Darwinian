"""
bottleneck_miner_node 单元测试

不打真实 API：mock build_concept_graph 和 LLM。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from darwinian.agents.bottleneck_miner import (
    bottleneck_miner_node,
    _apply_banned_keywords,
    _build_user_message,
    _format_entities,
    _format_limitations,
    _format_novel_pairs,
)
from darwinian.state import (
    ConceptGraph,
    Entity,
    EntityPair,
    FailedRecord,
    LimitationRef,
    PaperInfo,
    ResearchState,
)


def _sufficient_graph():
    """构造一个 is_sufficient=True 的 ConceptGraph"""
    entities = [
        Entity(canonical_name=f"method_{i}", type="method", paper_ids=["p1"])
        for i in range(25)
    ]
    papers = [PaperInfo(paper_id=f"p{i}", title=f"T{i}", abstract="x" * 200) for i in range(12)]
    return ConceptGraph(
        papers=papers,
        entities=entities,
        limitations=[LimitationRef(id="a1b2c3d4", text="slow", source_paper_id="p1")],
        novel_pair_hints=[EntityPair(entity_a="method_0", entity_b="method_1", score=5)],
        is_sufficient=True,
    )


class TestApplyBannedKeywords:
    def test_no_banned_passthrough(self):
        g = ConceptGraph(entities=[Entity(canonical_name="adam", type="method", paper_ids=["p1"])])
        out = _apply_banned_keywords(g, [])
        assert len(out.entities) == 1

    def test_exact_match_filtered(self):
        g = ConceptGraph(entities=[
            Entity(canonical_name="adam", type="method", paper_ids=["p1"]),
            Entity(canonical_name="sgd", type="method", paper_ids=["p2"]),
        ])
        out = _apply_banned_keywords(g, ["adam"])
        names = {e.canonical_name for e in out.entities}
        assert names == {"sgd"}

    def test_normalize_respects_punct(self):
        # "Adam Optimizer" 在 ban 列表里应该匹配实体 "adam optimizer"
        g = ConceptGraph(entities=[
            Entity(canonical_name="adam optimizer", type="method", paper_ids=["p1"]),
        ])
        out = _apply_banned_keywords(g, ["Adam-Optimizer"])
        assert out.entities == []

    def test_word_boundary_containment(self):
        # ban "adam" 应该匹配 "adam optimizer"
        g = ConceptGraph(entities=[
            Entity(canonical_name="adam optimizer", type="method", paper_ids=["p1"]),
            Entity(canonical_name="sgd momentum", type="method", paper_ids=["p2"]),
        ])
        out = _apply_banned_keywords(g, ["adam"])
        names = {e.canonical_name for e in out.entities}
        assert names == {"sgd momentum"}

    def test_bert_ban_does_not_kill_bertopic(self):
        g = ConceptGraph(entities=[
            Entity(canonical_name="bertopic", type="method", paper_ids=["p1"]),
        ])
        out = _apply_banned_keywords(g, ["bert"])
        # word-boundary 不匹配，bertopic 应当保留
        assert len(out.entities) == 1


class TestFormatters:
    def test_format_entities_groups_by_type(self):
        g = ConceptGraph(entities=[
            Entity(canonical_name="adam", type="method", paper_ids=["p1", "p2"]),
            Entity(canonical_name="imagenet", type="dataset", paper_ids=["p1"]),
        ])
        out = _format_entities(g)
        assert "[method]" in out and "adam" in out
        assert "[dataset]" in out and "imagenet" in out

    def test_format_entities_truncation(self):
        g = ConceptGraph(entities=[
            Entity(canonical_name=f"m{i}", type="method", paper_ids=["p1"])
            for i in range(20)
        ])
        out = _format_entities(g, max_per_type=5)
        assert "还有 15 个" in out

    def test_format_limitations(self):
        g = ConceptGraph(limitations=[
            LimitationRef(id="abc", text="过拟合", source_paper_id="p1"),
        ])
        out = _format_limitations(g)
        assert "[abc]" in out and "过拟合" in out

    def test_format_limitations_empty(self):
        assert _format_limitations(ConceptGraph()) == "（无）"

    def test_format_novel_pairs(self):
        g = ConceptGraph(novel_pair_hints=[
            EntityPair(entity_a="a", entity_b="b", score=3),
        ])
        out = _format_novel_pairs(g)
        assert "a" in out and "b" in out and "score=3" in out


class TestBuildUserMessage:
    def test_sufficient_graph_includes_data(self):
        g = _sufficient_graph()
        msg = _build_user_message("test direction", g, "无失败")
        assert "实体表" in msg
        assert "待解缺陷清单" in msg
        assert "结构洞" in msg
        assert "test direction" in msg
        assert "无失败" in msg

    def test_insufficient_graph_fallback(self):
        g = ConceptGraph(is_sufficient=False)
        msg = _build_user_message("test", g, "无失败")
        assert "数据不足" in msg or "先验知识" in msg


class TestBottleneckMinerNode:
    def test_uses_build_concept_graph_and_writes_state(self):
        # 两个 mock：build_concept_graph 和 invoke_with_retry
        fake_graph = _sufficient_graph()
        fake_llm = MagicMock()
        fake_response = MagicMock(content='{"core_problem":"长序列效率","evidence":["p1 — slow"]}')

        with patch("darwinian.agents.bottleneck_miner.build_concept_graph",
                   return_value=fake_graph):
            with patch("darwinian.agents.bottleneck_miner.invoke_with_retry",
                       return_value=fake_response):
                state = ResearchState(research_direction="test direction")
                out = bottleneck_miner_node(state, fake_llm)

        assert "concept_graph" in out
        assert out["concept_graph"].is_sufficient is True
        assert out["current_hypothesis"].core_problem == "长序列效率"
        assert out["current_hypothesis"].literature_support == ["p1 — slow"]

    def test_banned_keywords_filter_applied(self):
        """banned_keywords 会过滤 concept_graph.entities"""
        g = _sufficient_graph()
        # 加一个会被 ban 的实体
        g.entities.append(Entity(canonical_name="adam", type="method", paper_ids=["p1"]))

        fake_llm = MagicMock()
        fake_response = MagicMock(content='{"core_problem":"x","evidence":[]}')
        state = ResearchState(
            research_direction="test",
            failed_ledger=[FailedRecord(
                feature_vector=[0.1],
                error_summary="old fail",
                failure_type="MATH_ERROR",
                iteration=1,
                banned_keywords=["adam"],
            )],
        )

        with patch("darwinian.agents.bottleneck_miner.build_concept_graph", return_value=g):
            with patch("darwinian.agents.bottleneck_miner.invoke_with_retry",
                       return_value=fake_response):
                out = bottleneck_miner_node(state, fake_llm)

        names = {e.canonical_name for e in out["concept_graph"].entities}
        assert "adam" not in names

    def test_extractor_llm_optional(self):
        """没传 extractor_llm 时复用 main llm"""
        g = _sufficient_graph()
        fake_llm = MagicMock()
        fake_response = MagicMock(content='{"core_problem":"x","evidence":[]}')

        with patch("darwinian.agents.bottleneck_miner.build_concept_graph",
                   return_value=g) as mock_build:
            with patch("darwinian.agents.bottleneck_miner.invoke_with_retry",
                       return_value=fake_response):
                bottleneck_miner_node(ResearchState(research_direction="t"), fake_llm)
        # 构建时应当传入主 llm
        assert mock_build.call_args.kwargs["llm"] is fake_llm
