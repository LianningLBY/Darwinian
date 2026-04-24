"""
hypothesis_generator_node 单元测试

重点覆盖：
- 硬约束校验 (_validate_branches) 全部 error code
- 候选建议 (_suggest_entity_candidates)
- 降级路径（concept_graph is None / is_sufficient=False）
- step 7.5 组合查重
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.hypothesis_generator import (
    DuplicateHypothesisError,
    hypothesis_generator_node,
    _build_validation_feedback,
    _check_combination_novelty,
    _suggest_entity_candidates,
    _validate_branches,
)
from darwinian.state import (
    AbstractionBranch,
    ConceptGraph,
    Entity,
    Hypothesis,
    LimitationRef,
    PaperInfo,
    ResearchState,
)


# ---------------------------------------------------------------------------
# 小工具：构造一个常用的 sufficient graph
# ---------------------------------------------------------------------------

def _graph_with_two_task_types():
    """同时有 classification 和 language_modeling 两类论文"""
    entities = [
        Entity(canonical_name="adam", type="method", paper_ids=["p1", "p2"]),
        Entity(canonical_name="resnet", type="method", paper_ids=["p1", "p3"]),
        Entity(canonical_name="self attention", type="method", paper_ids=["p2", "p3"]),
    ]
    # 补足 is_sufficient 需要的实体数
    entities += [
        Entity(canonical_name=f"filler_{i}", type="method", paper_ids=["p1"])
        for i in range(20)
    ]
    papers = [
        PaperInfo(paper_id="p1", task_type="classification"),
        PaperInfo(paper_id="p2", task_type="language_modeling"),
        PaperInfo(paper_id="p3", task_type="classification"),
    ] + [PaperInfo(paper_id=f"p_extra_{i}", task_type="classification") for i in range(10)]
    limitations = [
        LimitationRef(id="L00001", text="slow", source_paper_id="p1"),
        LimitationRef(id="L00002", text="large memory", source_paper_id="p2"),
    ]
    return ConceptGraph(
        papers=papers,
        entities=entities,
        limitations=limitations,
        is_sufficient=True,
    )


def _valid_branch_pair():
    # 每个 branch 引用 ≥2 实体 from ≥2 papers，绑一条 valid limitation
    return [
        AbstractionBranch(
            name="B1", description="x", algorithm_logic="x", math_formulation="x",
            cited_entity_names=["adam", "self attention"],
            solved_limitation_id="L00001",
        ),
        AbstractionBranch(
            name="B2", description="x", algorithm_logic="x", math_formulation="x",
            cited_entity_names=["resnet", "self attention"],
            solved_limitation_id="L00002",
        ),
    ]


# ---------------------------------------------------------------------------
# _validate_branches
# ---------------------------------------------------------------------------

class TestValidateBranches:
    def test_valid_pair_no_errors(self):
        g = _graph_with_two_task_types()
        errors = _validate_branches(_valid_branch_pair(), g)
        assert errors == []

    def test_missing_entity_reported(self):
        g = _graph_with_two_task_types()
        branches = [
            AbstractionBranch(
                name="B", description="x", algorithm_logic="x", math_formulation="x",
                cited_entity_names=["adam", "mystery_entity"],
                solved_limitation_id="L00001",
            ),
            _valid_branch_pair()[1],
        ]
        errors = _validate_branches(branches, g)
        codes = [e[0] for e in errors]
        assert "MISSING_ENTITY" in codes

    def test_too_few_entities(self):
        g = _graph_with_two_task_types()
        branches = [
            AbstractionBranch(
                name="B", description="x", algorithm_logic="x", math_formulation="x",
                cited_entity_names=["adam"],   # 仅 1 个
                solved_limitation_id="L00001",
            ),
            _valid_branch_pair()[1],
        ]
        errors = _validate_branches(branches, g)
        codes = [e[0] for e in errors]
        assert "TOO_FEW_ENTITIES" in codes

    def test_invalid_limitation(self):
        g = _graph_with_two_task_types()
        branches = [
            AbstractionBranch(
                name="B", description="x", algorithm_logic="x", math_formulation="x",
                cited_entity_names=["adam", "self attention"],
                solved_limitation_id="NONEXISTENT",
            ),
            _valid_branch_pair()[1],
        ]
        errors = _validate_branches(branches, g)
        codes = [e[0] for e in errors]
        assert "INVALID_LIMITATION" in codes

    def test_cross_branch_task_type_insufficient(self):
        """整组 cited entities 都来自 task_type=classification 的论文"""
        # 让 adam 和 resnet 都只在 classification paper 里
        entities = [
            Entity(canonical_name="adam", type="method", paper_ids=["p1", "p3"]),
            Entity(canonical_name="resnet", type="method", paper_ids=["p1", "p3"]),
        ] + [
            Entity(canonical_name=f"f{i}", type="method", paper_ids=["p1"])
            for i in range(20)
        ]
        papers = [
            PaperInfo(paper_id="p1", task_type="classification"),
            PaperInfo(paper_id="p3", task_type="classification"),
        ] + [PaperInfo(paper_id=f"px{i}", task_type="classification") for i in range(10)]
        g = ConceptGraph(
            papers=papers, entities=entities,
            limitations=[LimitationRef(id="LX", text="x", source_paper_id="p1")],
            is_sufficient=True,
        )
        branches = [
            AbstractionBranch(
                name="B1", description="x", algorithm_logic="x", math_formulation="x",
                cited_entity_names=["adam", "resnet"],
                solved_limitation_id="LX",
            ),
            AbstractionBranch(
                name="B2", description="x", algorithm_logic="x", math_formulation="x",
                cited_entity_names=["adam", "resnet"],
                solved_limitation_id="LX",
            ),
        ]
        errors = _validate_branches(branches, g)
        codes = [e[0] for e in errors]
        assert "NOT_ENOUGH_TASK_TYPES" in codes

    def test_entity_name_normalization(self):
        """实体名带不同标点/大小写应该能匹配到图中 canonical"""
        g = _graph_with_two_task_types()
        branches = [
            AbstractionBranch(
                name="B", description="x", algorithm_logic="x", math_formulation="x",
                cited_entity_names=["Adam", "Self-Attention"],  # 不同大小写/标点
                solved_limitation_id="L00001",
            ),
            _valid_branch_pair()[1],
        ]
        errors = _validate_branches(branches, g)
        codes = [e[0] for e in errors]
        assert "MISSING_ENTITY" not in codes


# ---------------------------------------------------------------------------
# _suggest_entity_candidates
# ---------------------------------------------------------------------------

class TestSuggestCandidates:
    def test_word_boundary_substring_match(self):
        g = _graph_with_two_task_types()
        # "attention" 是 "self attention" 的子串（word boundary）
        cands = _suggest_entity_candidates("attention", g)
        names = [c.canonical_name for c in cands]
        assert "self attention" in names

    def test_fallback_to_popularity_when_no_match(self):
        g = _graph_with_two_task_types()
        cands = _suggest_entity_candidates("utterly_unmatched_xyz", g)
        assert len(cands) > 0  # 兜底返回按热度前 3

    def test_top_k_cap(self):
        g = _graph_with_two_task_types()
        cands = _suggest_entity_candidates("x", g, top_k=2)
        assert len(cands) <= 2


# ---------------------------------------------------------------------------
# 校验反馈文本
# ---------------------------------------------------------------------------

class TestBuildFeedback:
    def test_feedback_has_candidates(self):
        g = _graph_with_two_task_types()
        branches = [
            AbstractionBranch(
                name="B", description="x", algorithm_logic="x", math_formulation="x",
                cited_entity_names=["foobar"],
                solved_limitation_id="INVALID",
            ),
            _valid_branch_pair()[1],
        ]
        errors = _validate_branches(branches, g)
        feedback = _build_validation_feedback(errors, branches, g)
        assert "foobar" in feedback
        assert "候选" in feedback
        # 有 INVALID_LIMITATION 时列出可选 id
        assert "L00001" in feedback or "L00002" in feedback


# ---------------------------------------------------------------------------
# step 7.5: _check_combination_novelty
# ---------------------------------------------------------------------------

class TestCheckCombinationNovelty:
    def test_less_than_two_entities_not_existing(self):
        branch = AbstractionBranch(
            name="B", description="x", algorithm_logic="x", math_formulation="x",
            cited_entity_names=["only_one"],
        )
        _check_combination_novelty(branch)
        assert branch.existing_combination is False

    def test_all_terms_must_appear_in_title_or_abstract(self):
        branch = AbstractionBranch(
            name="B", description="x", algorithm_logic="x", math_formulation="x",
            cited_entity_names=["mamba", "flash_attention"],
        )
        fake_hits = [
            # 只含 mamba，不算命中
            {"paperId": "x1", "title": "mamba", "abstract": "nothing else"},
            # 标题含两个才算命中
            {"paperId": "x2", "title": "mamba meets flash_attention", "abstract": ""},
        ]
        with patch("darwinian.agents.hypothesis_generator.ss.search_papers", return_value=fake_hits):
            _check_combination_novelty(branch)
        assert branch.existing_combination is True
        assert "x2" in branch.existing_combination_refs
        assert "x1" not in branch.existing_combination_refs

    def test_no_hits_marks_false(self):
        branch = AbstractionBranch(
            name="B", description="x", algorithm_logic="x", math_formulation="x",
            cited_entity_names=["novel_a", "novel_b"],
        )
        with patch("darwinian.agents.hypothesis_generator.ss.search_papers", return_value=[]):
            _check_combination_novelty(branch)
        assert branch.existing_combination is False

    def test_s2_error_does_not_raise(self):
        branch = AbstractionBranch(
            name="B", description="x", algorithm_logic="x", math_formulation="x",
            cited_entity_names=["a", "b"],
        )
        with patch("darwinian.agents.hypothesis_generator.ss.search_papers",
                   side_effect=Exception("boom")):
            _check_combination_novelty(branch)   # 不抛
        assert branch.existing_combination is False


# ---------------------------------------------------------------------------
# hypothesis_generator_node 端到端
# ---------------------------------------------------------------------------

class TestHypothesisGeneratorNode:
    def _state(self, graph, core_problem="test"):
        return ResearchState(
            research_direction="test",
            concept_graph=graph,
            current_hypothesis=Hypothesis(core_problem=core_problem),
        )

    def test_raises_without_hypothesis(self):
        state = ResearchState(research_direction="x")
        with pytest.raises(ValueError):
            hypothesis_generator_node(state, MagicMock())

    def test_graceful_fallback_when_graph_absent(self):
        """concept_graph=None → 走降级路径（v1 prompt，不做硬约束）"""
        state = self._state(graph=None)
        fake_resp = MagicMock(content=(
            '{"core_problem":"x","abstraction_tree":[{'
            '"name":"B1","description":"x","algorithm_logic":"x","math_formulation":"x",'
            '"source_domain":"控制论"'
            '}],"confidence":0.5,"literature_support":[]}'
        ))
        with patch("darwinian.agents.hypothesis_generator.invoke_with_retry",
                   return_value=fake_resp):
            out = hypothesis_generator_node(state, MagicMock())
        assert out["current_hypothesis"].abstraction_tree[0].name == "B1"

    def test_insufficient_graph_fallback(self):
        g = ConceptGraph(is_sufficient=False)
        state = self._state(graph=g)
        fake_resp = MagicMock(content=(
            '{"core_problem":"x","abstraction_tree":[{'
            '"name":"B1","description":"x","algorithm_logic":"x","math_formulation":"x"'
            '}],"confidence":0.5,"literature_support":[]}'
        ))
        with patch("darwinian.agents.hypothesis_generator.invoke_with_retry",
                   return_value=fake_resp):
            out = hypothesis_generator_node(state, MagicMock())
        assert len(out["current_hypothesis"].abstraction_tree) == 1

    def test_v2_happy_path(self):
        """sufficient graph + valid LLM output + combination check"""
        g = _graph_with_two_task_types()
        state = self._state(g)
        valid_branches_json = (
            '{"core_problem":"x","abstraction_tree":['
            '{"name":"B1","description":"x","algorithm_logic":"x","math_formulation":"x",'
            ' "cited_entity_names":["adam","self attention"],"solved_limitation_id":"L00001"},'
            '{"name":"B2","description":"x","algorithm_logic":"x","math_formulation":"x",'
            ' "cited_entity_names":["resnet","self attention"],"solved_limitation_id":"L00002"}'
            '],"confidence":0.7,"literature_support":[]}'
        )
        fake_resp = MagicMock(content=valid_branches_json)
        with patch("darwinian.agents.hypothesis_generator.invoke_with_retry",
                   return_value=fake_resp):
            with patch("darwinian.agents.hypothesis_generator.ss.search_papers",
                       return_value=[]):   # step 7.5 查重返空
                out = hypothesis_generator_node(state, MagicMock())
        branches = out["current_hypothesis"].abstraction_tree
        assert len(branches) == 2
        assert all(b.cited_entity_names for b in branches)
        assert all(b.existing_combination is False for b in branches)

    def test_v2_validation_fail_then_retry_empty(self):
        """LLM 输出无效 cited_entity → 3 次重试后空 tree"""
        g = _graph_with_two_task_types()
        state = self._state(g)
        bad_json = (
            '{"core_problem":"x","abstraction_tree":['
            '{"name":"B1","description":"x","algorithm_logic":"x","math_formulation":"x",'
            ' "cited_entity_names":["nonexistent_x","nonexistent_y"],"solved_limitation_id":"NONE"}'
            ']}'
        )
        fake_resp = MagicMock(content=bad_json)
        with patch("darwinian.agents.hypothesis_generator.invoke_with_retry",
                   return_value=fake_resp):
            with patch("time.sleep"):   # 跳过 sleep 提速
                out = hypothesis_generator_node(state, MagicMock())
        # 3 次都失败 → 返回空 tree 触发 MATH_ERROR 路径
        assert out["current_hypothesis"].abstraction_tree == []
