"""
hook_writer 单元测试

不打真实 LLM —— mock invoke_with_retry，验证：
- 正常路径：每 EntityPair 生成一个 StructuralHoleHook
- relation_type 校验 + 反馈重试
- LLM 失败时跳过该 pair（不阻塞整体）
- 代表论文按 citation 排
- 空 pairs / 空 graph 安全返空
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.hook_writer import (
    write_structural_hole_hooks,
    _representative_paper_ids,
    _build_user_message,
)
from darwinian.state import (
    Entity,
    EntityPair,
    PaperInfo,
    StructuralHoleHook,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _mk_pair(a, b, score=5):
    return EntityPair(entity_a=a, entity_b=b, score=score)


def _mk_entity(name, paper_ids):
    return Entity(canonical_name=name, type="method", paper_ids=paper_ids)


def _mk_paper(pid, title="t", year=2024, citations=10):
    return PaperInfo(paper_id=pid, title=title, year=year, citation_count=citations)


def _good_response(hook_text="Generic mechanism-level hook", relation="divergence"):
    return MagicMock(content=_json.dumps({
        "hook_text": hook_text,
        "relation_type": relation,
    }))


# ---------------------------------------------------------------------------
# 主路径
# ---------------------------------------------------------------------------

class TestWriteHooksHappyPath:
    def test_single_pair_to_hook(self):
        pair = _mk_pair("quant_sensitivity", "draft_acceptance", score=8)
        entities = [
            _mk_entity("quant_sensitivity", ["P1"]),
            _mk_entity("draft_acceptance", ["P2"]),
        ]
        papers = [_mk_paper("P1", title="RAMP"), _mk_paper("P2", title="LayerSkip")]

        with patch("darwinian.agents.hook_writer.invoke_with_retry",
                   return_value=_good_response("RAMP measures perplexity but draft uses acceptance",
                                                "divergence")):
            hooks = write_structural_hole_hooks(
                [pair], entities, papers, "LLM speculative", MagicMock(),
            )
        assert len(hooks) == 1
        h = hooks[0]
        assert h.entity_a == "quant_sensitivity"
        assert h.entity_b == "draft_acceptance"
        assert h.score == 8
        assert "RAMP measures" in h.hook_text
        assert h.relation_type == "divergence"
        assert h.supporting_paper_ids_a == ["P1"]
        assert h.supporting_paper_ids_b == ["P2"]

    def test_max_hooks_limits(self):
        pairs = [_mk_pair(f"a{i}", f"b{i}") for i in range(10)]
        entities = []
        papers = []
        with patch("darwinian.agents.hook_writer.invoke_with_retry",
                   return_value=_good_response()):
            hooks = write_structural_hole_hooks(
                pairs, entities, papers, "x", MagicMock(), max_hooks=3,
            )
        assert len(hooks) == 3

    def test_empty_pairs_returns_empty(self):
        with patch("darwinian.agents.hook_writer.invoke_with_retry") as mock_llm:
            hooks = write_structural_hole_hooks([], [], [], "x", MagicMock())
        assert hooks == []
        # LLM 没被调
        assert mock_llm.call_count == 0


# ---------------------------------------------------------------------------
# relation_type 校验 + 反馈重试
# ---------------------------------------------------------------------------

class TestRelationTypeValidation:
    def test_invalid_relation_first_then_valid(self):
        """第 1 次 LLM 给 'random' → 反馈让它改 → 第 2 次给 divergence"""
        pair = _mk_pair("a", "b")
        responses = [
            _good_response("hook v1", "random"),       # invalid
            _good_response("hook v2", "convergence"),  # valid
        ]
        idx = {"i": 0}
        def fake(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r

        with patch("darwinian.agents.hook_writer.invoke_with_retry", side_effect=fake), \
             patch("time.sleep"):
            hooks = write_structural_hole_hooks(
                [pair], [], [], "x", MagicMock(), max_retries=1,
            )
        assert len(hooks) == 1
        assert hooks[0].relation_type == "convergence"
        assert idx["i"] == 2

    def test_invalid_relation_persistent_skipped(self):
        """LLM 死活给非法 relation → 该 pair 被跳过"""
        pair = _mk_pair("a", "b")
        with patch("darwinian.agents.hook_writer.invoke_with_retry",
                   return_value=_good_response("hook", "wrong_type")), \
             patch("time.sleep"):
            hooks = write_structural_hole_hooks(
                [pair], [], [], "x", MagicMock(), max_retries=1,
            )
        assert hooks == []   # 0 hook，但不抛异常

    def test_relation_case_insensitive(self):
        """LLM 输出 'Divergence' / 'DIVERGENCE' 也接受（lower 后比较）"""
        pair = _mk_pair("a", "b")
        with patch("darwinian.agents.hook_writer.invoke_with_retry",
                   return_value=_good_response("hook", "DIVERGENCE")):
            hooks = write_structural_hole_hooks(
                [pair], [], [], "x", MagicMock(),
            )
        assert hooks[0].relation_type == "divergence"

    def test_empty_hook_text_skipped(self):
        pair = _mk_pair("a", "b")
        with patch("darwinian.agents.hook_writer.invoke_with_retry",
                   return_value=_good_response("", "divergence")), \
             patch("time.sleep"):
            hooks = write_structural_hole_hooks(
                [pair], [], [], "x", MagicMock(), max_retries=0,
            )
        assert hooks == []


# ---------------------------------------------------------------------------
# 异常处理 / 部分失败
# ---------------------------------------------------------------------------

class TestExceptionHandling:
    def test_llm_failure_skips_pair(self):
        pair = _mk_pair("a", "b")
        with patch("darwinian.agents.hook_writer.invoke_with_retry",
                   side_effect=ConnectionError("net")), \
             patch("time.sleep"):
            hooks = write_structural_hole_hooks(
                [pair], [], [], "x", MagicMock(), max_retries=1,
            )
        assert hooks == []   # 不抛异常

    def test_unparseable_json_skipped(self):
        pair = _mk_pair("a", "b")
        bad = MagicMock(content="not json")
        with patch("darwinian.agents.hook_writer.invoke_with_retry",
                   return_value=bad), \
             patch("time.sleep"):
            hooks = write_structural_hole_hooks(
                [pair], [], [], "x", MagicMock(), max_retries=0,
            )
        assert hooks == []

    def test_partial_failure_continues(self):
        """3 个 pair, 第 2 个 LLM 挂 → 第 1 和 3 仍正常"""
        pairs = [_mk_pair(f"a{i}", f"b{i}") for i in range(3)]
        responses = [
            _good_response("hook 0", "divergence"),
            ConnectionError("net"),   # 第 2 个挂
            _good_response("hook 2", "transfer"),
        ]
        idx = {"i": 0}
        def fake(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            if isinstance(r, Exception):
                raise r
            return r

        with patch("darwinian.agents.hook_writer.invoke_with_retry", side_effect=fake), \
             patch("time.sleep"):
            hooks = write_structural_hole_hooks(
                pairs, [], [], "x", MagicMock(), max_retries=0,
            )
        assert len(hooks) == 2
        assert hooks[0].entity_a == "a0"
        assert hooks[1].entity_a == "a2"


# ---------------------------------------------------------------------------
# helper: _representative_paper_ids
# ---------------------------------------------------------------------------

class TestRepresentativePapers:
    def test_sorted_by_citation(self):
        """同 entity 多 paper → 按 citation 降序取 top-K"""
        entity = _mk_entity("foo", ["P_LOW", "P_HIGH", "P_MID"])
        papers_dict = {
            "P_LOW": _mk_paper("P_LOW", citations=10),
            "P_HIGH": _mk_paper("P_HIGH", citations=10000),
            "P_MID": _mk_paper("P_MID", citations=500),
        }
        result = _representative_paper_ids(
            "foo", {"foo": entity}, papers_dict, top_k=3,
        )
        assert result == ["P_HIGH", "P_MID", "P_LOW"]

    def test_top_k_limit(self):
        entity = _mk_entity("foo", [f"P{i}" for i in range(10)])
        papers_dict = {f"P{i}": _mk_paper(f"P{i}", citations=100 - i)
                       for i in range(10)}
        result = _representative_paper_ids(
            "foo", {"foo": entity}, papers_dict, top_k=3,
        )
        assert len(result) == 3
        assert result == ["P0", "P1", "P2"]

    def test_unknown_entity_returns_empty(self):
        result = _representative_paper_ids("nonexistent", {}, {})
        assert result == []

    def test_paper_not_in_dict_skipped(self):
        """entity.paper_ids 含 dict 里没有的 id → 跳过"""
        entity = _mk_entity("foo", ["P_OK", "P_MISSING"])
        papers_dict = {"P_OK": _mk_paper("P_OK", citations=10)}
        result = _representative_paper_ids(
            "foo", {"foo": entity}, papers_dict,
        )
        assert result == ["P_OK"]


# ---------------------------------------------------------------------------
# prompt 构造
# ---------------------------------------------------------------------------

class TestUserMessage:
    def test_includes_direction_and_pair(self):
        pair = _mk_pair("ent_a", "ent_b", score=7)
        msg = _build_user_message(pair, [], [], {}, "speculative decoding")
        assert "speculative decoding" in msg
        assert "ent_a" in msg
        assert "ent_b" in msg
        assert "score (min papers each side): 7" in msg

    def test_includes_paper_titles_for_context(self):
        pair = _mk_pair("a", "b")
        papers = {
            "P1": _mk_paper("P1", title="LayerSkip Paper", year=2024),
            "P2": _mk_paper("P2", title="RAMP Paper", year=2026),
        }
        msg = _build_user_message(pair, ["P1"], ["P2"], papers, "x")
        assert "LayerSkip Paper" in msg
        assert "RAMP Paper" in msg
        assert "(2024)" in msg
        assert "(2026)" in msg

    def test_no_papers_renders_placeholder(self):
        pair = _mk_pair("a", "b")
        msg = _build_user_message(pair, [], [], {}, "x")
        assert "未找到代表论文" in msg
