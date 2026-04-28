"""
proposal_tournament 单元测试

不打真实 LLM —— mock invoke_with_retry，验证：
- _select_anchors: phenomenon → hook → 复用 三档降级
- pairwise_compare: winner 校验、rubric 解析、异常吞掉
- compute_elo_rankings: Elo 数学正确（高 elo 赢低 elo 时 elo 差小）
- run_tournament: C(N,2) 配对、top_k 截取、空/单 proposal 兜底
- multi_elaborate: 调 elaborator N 次，N 个不同 anchor
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.proposal_tournament import (
    multi_elaborate,
    pairwise_compare,
    run_tournament,
    compute_elo_rankings,
    _select_anchors,
    _expected_score,
    _update_elo,
)
from darwinian.state import (
    AbstractionBranch,
    Phenomenon,
    ResearchConstraints,
    ResearchMaterialPack,
    ResearchProposal,
    StructuralHoleHook,
    TournamentMatch,
    TournamentResult,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _proposal(title="P", motivation="m", method="meth"):
    return ResearchProposal(
        skeleton=AbstractionBranch(
            name="x", description="x", algorithm_logic="x", math_formulation="x",
        ),
        title=title,
        elevator_pitch="pitch",
        motivation=motivation,
        proposed_method=method,
    )


def _phenomenon(desc="ph"):
    return Phenomenon(
        type="surprising_result",
        description=desc,
        supporting_quote="quote",
        paper_ids=["p1"],
    )


def _hook(name="h"):
    return StructuralHoleHook(
        entity_a=name + "_a", entity_b=name + "_b",
        hook_text="hook txt", relation_type="divergence",
    )


def _pack(n_phenomena=3, n_hooks=2):
    return ResearchMaterialPack(
        direction="dir",
        constraints=ResearchConstraints(),
        phenomena=[_phenomenon(f"ph{i}") for i in range(n_phenomena)],
        structural_hole_hooks=[_hook(f"h{i}") for i in range(n_hooks)],
    )


def _judge_response(winner="a", reasoning="A wins on novelty"):
    return MagicMock(content=_json.dumps({
        "winner": winner,
        "rubric_scores": {"novelty": "a", "feasibility": "tie", "impact": winner},
        "judge_reasoning": reasoning,
    }))


# ---------------------------------------------------------------------------
# _select_anchors
# ---------------------------------------------------------------------------

class TestSelectAnchors:
    def test_phenomena_first(self):
        """素材充足时，前 N 个 anchor 都是 phenomenon"""
        pack = _pack(n_phenomena=5, n_hooks=2)
        anchors = _select_anchors(pack, n_candidates=3)
        assert len(anchors) == 3
        # 每个 anchor pack 的 phenomena[0] 不同
        first_descs = [a.phenomena[0].description for a in anchors]
        assert first_descs == ["ph0", "ph1", "ph2"]

    def test_hooks_when_phenomena_run_out(self):
        pack = _pack(n_phenomena=2, n_hooks=3)
        anchors = _select_anchors(pack, n_candidates=4)
        # 0, 1 是 phenomenon anchor (phenomena[0] 不同)
        assert anchors[0].phenomena[0].description == "ph0"
        assert anchors[1].phenomena[0].description == "ph1"
        # 2, 3 是 hook anchor (structural_hole_hooks[0] 不同)
        assert anchors[2].structural_hole_hooks[0].entity_a == "h0_a"
        assert anchors[3].structural_hole_hooks[0].entity_a == "h1_a"

    def test_reuses_pack_when_no_anchors_left(self):
        pack = _pack(n_phenomena=1, n_hooks=1)
        anchors = _select_anchors(pack, n_candidates=4)
        # 0=phenomenon, 1=hook, 2,3=复用 pack（带 directive）
        assert len(anchors) == 4
        # 0/1 有强制 anchor directive
        assert "phenomenon" in anchors[0].anchor_directive.lower()
        assert "结构洞" in anchors[1].anchor_directive
        # 2/3 没强制 anchor 但有"自由发挥"directive 提示用不同 angle
        assert "candidate 3" in anchors[2].anchor_directive
        assert "candidate 4" in anchors[3].anchor_directive


# ---------------------------------------------------------------------------
# pairwise_compare
# ---------------------------------------------------------------------------

class TestPairwiseCompare:
    def test_happy_path(self):
        a, b = _proposal("A title"), _proposal("B title")
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   return_value=_judge_response(winner="a")):
            m = pairwise_compare(a, b, MagicMock())
        assert m.winner == "a"
        assert m.proposal_a_id == "A title"
        assert m.proposal_b_id == "B title"
        assert "novelty" in m.rubric_scores

    def test_winner_validation(self):
        bad = MagicMock(content=_json.dumps({"winner": "neither"}))
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   return_value=bad):
            m = pairwise_compare(_proposal(), _proposal(), MagicMock())
        assert m is None

    def test_winner_case_normalized(self):
        upper = MagicMock(content=_json.dumps({
            "winner": "TIE",
            "rubric_scores": {"novelty": "A"},
            "judge_reasoning": "x",
        }))
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   return_value=upper):
            m = pairwise_compare(_proposal(), _proposal(), MagicMock())
        assert m.winner == "tie"
        assert m.rubric_scores["novelty"] == "a"

    def test_invalid_rubric_filtered(self):
        resp = MagicMock(content=_json.dumps({
            "winner": "a",
            "rubric_scores": {"novelty": "wrong", "feasibility": "b", "impact": "tie"},
            "judge_reasoning": "x",
        }))
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   return_value=resp):
            m = pairwise_compare(_proposal(), _proposal(), MagicMock())
        assert "novelty" not in m.rubric_scores  # invalid 的剔除
        assert m.rubric_scores["feasibility"] == "b"
        assert m.rubric_scores["impact"] == "tie"

    def test_llm_failure_returns_none(self):
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   side_effect=ConnectionError("net")):
            m = pairwise_compare(_proposal(), _proposal(), MagicMock())
        assert m is None


# ---------------------------------------------------------------------------
# Elo 数学
# ---------------------------------------------------------------------------

class TestEloMath:
    def test_expected_score_equal_ratings(self):
        assert abs(_expected_score(1200, 1200) - 0.5) < 1e-9

    def test_expected_score_higher_wins_more(self):
        # rating 高 200 → ~76% win expected
        e = _expected_score(1400, 1200)
        assert 0.7 < e < 0.8

    def test_update_elo_winner_a(self):
        ratings = {"A": 1200, "B": 1200}
        _update_elo(ratings, "A", "B", winner="a")
        # A 升 16, B 降 16 (k=32, expected=0.5, score=1)
        assert ratings["A"] == 1216
        assert ratings["B"] == 1184

    def test_update_elo_tie_no_change_when_equal(self):
        ratings = {"A": 1200, "B": 1200}
        _update_elo(ratings, "A", "B", winner="tie")
        assert ratings["A"] == 1200
        assert ratings["B"] == 1200

    def test_compute_elo_rankings_sorts_by_elo(self):
        matches = [
            TournamentMatch(proposal_a_id="A", proposal_b_id="B", winner="a"),
            TournamentMatch(proposal_a_id="A", proposal_b_id="C", winner="a"),
            TournamentMatch(proposal_a_id="B", proposal_b_id="C", winner="tie"),
        ]
        rankings = compute_elo_rankings(["A", "B", "C"], matches)
        # A 赢两场 → elo 最高
        assert rankings[0]["proposal_id"] == "A"
        assert rankings[0]["wins"] == 2
        assert rankings[0]["losses"] == 0
        # B 和 C 战绩完全对称（各 0W 1L 1T）→ Elo 几乎相等，顺序无所谓
        assert {rankings[1]["proposal_id"], rankings[2]["proposal_id"]} == {"B", "C"}
        # 但 elo 都 < A
        assert rankings[1]["elo"] < rankings[0]["elo"]
        assert rankings[2]["elo"] < rankings[0]["elo"]

    def test_elo_unknown_id_skipped(self):
        """match 含未知 id 时不崩，跳过该 match"""
        matches = [
            TournamentMatch(proposal_a_id="A", proposal_b_id="GHOST", winner="a"),
            TournamentMatch(proposal_a_id="A", proposal_b_id="B", winner="a"),
        ]
        rankings = compute_elo_rankings(["A", "B"], matches)
        # 只 GHOST 的 match 被跳过，A vs B 仍生效
        assert rankings[0]["proposal_id"] == "A"
        assert rankings[0]["wins"] == 1


# ---------------------------------------------------------------------------
# run_tournament
# ---------------------------------------------------------------------------

class TestRunTournament:
    def test_n5_runs_C5_2_pairs(self):
        """5 个 proposal 跑 10 场 pairwise"""
        proposals = [_proposal(f"P{i}") for i in range(5)]
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   return_value=_judge_response("a")) as mock:
            result = run_tournament(proposals, MagicMock(), top_k=2)
        # C(5,2) = 10
        assert mock.call_count == 10
        assert len(result.matches) == 10
        assert len(result.top_k_ids) == 2

    def test_top_k_in_descending_elo(self):
        """top_k_ids 跟 elo_rankings 前 K 一致"""
        proposals = [_proposal(f"P{i}") for i in range(3)]
        # P0 永远赢
        def fake(*a, **k):
            return _judge_response("a")
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   side_effect=fake):
            result = run_tournament(proposals, MagicMock(), top_k=2)
        # P0 永远是 a，所以 top1 = P0
        assert result.top_k_ids[0] == "P0"
        assert result.top_k_ids == [r["proposal_id"] for r in result.elo_rankings[:2]]

    def test_single_proposal_no_matches(self):
        result = run_tournament([_proposal("solo")], MagicMock())
        assert result.matches == []
        assert result.top_k_ids == ["solo"]

    def test_empty_returns_empty(self):
        result = run_tournament([], MagicMock())
        assert result.matches == []
        assert result.top_k_ids == []

    def test_duplicate_titles_renamed(self):
        """同名 proposal 被加 _i 后缀避免冲突"""
        proposals = [_proposal("Same"), _proposal("Same"), _proposal("Different")]
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   return_value=_judge_response("a")):
            result = run_tournament(proposals, MagicMock(), top_k=3)
        ids_in_rankings = [r["proposal_id"] for r in result.elo_rankings]
        # 3 个不同 id（一个被改成 Same_1）
        assert len(set(ids_in_rankings)) == 3

    def test_failed_match_skipped(self):
        """单场失败不阻塞其他场"""
        proposals = [_proposal(f"P{i}") for i in range(3)]
        responses = [_judge_response("a"), None, _judge_response("b")]
        idx = {"i": 0}
        def fake(*a, **k):
            r = responses[idx["i"]]
            idx["i"] += 1
            if r is None:
                raise ConnectionError("net")
            return r
        with patch("darwinian.agents.proposal_tournament.invoke_with_retry",
                   side_effect=fake):
            result = run_tournament(proposals, MagicMock(), top_k=2)
        # 3 场预期，1 场失败 → 2 场 successful
        assert len(result.matches) == 2


# ---------------------------------------------------------------------------
# multi_elaborate
# ---------------------------------------------------------------------------

class TestMultiElaborate:
    def test_calls_elaborator_n_times(self):
        skel = AbstractionBranch(name="x", description="x",
                                  algorithm_logic="x", math_formulation="x")
        pack = _pack(n_phenomena=3, n_hooks=0)
        good = _proposal("P_x")
        with patch("darwinian.agents.proposal_tournament.elaborate_proposal_from_pack",
                   return_value=good) as mock_elab:
            result = multi_elaborate(skel, pack, MagicMock(), n_candidates=3)
        assert mock_elab.call_count == 3
        assert len(result) == 3
        # 每次 elaborator 收到不同的 material_pack（phenomenon[0] 不同）
        called_packs = [c.kwargs["material_pack"] for c in mock_elab.call_args_list]
        anchors = [p.phenomena[0].description for p in called_packs]
        assert anchors == ["ph0", "ph1", "ph2"]

    def test_failed_elaboration_skipped(self):
        skel = AbstractionBranch(name="x", description="x",
                                  algorithm_logic="x", math_formulation="x")
        pack = _pack(n_phenomena=3)
        results = [_proposal("P0"), None, _proposal("P2")]
        idx = {"i": 0}
        def fake(*a, **k):
            r = results[idx["i"]]
            idx["i"] += 1
            return r
        with patch("darwinian.agents.proposal_tournament.elaborate_proposal_from_pack",
                   side_effect=fake):
            result = multi_elaborate(skel, pack, MagicMock(), n_candidates=3)
        # 2 个成功
        assert len(result) == 2
        assert result[0].title == "P0"
        assert result[1].title == "P2"


# ===========================================================================
# Round 8b: anchor_directive 注入每个 candidate
# ===========================================================================

class TestAnchorDirectiveSetByMultiElaborate:
    def test_phenomenon_anchor_directive(self):
        pack = _pack(n_phenomena=3, n_hooks=0)
        anchors = _select_anchors(pack, n_candidates=3)
        # 每个 anchor pack 都有 directive 含对应 phenomenon description
        for i in range(3):
            assert "强制 anchor" in anchors[i].anchor_directive
            assert anchors[i].phenomena[0].description in anchors[i].anchor_directive
        # directive 各不相同
        directives = [a.anchor_directive for a in anchors]
        assert len(set(directives)) == 3

    def test_hook_anchor_directive(self):
        pack = _pack(n_phenomena=0, n_hooks=2)
        anchors = _select_anchors(pack, n_candidates=2)
        for a in anchors:
            assert "结构洞" in a.anchor_directive
            assert a.structural_hole_hooks[0].entity_a in a.anchor_directive

    def test_fallback_directive_when_no_anchor_left(self):
        pack = _pack(n_phenomena=0, n_hooks=0)
        anchors = _select_anchors(pack, n_candidates=3)
        # 所有 candidate 都用"自由发挥"directive
        for i, a in enumerate(anchors):
            assert f"candidate {i+1}" in a.anchor_directive
            assert "不同的 angle" in a.anchor_directive
