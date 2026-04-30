"""
proposal_debater (R19) 单元测试

不打真实 LLM —— mock invoke_with_retry，验证：
- _run_advocate / _run_challenger / _run_judge 单 call 解析
- debate_proposal 端到端：多轮 / 收敛 / 提前终止
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.proposal_debater import (
    _run_advocate,
    _run_challenger,
    _run_judge,
    debate_proposal,
)
from darwinian.state import (
    AbstractionBranch,
    DebateResult,
    MethodologyPhase,
    ResearchProposal,
)


def _proposal():
    return ResearchProposal(
        skeleton=AbstractionBranch(
            name="x", description="x", algorithm_logic="x", math_formulation="x",
        ),
        title="DriftRecast: encrypted traffic",
        elevator_pitch="pitch",
        motivation="motiv",
        proposed_method="method",
        technical_details="tech",
        target_venue="NeurIPS 2026",
        methodology_phases=[
            MethodologyPhase(
                phase_number=1, name="P1", description="d",
                expected_compute_hours=80.0,
            ),
        ],
        total_estimated_hours=80.0,
    )


def _adv_response(rate=0.30):
    return MagicMock(content=_json.dumps({
        "argument": "advocate argues this is novel and feasible.",
        "estimated_acceptance_rate": rate,
        "key_strengths": ["novel framing", "clean phases"],
    }))


def _chal_response():
    return MagicMock(content=_json.dumps({
        "argument": "challenger says rank correlation across architectures is questionable.",
        "weaknesses": ["cross-architecture comparison"],
        "potential_collisions": ["ACD-WFP 2024"],
    }))


def _judge_response(rate=0.25, revisions=None):
    return MagicMock(content=_json.dumps({
        "assessment": "judge sides 50/50, suggests revision.",
        "estimated_acceptance_rate": rate,
        "revisions_proposed": revisions or [
            "use ISCXVPN2016 instead of UNSW-NB15",
            "cap to 7B model",
        ],
    }))


# ---------------------------------------------------------------------------
# Single-call helpers
# ---------------------------------------------------------------------------

class TestSingleCalls:
    def test_advocate_parses(self):
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   return_value=_adv_response(rate=0.4)):
            out = _run_advocate(_proposal(), MagicMock(), [])
        assert out is not None
        assert out["rate"] == 0.4
        assert "novel" in out["argument"]

    def test_advocate_rate_clipped(self):
        bad = MagicMock(content=_json.dumps({
            "argument": "x", "estimated_acceptance_rate": 99.0,
            "key_strengths": [],
        }))
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   return_value=bad):
            out = _run_advocate(_proposal(), MagicMock(), [])
        assert out["rate"] == 1.0   # clipped

    def test_advocate_failure_returns_none(self):
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   side_effect=RuntimeError("boom")):
            out = _run_advocate(_proposal(), MagicMock(), [])
        assert out is None

    def test_advocate_uses_revisions_in_user_msg(self):
        captured: list = []

        def fake(llm, msgs):
            captured.append(msgs)
            return _adv_response()

        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   side_effect=fake):
            _run_advocate(_proposal(), MagicMock(),
                           ["use ISCXVPN2016", "cap to 7B"])
        user_content = captured[0][1].content
        assert "ISCXVPN2016" in user_content
        assert "上一轮" in user_content or "Judge" in user_content

    def test_challenger_parses(self):
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   return_value=_chal_response()):
            out = _run_challenger(_proposal(), "advocate text", MagicMock())
        assert out is not None
        assert "ACD-WFP" in out["collisions"][0]

    def test_judge_parses(self):
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   return_value=_judge_response(rate=0.20)):
            out = _run_judge(_proposal(), "adv", "chal", MagicMock())
        assert out is not None
        assert out["acceptance_rate"] == 0.20
        assert len(out["revisions"]) == 2


# ---------------------------------------------------------------------------
# debate_proposal end-to-end
# ---------------------------------------------------------------------------

class TestDebateE2E:
    def _mock_round(self, advocate_rate=0.30, judge_rate=0.25):
        """返回一个 side_effect 列表，模拟一轮三个 call"""
        return [
            _adv_response(rate=advocate_rate),
            _chal_response(),
            _judge_response(rate=judge_rate),
        ]

    def test_two_rounds_no_convergence(self):
        """rate 不收敛 → 跑满 max_rounds=2"""
        side_effects = (
            self._mock_round(advocate_rate=0.30, judge_rate=0.25)
            + self._mock_round(advocate_rate=0.40, judge_rate=0.45)  # delta=0.20
        )
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   side_effect=side_effects):
            result = debate_proposal(_proposal(), MagicMock(), max_rounds=2)
        assert len(result.rounds) == 2
        assert result.final_acceptance_rate == 0.45

    def test_early_convergence(self):
        """rate ≥ threshold AND |delta| < convergence_delta → 提前停 (但 max_rounds=2 至少跑 2 轮才能比 delta)"""
        side_effects = (
            self._mock_round(advocate_rate=0.40, judge_rate=0.40)
            + self._mock_round(advocate_rate=0.42, judge_rate=0.42)  # delta=0.02
        )
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   side_effect=side_effects):
            result = debate_proposal(
                _proposal(), MagicMock(), max_rounds=5,
                acceptance_threshold=0.30, convergence_delta=0.05,
            )
        # 跑了 2 轮（第二轮收敛）
        assert len(result.rounds) == 2
        assert result.converged is True
        assert result.final_acceptance_rate == 0.42

    def test_advocate_failure_aborts_round(self):
        """advocate 失败 → 该轮 break，前面累计的轮次保留"""
        side_effects = [
            *self._mock_round(advocate_rate=0.20, judge_rate=0.20),
            RuntimeError("LLM error"),  # 第二轮 advocate 抛错
        ]
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   side_effect=side_effects):
            result = debate_proposal(_proposal(), MagicMock(), max_rounds=3)
        assert len(result.rounds) == 1
        assert result.final_acceptance_rate == 0.20

    def test_all_failure_returns_empty(self):
        """全部失败 → 空 rounds，default rate=0"""
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   side_effect=RuntimeError("boom")):
            result = debate_proposal(_proposal(), MagicMock(), max_rounds=2)
        assert result.rounds == []
        assert result.final_acceptance_rate == 0.0
        assert result.converged is False

    def test_below_threshold_not_converged(self):
        """rate 稳定但低于 threshold → not converged"""
        side_effects = (
            self._mock_round(advocate_rate=0.10, judge_rate=0.10)
            + self._mock_round(advocate_rate=0.10, judge_rate=0.10)  # delta=0
        )
        with patch("darwinian.agents.proposal_debater.invoke_with_retry",
                   side_effect=side_effects):
            result = debate_proposal(
                _proposal(), MagicMock(), max_rounds=2,
                acceptance_threshold=0.30,
            )
        assert result.final_acceptance_rate == 0.10
        assert result.converged is False
