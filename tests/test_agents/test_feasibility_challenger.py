"""
feasibility_challenger 单元测试

不打真实 LLM —— mock invoke_with_retry，验证：
- _parse_risks: 合法/非法/缺字段 risk 解析
- _derive_verdict: 自动从 risks 推 verdict 的边界
- challenge_feasibility: 端到端 LLM mock 流程、严重性排序、verdict 兜底
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.feasibility_challenger import (
    _derive_verdict,
    _parse_risks,
    challenge_feasibility,
)
from darwinian.state import (
    AbstractionBranch,
    FeasibilityChallenge,
    FeasibilityRisk,
    MethodologyPhase,
    ResearchConstraints,
    ResearchProposal,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _proposal(title="P"):
    return ResearchProposal(
        skeleton=AbstractionBranch(
            name="x", description="x", algorithm_logic="x", math_formulation="x",
        ),
        title=title,
        elevator_pitch="pitch",
        motivation="motiv",
        proposed_method="method",
        technical_details="tech",
        methodology_phases=[
            MethodologyPhase(
                phase_number=1, name="phase1", description="d",
                expected_compute_hours=80.0,
            ),
        ],
        total_estimated_hours=80.0,
        fits_resource_budget=True,
        target_venue="NeurIPS 2026",
        target_deadline="2026-05-15",
    )


def _constraints():
    return ResearchConstraints(
        gpu_count=4, gpu_model="RTX PRO 6000",
        gpu_hours_budget=168.0, wall_clock_days=7,
        forbidden_techniques=["RLHF"],
    )


def _mock_llm(json_payload: dict):
    """构造 mock LLM 实例"""
    mock = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# _parse_risks
# ---------------------------------------------------------------------------

class TestParseRisks:
    def test_valid_risk_parsed(self):
        out = _parse_risks([{
            "category": "budget",
            "severity": "medium",
            "description": "phase 2 超预算 20h",
            "mitigation": "砍 ablation",
        }])
        assert len(out) == 1
        assert out[0].category == "budget"
        assert out[0].severity == "medium"
        assert "phase 2" in out[0].description

    def test_invalid_category_dropped(self):
        out = _parse_risks([{
            "category": "moonbeams",  # 非法
            "severity": "high",
            "description": "x",
        }])
        assert out == []

    def test_invalid_severity_dropped(self):
        out = _parse_risks([{
            "category": "budget",
            "severity": "critical",  # 非法
            "description": "x",
        }])
        assert out == []

    def test_empty_description_dropped(self):
        out = _parse_risks([{
            "category": "budget",
            "severity": "low",
            "description": "  ",
        }])
        assert out == []

    def test_non_dict_entry_dropped(self):
        out = _parse_risks([
            "not a dict",
            {"category": "budget", "severity": "low", "description": "ok"},
        ])
        assert len(out) == 1

    def test_sorted_by_severity_desc(self):
        out = _parse_risks([
            {"category": "budget", "severity": "low", "description": "L"},
            {"category": "budget", "severity": "high", "description": "H"},
            {"category": "budget", "severity": "medium", "description": "M"},
        ])
        assert [r.severity for r in out] == ["high", "medium", "low"]

    def test_description_truncated(self):
        out = _parse_risks([{
            "category": "budget", "severity": "low",
            "description": "x" * 600,
        }])
        assert len(out[0].description) <= 400

    def test_mitigation_optional(self):
        out = _parse_risks([{
            "category": "scope", "severity": "low",
            "description": "broad",
            # 无 mitigation
        }])
        assert out[0].mitigation == ""


# ---------------------------------------------------------------------------
# _derive_verdict
# ---------------------------------------------------------------------------

class TestDeriveVerdict:
    def test_no_risks_go(self):
        assert _derive_verdict([]) == "go"

    def test_only_low_go(self):
        risks = [FeasibilityRisk(category="budget", severity="low", description="d")]
        assert _derive_verdict(risks) == "go"

    def test_two_medium_go(self):
        risks = [
            FeasibilityRisk(category="budget", severity="medium", description="d"),
            FeasibilityRisk(category="data", severity="medium", description="d"),
        ]
        assert _derive_verdict(risks) == "go"

    def test_three_medium_mitigations(self):
        risks = [
            FeasibilityRisk(category="budget", severity="medium", description="d"),
            FeasibilityRisk(category="data", severity="medium", description="d"),
            FeasibilityRisk(category="scope", severity="medium", description="d"),
        ]
        assert _derive_verdict(risks) == "go_with_mitigations"

    def test_one_high_rework(self):
        risks = [
            FeasibilityRisk(category="dependency", severity="high", description="d"),
            FeasibilityRisk(category="data", severity="low", description="d"),
        ]
        assert _derive_verdict(risks) == "rework"


# ---------------------------------------------------------------------------
# challenge_feasibility (end-to-end with mock)
# ---------------------------------------------------------------------------

class TestChallengeFeasibility:
    def test_happy_path(self):
        payload = {
            "risks": [
                {
                    "category": "budget", "severity": "medium",
                    "description": "phase 2 超 20h",
                    "mitigation": "砍 ablation",
                },
            ],
            "overall_verdict": "go_with_mitigations",
            "summary": "可执行 但需要 ablation 缩减",
        }
        mock_response = MagicMock()
        mock_response.content = _json.dumps(payload)
        with patch(
            "darwinian.agents.feasibility_challenger.invoke_with_retry",
            return_value=mock_response,
        ):
            result = challenge_feasibility(_proposal(), _constraints(), MagicMock())
        assert isinstance(result, FeasibilityChallenge)
        assert len(result.risks) == 1
        assert result.overall_verdict == "go_with_mitigations"
        assert "ablation" in result.summary

    def test_invalid_verdict_falls_back_to_derived(self):
        payload = {
            "risks": [
                {"category": "dependency", "severity": "high",
                 "description": "deps unstable"},
            ],
            "overall_verdict": "ship_it",  # 非法
            "summary": "s",
        }
        mock_response = MagicMock()
        mock_response.content = _json.dumps(payload)
        with patch(
            "darwinian.agents.feasibility_challenger.invoke_with_retry",
            return_value=mock_response,
        ):
            result = challenge_feasibility(_proposal(), _constraints(), MagicMock())
        # 1 个 high → derive 出 rework
        assert result.overall_verdict == "rework"

    def test_llm_failure_returns_none(self):
        with patch(
            "darwinian.agents.feasibility_challenger.invoke_with_retry",
            side_effect=RuntimeError("boom"),
        ):
            result = challenge_feasibility(_proposal(), _constraints(), MagicMock())
        assert result is None

    def test_malformed_json_returns_none(self):
        mock_response = MagicMock()
        mock_response.content = "not json {{"
        with patch(
            "darwinian.agents.feasibility_challenger.invoke_with_retry",
            return_value=mock_response,
        ):
            result = challenge_feasibility(_proposal(), _constraints(), MagicMock())
        assert result is None

    def test_empty_risks_still_returns_challenge(self):
        payload = {"risks": [], "overall_verdict": "go", "summary": "clean"}
        mock_response = MagicMock()
        mock_response.content = _json.dumps(payload)
        with patch(
            "darwinian.agents.feasibility_challenger.invoke_with_retry",
            return_value=mock_response,
        ):
            result = challenge_feasibility(_proposal(), _constraints(), MagicMock())
        assert result is not None
        assert result.risks == []
        assert result.overall_verdict == "go"

    def test_severity_ordering_preserved(self):
        """LLM 给乱序 risks，输出必须按 severity 降序"""
        payload = {
            "risks": [
                {"category": "budget", "severity": "low", "description": "L"},
                {"category": "data", "severity": "high", "description": "H"},
                {"category": "scope", "severity": "medium", "description": "M"},
            ],
            "overall_verdict": "rework",
            "summary": "s",
        }
        mock_response = MagicMock()
        mock_response.content = _json.dumps(payload)
        with patch(
            "darwinian.agents.feasibility_challenger.invoke_with_retry",
            return_value=mock_response,
        ):
            result = challenge_feasibility(_proposal(), _constraints(), MagicMock())
        assert [r.severity for r in result.risks] == ["high", "medium", "low"]
