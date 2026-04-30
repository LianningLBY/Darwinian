"""
mechanism_alignment_checker (R11) 单元测试

不打真实 LLM —— mock invoke_with_retry，验证：
- _detect_cross_domain: 关键词正反例 + 中文
- _parse_dimensions: 5 维度解析、清洗、固定顺序
- _derive_overall: 全 aligned / ≥3 broken / 混合 → 三档兜底
- check_mechanism_alignment: 端到端 mock 流程、pre-filter、is_cross_domain=False 短路
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest

from darwinian.agents.mechanism_alignment_checker import (
    _CROSS_DOMAIN_PATTERNS,
    _derive_overall,
    _detect_cross_domain,
    _parse_dimensions,
    check_mechanism_alignment,
)
from darwinian.state import (
    AbstractionBranch,
    MechanismAlignment,
    MechanismAlignmentDimension,
    ResearchProposal,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _proposal(motivation="m", method="meth", title="P"):
    return ResearchProposal(
        skeleton=AbstractionBranch(
            name="x", description="x", algorithm_logic="x", math_formulation="x",
        ),
        title=title, elevator_pitch="p",
        motivation=motivation, proposed_method=method,
        technical_details="td",
    )


def _dim(name, verdict, expl="x"):
    return {"dimension": name, "verdict": verdict, "explanation": expl}


# ---------------------------------------------------------------------------
# _detect_cross_domain
# ---------------------------------------------------------------------------

class TestDetectCrossDomain:
    def test_inspired_by_triggers(self):
        assert _detect_cross_domain(
            "Our method is inspired by quantum error correction", ""
        )

    def test_derived_from_triggers(self):
        assert _detect_cross_domain(
            "We derive an accuracy bound directly from quantum decoder theory", ""
        )

    def test_analog_to_triggers(self):
        assert _detect_cross_domain(
            "By framing this as an analogy to encrypted traffic classification", ""
        )

    def test_transfer_from_triggers(self):
        assert _detect_cross_domain(
            "We transfer the accuracy-efficiency mechanism from surface code decoders", ""
        )

    def test_chinese_keywords(self):
        assert _detect_cross_domain("受量子纠错码启发，我们...", "")
        assert _detect_cross_domain("借鉴热力学第二定律", "")
        assert _detect_cross_domain("跨域迁移机制", "")

    def test_pure_incremental_no_trigger(self):
        """LayerSkip extends EAGLE 这种同领域改进不应触发"""
        assert not _detect_cross_domain(
            "We extend EAGLE-2 by adding a new draft model selection heuristic, "
            "improving acceptance rate by 12%.",
            "",
        )

    def test_method_field_also_checked(self):
        """trigger 在 proposed_method 而非 motivation 也要 catch"""
        assert _detect_cross_domain(
            "We propose SSMF for traffic classification.",
            "Inspired by surface code decoders, SSMF...",
        )

    # R13: implicit patterns（v3 LIVE 漏抓的措辞）

    def test_face_similar_implicit(self):
        """v3 LIVE 实测漏抓的 case：'face similar capacity-dependent vulnerabilities'"""
        assert _detect_cross_domain(
            "encrypted traffic models face similar capacity-dependent "
            "vulnerabilities under concept drift",
            "",
        )

    def test_exhibit_similar_implicit(self):
        assert _detect_cross_domain("transformers exhibit similar attention patterns", "")

    def test_suffer_similar_implicit(self):
        assert _detect_cross_domain("larger models suffer similar degradation", "")

    def test_same_mechanism_applies_implicit(self):
        assert _detect_cross_domain(
            "We argue the same mechanism applies in network classification", "",
        )

    def test_parallel_observation_implicit(self):
        assert _detect_cross_domain(
            "We make parallel observations on encrypted traffic", "",
        )

    def test_analogous_behavior_implicit(self):
        assert _detect_cross_domain(
            "Networks exhibit analogous behavior to attention layers", "",
        )

    def test_as_shown_in_we_hypothesize_implicit(self):
        """间接 transfer 措辞：'as shown in [paper], we hypothesize ...'"""
        assert _detect_cross_domain(
            "As shown in arxiv:2303.01037, we hypothesize that traffic models "
            "exhibit comparable degradation",
            "",
        )

    def test_we_hypothesize_similar_implicit(self):
        assert _detect_cross_domain(
            "we hypothesize traffic classifiers will show similar capacity scaling", "",
        )

    def test_chinese_implicit(self):
        assert _detect_cross_domain("我们认为同样的机制适用于加密流量", "")
        assert _detect_cross_domain("观察到平行观察现象", "")

    def test_pure_extends_no_trigger_against_implicit(self):
        """同领域 incremental 不应被 implicit pattern 误触发"""
        # "extends EAGLE" / "improves acceptance" 不含 implicit 暗示
        assert not _detect_cross_domain(
            "We extend EAGLE by adding a new draft selection heuristic that "
            "improves acceptance rate by 12% on standard benchmarks.",
            "",
        )


# ---------------------------------------------------------------------------
# _parse_dimensions
# ---------------------------------------------------------------------------

class TestParseDimensions:
    def test_valid_dimension(self):
        out = _parse_dimensions([
            _dim("formal_correspondence", "broken", "Hilbert space ≠ Euclidean"),
        ])
        assert len(out) == 1
        assert out[0].dimension == "formal_correspondence"
        assert out[0].verdict == "broken"

    def test_invalid_dimension_dropped(self):
        out = _parse_dimensions([_dim("moonbeam", "aligned", "x")])
        assert out == []

    def test_invalid_verdict_dropped(self):
        out = _parse_dimensions([_dim("formal_correspondence", "perfect", "x")])
        assert out == []

    def test_empty_explanation_dropped(self):
        out = _parse_dimensions([_dim("formal_correspondence", "aligned", "")])
        assert out == []

    def test_fixed_order(self):
        """LLM 给乱序，输出按 formal → assumption → metric → invariant → scaling"""
        out = _parse_dimensions([
            _dim("scaling_correspondence", "loose", "x"),
            _dim("formal_correspondence", "broken", "x"),
            _dim("metric_correspondence", "aligned", "x"),
            _dim("assumption_correspondence", "loose", "x"),
            _dim("invariant_correspondence", "broken", "x"),
        ])
        assert [d.dimension for d in out] == [
            "formal_correspondence",
            "assumption_correspondence",
            "metric_correspondence",
            "invariant_correspondence",
            "scaling_correspondence",
        ]

    def test_dedup_same_dimension(self):
        """同 dimension 出现多次 → 后者覆盖前者（last wins）"""
        out = _parse_dimensions([
            _dim("formal_correspondence", "broken", "first"),
            _dim("formal_correspondence", "aligned", "second"),
        ])
        assert len(out) == 1
        assert out[0].verdict == "aligned"
        assert out[0].explanation == "second"

    def test_non_dict_dropped(self):
        out = _parse_dimensions(["not a dict",
                                 _dim("formal_correspondence", "broken", "x")])
        assert len(out) == 1


# ---------------------------------------------------------------------------
# _derive_overall
# ---------------------------------------------------------------------------

class TestDeriveOverall:
    def test_empty_dims_not_applicable(self):
        assert _derive_overall([]) == "not_applicable"

    def test_all_aligned(self):
        dims = [
            MechanismAlignmentDimension(dimension="formal_correspondence",
                                          verdict="aligned", explanation="x"),
            MechanismAlignmentDimension(dimension="assumption_correspondence",
                                          verdict="aligned", explanation="x"),
        ]
        assert _derive_overall(dims) == "aligned"

    def test_three_broken_hand_waved(self):
        dims = [
            MechanismAlignmentDimension(dimension="formal_correspondence",
                                          verdict="broken", explanation="x"),
            MechanismAlignmentDimension(dimension="assumption_correspondence",
                                          verdict="broken", explanation="x"),
            MechanismAlignmentDimension(dimension="metric_correspondence",
                                          verdict="broken", explanation="x"),
            MechanismAlignmentDimension(dimension="invariant_correspondence",
                                          verdict="loose", explanation="x"),
            MechanismAlignmentDimension(dimension="scaling_correspondence",
                                          verdict="aligned", explanation="x"),
        ]
        assert _derive_overall(dims) == "hand_waved"

    def test_mostly_loose(self):
        dims = [
            MechanismAlignmentDimension(dimension="formal_correspondence",
                                          verdict="loose", explanation="x"),
            MechanismAlignmentDimension(dimension="assumption_correspondence",
                                          verdict="loose", explanation="x"),
            MechanismAlignmentDimension(dimension="metric_correspondence",
                                          verdict="aligned", explanation="x"),
        ]
        assert _derive_overall(dims) == "loose_analogy"


# ---------------------------------------------------------------------------
# check_mechanism_alignment
# ---------------------------------------------------------------------------

class TestCheckEndToEnd:
    def test_skip_filter_when_no_keyword_opt_in(self):
        """R13: pre-filter 改为 opt-in。显式 skip_if_no_cross_domain_keyword=True
        时无关键词不调 LLM。"""
        p = _proposal(motivation="LayerSkip extends EAGLE acceptance rate by 12%")
        with patch(
            "darwinian.agents.mechanism_alignment_checker.invoke_with_retry"
        ) as mock_llm:
            ma = check_mechanism_alignment(
                p, MagicMock(), skip_if_no_cross_domain_keyword=True,
            )
        assert mock_llm.call_count == 0
        assert ma is not None
        assert ma.is_cross_domain is False
        assert ma.overall_verdict == "not_applicable"

    def test_default_always_runs_llm(self):
        """R13: 默认 skip_if_no_cross_domain_keyword=False，无关键词也调 LLM"""
        p = _proposal(motivation="LayerSkip extends EAGLE acceptance rate by 12%")
        payload = {
            "is_cross_domain": False,
            "source_domain": "", "target_domain": "",
            "dimensions": [],
            "overall_verdict": "not_applicable",
            "recommendation": "no analogy",
        }
        mock_resp = MagicMock(content=_json.dumps(payload))
        with patch(
            "darwinian.agents.mechanism_alignment_checker.invoke_with_retry",
            return_value=mock_resp,
        ) as mock_llm:
            ma = check_mechanism_alignment(p, MagicMock())   # 用默认值
        assert mock_llm.call_count == 1
        assert ma.is_cross_domain is False

    def test_force_run_when_filter_disabled(self):
        """skip_if_no_cross_domain_keyword=False 时强制调 LLM"""
        p = _proposal(motivation="incremental improvement")
        payload = {
            "is_cross_domain": False,
            "source_domain": "",
            "target_domain": "",
            "dimensions": [],
            "overall_verdict": "not_applicable",
            "recommendation": "no analogy",
        }
        mock_resp = MagicMock(content=_json.dumps(payload))
        with patch(
            "darwinian.agents.mechanism_alignment_checker.invoke_with_retry",
            return_value=mock_resp,
        ) as mock_llm:
            ma = check_mechanism_alignment(
                p, MagicMock(), skip_if_no_cross_domain_keyword=False,
            )
        assert mock_llm.call_count == 1
        assert ma.is_cross_domain is False

    def test_quantum_to_traffic_hand_waved(self):
        """v2 LIVE 实测的红色 case：量子→加密流量。LLM 应判 hand_waved"""
        p = _proposal(motivation=(
            "We transfer the accuracy-efficiency tradeoff mechanism from surface code "
            "decoders to encrypted traffic classification under concept drift, derived "
            "from quantum error correction theory (arxiv:2203.15695)."
        ))
        payload = {
            "is_cross_domain": True,
            "source_domain": "quantum error correction",
            "target_domain": "encrypted traffic classification",
            "dimensions": [
                _dim("formal_correspondence", "broken",
                     "qubit Hilbert space ≠ continuous feature space; no homomorphism"),
                _dim("assumption_correspondence", "broken",
                     "Pauli channel assumes unitarity; NN are non-linear"),
                _dim("metric_correspondence", "broken",
                     "trace distance has no meaning on classifier output distributions"),
                _dim("invariant_correspondence", "loose",
                     "threshold theorem depends on stabilizer formalism"),
                _dim("scaling_correspondence", "loose",
                     "code distance d ≠ NN layer N"),
            ],
            "overall_verdict": "hand_waved",
            "recommendation": "重写 motivation：放弃量子类比，用 distribution shift 文献",
        }
        mock_resp = MagicMock(content=_json.dumps(payload))
        with patch(
            "darwinian.agents.mechanism_alignment_checker.invoke_with_retry",
            return_value=mock_resp,
        ):
            ma = check_mechanism_alignment(p, MagicMock())
        assert ma.is_cross_domain is True
        assert ma.overall_verdict == "hand_waved"
        assert ma.source_domain == "quantum error correction"
        assert len(ma.dimensions) == 5
        # 维度顺序固定
        assert [d.dimension for d in ma.dimensions][0] == "formal_correspondence"
        assert "重写 motivation" in ma.recommendation

    def test_invalid_overall_falls_back_to_derived(self):
        p = _proposal(motivation="inspired by thermodynamics")
        payload = {
            "is_cross_domain": True,
            "source_domain": "thermodynamics",
            "target_domain": "attention",
            "dimensions": [
                _dim("formal_correspondence", "broken", "x"),
                _dim("assumption_correspondence", "broken", "x"),
                _dim("metric_correspondence", "broken", "x"),
            ],
            "overall_verdict": "yolo",  # 非法
            "recommendation": "x",
        }
        mock_resp = MagicMock(content=_json.dumps(payload))
        with patch(
            "darwinian.agents.mechanism_alignment_checker.invoke_with_retry",
            return_value=mock_resp,
        ):
            ma = check_mechanism_alignment(p, MagicMock())
        # 3 个 broken → derive 出 hand_waved
        assert ma.overall_verdict == "hand_waved"

    def test_llm_failure_returns_none(self):
        p = _proposal(motivation="inspired by quantum")
        with patch(
            "darwinian.agents.mechanism_alignment_checker.invoke_with_retry",
            side_effect=RuntimeError("boom"),
        ):
            ma = check_mechanism_alignment(p, MagicMock())
        assert ma is None

    def test_malformed_json_returns_none(self):
        p = _proposal(motivation="inspired by quantum")
        mock_resp = MagicMock(content="not json {{")
        with patch(
            "darwinian.agents.mechanism_alignment_checker.invoke_with_retry",
            return_value=mock_resp,
        ):
            ma = check_mechanism_alignment(p, MagicMock())
        assert ma is None
