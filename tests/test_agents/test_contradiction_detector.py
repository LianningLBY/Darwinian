"""
contradiction_detector 单元测试 — 纯规则、零 LLM mock。

校验：
- _normalize_metric_name 折叠/去噪
- _parse_value_to_float 范围中点 / 单位 / 错误兜底
- _settings_overlap token 共享
- _is_divergent 阈值边界
- detect_cross_paper_contradictions 端到端
  * 同 paper 不算
  * setting 不重叠不算
  * 差异 < threshold 不算
  * 排序 + max_total 截取
"""

from __future__ import annotations

import pytest

from darwinian.agents.contradiction_detector import (
    _is_divergent,
    _normalize_metric_name,
    _parse_value_to_float,
    _settings_overlap,
    detect_cross_paper_contradictions,
)
from darwinian.state import PaperEvidence, QuantitativeClaim


def _ev(paper_id: str, claims: list[tuple[str, str, str]]) -> PaperEvidence:
    """方便构造 PaperEvidence: claims = [(metric_name, value, setting), ...]"""
    return PaperEvidence(
        paper_id=paper_id,
        title=f"Title {paper_id}",
        quantitative_claims=[
            QuantitativeClaim(metric_name=mn, metric_value=mv, setting=s)
            for mn, mv, s in claims
        ],
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestNormalizeMetricName:
    def test_lowercase(self):
        assert _normalize_metric_name("Speedup") == "speedup"

    def test_strip_modifiers(self):
        assert _normalize_metric_name("acceptance rate") == "acceptance"
        assert _normalize_metric_name("token acceptance ratio") == "token acceptance"

    def test_punctuation_stripped(self):
        assert _normalize_metric_name("top-1 accuracy") == "top-1 accuracy"
        # 标点变空格 后 split → 'ppl' + 'eval'，'eval' 不在 noise 列表所以保留
        assert _normalize_metric_name("PPL (eval)") == "ppl eval"

    def test_empty(self):
        assert _normalize_metric_name("") == ""

    def test_only_modifiers_returns_empty(self):
        assert _normalize_metric_name("rate ratio score") == ""


class TestParseValue:
    def test_simple_float(self):
        assert _parse_value_to_float("5.54") == 5.54

    def test_range_midpoint(self):
        assert _parse_value_to_float("2.16-2.62x") == pytest.approx(2.39)

    def test_unicode_dash_range(self):
        assert _parse_value_to_float("2.16–2.62x") == pytest.approx(2.39)

    def test_percent_stripped(self):
        assert _parse_value_to_float("85.3%") == 85.3

    def test_x_suffix(self):
        assert _parse_value_to_float("2.5x") == 2.5

    def test_no_number(self):
        assert _parse_value_to_float("n/a") is None
        assert _parse_value_to_float("") is None
        assert _parse_value_to_float("   ") is None


class TestSettingsOverlap:
    def test_overlap(self):
        assert _settings_overlap(
            "Llama-3.1-8B on MATH",
            "Llama-3.1-8B on GSM8K",
        )

    def test_no_overlap(self):
        assert not _settings_overlap(
            "Llama-3.1-8B on MATH",
            "GPT-4 on MMLU",
        )

    def test_both_empty_no_overlap(self):
        assert not _settings_overlap("", "")

    def test_one_empty_no_overlap(self):
        assert not _settings_overlap("Llama on MATH", "")

    def test_short_tokens_filtered(self):
        # "on" / "in" 这种 ≤2 char 不参与匹配
        assert not _settings_overlap("on a b", "in c d")


class TestIsDivergent:
    def test_just_below_threshold(self):
        # diff=0.6, max=2.6, ratio=0.23 → < 0.30 → not divergent
        assert not _is_divergent(2.0, 2.6, 0.30)

    def test_just_above_threshold(self):
        # diff=1.0, max=2.5, ratio=0.40 → ≥ 0.30 → divergent
        assert _is_divergent(1.5, 2.5, 0.30)

    def test_50pct_divergent(self):
        assert _is_divergent(2.0, 4.0, 0.30)  # ratio = 0.5

    def test_zero_ignored(self):
        assert not _is_divergent(0.0, 0.0, 0.30)

    def test_below_threshold(self):
        assert not _is_divergent(2.0, 2.05, 0.30)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

class TestDetectE2E:
    def test_contradiction_detected(self):
        evs = [
            _ev("p1", [("Speedup", "2.5x", "Llama-3.1-8B on MATH")]),
            _ev("p2", [("Speedup", "5.0x", "Llama-3.1-8B on MATH")]),
        ]
        out = detect_cross_paper_contradictions(evs, divergence_threshold=0.30)
        assert len(out) == 1
        assert out[0].type == "cross_paper_contradiction"
        assert set(out[0].paper_ids) == {"p1", "p2"}
        assert "speedup" in out[0].description.lower()

    def test_no_overlap_filtered(self):
        evs = [
            _ev("p1", [("Speedup", "2.5x", "Llama on MATH")]),
            _ev("p2", [("Speedup", "5.0x", "GPT-4 on MMLU")]),
        ]
        out = detect_cross_paper_contradictions(evs, divergence_threshold=0.30)
        assert out == []

    def test_below_threshold_filtered(self):
        evs = [
            _ev("p1", [("Speedup", "2.5x", "Llama-3.1-8B on MATH")]),
            _ev("p2", [("Speedup", "2.6x", "Llama-3.1-8B on MATH")]),
        ]
        out = detect_cross_paper_contradictions(evs, divergence_threshold=0.30)
        assert out == []

    def test_same_paper_skipped(self):
        evs = [
            _ev("p1", [
                ("Speedup", "2.5x", "Llama-8B on MATH"),
                ("Speedup", "5.0x", "Llama-8B on MATH"),  # 同 p1 内不同 config
            ]),
        ]
        out = detect_cross_paper_contradictions(evs, divergence_threshold=0.30)
        assert out == []

    def test_normalization_groups_metric_variants(self):
        """'Acceptance rate' 和 'acceptance' 应归到同组"""
        evs = [
            _ev("p1", [("Acceptance rate", "0.85", "Llama-8B on MATH")]),
            _ev("p2", [("acceptance", "0.40", "Llama-8B on MATH")]),
        ]
        out = detect_cross_paper_contradictions(evs, divergence_threshold=0.30)
        assert len(out) == 1
        assert set(out[0].paper_ids) == {"p1", "p2"}

    def test_max_total_truncates(self):
        """构造多个矛盾 metric，验证 max_total"""
        evs = []
        for i in range(8):
            evs.append(_ev(f"p{i}_a", [(f"metric{i}", "1.0", f"Llama-{i}B on MATH")]))
            evs.append(_ev(f"p{i}_b", [(f"metric{i}", "5.0", f"Llama-{i}B on MATH")]))
        out = detect_cross_paper_contradictions(
            evs, divergence_threshold=0.30, max_total=3,
        )
        assert len(out) == 3

    def test_empty_evidence(self):
        assert detect_cross_paper_contradictions([]) == []

    def test_unparseable_value_skipped(self):
        evs = [
            _ev("p1", [("Speedup", "n/a", "Llama-8B on MATH")]),
            _ev("p2", [("Speedup", "5.0x", "Llama-8B on MATH")]),
        ]
        out = detect_cross_paper_contradictions(evs)
        assert out == []

    def test_ranking_by_divergence(self):
        """大差异在前"""
        evs = [
            _ev("p1", [("MetricA", "1.0", "Llama-8B")]),
            _ev("p2", [("MetricA", "1.5", "Llama-8B")]),  # 33% diff
            _ev("p3", [("MetricB", "1.0", "Llama-8B")]),
            _ev("p4", [("MetricB", "10.0", "Llama-8B")]),  # 90% diff
        ]
        out = detect_cross_paper_contradictions(evs, divergence_threshold=0.30)
        assert len(out) == 2
        # MetricB 差异更大，应排第一
        assert "metricb" in out[0].description.lower()

    def test_max_per_metric_caps(self):
        """同 metric 的 candidate pair 上限"""
        evs = [
            _ev("p1", [("Speedup", "1.0x", "Llama-8B")]),
            _ev("p2", [("Speedup", "5.0x", "Llama-8B")]),
            _ev("p3", [("Speedup", "10.0x", "Llama-8B")]),  # 跟 p1 也是矛盾
            _ev("p4", [("Speedup", "20.0x", "Llama-8B")]),
        ]
        out = detect_cross_paper_contradictions(
            evs, divergence_threshold=0.30, max_per_metric=2,
        )
        assert len(out) == 2
        for ph in out:
            assert ph.type == "cross_paper_contradiction"
