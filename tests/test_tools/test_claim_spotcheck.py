"""
claim_spotcheck 单元测试

完全离线 — 数字提取 + 集合差集，无 LLM 调用。
"""

from __future__ import annotations

import pytest

from darwinian.tools.claim_spotcheck import (
    extract_numbers,
    spot_check_motivation_numbers,
)
from darwinian.state import PaperEvidence, QuantitativeClaim


class TestExtractNumbers:
    def test_basic_decimals(self):
        nums = extract_numbers("DEL achieves 2.16-2.62x speedup")
        assert "2.16-2.62x" in nums

    def test_percentage(self):
        nums = extract_numbers("85.3% accuracy")
        assert "85.3%" in nums

    def test_multiple(self):
        nums = extract_numbers("5.54 vs 5.60 PPL on Llama-2-7B")
        assert "5.54" in nums
        assert "5.60" in nums

    def test_short_phase_numbers_filtered(self):
        """1, 2, 3 这种 phase 序号不抽（无单位 + 太短）"""
        nums = extract_numbers("Phase 1, 2, 3")
        # 1/2/3 不应进结果
        assert all(len(n) >= 3 or "x" in n or "%" in n for n in nums)

    def test_x_suffix_kept(self):
        """2x / 4× 这种短数字+单位要保留"""
        nums = extract_numbers("2x faster, 4× memory")
        # 保留含 x/× 的
        assert any("x" in n or "×" in n for n in nums)

    def test_empty(self):
        assert extract_numbers("") == []
        assert extract_numbers(None) == []


class TestSpotCheckMotivationNumbers:
    def _make_evidence(self, claims):
        """claims = [(metric_name, metric_value, setting), ...]"""
        return PaperEvidence(
            paper_id="x", title="t", short_name="s", venue="v",
            quantitative_claims=[
                QuantitativeClaim(metric_name=mn, metric_value=mv, setting=sv)
                for mn, mv, sv in claims
            ],
            headline_result="r", relation_to_direction="extends",
        )

    def test_v9_27pct_caught_as_unverified(self):
        """v9 实测：motivation 写 27%，evidence 只有 3.01-3.76x / 2.66-2.89x"""
        ev = self._make_evidence([
            ("speedup", "3.01-3.76x", "T=0"),
            ("speedup", "2.66-2.89x", "T=1"),
        ])
        motivation = "EAGLE drops from 3.01-3.76x to 2.66-2.89x — a 27% reduction"
        suspicious = spot_check_motivation_numbers(motivation, [ev])
        assert "27%" in suspicious

    def test_evidence_numbers_not_flagged(self):
        """motivation 含 evidence 也有的数字 → 不报"""
        ev = self._make_evidence([
            ("speedup", "2.16-2.62x", "Llama-3.1-8B"),
        ])
        motivation = "DEL achieves 2.16-2.62x speedup"
        suspicious = spot_check_motivation_numbers(motivation, [ev])
        assert "2.16-2.62x" not in suspicious

    def test_setting_string_also_searched(self):
        """metric_value 没有但 setting 里有也算找到"""
        ev = self._make_evidence([
            ("speedup", "2x", "Llama-3.1-8B"),
        ])
        motivation = "Llama-3.1-8B is the primary target"
        suspicious = spot_check_motivation_numbers(motivation, [ev])
        # "3.1" 在 setting 里
        assert "3.1" not in suspicious

    def test_headline_result_searched(self):
        ev = self._make_evidence([])
        ev.headline_result = "5.54 PPL improvement"
        motivation = "5.54 baseline"
        suspicious = spot_check_motivation_numbers(motivation, [ev])
        assert "5.54" not in suspicious

    def test_empty_evidence_all_suspicious(self):
        motivation = "DEL achieves 2.16x speedup with 85% accuracy"
        suspicious = spot_check_motivation_numbers(motivation, [])
        assert "2.16x" in suspicious
        assert "85%" in suspicious

    def test_empty_motivation_no_suspicious(self):
        suspicious = spot_check_motivation_numbers("", [self._make_evidence([])])
        assert suspicious == []

    def test_returns_sorted(self):
        """输出确定性：按字典序排"""
        ev = self._make_evidence([])
        motivation = "9.5x then 2.1x then 5.5x"
        suspicious = spot_check_motivation_numbers(motivation, [ev])
        assert suspicious == sorted(suspicious)
