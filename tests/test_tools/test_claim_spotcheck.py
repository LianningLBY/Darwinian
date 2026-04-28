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


class TestPaperIdStripping:
    """Round 8: 抽数字前先 strip arxiv/S2 paper id"""

    def test_arxiv_id_not_extracted(self):
        """arxiv:2205.11916 不应被当 motivation 数字"""
        nums = extract_numbers("see arxiv:2205.11916 for details")
        assert "2205.11916" not in nums
        assert "11.916" not in nums   # 也不应被部分匹配

    def test_arxiv_bare_id_not_extracted(self):
        """裸 arxiv id (无前缀): 2404.16710"""
        nums = extract_numbers("paper 2404.16710 reports 78.7% accuracy")
        assert "2404.16710" not in nums
        assert "78.7%" in nums   # 真正的数字应保留

    def test_s2_hex_id_not_extracted(self):
        """S2 hex paperId 子串不被当数字: 047/093/789/810/2742"""
        # v10 实测的真实 paperId
        text = "see s2:1b6e810ce0afd0dd093f789d2b2742d047e316d5 for prior work"
        nums = extract_numbers(text)
        assert all(n not in ["047", "093", "789", "810", "2742", "316"]
                   for n in nums)

    def test_doi_not_extracted(self):
        nums = extract_numbers("DOI: 10.1145/3534678.3539107 reports 95% recall")
        assert "10.1145" not in str(nums)
        assert "95%" in nums

    def test_url_not_extracted(self):
        nums = extract_numbers("see https://arxiv.org/abs/2404.16710 for 2.5x speedup")
        assert "2404.16710" not in nums
        assert "2.5x" in nums

    def test_legitimate_numbers_still_caught(self):
        """v9 EGAT 的 27% cherry-pick 数字仍能被抓"""
        nums = extract_numbers(
            "see arxiv:2404.16710 — EAGLE drops from 3.01-3.76x to 2.66-2.89x, "
            "a 27% reduction"
        )
        assert "27%" in nums
        assert "3.01-3.76x" in nums
        assert "2.66-2.89x" in nums
        # arxiv ID 不在
        assert "2404.16710" not in nums


# ===========================================================================
# Round 9a: strip 模型规模 + training step + 范围倍数
# ===========================================================================

class TestStripModelSizes:
    def test_llama_size_not_extracted(self):
        nums = extract_numbers("Llama-7B achieves 78.7% on MultiArith")
        assert "7b" not in nums and "7B" not in nums
        assert "78.7%" in nums

    def test_decimal_size(self):
        nums = extract_numbers("GPT-2 1.5B and GPT-Neo 2.7B both fail")
        assert "1.5b" not in nums
        assert "2.7b" not in nums

    def test_size_range(self):
        """Llama 1-7B / PaLM 8-62B 这种范围"""
        nums = extract_numbers("PaLM 8-62B and 540B both improved")
        assert "8-62b" not in nums
        assert "540b" not in nums

    def test_uppercase_b(self):
        nums = extract_numbers("Mistral-7B and 13B variants")
        assert "7B" not in nums and "7b" not in nums
        assert "13B" not in nums and "13b" not in nums


class TestStripTrainingSteps:
    def test_steps_filtered(self):
        nums = extract_numbers("after 1000 steps and 4000 steps")
        assert "1000" not in nums
        assert "4000" not in nums

    def test_step_range(self):
        nums = extract_numbers("CoT degrades after 1000-2000 steps")
        assert "1000-2000" not in nums

    def test_epochs_also_caught(self):
        nums = extract_numbers("trained 50 epochs then evaluated")
        assert "50" not in nums

    def test_iterations_caught(self):
        nums = extract_numbers("ran 500 iterations on validation")
        assert "500" not in nums


class TestStripRangeMultipliers:
    def test_range_x_filtered(self):
        """2-4× / 3-5x 这种范围倍数（不是真 metric）"""
        nums = extract_numbers("explicit reasoning is 2-4× slower")
        assert "2-4×" not in nums
        assert "2-4x" not in nums

    def test_single_x_kept(self):
        """单个 2.5x speedup 仍保留（合法 metric）"""
        nums = extract_numbers("achieves 2.5x speedup over baseline")
        assert "2.5x" in nums


class TestV11RegressionAuditClean:
    """v11 实测的 unverified_numbers 这次应只剩真值得审的（去 1.5b/4000/2-4×）"""
    def test_v11_motivation_residue(self):
        """直接复现 v11 的 spot-check 输入 → 只剩百分数"""
        v11_motivation = (
            "Chain-of-thought boosts GSM8K by 17.9% (PaLM-540B) and 23.9% (LaMDA-137B). "
            "CoT-only fine-tuning degrades after 1000-2000 steps while ReAct continues to "
            "4000 steps on PaLM-8B/62B. "
            "Self-consistency shows +17.9% on PaLM-540B but only +3-6.8% on UL2-20B. "
            "Zero-shot-CoT on small models (GPT-2 1.5B: 2.2%, GPT-Neo 2.7B: 1.3%). "
            "explicit reasoning incurs 2-5× token overhead."
        )
        nums = extract_numbers(v11_motivation)
        # 这些 metadata 应被滤
        for noise in ["1.5b", "2.7b", "8b", "62b", "20b", "137b", "540b",
                      "1000-2000", "4000", "2-5×", "2-5x"]:
            assert noise not in [n.lower() for n in nums], f"误抽 {noise}"
        # 真 metric 仍保留
        kept = [n.lower() for n in nums]
        assert "17.9%" in kept
        assert "23.9%" in kept
        assert "2.2%" in kept
        assert "1.3%" in kept
