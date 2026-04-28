"""
venue_deadlines 单元测试

完全离线 — 静态 lookup 表行为校验。
"""

from __future__ import annotations

import pytest

from darwinian.utils.venue_deadlines import (
    VENUE_DEADLINES,
    normalize_venue,
    lookup_deadline,
    is_deadline_correct,
)


class TestNormalizeVenue:
    def test_lower_and_strip(self):
        assert normalize_venue(" NeurIPS 2026 ") == "neurips 2026"

    def test_strip_punctuation(self):
        assert normalize_venue("NeurIPS-2026 (main)") == "neurips 2026 main"

    def test_remove_the_prefix(self):
        assert normalize_venue("The NeurIPS 2026") == "neurips 2026"

    def test_empty(self):
        assert normalize_venue("") == ""
        assert normalize_venue(None) == ""


class TestLookupDeadline:
    def test_neurips_exact_match(self):
        info = lookup_deadline("NeurIPS 2026")
        assert info is not None
        assert info["deadline"] == "2026-05-15"

    def test_emnlp_exact(self):
        info = lookup_deadline("EMNLP 2026")
        assert info["deadline"] == "2026-05-25"

    def test_contain_match_extra_words(self):
        """venue 含额外词如 'NeurIPS 2026 conference' 也能命中"""
        info = lookup_deadline("NeurIPS 2026 main conference")
        assert info is not None

    def test_unknown_venue_returns_none(self):
        assert lookup_deadline("ICASSP 2027") is None
        assert lookup_deadline("NotARealVenue") is None

    def test_empty_returns_none(self):
        assert lookup_deadline("") is None


class TestIsDeadlineCorrect:
    def test_unknown_venue_passes(self):
        ok, ref = is_deadline_correct("Unknown Venue", "2026-05-15")
        assert ok is True
        assert ref is None

    def test_within_tolerance_passes(self):
        # NeurIPS 2026 真实 5/15，写 5/13 差 2 天
        ok, ref = is_deadline_correct("NeurIPS 2026", "2026-05-13")
        assert ok is True

    def test_v9_neurips_wrong_caught(self):
        """v9 实测：NeurIPS 2026 写 2026-09-01，差 100+ 天"""
        ok, ref = is_deadline_correct("NeurIPS 2026", "2026-09-01")
        assert ok is False
        assert ref == "2026-05-15"

    def test_custom_tolerance(self):
        # 5/13 差 2 天 → 0 day tolerance 应失败
        ok, _ = is_deadline_correct("NeurIPS 2026", "2026-05-13", tolerance_days=0)
        assert ok is False

    def test_invalid_date_format_passes(self):
        """格式错误时放行（其他 validator 抓）"""
        ok, _ = is_deadline_correct("NeurIPS 2026", "not a date")
        assert ok is True

    def test_rolling_venue_always_passes(self):
        """TMLR 是 rolling，任何 deadline 都不抓"""
        ok, _ = is_deadline_correct("TMLR", "2030-01-01")
        assert ok is True

    def test_empty_claimed_passes(self):
        ok, _ = is_deadline_correct("NeurIPS 2026", "")
        assert ok is True
