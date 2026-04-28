"""
Venue deadline lookup table — 写死的顶会 deadline 数据。

设计动机：v9 实测 LLM 写 NeurIPS 2026 deadline 为 2026-09-01，但实际 NeurIPS
通常 5 月截稿（9 月是 camera-ready）。LLM 凭训练数据记忆 deadline 经常错。
本模块提供可信源头让 elaborator 校验/参考。

数据维护：每 6 个月手动更新一次。来源以 official call for papers 为准。
当前数据更新于：2026-04-27（基于 2025 deadline 推断 2026 周期）。

调用方：
- elaborator _validate_v3 用 lookup_deadline 比对 LLM 输出
- LLM prompt 可在 _build_user_message_v3 里塞 known_deadlines 当 hint
"""

from __future__ import annotations

from datetime import date


# 顶会 paper-submission deadline 对照表
# key 用 normalized name (lowercase, 去标点)
# value: (year, deadline_date, notes)
VENUE_DEADLINES: dict[str, dict] = {
    # ---- NeurIPS ----
    "neurips 2026": {
        "deadline": "2026-05-15",
        "notes": "abstract 5/8, full paper 5/15 (推断, 基于 2025=5/16)",
    },
    "neurips 2027": {
        "deadline": "2027-05-15",
        "notes": "推断, 跟历史保持 5 月中旬",
    },

    # ---- ICML ----
    "icml 2026": {
        "deadline": "2026-01-30",
        "notes": "已过 (推断, 基于 2025=1/30)",
    },
    "icml 2027": {
        "deadline": "2027-01-29",
        "notes": "推断",
    },

    # ---- ICLR ----
    "iclr 2026": {
        "deadline": "2025-10-01",
        "notes": "已过 (推断, 基于 2025=10/1)",
    },
    "iclr 2027": {
        "deadline": "2026-10-01",
        "notes": "推断",
    },

    # ---- ACL / EMNLP / NAACL via ARR ----
    # ARR (ACL Rolling Review) 周期：每两月一次，commit deadline 跟 venue 挂钩
    "acl 2026": {
        "deadline": "2026-02-15",
        "notes": "已过 (commit ARR Dec 2025 → ACL 2026)",
    },
    "emnlp 2026": {
        "deadline": "2026-05-25",
        "notes": "ARR May 2026 commit → EMNLP 2026",
    },
    "naacl 2026": {
        "deadline": "2025-10-15",
        "notes": "已过",
    },

    # ---- AAAI ----
    "aaai 2027": {
        "deadline": "2026-08-01",
        "notes": "推断, 基于 2026=8/1",
    },

    # ---- COLM ----
    "colm 2026": {
        "deadline": "2026-03-25",
        "notes": "已过 (推断)",
    },

    # ---- CVPR / ICCV / ECCV ----
    "cvpr 2026": {
        "deadline": "2025-11-14",
        "notes": "已过",
    },
    "iccv 2027": {
        "deadline": "2027-03-08",
        "notes": "推断 (ICCV 双年)",
    },

    # ---- TMLR / JMLR ----
    "tmlr": {
        "deadline": "rolling",
        "notes": "rolling review, 无 fixed deadline",
    },
}


def normalize_venue(venue: str) -> str:
    """归一化 venue 名字: 小写 + 去标点 + 去多余空格"""
    if not venue:
        return ""
    import re
    s = venue.lower().strip()
    s = re.sub(r"[,;\-\(\)\[\]]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # 去常见前缀
    for prefix in ("the ",):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def lookup_deadline(venue: str) -> dict | None:
    """
    根据 venue 名字查 deadline。fuzzy 匹配（contain 关系）。

    Args:
        venue: 如 "NeurIPS 2026" / "EMNLP 2026 (ARR)" / "neurips 2026 conference"
    Returns:
        {"deadline": "YYYY-MM-DD", "notes": "..."} 或 None（未知 venue）
    """
    norm = normalize_venue(venue)
    if not norm:
        return None
    # 精确命中
    if norm in VENUE_DEADLINES:
        return VENUE_DEADLINES[norm]
    # contain 命中（如 "neurips 2026 conference" 包含 "neurips 2026"）
    for key, info in VENUE_DEADLINES.items():
        if key in norm:
            return info
    return None


def is_deadline_correct(
    venue: str,
    claimed_deadline: str,
    *,
    tolerance_days: int = 30,
) -> tuple[bool, str | None]:
    """
    校验 LLM 输出的 deadline 是否在已知真实 deadline 的 ±tolerance_days 内。

    Args:
        venue: 如 "NeurIPS 2026"
        claimed_deadline: LLM 输出的 deadline (ISO 'YYYY-MM-DD')
        tolerance_days: 默认 30 天容差

    Returns:
        (is_ok, reference_deadline_or_none):
        - is_ok=True 表示通过校验（含未知 venue 默认放行 + tolerance 内）
        - is_ok=False 时 reference_deadline 提供真实 deadline 供反馈
    """
    info = lookup_deadline(venue)
    if info is None:
        return True, None   # 未知 venue 放行（不阻塞）
    truth = info.get("deadline", "")
    if truth == "rolling":
        return True, None   # rolling review 无 deadline
    if not claimed_deadline or not truth:
        return True, None
    try:
        claimed = date.fromisoformat(claimed_deadline.strip())
        actual = date.fromisoformat(truth)
    except (ValueError, TypeError):
        return True, None   # 格式错放行（其他 validator 会抓）
    diff = abs((claimed - actual).days)
    if diff > tolerance_days:
        return False, truth
    return True, None
