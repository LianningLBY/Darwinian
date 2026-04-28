"""
Claim Spot-Check (Pri-4) — 检查 proposal motivation 里的数字是否在 PaperEvidence 里有依据。

设计动机：v9 EGAT 实测 motivation 写 "27% reduction in EAGLE speedup"，但 paper_evidence
里只有 3.01-3.76x / 2.66-2.89x，没有 27%。LLM 自己算了一个 cherry-pick 数字
（计算方式还有问题，文中写的范围给出 4-29%，27% 是其中一个特定配置）。

策略 (non-blocking):
- 抽 motivation 段所有数字 token (含 X%/Yx/小数/范围)
- 检查每个是否在 paper_evidence.quantitative_claims 或 headline_result 里出现
- 不出现的 → 标到 ResearchProposal.unverified_numbers
- 仅警告，不触发 elaborator retry（避免合理派生数字被误杀）

为什么 non-blocking：
- LLM 可能合法做派生（avg(2.16, 2.62) = 2.39）
- 数字格式细微差异（"4×" vs "4x"）需 robust 匹配
- 触发 retry 容易陷入"refine 后还是有派生数字"循环
"""

from __future__ import annotations

import re

from darwinian.state import PaperEvidence


# 数字 token 正则：匹配
# - 整数 / 浮点：5, 5.54, 3.01
# - 范围：3.01-3.76, 5.54-5.60
# - 单位：x / × / % / B (billion params) / GB
# - acceptance ratio 风格：0.85
_NUMBER_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:[\-–]\d+(?:\.\d+)?)?(?:\s*[xX×%]|\s*B|\s*GB|\s*M|\s*hours?)?",
)


def extract_numbers(text: str) -> list[str]:
    """
    从文本提取数字 token。返回 normalized 字符串（小写、去内空格）。

    例：
      "DEL 2.16-2.62x speedup, 5.54 vs 5.60 PPL"
      → ["2.16-2.62x", "5.54", "5.60"]
    """
    if not text:
        return []
    raw = _NUMBER_PATTERN.findall(text)
    out = []
    for n in raw:
        norm = n.strip().lower().replace(" ", "")
        # 去掉太短的（1-2 位数 + 无单位 = 可能是 phase number / list index）
        if len(norm) < 3 and not any(c in norm for c in "x%×"):
            continue
        out.append(norm)
    return out


def spot_check_motivation_numbers(
    motivation: str,
    paper_evidence: list[PaperEvidence],
) -> list[str]:
    """
    返回 motivation 中出现但 paper_evidence 里"找不到"的数字 token 列表。

    匹配规则：motivation 里的数字 token 严格相等（normalize 后）出现在任何
    PaperEvidence.quantitative_claims[*].metric_value 或 headline_result 里
    → 算"找到"。

    Args:
        motivation: proposal.motivation 段
        paper_evidence: pack.paper_evidence

    Returns:
        list[str]: 可疑数字（不在 evidence 里的）。空 list 表示全部可追溯。
    """
    motivation_nums = set(extract_numbers(motivation))
    if not motivation_nums:
        return []

    evidence_nums: set[str] = set()
    for ev in paper_evidence:
        for c in ev.quantitative_claims:
            evidence_nums.update(extract_numbers(c.metric_value))
            evidence_nums.update(extract_numbers(c.setting))
        evidence_nums.update(extract_numbers(ev.headline_result))

    suspicious = sorted(motivation_nums - evidence_nums)
    return suspicious
