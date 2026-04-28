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

# Round 8/9a fix: 去掉非数据性数字模式
# 顺序敏感：先去最长 / 最具结构的，避免短模式提前匹配
_PAPER_ID_PATTERNS = [
    # arxiv id with optional version: 2205.11916, 2404.16710v2
    re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b"),
    # arxiv: / s2: 前缀 + hex
    re.compile(r"(?:s2|arxiv):[a-f0-9]{10,}", re.IGNORECASE),
    # 裸 hex paper id (≥16 hex chars，避免误杀短数字)
    re.compile(r"\b[a-f0-9]{16,}\b", re.IGNORECASE),
    # DOI: 10.xxxx/yyyy
    re.compile(r"\b10\.\d{4,9}/[\w.\-]+\b"),
    # URL fragments
    re.compile(r"https?://\S+"),

    # Round 9a: 模型规模 (Llama-7B / 1.5B / GPT-Neo-2.7B / 540B / 175B / 70b /
    #          1B/2B/13b/62B/175B), 也覆盖范围 (1-7B / 2-20B)
    re.compile(r"\b\d+(?:\.\d+)?(?:[\-–]\d+(?:\.\d+)?)?\s*[BMK]\b", re.IGNORECASE),
    # Round 9a: training steps (1000 steps / 1000-2000 steps / 4000-step)
    re.compile(
        r"\b\d+(?:[\-–]\d+)?[\-\s]*(?:steps?|epochs?|iter(?:ations)?)\b",
        re.IGNORECASE,
    ),
    # Round 9a: 倍数对比含范围 (2-4× / 2-5x), 仅丢含 hyphen 的范围避免误吞独立 "2.5x speedup"
    # 注意：× 是 Unicode 字符，不是 \w，所以末尾不能用 \b（会被立即匹配失败）
    re.compile(r"\b\d+[\-–]\d+\s*(?:[×]|x\b)", re.IGNORECASE),
    # Round 9a: 单独的 4000 / 1000 这种 step count 数字（无单位）
    # 范围 4000-step 上面已抓；这里抓"after 1000 steps" 中的 1000 (前面规则已抓)
    # 不再加规则避免误吞 motivation 真数字
]


def _strip_paper_ids(text: str) -> str:
    """先去掉 paperId / arxiv id / DOI / URL，避免它们的数字片段被误抽"""
    cleaned = text
    for pat in _PAPER_ID_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    return cleaned


def extract_numbers(text: str) -> list[str]:
    """
    从文本提取数字 token。返回 normalized 字符串（小写、去内空格）。

    Round 8 fix: 抽数字前先 strip paper IDs（v10 实测把 arxiv:2205.11916 的
    "2205.11916" 和 S2 hex paperId 的 "047/093/316/789/810/2742" 子串误抽
    成 motivation 数字）

    例：
      "DEL 2.16-2.62x speedup, 5.54 vs 5.60 PPL"
      → ["2.16-2.62x", "5.54", "5.60"]
      "see arxiv:2205.11916 for 78.7% accuracy"
      → ["78.7%"]  (2205.11916 被 strip 掉)
    """
    if not text:
        return []
    cleaned = _strip_paper_ids(text)
    raw = _NUMBER_PATTERN.findall(cleaned)
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
