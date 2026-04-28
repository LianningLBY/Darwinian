"""
Cross-Paper Contradiction Detector (R9d) — 扫 PaperEvidence.quantitative_claims，
找同 metric + 重叠 setting 但数值显著分歧的对，emit Phenomenon(type='cross_paper_contradiction').

设计动机：phenomenon_miner v1 的 docstring 已声明 cross_paper_contradiction 是更深的
idea seed（"为什么同样跑 MATH，A 报 5.54 PPL 而 B 报 7.2 PPL"），但 v1 留下未实现。
本模块补上 — 纯规则、零 LLM call，O(N²) 但 N 通常 ≤ 30 篇。

策略：
  1. 归一化 metric_name（lowercase, strip 'rate'/'ratio'/'score' 等修饰，去标点）
  2. 解析 metric_value 为单一浮点（取范围中点；剥 'x'/'×'/'%'）
  3. 按 metric 分组；组内做 pairwise compare：
     - 必须不同 paper_id
     - settings 必须 "overlap"（共享至少一个 ≥3 字符的 token，或两边都为空时不算）
     - 相对差 |a-b|/max(a,b) ≥ threshold（默认 0.30）
  4. 命中 → Phenomenon(type='cross_paper_contradiction', paper_ids=[a,b])

为什么纯规则：
  - LLM call 太贵 / 慢（N 篇论文有 N² 对，30 篇 = 870 对）
  - 矛盾的判定标准本身简单：同名 metric + 同 setting + 数值差 30%+
  - 误报由下游 elaborator 决定是否采用（不阻塞 pipeline）

为什么 setting overlap 必须强制：
  - 不同模型 / 不同 dataset 上的 metric 数值差异是 expected，不是矛盾
  - 没有 overlap check 会把 "Llama-7B speedup 2x" vs "GPT-4 speedup 5x" 当矛盾
"""

from __future__ import annotations

import re

from darwinian.state import PaperEvidence, Phenomenon


# 归一化时去掉的常见修饰词
_METRIC_NOISE = {
    "rate", "ratio", "score", "value", "metric", "result",
    "performance", "average", "avg", "mean",
}
# 解析 value 的正则：抓首个 数字 / 范围 数字-数字
_VALUE_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*[\-–]\s*(\d+(?:\.\d+)?))?")
# 抽 setting 关键 token（≥3 char 字母数字）
_SETTING_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _normalize_metric_name(name: str) -> str:
    """lowercase + 去常见修饰词 + 折叠空格"""
    if not name:
        return ""
    # 去标点
    cleaned = re.sub(r"[^\w\s\-]", " ", name.lower())
    tokens = [t for t in cleaned.split() if t and t not in _METRIC_NOISE]
    return " ".join(tokens).strip()


def _parse_value_to_float(value_str: str) -> float | None:
    """
    把 metric_value 解成单一浮点。
    - "2.16-2.62x" → 2.39 (范围中点)
    - "85.3%" → 85.3
    - "5.54" → 5.54
    - "n/a" / 空 → None
    """
    if not value_str:
        return None
    m = _VALUE_NUM_RE.search(value_str)
    if not m:
        return None
    lo = float(m.group(1))
    if m.group(2):
        hi = float(m.group(2))
        return (lo + hi) / 2.0
    return lo


def _setting_tokens(setting: str) -> set[str]:
    """抽 setting 里的 ≥3 char 字母数字 token（小写）"""
    if not setting:
        return set()
    return {t.lower() for t in _SETTING_TOKEN_RE.findall(setting)}


def _settings_overlap(setting_a: str, setting_b: str) -> bool:
    """两个 setting 是否共享至少一个 token"""
    ta = _setting_tokens(setting_a)
    tb = _setting_tokens(setting_b)
    return bool(ta & tb)


def _is_divergent(val_a: float, val_b: float, threshold: float) -> bool:
    """相对差 ≥ threshold（避免被零除）"""
    denom = max(abs(val_a), abs(val_b))
    if denom == 0:
        return False
    return abs(val_a - val_b) / denom >= threshold


def detect_cross_paper_contradictions(
    paper_evidence: list[PaperEvidence],
    *,
    divergence_threshold: float = 0.30,
    max_per_metric: int = 2,
    max_total: int = 5,
) -> list[Phenomenon]:
    """
    扫所有 quantitative_claims 找跨论文数值矛盾。

    Args:
        paper_evidence: pack.paper_evidence
        divergence_threshold: 相对差阈值（默认 0.30 = 30%）。低于此值认为是
            正常的 noise / 不同 setting 引起的小差。
        max_per_metric: 每个归一化 metric 最多 emit 几条（防同一 metric 刷屏）
        max_total: 全局上限（避免 ResearchMaterialPack 被矛盾淹没）

    Returns:
        list[Phenomenon]，type 都是 'cross_paper_contradiction'，paper_ids 含双方。
        按差异度降序。
    """
    # Step 1: 拍平所有 claims，附带 paper_id
    claim_records: list[dict] = []
    for ev in paper_evidence:
        for c in ev.quantitative_claims:
            val = _parse_value_to_float(c.metric_value)
            if val is None:
                continue
            metric_norm = _normalize_metric_name(c.metric_name)
            if not metric_norm:
                continue
            claim_records.append({
                "paper_id": ev.paper_id,
                "metric_norm": metric_norm,
                "metric_value_raw": c.metric_value,
                "metric_value_num": val,
                "setting": c.setting,
            })

    # Step 2: 按 metric 分组
    groups: dict[str, list[dict]] = {}
    for r in claim_records:
        groups.setdefault(r["metric_norm"], []).append(r)

    # Step 3: 组内 pairwise 比较
    candidates: list[tuple[float, Phenomenon]] = []
    for metric_norm, recs in groups.items():
        if len(recs) < 2:
            continue
        emitted_for_metric = 0
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if a["paper_id"] == b["paper_id"]:
                    continue
                if not _settings_overlap(a["setting"], b["setting"]):
                    continue
                if not _is_divergent(
                    a["metric_value_num"], b["metric_value_num"], divergence_threshold,
                ):
                    continue
                rel_diff = abs(a["metric_value_num"] - b["metric_value_num"]) / max(
                    abs(a["metric_value_num"]), abs(b["metric_value_num"]),
                )
                desc = (
                    f"Same metric '{metric_norm}' on overlapping setting yields divergent "
                    f"values: {a['paper_id']} reports {a['metric_value_raw']} (setting: "
                    f"{a['setting'][:80]}) vs {b['paper_id']} reports {b['metric_value_raw']} "
                    f"(setting: {b['setting'][:80]}). Relative diff: {rel_diff*100:.0f}%."
                )
                quote = (
                    f"[{a['paper_id']}] {a['metric_value_raw']} | "
                    f"[{b['paper_id']}] {b['metric_value_raw']}"
                )
                question = (
                    f"Why does '{metric_norm}' diverge by {rel_diff*100:.0f}% across "
                    f"these papers — are evaluation protocols truly aligned, or is there "
                    f"an unreported setting that drives the gap?"
                )
                ph = Phenomenon(
                    type="cross_paper_contradiction",
                    description=desc[:500],
                    supporting_quote=quote[:500],
                    paper_ids=[a["paper_id"], b["paper_id"]],
                    suggested_question=question[:300],
                )
                candidates.append((rel_diff, ph))
                emitted_for_metric += 1
                if emitted_for_metric >= max_per_metric:
                    break
            if emitted_for_metric >= max_per_metric:
                break

    # Step 4: 按差异度降序，截 max_total
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [ph for _, ph in candidates[:max_total]]
