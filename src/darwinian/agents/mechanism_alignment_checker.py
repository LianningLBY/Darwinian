"""
Mechanism Alignment Checker (R11) — adversarial pass on cross-domain analogies.

设计动机（这个洞是朋友 review 时抓出来的）：
- v2 LIVE 实测「量子纠错码 decoder 在 i.ni.d 噪声下 40-95% decrement → 加密流量
  classifier 在 concept drift 下也会 degrade，因此可以 transfer rMWPM 的 27-79%
  improvement 机制」这条 idea 工程上完全跑通：4×GPU 够、有数据、phase 时间够
- R9c (Feasibility Challenger) 只把它标 MEDIUM ("quantum analogy hand-waved")
- 但它真正的硬伤是**数学结构不对应**：
  * qubit Hilbert space ≠ 连续特征空间（无同构映射）
  * depolarizing/Pauli channel ≠ 分布漂移（measure 完全不同）
  * surface code threshold theorem ≠ NN 泛化（前者依赖 stabilizer formalism）
- R9c 攻击工程可行性 (能不能跑通)；本模块攻击科学合理性 (transfer 在数学上是否成立)
- 两者互补，缺一不可

为什么独有：HindSight 2026 / Si et al. (ICLR 2025) / DeepReview / Co-Scientist
都没专门做 cross-domain mechanism alignment。Darwinian 唯一差异化能力。

策略：
1. Pre-filter：仅在 motivation 含 cross-domain 关键词时触发 LLM call（省钱）
2. LLM 在 5 个维度独立打分：
   - formal_correspondence: A 域对象 ↔ B 域对象的同构/同态映射存在吗
   - assumption_correspondence: A 域关键假设在 B 域成立吗
   - metric_correspondence: A 域距离/范数在 B 域有意义吗
   - invariant_correspondence: A 域核心定理在 B 域可证吗
   - scaling_correspondence: A 域 scaling law 跟 B 域是否真结构相似
3. 综合 verdict 三档：aligned / loose_analogy / hand_waved
4. Non-blocking：只警告 PI，渲染到 seed.md
"""

from __future__ import annotations

import re
import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import (
    MechanismAlignment,
    MechanismAlignmentDimension,
    ResearchProposal,
)
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry


_VALID_DIMENSIONS = {
    "formal_correspondence",
    "assumption_correspondence",
    "metric_correspondence",
    "invariant_correspondence",
    "scaling_correspondence",
}
_VALID_DIM_VERDICTS = {"aligned", "loose", "broken"}
_VALID_OVERALL = {"aligned", "loose_analogy", "hand_waved", "not_applicable"}

# 探测 cross-domain 类比的关键词（中英混合，因为 elaborator 可能任一）
# 命中任一即触发 LLM check（false positive 也只是浪费 1 次 ¥0.05 LLM call）
_CROSS_DOMAIN_PATTERNS = [
    re.compile(r"\binspired by\b", re.IGNORECASE),
    # derive / derived / deriving from（动词时态全覆盖）
    re.compile(r"\bderiv(?:e|ed|es|ing) .{0,80}\bfrom\b", re.IGNORECASE),
    re.compile(r"\bbased on .{1,40} (theory|principle|framework|formalism)\b", re.IGNORECASE),
    re.compile(r"\banalog(?:ous|y) to\b", re.IGNORECASE),
    re.compile(r"\btransfer .{1,40} from\b", re.IGNORECASE),
    re.compile(r"\bcross[- ]domain\b", re.IGNORECASE),
    re.compile(r"\bborrow(?:ing|ed)? .{1,30} from\b", re.IGNORECASE),
    re.compile(r"\bdraw(?:s|n|ing)? .{1,30} from\b", re.IGNORECASE),
    re.compile(r"borrowing intuition", re.IGNORECASE),
    # 中文：'借鉴' / '启发' (含'受X启发'/'X启发的') / '类比' / '跨域' / '跨学科'
    re.compile(r"借鉴|启发|类比|跨域|跨学科"),
]


def _detect_cross_domain(motivation: str, proposed_method: str = "") -> bool:
    """
    Pre-filter: 文本里有没有 cross-domain analogy 关键词。
    命中即返 True (LLM 进一步判定是否真 cross-domain)。
    """
    text = f"{motivation}\n{proposed_method}"
    return any(pat.search(text) for pat in _CROSS_DOMAIN_PATTERNS)


_CHECKER_PROMPT = """你是科研 mechanism alignment 严苛 reviewer，专门攻击 cross-domain
类比的**数学结构对应**。你的任务不是质疑工程可行性（那是 Feasibility Challenger 的事），
而是问：**类比的数学基础站得住吗？**

【背景】
LLM-generated proposals 经常硬拗 cross-domain 类比来凑 novelty，例如：
- "我们将量子纠错码的 threshold theorem 用到神经网络泛化"
  问题：stabilizer formalism 不适用 NN，threshold theorem 依赖 Pauli channel
- "我们用热力学第二定律解释 transformer attention 的 information loss"
  问题：信息论 entropy 不等于热力学 entropy（不同 measure 空间）

【任务】
1. 先判定 motivation/method 里**是否真有 cross-domain 类比**。如果只是同领域的
   incremental work（如 LayerSkip extends EAGLE），输出 is_cross_domain=false 直接结束。
2. 如果真有 cross-domain 类比：
   - 标出 source_domain (借的那个域) 和 target_domain (要做的那个域)
   - 在 5 个维度独立打分：formal / assumption / metric / invariant / scaling
   - 综合判 verdict：aligned (有真数学对应) / loose_analogy (只是启发) /
     hand_waved (数学不对应)

【5 维度定义】
1. formal_correspondence — A 域的 state space / operator 跟 B 域的对象有没有
   同构/同态/可证 transfer 映射？
   * aligned: 存在显式 isomorphism / homomorphism / functor
   * loose: 概念上相似但映射只是隐喻（如 "noise channel" ↔ "data noise"）
   * broken: 数学结构不同（如 Hilbert space vs Euclidean，无映射）

2. assumption_correspondence — A 域核心假设（i.i.d. noise / unitarity / smoothness
   / convexity）在 B 域是否仍然成立？
   * aligned: 假设直接 transfer 或可证条件下 transfer
   * loose: 假设大致类似但定义域不同
   * broken: A 假设在 B 中明确不成立（如 unitarity 之于非线性 NN）

3. metric_correspondence — A 域的距离/范数（trace distance / fidelity /
   Wasserstein）在 B 域有定义且有意义吗？
   * aligned: 同 metric 直接可用 / 有 measurable equivalence
   * loose: 不同 metric 但定性方向一致
   * broken: A metric 在 B 中无意义或会误导（如 trace distance 之于 vector）

4. invariant_correspondence — A 域核心定理（threshold theorem / decoupling /
   conservation law）在 B 域是否还可证？
   * aligned: 定理直接 transfer 或在 B 域已被证（给 citation）
   * loose: 定理结论被借用作 hypothesis 但未证
   * broken: A 定理依赖的前提在 B 中不成立

5. scaling_correspondence — A 域的 scaling law（distance d / system size N）
   跟 B 域 (model_params / dataset_size) 是否真有结构相似？
   * aligned: 显式映射 + 可定量比较的 scaling exponent
   * loose: 都"随规模变好"但 exponent 来源不同
   * broken: A 的 scaling 跟 B 完全无对应（如 code distance 之于 layer number）

【综合 verdict 规则】
- 5 维度全 aligned → aligned
- ≥3 维度 broken → hand_waved
- 否则（混合，多 loose） → loose_analogy

【输出严格 JSON】
{
  "is_cross_domain": true | false,
  "source_domain": "...",   // is_cross_domain=false 时填空字符串
  "target_domain": "...",
  "dimensions": [            // is_cross_domain=false 时填 []
    {"dimension": "formal_correspondence", "verdict": "broken",
     "explanation": "..."},
    ...
  ],
  "overall_verdict": "aligned" | "loose_analogy" | "hand_waved" | "not_applicable",
  "recommendation": "..."
}

is_cross_domain=false → overall_verdict 必须是 not_applicable，dimensions=[]。
explanation 必须引用具体对象/算子/假设名。
recommendation 必须 actionable（具体怎么改 motivation）。
不要 markdown code fence。
"""


def check_mechanism_alignment(
    proposal: ResearchProposal,
    llm: BaseChatModel,
    *,
    skip_if_no_cross_domain_keyword: bool = True,
) -> MechanismAlignment | None:
    """
    对 single proposal 跑 mechanism alignment critique。

    Args:
        proposal: 通常是 tournament top-1
        llm: 推荐 cheap-mid（attacker prompt 不长）
        skip_if_no_cross_domain_keyword: True (默认) 时先 regex 探测；命中才调 LLM。
            False 时强制跑 LLM (用于测试或生产 paranoid mode)

    Returns:
        MechanismAlignment 或 None（LLM 失败 / 预过滤跳过）
    """
    # Pre-filter: 仅 motivation 含 cross-domain 关键词时调 LLM
    if skip_if_no_cross_domain_keyword:
        if not _detect_cross_domain(proposal.motivation, proposal.proposed_method):
            print(
                "[mechanism_alignment] skip: motivation 未检出 cross-domain 关键词",
                file=sys.stderr,
            )
            return MechanismAlignment(
                is_cross_domain=False,
                overall_verdict="not_applicable",
                recommendation="(无 cross-domain 类比 — 跳过)",
            )

    user_msg = (
        f"【Proposal title】\n{proposal.title}\n\n"
        f"【Motivation (摘)】\n{proposal.motivation[:1000]}\n\n"
        f"【Proposed method (摘)】\n{proposal.proposed_method[:600]}\n\n"
        f"【Technical details (摘)】\n{proposal.technical_details[:400]}\n\n"
        "请按 SYSTEM_PROMPT 输出 JSON。"
    )
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_CHECKER_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(
            f"[mechanism_alignment] 失败: {type(e).__name__}: {str(e)[:120]}",
            file=sys.stderr,
        )
        return None

    is_cross = bool(raw.get("is_cross_domain", False))
    if not is_cross:
        return MechanismAlignment(
            is_cross_domain=False,
            overall_verdict="not_applicable",
            recommendation=str(raw.get("recommendation", "")).strip()[:300],
        )

    dimensions = _parse_dimensions(raw.get("dimensions") or [])
    overall = str(raw.get("overall_verdict", "")).strip().lower()
    if overall not in _VALID_OVERALL:
        overall = _derive_overall(dimensions)
    return MechanismAlignment(
        is_cross_domain=True,
        source_domain=str(raw.get("source_domain", "")).strip()[:120],
        target_domain=str(raw.get("target_domain", "")).strip()[:120],
        dimensions=dimensions,
        overall_verdict=overall,
        recommendation=str(raw.get("recommendation", "")).strip()[:400],
    )


def _parse_dimensions(raw_dims: list) -> list[MechanismAlignmentDimension]:
    """parse + 清洗 LLM 给的 dimensions 列表，按固定 5 维度顺序排"""
    out_by_name: dict[str, MechanismAlignmentDimension] = {}
    for d in raw_dims:
        if not isinstance(d, dict):
            continue
        name = str(d.get("dimension", "")).strip().lower()
        verdict = str(d.get("verdict", "")).strip().lower()
        explanation = str(d.get("explanation", "")).strip()
        if name not in _VALID_DIMENSIONS:
            continue
        if verdict not in _VALID_DIM_VERDICTS:
            continue
        if not explanation:
            continue
        out_by_name[name] = MechanismAlignmentDimension(
            dimension=name,
            verdict=verdict,
            explanation=explanation[:400],
        )

    # 按固定顺序输出 (formal → assumption → metric → invariant → scaling)
    order = [
        "formal_correspondence",
        "assumption_correspondence",
        "metric_correspondence",
        "invariant_correspondence",
        "scaling_correspondence",
    ]
    return [out_by_name[k] for k in order if k in out_by_name]


def _derive_overall(dimensions: list[MechanismAlignmentDimension]) -> str:
    """LLM 没给合法 overall_verdict 时按 dimensions 自动推断"""
    if not dimensions:
        return "not_applicable"
    n_aligned = sum(1 for d in dimensions if d.verdict == "aligned")
    n_broken = sum(1 for d in dimensions if d.verdict == "broken")
    if n_aligned == len(dimensions):
        return "aligned"
    if n_broken >= 3:
        return "hand_waved"
    return "loose_analogy"
