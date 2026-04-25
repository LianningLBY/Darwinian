"""
论文证据抽取器 (Phase A 核心抽取层)

职责：从单篇论文（abstract 或全文 LaTeX）抽取结构化证据 PaperEvidence。
设计借鉴 Microsoft Claimify pattern + LLM-NERRE schema-conditioned extraction。

跟现有 batch_extract_entities (knowledge_graph.py) 的关系：
  - batch_extract_entities: 浅抽取，仅 method/dataset/metric 名字 → 用于 ConceptGraph 实体表
  - paper_evidence_extractor: 深抽取，含精确数据 + 上下文 + 关系 → 用于 ResearchProposal

输入：
  - paper_id / title / abstract（必填）
  - full_text（可选，arxiv_latex_fetcher.render_for_llm 的输出）
  - direction（用于让 LLM 判 relation_to_direction）

输出：PaperEvidence 或 None（连续校验失败）
"""

from __future__ import annotations

import json as _json
import re
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from darwinian.state import PaperEvidence, QuantitativeClaim
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位结构化论文证据抽取专家。给定一篇论文的标题、摘要和（可选）全文，你要抽取出可追溯的精确数据。

输出严格 JSON，包含以下字段：

{
  "short_name": "论文/方法的口语化简称（如 'LayerSkip' / 'DEL' / 'QSpec'，不带描述词）",
  "venue": "发表会议/期刊（如 'ACL 2024' / 'arxiv preprint 2025'）。abstract 没说的，从 paper_id 推断 (arxiv id 头 4 位是 yymm)",
  "year": 2024,
  "category": "本论文所属研究类别（句子化短语，3-7 词），用于按类别分组研究现状。如 'Layer-skipping self-speculative methods'",
  "method_names": ["论文的具体方法名 list，全小写规范"],
  "datasets": ["benchmark 数据集名 list"],
  "metrics": ["评估指标名 list（不带值），如 'speedup', 'accuracy', 'PPL'"],
  "quantitative_claims": [
    {"metric_name": "speedup", "metric_value": "2.16-2.62x", "setting": "Llama-3.1-8B on MATH"},
    ...
  ],
  "headline_result": "**一句话**核心数据，用于研究现状里的简短引用，如 '2.16-2.62x speedup' 或 'reduces PPL from 5.60 to 5.54 on Llama-2-7B'。**必须含具体数字**",
  "limitations": ["论文自己承认的局限/缺陷 list"],
  "relation_to_direction": "本论文与给定研究方向的关系。**只能从这 5 个枚举里选一个**: 'extends' / 'baseline' / 'inspires' / 'orthogonal' / 'reproduces'"
}

【关键硬约束】
1. headline_result **必须包含至少一个数字**（速度倍数 / 精度百分比 / loss 数值等），不要写 "improves performance" 这种空话
2. quantitative_claims 至少 1 条；每条的 metric_value **必须含数字字面量**
3. relation_to_direction 五选一，写其他视为无效
4. 全文 LaTeX 提供时，优先从 Experiments / Results 节抽数字；只有 abstract 时尽力而为，但 headline_result 仍要求数字

输出严格 JSON，不要 markdown 包裹，不要任何额外说明。
"""


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def extract_evidence(
    paper_id: str,
    title: str,
    abstract: str,
    direction: str,
    llm: BaseChatModel,
    *,
    full_text: str = "",
    max_retries: int = 2,
) -> PaperEvidence | None:
    """
    从单篇论文抽 PaperEvidence。

    Args:
        paper_id: 'arxiv:2404.16710' 或 's2:xxx' 格式
        title / abstract: 必填
        direction: 当前研究方向，用于让 LLM 判 relation_to_direction
        llm: 主 LLM（推荐用 Opus / GPT-4 级别强模型）
        full_text: 可选 —— arxiv_latex_fetcher.render_for_llm 的输出（method+experiments+conclusion）
                   提供时 quantitative_claims 精度大幅提升
        max_retries: 校验失败带反馈重试次数（默认 2）

    Returns:
        PaperEvidence 或 None（解析/校验全部失败）
    """
    if not paper_id or not title:
        return None

    user_message = _build_user_message(paper_id, title, abstract, direction, full_text)
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    last_evidence: PaperEvidence | None = None
    for attempt in range(max_retries + 1):
        try:
            response = invoke_with_retry(llm, messages)
            raw = parse_llm_json(response.content)
            evidence = _build_evidence(raw, paper_id, title, full_text)
        except Exception as e:
            print(f"[paper_evidence_extractor] 第 {attempt+1}/{max_retries+1} 次解析失败: {type(e).__name__}: {e}")
            time.sleep(2 * (attempt + 1))
            continue

        errors = _validate(evidence)
        if not errors:
            return evidence

        last_evidence = evidence
        if attempt < max_retries:
            feedback = _build_feedback(errors)
            print(f"[paper_evidence_extractor] 校验失败 ({attempt+1}/{max_retries+1}): {[e[0] for e in errors]}")
            messages.append(response)
            messages.append(HumanMessage(content=feedback))

    return last_evidence


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

_VALID_RELATIONS = {"extends", "baseline", "inspires", "orthogonal", "reproduces"}


def _build_user_message(
    paper_id: str,
    title: str,
    abstract: str,
    direction: str,
    full_text: str,
) -> str:
    parts = [
        f"【当前研究方向】\n{direction}",
        f"\n【论文 paper_id】\n{paper_id}",
        f"\n【标题】\n{title}",
        f"\n【摘要】\n{abstract or '(无)'}",
    ]
    if full_text:
        parts.append(f"\n【全文（method+experiments+conclusion 节选）】\n{full_text}")
        parts.append("\n注意：抽 quantitative_claims 时**优先从全文 Experiments 节找具体数字**。")
    else:
        parts.append("\n注意：本论文只有 abstract（无全文），尽力从 abstract 抽数字。")

    parts.append(
        "\n请按 SYSTEM_PROMPT 的 JSON schema 输出。relation_to_direction 必须从 "
        "['extends', 'baseline', 'inspires', 'orthogonal', 'reproduces'] 中选。"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 解析 LLM 输出 → PaperEvidence
# ---------------------------------------------------------------------------

def _build_evidence(
    raw: dict,
    paper_id: str,
    title: str,
    full_text: str,
) -> PaperEvidence:
    """从 LLM 解析的 dict 组装 PaperEvidence"""
    claims_raw = raw.get("quantitative_claims") or []
    claims = []
    for c in claims_raw:
        if not isinstance(c, dict):
            continue
        claims.append(QuantitativeClaim(
            metric_name=str(c.get("metric_name", "")),
            metric_value=str(c.get("metric_value", "")),
            setting=str(c.get("setting", "")),
        ))

    return PaperEvidence(
        paper_id=paper_id,
        title=title,
        short_name=str(raw.get("short_name", "")),
        venue=str(raw.get("venue", "")),
        year=int(raw.get("year", 0) or 0),
        category=str(raw.get("category", "")),
        method_names=raw.get("method_names") or [],
        datasets=raw.get("datasets") or [],
        metrics=raw.get("metrics") or [],
        quantitative_claims=claims,
        headline_result=str(raw.get("headline_result", "")),
        limitations=raw.get("limitations") or [],
        relation_to_direction=str(raw.get("relation_to_direction", "")),
        full_text_used=bool(full_text),
    )


# ---------------------------------------------------------------------------
# 硬约束校验
# ---------------------------------------------------------------------------

# error code 规范:
#   HEADLINE_NO_NUMBER: headline_result 没含数字
#   NO_QUANT_CLAIMS:    quantitative_claims 为空
#   QUANT_VALUE_NO_NUMBER: 某个 claim 的 metric_value 没含数字
#   INVALID_RELATION:   relation_to_direction 不在 5 个枚举里
#   MISSING_SHORT_NAME: short_name 空


def _has_number(text: str) -> bool:
    """简单启发：text 含至少一个数字字面量（支持小数、范围、百分号、x 倍号）"""
    return bool(re.search(r"\d", text or ""))


def _validate(evidence: PaperEvidence) -> list[tuple]:
    errors: list[tuple] = []

    if not evidence.short_name.strip():
        errors.append(("MISSING_SHORT_NAME", None))

    if not _has_number(evidence.headline_result):
        errors.append(("HEADLINE_NO_NUMBER", evidence.headline_result))

    if not evidence.quantitative_claims:
        errors.append(("NO_QUANT_CLAIMS", None))
    else:
        for i, c in enumerate(evidence.quantitative_claims):
            if not _has_number(c.metric_value):
                errors.append(("QUANT_VALUE_NO_NUMBER", (i, c.metric_value)))

    if evidence.relation_to_direction not in _VALID_RELATIONS:
        errors.append(("INVALID_RELATION", evidence.relation_to_direction))

    return errors


def _build_feedback(errors: list[tuple]) -> str:
    """带具体错误 + 修正指引的反馈，塞回 prompt 让 LLM 重抽"""
    lines = ["你上次输出未通过校验，请按以下反馈修正后重出："]
    for code, detail in errors:
        if code == "MISSING_SHORT_NAME":
            lines.append("  ❌ short_name 不能为空。从标题/摘要里提取方法的口语化简称（如 'LayerSkip'）")
        elif code == "HEADLINE_NO_NUMBER":
            lines.append(
                f"  ❌ headline_result '{detail}' 没含数字。必须含具体数字，"
                f"如 '2.16-2.62x speedup' / 'reduces PPL from 5.60 to 5.54' / "
                f"'85.3% accuracy'。如果 abstract 里真的没数字，从全文 Experiments 节找"
            )
        elif code == "NO_QUANT_CLAIMS":
            lines.append("  ❌ quantitative_claims 不能为空。至少抽 1 条精确定量结果")
        elif code == "QUANT_VALUE_NO_NUMBER":
            i, val = detail
            lines.append(f"  ❌ quantitative_claims[{i}].metric_value='{val}' 没含数字")
        elif code == "INVALID_RELATION":
            lines.append(
                f"  ❌ relation_to_direction='{detail}' 不在枚举里。"
                f"只能从 [extends, baseline, inspires, orthogonal, reproduces] 五选一"
            )

    lines.append("\n请保持其他字段不变，只修正以上问题。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch 接口（多篇论文一次过）
# ---------------------------------------------------------------------------

def batch_extract_evidence(
    papers: list[dict],
    direction: str,
    llm: BaseChatModel,
    *,
    full_text_provider=None,
) -> list[PaperEvidence]:
    """
    批量抽取多篇论文的 PaperEvidence。

    Args:
        papers: 输入列表，每项是 {"paper_id": ..., "title": ..., "abstract": ...}
        direction: 当前研究方向
        llm: 主 LLM
        full_text_provider: 可选 callable(paper_id) -> str
            如果提供，对每篇论文调一次得到 full_text；返回空字符串 = 该论文降级 abstract-only
            典型用法：传入 lambda 调 arxiv_latex_fetcher + render_for_llm

    Returns:
        list[PaperEvidence]，长度与 papers 一致；失败的位置返 None
    """
    results: list[PaperEvidence] = []
    for p in papers:
        full_text = ""
        if full_text_provider is not None:
            try:
                full_text = full_text_provider(p["paper_id"]) or ""
            except Exception:
                full_text = ""

        evidence = extract_evidence(
            paper_id=p["paper_id"],
            title=p.get("title", ""),
            abstract=p.get("abstract", ""),
            direction=direction,
            llm=llm,
            full_text=full_text,
        )
        if evidence is not None:
            results.append(evidence)
    return results
