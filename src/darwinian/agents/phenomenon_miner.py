"""
Phenomenon Miner — 从论文 full text 抽"未解释/意外/矛盾"现象。

设计动机：13 个 SOTA auto-research 系统都从概念组合生成 idea，导致 idea 普遍
"新颖但不可行" (Si et al ICLR 2025)。Phenomenon 是更强的 idea 来源——
真正的研究往往从"这个现象为什么会这样"开始。

策略：
- 单论文挖掘：每篇 full text → 1 次 LLM call → 抽 unexplained_trend / surprising_result
- 跨论文矛盾：扫 PaperEvidence.quantitative_claims，同 metric+相似 setting 但数值差异显著的对，
  作 cross_paper_contradiction（v1 暂只 schema 占位，cross-paper detection 留下迭代）

成本：~8 次 LLM 调用 / 一份 seed (top-K paper 各 1 次)，~30 秒
"""

from __future__ import annotations

import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import PaperEvidence, Phenomenon
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry


_VALID_TYPES = {"unexplained_trend", "surprising_result", "cross_paper_contradiction"}


SYSTEM_PROMPT = """你是科研现象挖掘专家。给定一篇论文的 title + abstract + 全文节选，
你要从中识别**真实存在的"未解释/意外/矛盾"现象**作为后续研究 idea 的 seed。

输出严格 JSON：
{
  "phenomena": [
    {
      "type": "unexplained_trend" | "surprising_result",
      "description": "一句话描述（≤ 60 词）含 model/dataset/具体数字",
      "supporting_quote": "原文引用（≤ 200 字），必须是论文里**实际出现的句子**",
      "suggested_question": "这个现象引出的 'why X happens?' 研究问题"
    },
    ...
  ]
}

【三种现象类型定义】
- unexplained_trend: 作者明确承认观察到 X 但**留给 future work**或**未深入解释**
  关键短语：'we leave investigation to future work' / 'further study is needed' /
            'the underlying reason is unclear'
- surprising_result: 反直觉发现，作者用 'surprisingly' / 'counterintuitively' /
            'against our expectation' 等明显标记
- cross_paper_contradiction: **本任务不抽这一类**（需要多篇论文交叉对比，由其他工具处理）

【关键硬约束】
1. supporting_quote **必须是论文原文**，不要改写或编造
2. 没找到真实现象就返空 phenomena: []，**不要硬凑**
3. description 必须含具体数字 / 模型名 / 数据集名（让现象可追溯）
4. 每篇论文最多抽 3 条，质量优于数量
5. 如果论文是"工程优化报告"类（无 surprising 结果，无 future work 段），返空

输出严格 JSON，不要 markdown 包裹。
"""


def mine_phenomena(
    paper_id: str,
    title: str,
    full_text: str,
    abstract: str,
    llm: BaseChatModel,
    *,
    max_per_paper: int = 3,
) -> list[Phenomenon]:
    """
    从单篇论文 full text 挖现象。

    Args:
        paper_id: 论文 ID（'arxiv:xxx' 或 's2:xxx'）
        title: 论文标题
        full_text: arxiv_latex_fetcher.render_for_llm 输出（method+experiments+conclusion）
        abstract: 论文摘要
        llm: 便宜 LLM（推荐 Haiku 级）
        max_per_paper: 每篇最多抽几条（默认 3）

    Returns:
        list[Phenomenon]，全部 type 在 _VALID_TYPES 里。LLM 失败时返 []。
    """
    if not full_text and not abstract:
        return []

    user_message = _build_user_message(paper_id, title, abstract, full_text)
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        raw = parse_llm_json(response.content)
    except Exception as e:
        print(f"[phenomenon_miner] {paper_id} 解析失败: {type(e).__name__}: {e}",
              file=sys.stderr)
        return []

    raw_phenomena = raw.get("phenomena") or []
    out: list[Phenomenon] = []
    for p in raw_phenomena:
        if len(out) >= max_per_paper:
            break
        if not isinstance(p, dict):
            continue
        ptype = str(p.get("type", "")).strip().lower()
        if ptype not in _VALID_TYPES:
            continue
        # cross_paper_contradiction 不在单论文挖掘范围
        if ptype == "cross_paper_contradiction":
            continue
        desc = str(p.get("description", "")).strip()
        quote = str(p.get("supporting_quote", "")).strip()
        if not desc or not quote:
            continue
        out.append(Phenomenon(
            type=ptype,
            description=desc,
            supporting_quote=quote[:500],   # 防 LLM 偶尔超长
            paper_ids=[paper_id],
            suggested_question=str(p.get("suggested_question", "")).strip(),
        ))
    return out


def batch_mine_phenomena(
    papers: list[dict],
    llm: BaseChatModel,
    *,
    full_text_provider=None,
    max_per_paper: int = 3,
) -> list[Phenomenon]:
    """
    批量挖：对每篇 paper 调一次 LLM。

    Args:
        papers: 列表 of {"paper_id": ..., "title": ..., "abstract": ...}
        llm: 便宜 LLM
        full_text_provider: callable(paper_id) -> str；None 时降级仅用 abstract
        max_per_paper: 每篇上限

    Returns:
        合并后的 list[Phenomenon]（不去重，由调用方决定）
    """
    all_phenomena: list[Phenomenon] = []
    for p in papers:
        full_text = ""
        if full_text_provider is not None:
            try:
                full_text = full_text_provider(p["paper_id"]) or ""
            except Exception:
                full_text = ""
        phenomena = mine_phenomena(
            paper_id=p["paper_id"],
            title=p.get("title", ""),
            full_text=full_text,
            abstract=p.get("abstract", ""),
            llm=llm,
            max_per_paper=max_per_paper,
        )
        all_phenomena.extend(phenomena)
    return all_phenomena


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def _build_user_message(
    paper_id: str,
    title: str,
    abstract: str,
    full_text: str,
) -> str:
    parts = [
        f"【paper_id】{paper_id}",
        f"\n【title】{title}",
        f"\n【abstract】\n{abstract or '(无)'}",
    ]
    if full_text:
        # full_text 可能很长，截到 ~20K 字符（足够 method+conclusion）
        parts.append(f"\n【full text 节选 (method+experiments+conclusion)】\n{full_text[:20000]}")
    else:
        parts.append("\n【说明】只有 abstract，无全文 → 仅抽 abstract 里的 surprising 标记")

    parts.append(
        "\n请按 SYSTEM_PROMPT 的 JSON schema 输出 phenomena 列表。"
        "宁缺勿滥——没真实现象就返 phenomena: []。"
        "supporting_quote 必须是原文实际句子。"
    )
    return "\n".join(parts)
