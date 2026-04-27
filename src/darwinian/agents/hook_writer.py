"""
Hook Writer — 把 EntityPair 升级为 StructuralHoleHook（带叙事 + relation_type）。

设计动机：bottleneck_miner 算出来的 novel_pair_hints 只是两个 entity 名 + 一个 score。
elaborator 看到这个不知道为什么这个交叉值得做，写不出 QuantSkip 风格的
"X 没人跟 Y 比" 深度交叉论证。

本工具对 top-K EntityPair 各调一次便宜 LLM，结合两端的代表论文上下文，
让 LLM 写一句话 hook_text 并标 relation_type。

成本：~5 次 LLM 调用 / 一份 seed，~10 秒，0 次 S2 调用。
"""

from __future__ import annotations

import sys
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import (
    Entity,
    EntityPair,
    PaperInfo,
    StructuralHoleHook,
)
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry


_VALID_RELATIONS = {"divergence", "convergence", "transfer"}


SYSTEM_PROMPT = """你是科研策略专家。给定研究方向 + 一对"高频但从未共现的实体" + 各自代表论文，
你要写一句话研究 hook，解释为什么这个交叉值得做。

输出严格 JSON：
{
  "hook_text": "一句话 hook（30-80 词），含具体机制/对比/数字暗示",
  "relation_type": "divergence" | "convergence" | "transfer"
}

【relation_type 选择指引】
- divergence: 两端可能发散，需要对照测量（如"X 用 metric_A 优化，Y 用 metric_B 衡量，
  没人测 A 和 B 的 rank correlation 是否一致"）
- convergence: 两端可能等价，值得理论分析（如"X 和 Y 殊途同归，证明等价性能简化设计空间"）
- transfer:    一端方法迁移到另一端任务（如"X 在领域 A 用得很好，未试过用在 B 上"）

【hook_text 写作要求】
1. 含**机制级**对比（不是"X 和 Y 都很重要"这种空话）
2. 引用代表论文的具体方法/数据/局限（如 "RAMP 用 perplexity 测，DEL 用 acceptance rate"）
3. 长度 30-80 词，一句话或两句短句
4. 用英文写（与 elaborator 输出一致）

输出严格 JSON，不要 markdown 包裹。
"""


def write_structural_hole_hooks(
    pairs: list[EntityPair],
    entities: list[Entity],
    papers: list[PaperInfo],
    direction: str,
    llm: BaseChatModel,
    *,
    max_hooks: int = 5,
    max_retries: int = 1,
) -> list[StructuralHoleHook]:
    """
    对 top-K EntityPair 调 LLM 写 hook，返回 StructuralHoleHook 列表。

    Args:
        pairs: ConceptGraph.novel_pair_hints
        entities: ConceptGraph.entities（用于反查 entity → paper_ids）
        papers: ConceptGraph.papers（用于附 title 上下文给 LLM）
        direction: 研究方向原文（让 LLM 写 hook 时贴方向）
        llm: 便宜 LLM（推荐 Haiku 级）
        max_hooks: 最多生成多少 hook（pairs 已按 score 排序，取头）
        max_retries: relation_type 校验失败时重试次数（默认 1）

    Returns:
        list[StructuralHoleHook]，长度 ≤ min(len(pairs), max_hooks)。
        LLM 失败的 pair 会被跳过（不阻塞整体 pipeline）。
    """
    if not pairs:
        return []

    entity_by_name = {e.canonical_name: e for e in entities}
    paper_by_id = {p.paper_id: p for p in papers}

    hooks: list[StructuralHoleHook] = []
    for pair in pairs[:max_hooks]:
        hook = _write_single_hook(
            pair, entity_by_name, paper_by_id, direction, llm, max_retries,
        )
        if hook is not None:
            hooks.append(hook)
    return hooks


# ---------------------------------------------------------------------------
# 单 pair 处理
# ---------------------------------------------------------------------------

def _write_single_hook(
    pair: EntityPair,
    entity_by_name: dict[str, Entity],
    paper_by_id: dict[str, PaperInfo],
    direction: str,
    llm: BaseChatModel,
    max_retries: int,
) -> StructuralHoleHook | None:
    # 反查两端的代表 paper_ids
    paper_ids_a = _representative_paper_ids(pair.entity_a, entity_by_name, paper_by_id)
    paper_ids_b = _representative_paper_ids(pair.entity_b, entity_by_name, paper_by_id)

    user_message = _build_user_message(
        pair, paper_ids_a, paper_ids_b, paper_by_id, direction,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    for attempt in range(max_retries + 1):
        try:
            response = invoke_with_retry(llm, messages)
            raw = parse_llm_json(response.content)
            hook_text = str(raw.get("hook_text", "")).strip()
            relation_type = str(raw.get("relation_type", "")).strip().lower()
        except Exception as e:
            print(f"[hook_writer] {pair.entity_a} × {pair.entity_b} 解析失败"
                  f" ({attempt+1}/{max_retries+1}): {type(e).__name__}", file=sys.stderr)
            time.sleep(1.0 * (attempt + 1))
            continue

        # 校验
        if not hook_text:
            print(f"[hook_writer] {pair.entity_a} × {pair.entity_b}: hook_text 空",
                  file=sys.stderr)
            continue
        if relation_type not in _VALID_RELATIONS:
            if attempt < max_retries:
                # 反馈让 LLM 改 relation_type
                messages.append(response)
                messages.append(HumanMessage(content=(
                    f"relation_type='{relation_type}' 不在枚举里。"
                    f"只能从 [divergence, convergence, transfer] 三选一。"
                    f"保持 hook_text 不变，只改 relation_type 后重出 JSON。"
                )))
                continue
            print(f"[hook_writer] {pair.entity_a} × {pair.entity_b}: "
                  f"relation_type='{relation_type}' 仍非法，跳过", file=sys.stderr)
            return None

        return StructuralHoleHook(
            entity_a=pair.entity_a,
            entity_b=pair.entity_b,
            score=pair.score,
            hook_text=hook_text,
            relation_type=relation_type,
            supporting_paper_ids_a=paper_ids_a[:5],
            supporting_paper_ids_b=paper_ids_b[:5],
        )

    return None


def _representative_paper_ids(
    entity_name: str,
    entity_by_name: dict[str, Entity],
    paper_by_id: dict[str, PaperInfo],
    *,
    top_k: int = 5,
) -> list[str]:
    """
    取该 entity 出现在哪几篇 paper 里，按 citation_count 取 top-K。
    LLM 看到代表论文标题才能判 relation_type 准。
    """
    entity = entity_by_name.get(entity_name)
    if not entity:
        return []
    candidate_ids = entity.paper_ids[:20]
    candidates = [
        (pid, paper_by_id.get(pid).citation_count or 0)
        for pid in candidate_ids if pid in paper_by_id
    ]
    candidates.sort(key=lambda t: t[1], reverse=True)
    return [pid for pid, _ in candidates[:top_k]]


def _build_user_message(
    pair: EntityPair,
    paper_ids_a: list[str],
    paper_ids_b: list[str],
    paper_by_id: dict[str, PaperInfo],
    direction: str,
) -> str:
    a_lines = _render_paper_block(paper_ids_a, paper_by_id)
    b_lines = _render_paper_block(paper_ids_b, paper_by_id)
    return (
        f"【研究方向】\n{direction}\n\n"
        f"【实体对】\nentity_a: {pair.entity_a}\nentity_b: {pair.entity_b}\n"
        f"score (min papers each side): {pair.score}\n\n"
        f"【entity_a 的代表论文】\n{a_lines}\n\n"
        f"【entity_b 的代表论文】\n{b_lines}\n\n"
        "请按 SYSTEM_PROMPT 的 JSON schema 输出 hook_text + relation_type。"
        "hook_text 必须含机制级对比和具体论文方法引用，不要写空话。"
    )


def _render_paper_block(paper_ids: list[str], paper_by_id: dict[str, PaperInfo]) -> str:
    if not paper_ids:
        return "  （未找到代表论文）"
    lines = []
    for pid in paper_ids[:5]:
        p = paper_by_id.get(pid)
        if p is None:
            continue
        title = (p.title or "")[:80]
        lines.append(f"  - [{pid}] ({p.year or '?'}) {title}")
    return "\n".join(lines) or "  （论文 lookup 全部失败）"
