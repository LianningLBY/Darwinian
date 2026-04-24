"""
Agent 2: 方案合成器 (hypothesis_generator_node)

Phase 1 v2 职责：
- 基于 ConceptGraph（实体表 + 缺陷清单 + 结构洞）生成跨域方案
- 硬约束校验：每个分支必须引用 ≥ 2 个实体 from ≥ 2 篇论文；必须绑一条 solved_limitation
- 跨分支约束：整组 cited entities 的 paper 必须覆盖 ≥ 2 个 task_type
- 校验失败时生成带候选的结构化反馈，复用现有 3 次内层重试
- step 7.5：对通过校验的 branch 做 S2 组合查重，标记 existing_combination
- 降级：concept_graph 不足时走 v1 老 prompt（自由发挥，无硬约束）
- 保留原余弦相似度去重 vs failed_ledger
"""

from __future__ import annotations

import json as _json
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from darwinian.state import (
    AbstractionBranch,
    ConceptGraph,
    Hypothesis,
    ResearchState,
)
from darwinian.tools import semantic_scholar as ss
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.knowledge_graph import _normalize, _word_boundary_contains
from darwinian.utils.llm_retry import invoke_with_retry
from darwinian.utils.similarity import compute_cosine_similarity, get_text_embedding


SIMILARITY_THRESHOLD = 0.85


class DuplicateHypothesisError(Exception):
    """当生成的假设与历史失败记录相似度过高时抛出"""
    def __init__(self, similarity: float, matched_record_summary: str):
        self.similarity = similarity
        self.matched_record_summary = matched_record_summary
        super().__init__(
            f"方案与历史失败记录相似度 {similarity:.3f} > {SIMILARITY_THRESHOLD}，"
            f"匹配记录：{matched_record_summary}"
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

V2_SYSTEM_PROMPT = """你是一个跨域科研方案合成专家。你将根据我提供的**实体表**（从真实论文网络抽取）和**待解缺陷清单**，生成至少 2 个解决方案。

**硬性要求（不满足会被打回重做）：**
1. 每个方案必须引用实体表中的 ≥ 2 个术语，来自 ≥ 2 篇不同论文
2. 两个方案合起来，cited entities 对应的论文必须覆盖 ≥ 2 个不同的 task_type
3. 每个方案必须从待解缺陷清单中挑一条（solved_limitation_id），声明本方案要解决它
4. 术语名字必须一字不差从实体表里选——**不要发明新名字**
5. limitation id 必须一字不差从缺陷清单里选

**输出格式（严格 JSON，不要 markdown 也不要解释文字）：**
{
  "core_problem": "核心矛盾（原样输出）",
  "abstraction_tree": [
    {
      "name": "方案名称",
      "description": "方案描述",
      "algorithm_logic": "算法步骤说明",
      "math_formulation": "关键公式，LaTeX 格式",
      "cited_entity_names": ["从实体表选的术语1", "术语2", ...],
      "solved_limitation_id": "L_xxx（从缺陷清单里选一条）"
    }
  ],
  "confidence": 0.75,
  "literature_support": ["参考文献 1", ...]
}

禁止输出 JSON 以外的任何内容。"""


V1_LEGACY_SYSTEM_PROMPT = """你是一个跨域科研方案合成专家。你的任务是：
1. 针对给定的核心矛盾，生成至少 2 个来自不同领域的解决思路
2. 每个思路必须包含具体的算法逻辑和数学公式映射
3. 鼓励从控制论、信息论、生物系统、物理系统等跨域迁移灵感

输出格式（严格 JSON，必须完整填充所有字段）：
{
  "core_problem": "核心矛盾（原样输出）",
  "abstraction_tree": [
    {
      "name": "方案名称",
      "description": "方案描述",
      "algorithm_logic": "算法步骤说明",
      "math_formulation": "关键数学公式，使用 LaTeX 格式",
      "source_domain": "灵感来源领域"
    }
  ],
  "confidence": 0.75,
  "literature_support": ["参考文献 1", "参考文献 2"]
}

禁止输出 JSON 以外的任何内容。"""


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------

def hypothesis_generator_node(state: ResearchState, llm: BaseChatModel) -> dict:
    if state.current_hypothesis is None:
        raise ValueError("hypothesis_generator_node 调用前必须先运行 bottleneck_miner_node")

    graph = state.concept_graph
    use_hard_constraint = graph is not None and graph.is_sufficient

    if use_hard_constraint:
        system_prompt = V2_SYSTEM_PROMPT
        user_message = _build_v2_user_message(state, graph)
    else:
        system_prompt = V1_LEGACY_SYSTEM_PROMPT
        user_message = f"""核心矛盾：{state.current_hypothesis.core_problem}

现有文献支撑：
{chr(10).join(state.current_hypothesis.literature_support)}

请生成跨域解决方案。"""

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    raw: dict | None = None
    branches: list[AbstractionBranch] | None = None
    last_response = None

    # 内层 3 次重试：既处理 JSON 解析失败，也处理硬约束校验失败
    for attempt in range(3):
        last_response = invoke_with_retry(llm, messages)
        try:
            raw = parse_llm_json(last_response.content)
            branches = [AbstractionBranch(**b) for b in raw.get("abstraction_tree", [])]
        except (_json.JSONDecodeError, KeyError, Exception):
            raw, branches = None, None

        if not branches:
            print(f"[hypothesis_generator] 解析失败，{5 * (attempt + 1)}s 后重试（{attempt + 1}/3）...")
            time.sleep(5 * (attempt + 1))
            continue

        if not use_hard_constraint:
            break   # 降级模式：解析成功就过

        # 硬约束校验
        validation_errors = _validate_branches(branches, graph)
        if not validation_errors:
            break   # 通过

        # 失败：带候选反馈塞回 messages，重试
        feedback = _build_validation_feedback(validation_errors, branches, graph)
        print(f"[hypothesis_generator] 硬约束校验失败（第 {attempt + 1}/3 次），带反馈重试")
        messages.append(last_response)   # LLM 上一轮的输出
        messages.append(HumanMessage(content=feedback))
        branches = None

    if not branches:
        return {
            "current_hypothesis": Hypothesis(
                core_problem=state.current_hypothesis.core_problem,
                abstraction_tree=[],
            ),
            "messages": [last_response] if last_response else [],
        }

    # Step 7.5: 组合新颖性查重（对每个 branch 做一次 S2 搜索）
    for branch in branches:
        _check_combination_novelty(branch)

    new_hypothesis = Hypothesis(
        core_problem=raw["core_problem"] if raw else state.current_hypothesis.core_problem,
        abstraction_tree=branches,
        confidence=(raw or {}).get("confidence", 0.5),
        literature_support=(raw or {}).get("literature_support", []),
    )

    # 原有：余弦相似度去重 vs failed_ledger
    _check_duplicate(new_hypothesis, state)

    return {
        "current_hypothesis": new_hypothesis,
        "messages": [last_response],
    }


# ---------------------------------------------------------------------------
# Prompt 构建（v2 路径）
# ---------------------------------------------------------------------------

def _build_v2_user_message(state: ResearchState, graph: ConceptGraph) -> str:
    entities_block = _render_entities_by_type(graph)
    limitations_block = _render_limitations(graph)
    pairs_block = _render_novel_pairs(graph)
    literature = "\n".join(state.current_hypothesis.literature_support) or "（无）"
    return f"""核心矛盾：{state.current_hypothesis.core_problem}

【实体表】(从 {len(graph.papers)} 篇论文抽取，按 type 分组)
{entities_block}

【待解缺陷清单】
{limitations_block}

【潜在结构洞（高频但从未共现，可作为跨域组合启发）】
{pairs_block}

【参考文献】
{literature}

请严格按系统提示的 JSON 格式，生成 2 个方案分支。务必：
- cited_entity_names 只从实体表里挑
- solved_limitation_id 只从缺陷清单里挑
- 两个分支合起来覆盖 ≥ 2 个 task_type
"""


def _render_entities_by_type(graph: ConceptGraph, max_per_type: int = 20) -> str:
    by_type: dict[str, list] = {}
    for e in graph.entities:
        by_type.setdefault(e.type, []).append(e)
    lines = []
    for etype in ["method", "dataset", "metric", "task_type"]:
        group = sorted(by_type.get(etype, []), key=lambda e: len(e.paper_ids), reverse=True)
        lines.append(f"\n  [{etype}] ({len(group)} 个)")
        for e in group[:max_per_type]:
            lines.append(f"    - {e.canonical_name}  (在 {len(e.paper_ids)} 篇)")
        if len(group) > max_per_type:
            lines.append(f"    ...（还有 {len(group) - max_per_type} 个未显示）")
    return "\n".join(lines)


def _render_limitations(graph: ConceptGraph, max_count: int = 20) -> str:
    if not graph.limitations:
        return "  （无）"
    lines = []
    for lim in graph.limitations[:max_count]:
        lines.append(f"  - id={lim.id}  paper={lim.source_paper_id}  text={lim.text}")
    if len(graph.limitations) > max_count:
        lines.append(f"  ...（还有 {len(graph.limitations) - max_count} 条）")
    return "\n".join(lines)


def _render_novel_pairs(graph: ConceptGraph) -> str:
    if not graph.novel_pair_hints:
        return "  （无）"
    return "\n".join(
        f"  - ({p.entity_a} × {p.entity_b})  score={p.score}"
        for p in graph.novel_pair_hints
    )


# ---------------------------------------------------------------------------
# 硬约束校验
# ---------------------------------------------------------------------------

# error tuples format: (code, branch_index, detail)
# codes: MISSING_ENTITY / TOO_FEW_ENTITIES / NOT_ENOUGH_PAPERS
#        INVALID_LIMITATION / NOT_ENOUGH_TASK_TYPES (cross-branch)

def _find_entity(graph: ConceptGraph, name: str):
    norm = _normalize(name)
    return graph.entity_by_name(norm)


def _validate_branches(branches: list[AbstractionBranch], graph: ConceptGraph) -> list[tuple]:
    errors: list[tuple] = []
    paper_id_to_task_type = {p.paper_id: p.task_type for p in graph.papers}

    for idx, branch in enumerate(branches):
        # 1. 每个 cited entity 必须在图里
        for name in branch.cited_entity_names:
            if _find_entity(graph, name) is None:
                errors.append(("MISSING_ENTITY", idx, name))

        # 2. 至少 2 个 entity
        if len(branch.cited_entity_names) < 2:
            errors.append(("TOO_FEW_ENTITIES", idx, len(branch.cited_entity_names)))

        # 3. 覆盖 ≥ 2 篇不同论文
        papers_covered: set[str] = set()
        for name in branch.cited_entity_names:
            e = _find_entity(graph, name)
            if e:
                papers_covered.update(e.paper_ids)
        if len(papers_covered) < 2:
            errors.append(("NOT_ENOUGH_PAPERS", idx, len(papers_covered)))

        # 4. limitation 必须存在
        if not branch.solved_limitation_id or graph.limitation_by_id(branch.solved_limitation_id) is None:
            errors.append(("INVALID_LIMITATION", idx, branch.solved_limitation_id))

    # 5. 跨分支：task_type 覆盖 ≥ 2
    all_papers: set[str] = set()
    for branch in branches:
        for name in branch.cited_entity_names:
            e = _find_entity(graph, name)
            if e:
                all_papers.update(e.paper_ids)
    task_types = {paper_id_to_task_type.get(pid, "") for pid in all_papers}
    task_types.discard("")
    if len(task_types) < 2:
        errors.append(("NOT_ENOUGH_TASK_TYPES", -1, sorted(task_types)))

    return errors


def _build_validation_feedback(
    errors: list[tuple],
    branches: list[AbstractionBranch],
    graph: ConceptGraph,
) -> str:
    lines = ["你上次输出未通过硬约束校验。具体问题及候选："]

    # 按 branch 分组
    by_branch: dict[int, list[tuple]] = {}
    for e in errors:
        by_branch.setdefault(e[1], []).append(e)

    for bidx in sorted(by_branch.keys()):
        if bidx < 0:
            continue   # 跨分支在外面处理
        name = branches[bidx].name if bidx < len(branches) else f"branch #{bidx}"
        lines.append(f"\n[分支 {bidx}: {name}]")
        for code, _, detail in by_branch[bidx]:
            if code == "MISSING_ENTITY":
                cands = _suggest_entity_candidates(detail, graph)
                cand_str = ", ".join(f"{c.canonical_name} (在 {len(c.paper_ids)} 篇)" for c in cands)
                lines.append(f"  ❌ 引用的实体 \"{detail}\" 不在实体表。候选（同类型，按论文数降序 top 3）：{cand_str or '无'}")
            elif code == "TOO_FEW_ENTITIES":
                lines.append(f"  ❌ 只引用了 {detail} 个实体，要求 ≥ 2")
            elif code == "NOT_ENOUGH_PAPERS":
                lines.append(f"  ❌ 引用的实体仅覆盖 {detail} 篇论文，要求 ≥ 2")
            elif code == "INVALID_LIMITATION":
                ids = [l.id for l in graph.limitations[:10]]
                lines.append(f"  ❌ solved_limitation_id \"{detail}\" 不存在。可选 id（列出前 10 个）：{', '.join(ids)}")

    # 跨分支错误
    cross = [e for e in errors if e[0] == "NOT_ENOUGH_TASK_TYPES"]
    if cross:
        _, _, found_types = cross[0]
        lines.append(f"\n[跨分支]")
        lines.append(f"  ❌ 两个分支合起来仅覆盖 task_type: {found_types}，要求 ≥ 2 个不同。")
        lines.append(f"     建议让两个分支选自不同 task_type 的实体（见实体表 task_type 分组）。")

    lines.append("\n请严格从候选里挑选（不要发明新名字），按同样的 JSON schema 重新输出。")
    return "\n".join(lines)


def _suggest_entity_candidates(missing_name: str, graph: ConceptGraph, top_k: int = 3):
    """
    找候选 entity：同类型优先；按 word-boundary substring 包含 + 按 paper_ids 数降序。
    如果匹配不到任何同类型候选，降级为按 paper 数全局 top-K。
    """
    norm = _normalize(missing_name)
    # 先找词边界包含的（双向）
    candidates = []
    for e in graph.entities:
        en = e.canonical_name
        if norm and (_word_boundary_contains(norm, en) or _word_boundary_contains(en, norm)):
            candidates.append(e)
    # 按 paper 数降序
    candidates.sort(key=lambda e: len(e.paper_ids), reverse=True)
    if candidates:
        return candidates[:top_k]
    # 兜底：全部实体 top-K by paper count
    return sorted(graph.entities, key=lambda e: len(e.paper_ids), reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Step 7.5: 组合新颖性查重（S2）
# ---------------------------------------------------------------------------

def _check_combination_novelty(branch: AbstractionBranch) -> None:
    """
    用 branch.cited_entity_names 组合拼成 query 搜 S2，做标题/摘要"同时包含所有术语"后过滤。
    命中则 existing_combination=True，传给 Agent 3 作为 NOT_NOVEL 判据输入。

    失败（网络/无 entities）时默默放过，不阻断主流程。
    """
    if len(branch.cited_entity_names) < 2:
        branch.existing_combination = False
        return
    try:
        query = " ".join(branch.cited_entity_names)
        hits = ss.search_papers(query, limit=10)
    except Exception:
        hits = []

    matched = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        text = ((h.get("title") or "") + " " + (h.get("abstract") or "")).lower()
        if all(term.lower() in text for term in branch.cited_entity_names):
            matched.append(h)

    branch.existing_combination = len(matched) > 0
    branch.existing_combination_refs = [h.get("paperId", "") for h in matched[:3] if h.get("paperId")]


# ---------------------------------------------------------------------------
# 失败记账（保留 v1 行为）
# ---------------------------------------------------------------------------

def _check_duplicate(hypothesis: Hypothesis, state: ResearchState) -> None:
    """检查新假设是否与 failed_ledger 中的历史记录过于相似"""
    if not state.failed_ledger:
        return

    hypothesis_text = f"{hypothesis.core_problem} " + " ".join(
        f"{b.name} {b.algorithm_logic}" for b in hypothesis.abstraction_tree
    )
    hypothesis_vector = get_text_embedding(hypothesis_text)

    for record in state.failed_ledger:
        if not record.feature_vector:
            continue
        similarity = compute_cosine_similarity(hypothesis_vector, record.feature_vector)
        if similarity > SIMILARITY_THRESHOLD:
            raise DuplicateHypothesisError(
                similarity=similarity,
                matched_record_summary=record.error_summary,
            )
