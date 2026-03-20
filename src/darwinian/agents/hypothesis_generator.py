"""
Agent 2: 方案合成器 (hypothesis_generator_node)

职责：
- 接收核心矛盾，生成跨域解决思路
- 必须输出完整的 Hypothesis 强类型对象
- 内置余弦相似度去重，与 failed_ledger 比较，相似度 > 0.85 时触发 DuplicateError
"""

from __future__ import annotations

import json
from darwinian.utils.json_parser import parse_llm_json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

import json as _json
from darwinian.state import ResearchState, Hypothesis, AbstractionBranch
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


SYSTEM_PROMPT = """你是一个跨域科研方案合成专家。你的任务是：
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


def hypothesis_generator_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 2 节点函数。
    输入: current_hypothesis.core_problem
    输出: 更新 current_hypothesis（填充 abstraction_tree 等字段）
    """
    if state.current_hypothesis is None:
        raise ValueError("hypothesis_generator_node 调用前必须先运行 bottleneck_miner_node")

    user_message = f"""核心矛盾：{state.current_hypothesis.core_problem}

现有文献支撑：
{chr(10).join(state.current_hypothesis.literature_support)}

请生成跨域解决方案。"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    try:
        raw = parse_llm_json(response.content)
        branches = [AbstractionBranch(**b) for b in raw["abstraction_tree"]]
    except (_json.JSONDecodeError, KeyError, Exception):
        # LLM 输出截断（如只有 <think> 块）或结构缺失，返回空 abstraction_tree
        # Agent 3 会捕获空树 → MATH_ERROR → 触发重试，不让整个 pipeline 崩溃
        return {
            "current_hypothesis": Hypothesis(
                core_problem=state.current_hypothesis.core_problem,
                abstraction_tree=[],
            ),
            "messages": [response],
        }

    new_hypothesis = Hypothesis(
        core_problem=raw["core_problem"],
        abstraction_tree=branches,
        confidence=raw.get("confidence", 0.5),
        literature_support=raw.get("literature_support", []),
    )

    # 余弦相似度去重检查
    _check_duplicate(new_hypothesis, state)

    return {
        "current_hypothesis": new_hypothesis,
        "messages": [response],
    }


def _check_duplicate(hypothesis: Hypothesis, state: ResearchState) -> None:
    """检查新假设是否与 failed_ledger 中的历史记录过于相似"""
    if not state.failed_ledger:
        return

    # 将假设文本向量化
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
