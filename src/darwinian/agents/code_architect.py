"""
Agent 4: 代码架构师 (code_architect_node)

职责：
- 接收验证通过的 Hypothesis 和数据集 Schema
- 编写包含 Baseline 和 Proposed Method 的实验级 Python 代码
- 代码必须包含完整的依赖声明和防御性 shape 检查
"""

from __future__ import annotations

import json
from darwinian.utils.json_parser import parse_llm_json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState, ExperimentCode


SYSTEM_PROMPT = """你是一个科研实验代码架构师。你的任务是编写可直接运行的 Python 实验代码。

代码规范（强制要求）：
1. **完整依赖声明**：在代码头部注释中列出所有 pip 依赖
2. **防御性检查**：每次 tensor/array 操作前必须检查 shape，出错时打印清晰的错误信息
3. **标准化输出**：每个模型最后必须 print 一行 JSON：{"model": "baseline"|"proposed", "metrics": {...}}
4. **超时保护**：单次训练不超过 300 秒，超时后保存中间结果并退出
5. **独立文件结构**：
   - baseline_code：独立可运行，实现最经典的基准方法
   - proposed_code：独立可运行，实现假设中选定的方案
   - dataset_loader_code：独立可运行，加载数据并打印数据统计

输出格式（严格 JSON）：
{
  "baseline_code": "完整的 Python 代码字符串",
  "proposed_code": "完整的 Python 代码字符串",
  "dataset_loader_code": "完整的 Python 代码字符串",
  "requirements": ["numpy", "scikit-learn", ...]
}

禁止输出 JSON 以外的任何内容。"""


def code_architect_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 4 节点函数。
    输入: current_hypothesis (selected_branch) + dataset_schema
    输出: 更新 experiment_code
    """
    if state.current_hypothesis is None or state.current_hypothesis.selected_branch is None:
        raise ValueError("code_architect_node 调用前必须先通过 theoretical_critic_node")

    hypothesis = state.current_hypothesis
    branch = hypothesis.selected_branch

    # 如果是修复重试，附上上次的错误信息
    retry_context = ""
    if (
        state.experiment_code is not None
        and state.experiment_result is not None
        and state.experiment_code.retry_count > 0
    ):
        retry_context = f"""
上次代码执行的错误（第 {state.experiment_code.retry_count} 次重试）：
STDERR: {state.experiment_result.stderr[:2000]}
诊断结论：{state.experiment_result.diagnosis}

请修复上述错误，不要改变整体方法逻辑。"""

    user_message = f"""研究假设：
核心问题：{hypothesis.core_problem}

选定方案：
- 名称：{branch.name}
- 算法逻辑：{branch.algorithm_logic}
- 数学公式：{branch.math_formulation}
- 灵感来源：{branch.source_domain}

数据集 Schema：
{json.dumps(state.dataset_schema, ensure_ascii=False, indent=2)}

{retry_context}

请编写完整的实验代码。"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw = parse_llm_json(response.content)

    retry_count = 0
    if state.experiment_code is not None:
        retry_count = state.experiment_code.retry_count + 1

    new_code = ExperimentCode(
        baseline_code=raw["baseline_code"],
        proposed_code=raw["proposed_code"],
        dataset_loader_code=raw["dataset_loader_code"],
        requirements=raw.get("requirements", []),
        retry_count=retry_count,
    )

    return {
        "experiment_code": new_code,
        "messages": [response],
    }
