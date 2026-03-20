"""
Agent 4: 代码架构师 (code_architect_node)

职责：
- 接收验证通过的 Hypothesis 和数据集 Schema
- 分三次独立调用 LLM，分别生成：
    1. dataset_loader_code（数据加载）
    2. baseline_code（基准方法）
    3. proposed_code（假设方案）
- 每次只生成一段代码，避免单次输出过长导致 token 截断
"""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState, ExperimentCode


# ── 公共 system prompt ──────────────────────────────────────────────────────

_BASE_SYSTEM = """你是一个科研实验代码架构师。你的任务是编写可直接运行的 Python 实验代码。

代码规范（强制要求）：
1. 完整依赖声明：在代码头部注释中列出所有 pip 依赖
2. 防御性检查：每次 tensor/array 操作前必须检查 shape
3. 标准化输出：每个模型最后必须 print 一行 JSON：{"model": "baseline"|"proposed"|"loader", "metrics": {...}}
4. 超时保护：单次训练不超过 300 秒，超时后保存中间结果并退出
5. 代码长度：单段代码不超过 150 行，复杂逻辑拆分为小函数

数据集加载规则：
- 若提供了 selected_dataset（HuggingFace），使用 `datasets` 库：`from datasets import load_dataset`，代码可以联网下载
- 若提供了用户上传路径（user_upload），从 `/data/<文件名>` 读取（已只读挂载）
- 若两者都没有，使用 scikit-learn 内置数据集或随机生成数据

只输出纯 Python 代码，不要任何 markdown、注释说明或 JSON 包装。"""


# ── 三段代码的独立 prompt ───────────────────────────────────────────────────

def _make_context(hypothesis, branch, dataset_section: str, retry_context: str) -> str:
    return f"""研究假设：
核心问题：{hypothesis.core_problem}

选定方案：
- 名称：{branch.name}
- 算法逻辑：{branch.algorithm_logic}
- 数学公式：{branch.math_formulation}
- 灵感来源：{branch.source_domain}

{dataset_section}
{retry_context}"""


def _call_llm(llm: BaseChatModel, system: str, user: str) -> str:
    """单次 LLM 调用，剥离可能的 markdown 代码块包裹。"""
    import re
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    text = response.content.strip()
    # 剥离 <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    # 剥离 ```python ... ``` 或 ``` ... ```
    block = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if block:
        text = block.group(1).strip()
    return text


# ── 节点主函数 ──────────────────────────────────────────────────────────────

def code_architect_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 4 节点函数。
    分三次调用 LLM，每次只生成一段代码，避免单次 token 过长截断。
    """
    if state.current_hypothesis is None or state.current_hypothesis.selected_branch is None:
        raise ValueError("code_architect_node 调用前必须先通过 theoretical_critic_node")

    hypothesis = state.current_hypothesis
    branch = hypothesis.selected_branch

    # 修复重试时附上错误信息
    retry_context = ""
    if (
        state.experiment_code is not None
        and state.experiment_result is not None
        and state.experiment_code.retry_count > 0
    ):
        retry_context = f"""
上次代码执行的错误（第 {state.experiment_code.retry_count} 次重试）：
STDERR: {state.experiment_result.stderr[:1500]}
诊断结论：{state.experiment_result.diagnosis}

请修复上述错误，不要改变整体方法逻辑。"""

    # 构建数据集描述
    if state.selected_dataset:
        ds = state.selected_dataset
        if ds.source == "user_upload":
            dataset_section = f"""数据集（用户上传）：
- 文件名：{ds.dataset_id}（已挂载到 Docker 容器 /data/{ds.dataset_id}）
- 任务类型：{ds.task_type}
- 加载示例：{ds.load_instruction}"""
        elif ds.source == "huggingface":
            dataset_section = f"""数据集（HuggingFace）：
- ID：{ds.dataset_id}
- 描述：{ds.description}
- 任务类型：{ds.task_type}
- 加载示例：{ds.load_instruction}
注意：代码运行时可以联网，直接使用 `from datasets import load_dataset` 下载。"""
        else:
            dataset_section = f"数据集：{ds.load_instruction}"
    else:
        dataset_section = f"""数据集 Schema（使用 sklearn 内置或合成数据）：
{json.dumps(state.dataset_schema, ensure_ascii=False, indent=2)}"""

    ctx = _make_context(hypothesis, branch, dataset_section, retry_context)

    # ── 第 1 次调用：dataset_loader_code ──
    loader_code = _call_llm(
        llm,
        system=_BASE_SYSTEM,
        user=f"""{ctx}

任务：只编写「数据加载脚本」（dataset_loader_code）。
要求：加载数据集，打印样本数量、特征维度、标签分布等统计信息，最后 print 一行 JSON。
不超过 60 行。只输出 Python 代码。""",
    )

    # ── 第 2 次调用：baseline_code ──
    baseline_code = _call_llm(
        llm,
        system=_BASE_SYSTEM,
        user=f"""{ctx}

任务：只编写「Baseline 基准方法代码」（baseline_code）。
要求：实现最经典的基准方法（如 MLP、GNN、Logistic Regression 等），完整训练+评估流程，最后 print 一行 JSON 包含 metrics。
不超过 150 行。只输出 Python 代码。""",
    )

    # ── 第 3 次调用：proposed_code ──
    proposed_code = _call_llm(
        llm,
        system=_BASE_SYSTEM,
        user=f"""{ctx}

任务：只编写「Proposed Method 实验代码」（proposed_code）。
要求：实现假设中选定的方案（{branch.name}），完整训练+评估流程，最后 print 一行 JSON 包含 metrics。
不超过 150 行。只输出 Python 代码。""",
    )

    # ── 第 4 次调用：requirements ──
    req_raw = _call_llm(
        llm,
        system="你是 Python 依赖分析专家。只输出 JSON 数组，不要任何其他内容。",
        user=f"""分析以下三段 Python 代码，列出所有需要 pip install 的第三方库（不包括标准库）。

--- dataset_loader ---
{loader_code[:500]}

--- baseline ---
{baseline_code[:500]}

--- proposed ---
{proposed_code[:500]}

只输出 JSON 数组，例如：["numpy", "torch", "scikit-learn"]""",
    )
    try:
        import re
        # 找到 [ ... ] 数组
        arr_match = re.search(r"\[[\s\S]*?\]", req_raw)
        requirements = json.loads(arr_match.group()) if arr_match else []
    except Exception:
        requirements = []

    retry_count = 0
    if state.experiment_code is not None:
        retry_count = state.experiment_code.retry_count + 1

    new_code = ExperimentCode(
        baseline_code=baseline_code,
        proposed_code=proposed_code,
        dataset_loader_code=loader_code,
        requirements=requirements,
        retry_count=retry_count,
    )

    return {"experiment_code": new_code}
