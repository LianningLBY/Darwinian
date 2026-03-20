"""
Agent 4: 代码架构师 (code_architect_node)

职责：
- 分五次独立调用 LLM，分别生成：
    1. dataset_loader_code（数据加载）
    2. baseline_code（基准方法，固定随机种子 + 多 seed 统计）
    3. proposed_code（假设方案，固定随机种子 + 多 seed 统计）
    4. ablation_code（消融实验，逐一去掉各关键组件）
    5. requirements（依赖列表）
- 每次只生成一段，避免 token 截断
- 所有代码强制固定随机种子，保证可复现性
"""

from __future__ import annotations

import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState, ExperimentCode


_BASE_SYSTEM = """你是一个科研实验代码架构师。你的任务是编写可直接运行的 Python 实验代码。

代码规范（强制要求）：
1. 可复现性：文件开头必须设置 SEED = 42，并对所有随机库统一固定：
   import numpy as np, random, os
   SEED = 42
   np.random.seed(SEED); random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
   如使用 torch：import torch; torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
2. 多 seed 统计：基准和提出方法均需在 SEEDS = [42, 123, 456] 三个种子下各跑一次，
   报告均值和标准差，格式：{"model": "baseline", "metrics": {"accuracy_mean": 0.85, "accuracy_std": 0.02}}
3. 防御性检查：每次 tensor/array 操作前检查 shape
4. 标准化输出：最后一行 print 一行 JSON
5. 超时保护：单次训练不超过 100 秒（每个 seed），超时后跳过该 seed 继续
6. 代码长度：单段不超过 150 行

数据集加载规则：
- HuggingFace：`from datasets import load_dataset`，可联网下载
- 用户上传：从 `/data/<文件名>` 读取（只读挂载）
- 无数据集：使用 scikit-learn 内置数据集或随机生成

只输出纯 Python 代码，不要任何 markdown 或说明文字。"""


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
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    text = response.content.strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    block = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if block:
        text = block.group(1).strip()
    return text


def code_architect_node(state: ResearchState, llm: BaseChatModel) -> dict:
    if state.current_hypothesis is None or state.current_hypothesis.selected_branch is None:
        raise ValueError("code_architect_node 调用前必须先通过 theoretical_critic_node")

    hypothesis = state.current_hypothesis
    branch = hypothesis.selected_branch

    retry_context = ""
    if (
        state.experiment_code is not None
        and state.experiment_result is not None
        and state.experiment_code.retry_count > 0
    ):
        stderr_snippet = state.experiment_result.stderr[:1500]
        # 提取维度信息，帮助 LLM 精确修复
        import re as _re
        dim_hints = ""
        dims = _re.findall(r"(\d+)\s*[x×,]\s*(\d+)", stderr_snippet)
        if dims:
            dim_hints = f"\n⚠️  检测到维度不匹配：{dims}。必须确保 loader → baseline → proposed 的 n_features 一致。"
        retry_context = f"""
⚠️  上次代码执行失败（第 {state.experiment_code.retry_count} 次重试）：
STDERR: {stderr_snippet}
诊断结论：{state.experiment_result.diagnosis}
{dim_hints}
修复要求：
1. 所有脚本（loader / baseline / proposed）使用相同的输入维度
2. 每层矩阵乘法前用 assert 检查 shape
3. 不要改变整体方法逻辑，只修复错误"""

    if state.selected_dataset:
        ds = state.selected_dataset
        if ds.source == "user_upload":
            dataset_section = f"""数据集（用户上传）：
- 文件名：{ds.dataset_id}（挂载至 /data/{ds.dataset_id}）
- 任务类型：{ds.task_type}
- 加载示例：{ds.load_instruction}"""
        elif ds.source == "huggingface":
            dataset_section = f"""数据集（HuggingFace）：
- ID：{ds.dataset_id}
- 描述：{ds.description}
- 任务类型：{ds.task_type}
- 加载示例：{ds.load_instruction}
注意：可联网使用 `from datasets import load_dataset` 下载。"""
        else:
            dataset_section = f"数据集：{ds.load_instruction}"
    else:
        dataset_section = f"""数据集 Schema（使用 sklearn 内置或合成数据）：
{json.dumps(state.dataset_schema, ensure_ascii=False, indent=2)}"""

    ctx = _make_context(hypothesis, branch, dataset_section, retry_context)

    is_retry = state.experiment_code is not None and state.experiment_code.retry_count > 0

    # ── 第 1 次：dataset_loader_code ──
    # 重试时直接复用上一轮 loader（它不是出错原因），避免维度再次漂移
    if is_retry and state.experiment_code and state.experiment_code.dataset_loader_code:
        loader_code = state.experiment_code.dataset_loader_code
    else:
        loader_code = _call_llm(
            llm, system=_BASE_SYSTEM,
            user=f"""{ctx}

任务：只编写「数据加载脚本」。
要求：
- 加载数据集，打印样本数、特征维度、标签分布
- 必须暴露 load_data() 函数供其他脚本调用，返回 (X_train, X_test, y_train, y_test)
- 最后 print 一行 JSON: {{"model": "loader", "metrics": {{"n_samples": ..., "n_features": ...}}}}
不超过 60 行。只输出 Python 代码。""",
        )

    # ── 第 2 次：baseline_code ──
    # 重试时同样复用上一轮 baseline（维度来源与 loader 一致）
    if is_retry and state.experiment_code and state.experiment_code.baseline_code:
        baseline_code = state.experiment_code.baseline_code
    else:
        baseline_code = _call_llm(
            llm, system=_BASE_SYSTEM,
            user=f"""{ctx}

任务：只编写「Baseline 基准方法代码」。
要求：
- 实现最经典的基准方法
- 在 SEEDS = [42, 123, 456] 三个种子下各训练一次
- 计算每个指标的均值和标准差
- 最后 print 一行 JSON: {{"model": "baseline", "metrics": {{"accuracy_mean": ..., "accuracy_std": ...}}}}
不超过 150 行。只输出 Python 代码。""",
        )

    # ── 第 3 次：proposed_code ──
    # 始终将 loader_code 传入，让 LLM 能看到实际 n_features，杜绝维度猜测
    proposed_code = _call_llm(
        llm, system=_BASE_SYSTEM,
        user=f"""{ctx}

以下是本次实验的 dataset_loader.py（必须与此保持维度一致）：
```python
{loader_code[:600]}
```

任务：只编写「Proposed Method 实验代码」（{branch.name}）。
要求：
- 实现假设中的方案，输入维度必须与 load_data() 返回的 X_train.shape[1] 一致
- 在 SEEDS = [42, 123, 456] 三个种子下各训练一次
- 计算每个指标的均值和标准差
- 最后 print 一行 JSON: {{"model": "proposed", "metrics": {{"accuracy_mean": ..., "accuracy_std": ...}}}}
不超过 150 行。只输出 Python 代码。""",
    )

    # ── 第 4 次：ablation_code ──
    ablation_code = _call_llm(
        llm, system=_BASE_SYSTEM,
        user=f"""{ctx}

以下是本次实验的 dataset_loader.py（维度来源）：
```python
{loader_code[:400]}
```

任务：只编写「消融实验代码」（Ablation Study）。
要求：
- 识别方案 {branch.name} 的 2-3 个关键组件
- 每个变体去掉一个组件，其余保持不变，输入维度与 loader 一致
- 每个变体用 SEED=42 跑一次，输出结果
- 每个变体 print 一行 JSON: {{"model": "ablation_no_<组件名>", "metrics": {{...}}}}
- 最后 print {{"model": "ablation_summary", "metrics": {{"n_variants": ...}}}}
不超过 150 行。只输出 Python 代码。""",
    )

    # ── 第 5 次：requirements ──
    req_raw = _call_llm(
        llm,
        system="你是 Python 依赖分析专家。只输出 JSON 数组，不要任何其他内容。",
        user=f"""分析以下代码，列出所有需要 pip install 的第三方库（不包括标准库）。

--- loader ---
{loader_code[:400]}
--- baseline ---
{baseline_code[:400]}
--- proposed ---
{proposed_code[:400]}

只输出 JSON 数组，例如：["numpy", "scikit-learn", "torch"]""",
    )
    try:
        arr_match = re.search(r"\[[^\]]*\]", req_raw)
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
        ablation_code=ablation_code,
        requirements=requirements,
        retry_count=retry_count,
    )

    return {"experiment_code": new_code}
