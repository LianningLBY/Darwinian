"""
数据集搜索工具

流程：
1. 根据研究方向 + 假设核心问题，调用 HuggingFace Hub API 搜索相关公开数据集
2. 将候选列表交给 LLM 选出最合适的一个
3. 返回 DatasetInfo，供 Agent 4 生成下载代码
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

from darwinian.state import DatasetInfo, ResearchState
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


HF_API_URL = "https://huggingface.co/api/datasets"
SEARCH_TIMEOUT = 10  # 秒


def search_hf_datasets(query: str, limit: int = 8) -> list[dict]:
    """
    调用 HuggingFace Hub 数据集搜索 API。

    Returns:
        候选数据集列表，每项含 id / description / tags / downloads
    """
    try:
        resp = requests.get(
            HF_API_URL,
            params={"search": query, "limit": limit, "sort": "downloads", "direction": -1},
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
        results = []
        for item in raw:
            results.append({
                "id":          item.get("id", ""),
                "description": (item.get("description") or item.get("cardData", {}).get("pretty_name") or "")[:200],
                "tags":        item.get("tags", [])[:8],
                "downloads":   item.get("downloads", 0),
            })
        return results
    except Exception as e:
        return []


PICKER_SYSTEM = """你是一个科研数据集选择专家。给定研究课题和候选数据集列表，选出最适合做实验的那一个。

输出严格 JSON（禁止其他内容）：
{
  "dataset_id": "数据集的 HuggingFace ID（如 scikit-learn/iris 或 rajpurkar/squad）",
  "description": "一句话说明为什么选它",
  "task_type": "classification|regression|nlp|graph|cv|other",
  "load_instruction": "datasets.load_dataset('dataset_id') 或其他加载示例代码（一行）"
}"""


def pick_dataset_with_llm(
    llm: BaseChatModel,
    research_direction: str,
    core_problem: str,
    candidates: list[dict],
) -> DatasetInfo:
    """让 LLM 从候选列表中选出最合适的数据集。"""
    from darwinian.utils.json_parser import parse_llm_json

    if not candidates:
        # 搜索失败时降级为 sklearn 内置
        return DatasetInfo(
            source="builtin",
            dataset_id="sklearn-builtin",
            description="HuggingFace 搜索失败，使用 sklearn 内置数据集",
            task_type="classification",
            load_instruction="from sklearn.datasets import make_classification; X, y = make_classification(n_samples=1000)",
        )

    candidates_text = "\n".join(
        f"{i+1}. [{c['id']}] {c['description']} | tags: {c['tags']} | downloads: {c['downloads']}"
        for i, c in enumerate(candidates)
    )

    user_msg = f"""研究方向：{research_direction}
核心问题：{core_problem}

HuggingFace 候选数据集：
{candidates_text}

请选出最适合验证上述研究假设的数据集。"""

    response = llm.invoke([
        SystemMessage(content=PICKER_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    raw = parse_llm_json(response.content)
    return DatasetInfo(
        source="huggingface",
        dataset_id=raw["dataset_id"],
        description=raw.get("description", ""),
        task_type=raw.get("task_type", ""),
        load_instruction=raw.get("load_instruction", f"datasets.load_dataset('{raw['dataset_id']}')"),
    )


def dataset_finder_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    数据集搜索节点。

    - 若用户已上传数据（user_data_path 非空），直接构造 DatasetInfo，跳过搜索。
    - 否则调用 HuggingFace API 搜索并让 LLM 选择。
    """
    # 用户上传优先
    if state.user_data_path and Path(state.user_data_path).exists():
        filename = Path(state.user_data_path).name
        return {
            "selected_dataset": DatasetInfo(
                source="user_upload",
                dataset_id=filename,
                description=f"用户上传的数据集：{filename}",
                task_type=state.dataset_schema.get("task", "unknown"),
                load_instruction=_infer_load_instruction(filename),
            )
        }

    # 构建搜索关键词
    core_problem = ""
    if state.current_hypothesis:
        core_problem = state.current_hypothesis.core_problem
    query = f"{state.research_direction} {core_problem}".strip()[:120]

    candidates = search_hf_datasets(query)
    dataset_info = pick_dataset_with_llm(llm, state.research_direction, core_problem, candidates)
    return {"selected_dataset": dataset_info}


def _infer_load_instruction(filename: str) -> str:
    """根据文件扩展名生成加载代码提示。"""
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return f"import pandas as pd; df = pd.read_csv('/data/{filename}')"
    elif ext in (".parquet", ".pq"):
        return f"import pandas as pd; df = pd.read_parquet('/data/{filename}')"
    elif ext in (".npz", ".npy"):
        return f"import numpy as np; data = np.load('/data/{filename}')"
    elif ext in (".json", ".jsonl"):
        return f"import pandas as pd; df = pd.read_json('/data/{filename}', lines={str(ext=='.jsonl')})"
    else:
        return f"# 请根据文件格式加载 /data/{filename}"
