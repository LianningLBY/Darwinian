"""
Agent 5: 诊断分析师 (diagnostician_node)

职责：
- 读取代码执行的 stdout/stderr 日志
- 区分 code_bug（语法/运行时错误）vs logic_flaw（方法无效）
- code_bug → 驱动内层循环让 Agent 4 修复
- logic_flaw (insufficient) → 写入 failed_ledger，终止本轮
"""

from __future__ import annotations

import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from darwinian.state import ResearchState, ExecutionVerdict


SYSTEM_PROMPT = """你是一个科研实验诊断分析师。你的任务是分析代码执行日志，判断失败原因。

判断规则：
- **code_error**: stdout/stderr 中出现 Traceback、SyntaxError、ImportError、shape mismatch 等程序级错误
- **insufficient**: 代码成功运行，但提出方法的指标 <= 基准方法的指标（即方法本身无效）
- **success**: 代码成功运行，且提出方法的指标 > 基准方法的指标

输出格式（严格 JSON）：
{
  "verdict": "code_error" | "insufficient" | "success",
  "diagnosis": "详细诊断说明",
  "baseline_metrics": {"metric_name": value, ...},
  "proposed_metrics": {"metric_name": value, ...},
  "failure_summary": "如果失败，用一句话总结失败原因（用于写入 failed_ledger）",
  "error_keywords": ["失败关键词，用于 failed_ledger 的 banned_keywords"]
}

判断 insufficient 时必须同时解析出两个模型的指标数值，不能仅凭 stderr 判断。
禁止输出 JSON 以外的任何内容。"""


def diagnostician_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Agent 5 节点函数。
    输入: experiment_result.stdout/stderr（由 code_execute 工具填入）
    输出: 更新 experiment_result（填充 execution_verdict、diagnosis 和指标）
    """
    if state.experiment_result is None:
        raise ValueError("diagnostician_node 调用前必须先执行 code_execute")

    result = state.experiment_result

    user_message = f"""代码执行输出：

STDOUT（前 3000 字符）：
{result.stdout[:3000]}

STDERR（前 2000 字符）：
{result.stderr[:2000]}

请诊断执行结果。"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw = json.loads(response.content)
    verdict = ExecutionVerdict(raw["verdict"])

    updated_result = result.model_copy(update={
        "execution_verdict": verdict,
        "diagnosis": raw.get("diagnosis", ""),
        "baseline_metrics": raw.get("baseline_metrics", {}),
        "proposed_metrics": raw.get("proposed_metrics", {}),
    })

    return {
        "experiment_result": updated_result,
        "messages": [response],
        # 供路由函数使用
        "_failure_summary": raw.get("failure_summary", ""),
        "_error_keywords": raw.get("error_keywords", []),
    }
