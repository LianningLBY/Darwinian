"""
DarwinianTool — 将完整科研 Pipeline 封装为 LangChain Tool

可被任何兼容 LangChain 的 Agent 直接调用：
- LangChain AgentExecutor
- LangGraph 自定义 Agent 节点
- Claude / OpenAI Function Calling（通过 LangChain 适配层）

用法示例：
    from darwinian.tools.darwinian_tool import build_darwinian_tool
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o")
    tool = build_darwinian_tool(llm)

    # 直接调用
    result = tool.invoke("图神经网络在分子属性预测中的应用")

    # 挂载到 Agent
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    agent = create_tool_calling_agent(llm, [tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[tool])
"""

from __future__ import annotations

import json
from typing import Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 输入 Schema（用于 Function Calling 的参数描述）
# ---------------------------------------------------------------------------

class DarwinianInput(BaseModel):
    research_direction: str = Field(
        description="研究方向描述，例如：'图神经网络在分子属性预测中的应用'"
    )
    max_outer_loops: int = Field(
        default=3,
        description="最大迭代轮数（每轮包含假设生成 + 实验验证），默认 3",
        ge=1,
        le=10,
    )
    user_data_path: Optional[str] = Field(
        default=None,
        description="用户自定义数据集路径（可选），不填则自动搜索公开数据集",
    )


# ---------------------------------------------------------------------------
# Tool 实现
# ---------------------------------------------------------------------------

class DarwinianTool(BaseTool):
    """
    将 Darwinian 多智能体科研 Pipeline 封装为可调用 Tool。

    输入：研究方向（自然语言字符串）
    输出：研究报告（Markdown 字符串）或失败摘要
    """

    name: str = "darwinian_research"
    description: str = (
        "自动化科研工具。输入一个研究方向，系统自主完成文献挖掘、跨域假设生成、"
        "理论审查、实验代码生成与执行、鲁棒性验证，返回完整研究报告。"
        "适用于需要快速验证科研假设、生成实验基线或探索新研究方向的场景。"
        "输入示例：'图神经网络在分子属性预测中的应用'"
    )
    args_schema: Type[BaseModel] = DarwinianInput
    return_direct: bool = False  # False = 结果返回给上层 Agent 继续处理

    # LLM 实例（通过 build_darwinian_tool 工厂函数注入）
    llm: BaseChatModel = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        research_direction: str,
        max_outer_loops: int = 3,
        user_data_path: Optional[str] = None,
    ) -> str:
        """同步执行科研 Pipeline，返回研究报告或失败摘要。"""
        from darwinian.graphs.main_graph import build_main_graph
        from darwinian.state import ResearchState

        graph = build_main_graph(self.llm)

        initial_state = ResearchState(
            research_direction=research_direction,
            max_outer_loops=max_outer_loops,
            user_data_path=user_data_path or "",
        )

        try:
            final_state = graph.invoke(initial_state)
        except Exception as e:
            return f"[DarwinianTool] Pipeline 执行异常：{type(e).__name__}: {e}"

        return _format_result(final_state)

    async def _arun(
        self,
        research_direction: str,
        max_outer_loops: int = 3,
        user_data_path: Optional[str] = None,
    ) -> str:
        """异步执行（复用同步实现，避免阻塞事件循环需自行包装 run_in_executor）。"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._run(research_direction, max_outer_loops, user_data_path),
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_darwinian_tool(llm: BaseChatModel) -> DarwinianTool:
    """
    构建 DarwinianTool 实例。

    Args:
        llm: LangChain 兼容的 chat model（如 ChatOpenAI、ChatAnthropic）

    Returns:
        可直接挂载到任何 LangChain Agent 的 Tool 实例

    示例：
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-opus-4-6")
        tool = build_darwinian_tool(llm)
    """
    return DarwinianTool(llm=llm)


# ---------------------------------------------------------------------------
# 辅助：格式化输出
# ---------------------------------------------------------------------------

def _format_result(state) -> str:
    """将最终 ResearchState 格式化为可读的字符串结果。"""
    from darwinian.state import FinalVerdict

    if state.final_verdict == FinalVerdict.PUBLISH_READY and state.final_report:
        return state.final_report

    # 未达到发表标准，返回失败摘要
    lines = ["[DarwinianTool] 研究未达到发表标准，摘要如下：\n"]

    if state.current_hypothesis:
        lines.append(f"**最后尝试的核心问题**：{state.current_hypothesis.core_problem}\n")
        if state.current_hypothesis.selected_branch:
            b = state.current_hypothesis.selected_branch
            lines.append(f"**最后尝试的方案**：{b.name} — {b.algorithm_logic[:100]}...\n")

    if state.experiment_result:
        r = state.experiment_result
        lines.append(f"**实验诊断**：{r.diagnosis}\n")
        if r.baseline_metrics:
            lines.append(f"**Baseline 指标**：{json.dumps(r.baseline_metrics, ensure_ascii=False)}\n")
        if r.proposed_metrics:
            lines.append(f"**Proposed 指标**：{json.dumps(r.proposed_metrics, ensure_ascii=False)}\n")

    lines.append(f"\n**总循环轮数**：{state.outer_loop_count} / {state.max_outer_loops}")
    lines.append(f"**认知账本记录**：{len(state.failed_ledger)} 条失败经验")

    return "\n".join(lines)
