"""
LLM 调用重试工具

对流式连接断开、网络超时等瞬时错误自动重试。
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


_RETRYABLE_KEYWORDS = (
    "RemoteProtocolError",
    "incomplete chunked",
    "peer closed",
    "ConnectionError",
    "ReadTimeout",
    "ConnectTimeout",
    "ProtocolError",
)


def invoke_with_retry(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    max_retries: int = 3,
    base_wait: float = 5.0,
    **kwargs: Any,
) -> Any:
    """
    调用 llm.invoke，遇到网络瞬时错误时自动重试。

    Args:
        llm: LangChain 兼容的 chat model
        messages: 消息列表
        max_retries: 最大重试次数（不含首次调用）
        base_wait: 首次重试等待秒数，后续线性增加
        **kwargs: 传给 llm.invoke 的额外参数

    Returns:
        llm.invoke 的返回值

    Raises:
        最后一次异常（若全部重试均失败）
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages, **kwargs)
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)
            is_retryable = any(kw in exc_str or kw in type(exc).__name__ for kw in _RETRYABLE_KEYWORDS)
            if is_retryable and attempt < max_retries:
                wait = base_wait * (attempt + 1)
                print(f"[llm_retry] 网络异常（{type(exc).__name__}），{wait:.0f}s 后重试（{attempt + 1}/{max_retries}）...")
                time.sleep(wait)
                continue
            raise
    raise last_exc  # unreachable, but keeps type checker happy
