"""
MiniMax LLM 包装器

使用 langchain 的 ChatOpenAI 兼容接口连接 MiniMax API
"""

from langchain_openai import ChatOpenAI

__all__ = ["ChatMiniMax"]


class ChatMiniMax(ChatOpenAI):
    """
    MiniMax Chat 模型
    
    使用方式与 ChatOpenAI 相同，只是 API 基础 URL 不同
    
    示例:
        from darwinian.llms import ChatMiniMax
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        llm = ChatMiniMax(
            model="MiniMax-M2.7",
            api_key=os.environ["MINIMAX_API_KEY"],
            max_tokens=8192,
        )
    """
    
    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        api_key: str | None = None,
        base_url: str = "https://api.minimax.chat/v1",
        **kwargs
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
