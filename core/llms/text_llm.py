from __future__ import annotations

import logging
from typing import Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.config.schema import LLMConfig
from core.utils.retry import RetryPolicy, classify_http_exception

from langchain_core.messages import BaseMessage
T = TypeVar("T", bound=BaseModel)


class TextLLM:
    """文本 LLM 封装：支持 Pydantic 结构化输出。"""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._model = ChatOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.request_timeout,
        )
        # 统一重试策略：任意异常最多重试 5 次，退避 2s
        self._retry = RetryPolicy(max_retries=5)

    def structured_invoke(self, schema: Type[T], prompt_messages: list, json_mode: bool = False) -> T:
        """以结构化输出方式执行调用。

        - schema: Pydantic BaseModel 子类
        - prompt_messages: LangChain 消息数组（System/Human 等）
        - json_mode: True 时启用模型内置 JSON 模式（若支持）
        """
        # 构建结构化输出模型：
        # - 当 json_mode=True 时，显式使用 json_mode
        # - 否则不传 method（让 LangChain 选择最佳实现）；若构建异常，再回退到 function_calling
        log = logging.getLogger("note_gen.llms.text")
        if json_mode:
            model_with_schema = self._model.with_structured_output(schema, method="json_mode")
        else:
            model_with_schema = self._model.with_structured_output(schema)

        @self._retry
        def _invoke() -> T:
            log.info("调用文本 LLM（结构化输出）", extra={"json_mode": json_mode, "schema": schema.__name__})
            return model_with_schema.invoke(prompt_messages)  # type: ignore[return-value]

        return _invoke()

    def invoke(self, prompt_messages: list) -> str:
        """以普通文本形式执行调用，返回模型原始内容。"""
        log = logging.getLogger("note_gen.llms.text")

        @self._retry
        def _invoke() -> str:
            log.info("调用文本 LLM（普通输出）")
            message = self._model.invoke(prompt_messages)
            if isinstance(message, BaseMessage):
                content = message.content
            elif isinstance(message, str):
                content = message
            else:
                content = getattr(message, "content", None)
            if not isinstance(content, str) or not content:
                raise ValueError("模型返回内容为空或类型异常")
            return content

        return _invoke()
