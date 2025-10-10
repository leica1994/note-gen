from __future__ import annotations

import logging
from typing import Type, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.config.schema import LLMConfig
from core.llms.concurrency import ConcurrencyLimiter

T = TypeVar("T", bound=BaseModel)

class TextLLM:
    """文本 LLM 封装：支持 Pydantic 结构化输出。"""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._limiter = ConcurrencyLimiter(cfg.concurrency)
        self._model = ChatOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.request_timeout,
        )

    def structured_invoke(self, schema: Type[T], prompt_messages: list, json_mode: bool = False) -> T:
        """以结构化输出方式执行调用。"""
        log = logging.getLogger("note_gen.llms.text")
        log.info("调用文本 LLM（结构化输出）", extra={"json_mode": json_mode, "schema": schema.__name__})

        def _call() -> T:
            if json_mode:
                model_with_schema = self._model.with_structured_output(schema, method="json_mode")
            else:
                model_with_schema = self._model.with_structured_output(schema)
            return model_with_schema.invoke(prompt_messages)  # type: ignore[return-value]

        return self._limiter.run(_call)

    def invoke(self, prompt_messages: list) -> str:
        """以普通文本形式执行调用，返回模型原始内容。"""
        log = logging.getLogger("note_gen.llms.text")
        log.info("调用文本 LLM（普通输出）")

        def _call() -> str:
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

        return self._limiter.run(_call)
