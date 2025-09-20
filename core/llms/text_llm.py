from __future__ import annotations

from typing import Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.config.schema import LLMConfig
from core.utils.retry import RetryPolicy, classify_http_exception
import logging

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
        self._retry = RetryPolicy(max_retries=3)

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
            try:
                model_with_schema = self._model.with_structured_output(schema)
            except Exception as e:
                # 某些 OpenAI 兼容端点可能需要显式声明 function_calling
                try:
                    log.info("with_structured_output 回退到 function_calling 模式")
                    model_with_schema = self._model.with_structured_output(schema, method="function_calling")
                except Exception:
                    raise e

        @self._retry(classify_http_exception)
        def _invoke() -> T:
            log.info("调用文本 LLM（结构化输出）", extra={"json_mode": json_mode, "schema": schema.__name__})
            return model_with_schema.invoke(prompt_messages)  # type: ignore[return-value]

        return _invoke()
