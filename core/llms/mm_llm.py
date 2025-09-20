from __future__ import annotations

import base64
from typing import Type, TypeVar
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from core.config.schema import MultiModalConfig
from core.utils.retry import RetryPolicy, classify_http_exception
import logging

T = TypeVar("T", bound=BaseModel)


class MultiModalLLM:
    """多模态 LLM 封装：传入文本与图片（URL 或 base64）。"""

    def __init__(self, cfg: MultiModalConfig) -> None:
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

    def _image_block(self, image_path: str) -> dict:
        if self.cfg.use_base64_image:
            data = Path(image_path).read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return {
                "type": "image",
                "source_type": "base64",
                "data": b64,
                "mime_type": "image/jpeg",
            }
        else:
            return {
                "type": "image",
                "source_type": "url",
                "url": Path(image_path).as_uri(),
            }

    def structured_choose(self, schema: Type[T], instruction: str, image_path: str) -> T:
        """将九宫格图与文字说明一起发送，返回结构化结果。"""
        log = logging.getLogger("note_gen.llms.mm")
        blocks = [
            {"type": "text", "text": instruction},
            self._image_block(image_path),
        ]
        msg = HumanMessage(content=blocks)
        model_with_schema = self._model.with_structured_output(schema)

        @self._retry(classify_http_exception)
        def _invoke() -> T:
            log.info("调用多模态 LLM（结构化输出）", extra={"schema": schema.__name__, "image": image_path})
            return model_with_schema.invoke([msg])  # type: ignore[return-value]

        return _invoke()
