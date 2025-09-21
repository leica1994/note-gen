from __future__ import annotations

import base64
from typing import Type, TypeVar
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

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

        # 外层增加结构化解析容错重试（最多 3 次）；HTTP 类错误的重试由装饰器处理
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                return _invoke()
            except Exception as e:  # noqa: BLE001
                last_err = e
                # 装饰器已对 429/5xx/timeout 执行过退避与重试；若仍抛出则直接跳出
                category = classify_http_exception(e)
                if category is not None:
                    log.error("多模态结构化调用失败（HTTP类）", extra={
                        "attempt": attempt,
                        "category": category,
                        "error_cls": e.__class__.__name__,
                        "error": str(e)[:800],
                    })
                    break
                # 结构化解析/校验异常重试（常见于返回带 ```json 围栏的内容）
                msg_text = str(e)
                is_parse_error = isinstance(e, ValidationError) or (
                    "json_invalid" in msg_text or "Invalid JSON" in msg_text or "validation error" in msg_text
                )
                if is_parse_error and attempt < 3:
                    log.warning("多模态结构化解析失败，准备重试", extra={
                        "attempt": attempt,
                        "max": 3,
                        "error_cls": e.__class__.__name__,
                        "error": msg_text[:800],
                    })
                    continue
                # 其他异常：记录并退出
                log.error("多模态结构化调用失败（非HTTP/解析类）", extra={
                    "attempt": attempt,
                    "error_cls": e.__class__.__name__,
                    "error": str(e)[:800],
                })
                break
        assert last_err is not None
        log.error("多模态结构化调用最终失败", extra={
            "attempts": 3,
            "error_cls": last_err.__class__.__name__,
            "error": str(last_err)[:1200],
        })
        raise last_err
