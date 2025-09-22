from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Type, TypeVar

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from core.config.schema import MultiModalConfig
from core.utils.retry import RetryPolicy, classify_http_exception

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
        # 提升 LLM 查询的失败重试次数：
        # - 429（限流）：由原来的 3 次提升到 5 次（遵循合规退避 20s）
        # - 5xx/timeout：仍保持最多 1 次重试（退避 2s）
        # 说明：仅提升 LLM 相关重试次数，不影响其他模块（如 FFmpeg）。
        self._retry = RetryPolicy(
            max_retries=5,
            category_max={"429": 5, "5xx": 1, "timeout": 1},
        )

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
                    break
                # 结构化解析/校验异常重试（常见于返回带 ```json 围栏的内容）
                msg_text = str(e)
                is_parse_error = isinstance(e, ValidationError) or (
                        "json_invalid" in msg_text or "Invalid JSON" in msg_text or "validation error" in msg_text
                )
                if is_parse_error and attempt < 3:
                    log.info("结构化解析失败，准备重试", extra={"attempt": attempt, "max": 3})
                    continue
                break
        assert last_err is not None
        raise last_err

    def choose_index(self, instruction: str, image_path: str) -> int:
        """将九宫格图与文字说明一起发送，要求模型只返回 1..9 的数字。

        - 不使用结构化输出，避免多模态端点的 JSON 解析不稳定问题。
        - 对非数字响应进行至多 3 次解析重试；HTTP/429/5xx 的退避与重试由 RetryPolicy 处理。
        - 提示词建议在调用侧进一步强调“只返回一个数字，不要任何其他字符/空格/换行”。
        """
        log = logging.getLogger("note_gen.llms.mm")
        blocks = [
            {"type": "text", "text": instruction},
            self._image_block(image_path),
        ]
        msg = HumanMessage(content=blocks)

        @self._retry(classify_http_exception)
        def _invoke_text() -> str:
            log.info("调用多模态 LLM（纯文本输出）", extra={"image": image_path})
            resp = self._model.invoke([msg])
            content = getattr(resp, "content", None)
            if isinstance(content, str):
                return content.strip()
            return str(content)

        import re

        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                text = _invoke_text()
                m = re.search(r"([1-9])", text)
                if not m:
                    raise ValueError(f"非数字响应：{text!r}")
                return int(m.group(1))
            except Exception as e:  # noqa: BLE001
                last_err = e
                category = classify_http_exception(e)
                if category is not None:
                    break
                if attempt < 3:
                    log.info("解析数字失败，准备重试", extra={"attempt": attempt, "max": 3})
                    continue
                break
        assert last_err is not None
        raise last_err
