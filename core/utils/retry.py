from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


class RetryPolicy:
    """
    通用重试策略（简化版）：
    - 任意异常统一最多重试 `max_retries` 次（不含首次）；
    - 每次重试固定退避 2 秒；
    - 作为装饰器使用：`@retry` 或 `@retry()` 均可。
    """

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries

    def __call__(self, func: Callable[P, R] | None = None) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
        def decorator(f: Callable[P, R]) -> Callable[P, R]:
            @wraps(f)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                """对任意异常进行最多 `max_retries` 次重试（统一退避 2s）。"""
                import logging

                log = logging.getLogger("note_gen.retry")
                attempts = 0  # 已重试次数（不含首次）
                last_err: Exception | None = None
                while True:
                    try:
                        return f(*args, **kwargs)
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                        if attempts >= self.max_retries:
                            break
                        attempts += 1
                        log.info("准备重试", extra={"attempt": attempts, "max": self.max_retries})
                        time.sleep(2)
                        continue
                assert last_err is not None
                raise last_err

            return wrapper

        # 允许两种用法：@retry 和 @retry()
        if func is not None and callable(func):
            return decorator(func)
        return decorator


def classify_http_exception(e: Exception) -> str | None:
    """基于常见异常信息的简单分类（OpenAI/HTTP 风格）。"""
    msg = str(e).lower()
    if "429" in msg or "rate limit" in msg:
        return "429"
    if "timeout" in msg:
        return "timeout"
    if " 5" in msg and "server" in msg:
        return "5xx"
    if " 5" in msg and "internal" in msg:
        return "5xx"
    return None
