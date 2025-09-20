from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Any


P = ParamSpec("P")
R = TypeVar("R")


class RetryPolicy:
    """
    简单重试策略：
    - 最大重试次数（不含首次）
    - 针对 HTTP 429 退避 20s
    - 针对 5xx / 超时退避 2s，最多一次额外重试
    - 用户自定义判定函数（返回 '429'/'5xx'/None）
    - 其他错误不重试，直接抛出
    """

    def __init__(self, max_retries: int = 3, *, category_max: dict[str, int] | None = None) -> None:
        """
        max_retries: 默认类别最大重试次数（不含首次）。
        category_max: 按类别覆盖的最大重试次数配置，如 {"429":3, "5xx":1, "timeout":1}
        说明：根据合规要求，5xx/timeout 仅允许最多一次重试；429 可多次（默认 3）。
        """
        self.max_retries = max_retries
        self.category_max = category_max or {"429": 3, "5xx": 1, "timeout": 1}

    def __call__(self, classify: Callable[[Exception], str | None]):
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # 记录各类别已重试次数
                retried_by_cat: dict[str, int] = {k: 0 for k in self.category_max}
                last_err: Exception | None = None
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                        category = classify(e)
                        if category in {"429", "5xx", "timeout"}:
                            # 检查是否达到该类别最大重试次数
                            current = retried_by_cat.get(category, 0)
                            limit = self.category_max.get(category, self.max_retries)
                            if current >= limit:
                                break
                            # 退避
                            if category == "429":
                                time.sleep(20)
                            else:  # 5xx 或 timeout
                                time.sleep(2)
                            retried_by_cat[category] = current + 1
                            continue
                        # 其他错误不重试
                        break
                assert last_err is not None
                raise last_err

            return wrapper

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
