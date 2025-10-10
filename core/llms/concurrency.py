"""并发控制工具，确保对同一模型的调用数量不超过配置上限。"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Generator, TypeVar

T = TypeVar("T")


class ConcurrencyLimiter:
    """基于信号量的并发限制器。"""

    def __init__(self, limit: int) -> None:
        safe_limit = max(1, int(limit))
        self._limit = safe_limit
        self._semaphore = threading.Semaphore(safe_limit)
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        """返回当前并发上限。"""
        return self._limit

    @contextmanager
    def acquire(self) -> Generator[None, None, None]:
        """获取一个并发配额，用于 with 语句控制调用范围。"""
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    def run(self, func: Callable[..., T], *args, **kwargs) -> T:
        """在并发限制内执行函数。"""
        with self.acquire():
            return func(*args, **kwargs)

    def update_limit(self, new_limit: int) -> None:
        """增加并发上限；若需要缩减请外部重建实例以避免资源悬挂。"""
        safe_limit = max(1, int(new_limit))
        with self._lock:
            if safe_limit <= self._limit:
                self._limit = safe_limit
                return
            for _ in range(safe_limit - self._limit):
                self._semaphore.release()
            self._limit = safe_limit

