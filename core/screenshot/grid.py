from __future__ import annotations

from typing import List

from core.config.schema import ScreenshotConfig


def generate_grid_timestamps(start_sec: float, end_sec: float, cfg: ScreenshotConfig) -> List[float]:
    """在区间内生成九宫格时间点。

    - 默认包含两端点（可配置）
    - 要求能产生 cfg.grid_count 个“非降序且尽可能唯一”的点
    - 若区间太短无法取足够多的不同帧，抛出异常（不兜底）
    """
    if end_sec <= start_sec:
        raise ValueError("时间区间非法")
    n = cfg.grid_count
    if cfg.include_endpoints:
        step = (end_sec - start_sec) / (n - 1)
        arr = [round(start_sec + i * step, 3) for i in range(n)]
    else:
        step = (end_sec - start_sec) / (n + 1)
        arr = [round(start_sec + (i + 1) * step, 3) for i in range(n)]

    # 唯一性与单调性粗检（避免由于过短区间产生重复）
    if len(set(arr)) < n:
        raise ValueError("区间过短，无法生成足够的唯一时间点")
    for i in range(1, n):
        if not (arr[i] > arr[i - 1]):
            raise ValueError("时间点非严格递增")
    return arr

