from __future__ import annotations

from typing import List

from core.config.schema import ScreenshotConfig


def generate_grid_timestamps(start_sec: float, end_sec: float, cfg: ScreenshotConfig) -> List[float]:
    """在区间内生成九宫格时间点。

    设计目标（更新）：
    - 固定使用区间起始、结束时间作为首尾两张截图；
    - 其余 `grid_count - 2` 张按照时间区间均匀分割；
    - 输出时间点严格递增，尽可能保持高精度，避免重复。
    """
    if end_sec <= start_sec:
        raise ValueError("时间区间非法")

    n = cfg.grid_count
    if n <= 0:
        raise ValueError("九宫格数量配置非法")
    if n == 1:
        # 极端配置下仅返回起始帧
        return [round(start_sec, 3)]

    span = float(end_sec) - float(start_sec)
    step = span / float(n - 1)
    raw_points = [start_sec + step * i for i in range(n)]
    raw_points[0] = start_sec
    raw_points[-1] = end_sec

    rounded = [round(pt, 3) for pt in raw_points]
    if len(set(rounded)) < n:
        rounded = [round(pt, 6) for pt in raw_points]
    if len(set(rounded)) < n:
        rounded = raw_points

    for i in range(1, n):
        if rounded[i] <= rounded[i - 1]:
            raise ValueError("时间点非严格递增")
    return rounded
