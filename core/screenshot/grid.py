from __future__ import annotations

from typing import List

from core.config.schema import ScreenshotConfig


def generate_grid_timestamps(start_sec: float, end_sec: float, cfg: ScreenshotConfig) -> List[float]:
    """在区间内生成九宫格时间点（避开过渡帧）。

    设计目标：
    - 默认包含两端点（可配置），但实际取“收缩后的端点”，以避免淡入/淡出等过渡帧；
    - 必须生成 `cfg.grid_count` 个严格递增且尽可能唯一的时间点；
    - 当区间过短时，自动缩小边缘留白，保证可用。
    """
    if end_sec <= start_sec:
        raise ValueError("时间区间非法")

    n = cfg.grid_count
    # 根据区间长度自适应边缘留白，最多占用 1/4 区间
    span = float(end_sec) - float(start_sec)
    adaptive_margin = min(max(0.0, cfg.edge_margin_sec), max(0.0, span / 4))
    s = start_sec + adaptive_margin
    e = end_sec - adaptive_margin

    # 若缩减后出现反转（极短区间），退化为不加留白
    if e <= s:
        s, e = start_sec, end_sec

    if cfg.include_endpoints:
        step = (e - s) / (n - 1)
        arr = [round(s + i * step, 3) for i in range(n)]
    else:
        step = (e - s) / (n + 1)
        arr = [round(s + (i + 1) * step, 3) for i in range(n)]

    # 唯一性与单调性粗检（避免由于过短区间产生重复）
    if len(set(arr)) < n:
        # 再次兜底：均匀采样在 (start, end) 内，偏向中部
        if span <= 0:
            raise ValueError("时间区间非法")
        # 采用不含端点的方式重算
        step = span / (n + 1)
        arr = [round(start_sec + (i + 1) * step, 3) for i in range(n)]

    for i in range(1, n):
        if not (arr[i] > arr[i - 1]):
            raise ValueError("时间点非严格递增")
    return arr
