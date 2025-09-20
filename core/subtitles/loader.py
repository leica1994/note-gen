from __future__ import annotations

from pathlib import Path
from typing import List

import pysubs2  # type: ignore
import webvtt  # type: ignore

from .models import SubtitleDocument, SubtitleSegment


def _to_seconds(ms: float) -> float:
    return round(ms / 1000.0, 3)


def load_subtitle(path: str | Path) -> SubtitleDocument:
    """加载并归一化字幕（ASS/VTT/SRT）。

    - ASS/SRT 使用 pysubs2
    - VTT 使用 webvtt
    - 返回统一的 SubtitleDocument，按开始时间排序，行号重新编号
    - 严格校验：时间戳必须递增且无重叠
    """
    p = Path(path)
    ext = p.suffix.lower()
    items: List[SubtitleSegment] = []
    fmt = ext.lstrip(".")

    if ext in {".srt", ".ass", ".ssa"}:
        subs = pysubs2.load(str(p))
        for i, event in enumerate(sorted(subs, key=lambda e: e.start), start=1):
            start_sec = _to_seconds(event.start)
            end_sec = _to_seconds(event.end)
            text = pysubs2.get_text(event)
            items.append(SubtitleSegment(line_no=i, start_sec=start_sec, end_sec=end_sec, text=text))
        fmt = "ass" if ext in {".ass", ".ssa"} else "srt"

    elif ext == ".vtt":
        vtt = webvtt.read(str(p))
        def ts_to_sec(ts: str) -> float:
            h, m, s = ts.split(":")
            sec = float(s.replace(",", "."))
            return int(h) * 3600 + int(m) * 60 + sec

        for i, c in enumerate(vtt, start=1):
            start_sec = ts_to_sec(c.start)
            end_sec = ts_to_sec(c.end)
            text = c.text
            items.append(SubtitleSegment(line_no=i, start_sec=start_sec, end_sec=end_sec, text=text))
        fmt = "vtt"

    else:
        raise ValueError(f"不支持的字幕格式：{ext}")

    # 重新编号并排序
    items = sorted(items, key=lambda s: (s.start_sec, s.end_sec))
    for i, seg in enumerate(items, start=1):
        seg.line_no = i

    # 校验时间戳不重叠
    last_end = -1.0
    for seg in items:
        if seg.start_sec < 0 or seg.end_sec <= seg.start_sec:
            raise ValueError(f"非法时间范围：{seg}")
        if last_end > 0 and seg.start_sec < last_end:
            raise ValueError(f"时间重叠：上一条结束 {last_end}s，当前开始 {seg.start_sec}s，行号 {seg.line_no}")
        last_end = seg.end_sec

    return SubtitleDocument(items=items, source_path=p, format=fmt)

