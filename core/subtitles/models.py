from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class SubtitleSegment(BaseModel):
    """统一的字幕片段模型。"""

    line_no: int = Field(description="行号（连续递增，从1开始）")
    start_sec: float = Field(ge=0, description="开始时间（秒）")
    end_sec: float = Field(ge=0, description="结束时间（秒，需 > 开始时间）")
    text: str = Field(description="字幕文本（保留原文，不做摘要）")


class SubtitleDocument(BaseModel):
    """统一的字幕文档。"""

    items: List[SubtitleSegment] = Field(default_factory=list)
    source_path: Path
    format: str = Field(description="来源格式：ass|srt|vtt")

