from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from core.subtitles.models import SubtitleSegment, SubtitleDocument


class ParagraphImage(BaseModel):
    grid_image_path: Optional[Path] = None
    grid_timestamps_sec: List[float] = Field(default_factory=list)
    chosen_index: Optional[int] = None  # 1..9
    chosen_timestamp_sec: Optional[float] = None
    hi_res_image_path: Optional[Path] = None


class Paragraph(BaseModel):
    title: str
    start_sec: float
    end_sec: float
    lines: List[SubtitleSegment]
    children: List["Paragraph"] = Field(default_factory=list)
    image: Optional[ParagraphImage] = None
    optimized: List[str] = Field(default_factory=list,
                                 description="AI 优化后的段落内容（去语气词、加标点、组合为流畅句子；可分段则多项；可含 Markdown 标记重点）")


class Chapter(BaseModel):
    title: str
    start_sec: float
    end_sec: float
    paragraphs: List[Paragraph]


class Note(BaseModel):
    video_path: Path
    subtitle: SubtitleDocument
    chapters: List[Chapter]
    meta: dict = Field(default_factory=dict)


class GenerationInputMeta(BaseModel):
    video_path: Path
    subtitle: SubtitleDocument
    params: dict = Field(default_factory=dict)


# LLM 结构化输出契约（分章）
class ChapterBoundary(BaseModel):
    title: str
    start_line_no: int
    end_line_no: int


class ChaptersSchema(BaseModel):
    chapters: List[ChapterBoundary]


# LLM 结构化输出契约（分段）
class ParagraphLine(BaseModel):
    line_no: int
    start_sec: float
    end_sec: float
    text: str


class ParagraphSchema(BaseModel):
    title: str
    start_sec: float
    end_sec: float
    lines: List[ParagraphLine]
    children: List["ParagraphSchema"] = Field(default_factory=list)
    optimized: List[str] = Field(default_factory=list)

    @field_validator("children", mode="before")
    @classmethod
    def _sanitize_children(cls, v):
        """在模型验证前清洗 children：
        - None → []
        - 过滤空对象或缺少必填字段的条目（避免上游模型偶发输出 {}）。
        """
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                if not item:
                    continue
                if isinstance(item, dict):
                    required = {"title", "start_sec", "end_sec", "lines"}
                    if not required.issubset(set(item.keys())):
                        continue
                cleaned.append(item)
            return cleaned
        return []


class ParagraphsSchema(BaseModel):
    paragraphs: List[ParagraphSchema]

    @field_validator("paragraphs", mode="before")
    @classmethod
    def _sanitize_paragraphs(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                if not item:
                    continue
                if isinstance(item, dict):
                    required = {"title", "start_sec", "end_sec", "lines"}
                    if not required.issubset(set(item.keys())):
                        continue
                cleaned.append(item)
            return cleaned
        return []


class SelectedIndexSchema(BaseModel):
    index: int = Field(ge=1, le=9)
