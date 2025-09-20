from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """
    通用文本 LLM 配置。

    说明：
    - 采用 OpenAI 兼容接口参数，适配 langchain_openai.ChatOpenAI。
    - base_url 可为空（使用默认），model 必填，api_key 建议从 GUI 输入。
    """

    base_url: Optional[str] = Field(default=None, description="OpenAI 兼容服务地址")
    api_key: Optional[str] = Field(default=None, description="API Key（GUI 输入，日志脱敏）")
    model: str = Field(default="gpt-4o", description="模型名称，如 gpt-4o / o4-mini 等")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="采样温度")
    max_tokens: Optional[int] = Field(default=None, description="最大生成 token 数")
    request_timeout: int = Field(default=120, description="请求超时（秒）")
    concurrency: int = Field(default=1, ge=1, le=4, description="并发上限（当前串行处理建议为1）")


class MultiModalConfig(LLMConfig):
    """多模态 LLM 配置（继承通用配置）。"""

    use_base64_image: bool = Field(default=True, description="是否使用 base64 方式传图")


class ScreenshotConfig(BaseModel):
    """截图与九宫格配置。"""

    ffmpeg_path: str = Field(default="ffmpeg", description="ffmpeg 可执行文件路径")
    low_width: int = Field(default=320, description="九宫格单格低清宽度")
    low_height: int = Field(default=180, description="九宫格单格低清高度")
    grid_columns: int = Field(default=3, description="九宫格列数")
    grid_rows: int = Field(default=3, description="九宫格行数")
    hi_quality: int = Field(default=2, ge=2, le=31, description="ffmpeg -q:v 值，数值越小质量越高")
    include_endpoints: bool = Field(default=True, description="九宫格是否包含区间两端点")

    @property
    def grid_count(self) -> int:
        return self.grid_columns * self.grid_rows


class ExportConfig(BaseModel):
    """导出与目录配置。"""

    outputs_root: Path = Field(default=Path("outputs"), description="输出根目录")
    evidence_root: Path = Field(default=Path("evidence"), description="证据根目录")
    logs_root: Path = Field(default=Path("logs"), description="日志根目录")
    save_prompts_and_raw: bool = Field(default=True, description="保存提示词与原始返回以供审计")

    @field_validator("outputs_root", "evidence_root", "logs_root")
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class AppConfig(BaseModel):
    """
    应用总配置（支持 cache.json 记忆）。
    """

    text_llm: LLMConfig = Field(default_factory=LLMConfig)
    mm_llm: MultiModalConfig = Field(default_factory=MultiModalConfig)
    screenshot: ScreenshotConfig = Field(default_factory=ScreenshotConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
