from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

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
    # 采样温度：默认 0.7（与 cache.json 缺省一致）
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    max_tokens: Optional[int] = Field(default=None, description="最大生成 token 数")
    request_timeout: int = Field(default=300, description="请求超时（秒）")
    # 放开并发上限约束：仅要求 >=1；具体风险由使用方自行控制
    concurrency: int = Field(default=10, ge=1, description="并发数（默认10，按需限流）")


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
    # 为降低“过渡帧/淡入淡出”被选中的概率，在生成九宫格时间点时对区间首尾预留边缘安全距离。
    # 说明：
    # - 该值表示在 [start, end] 区间两端各向内收缩的秒数；
    # - 实际使用时会与区间长度联动，避免区间过短导致无法取满帧。
    edge_margin_sec: float = Field(default=0.5, ge=0.0, description="时间点生成的首尾边缘留白（秒）")
    max_workers: int = Field(default=1, ge=1, le=64, description="截图/段落处理并发数（固定串行建议设为1）")
    # 硬件加速（可选）：例如 cuda/qsv/dxva2/d3d11va/vaapi 等；默认不启用
    hwaccel: Optional[str] = Field(default=None,
                                   description="ffmpeg -hwaccel 参数（如 cuda/qsv/dxva2/vaapi），默认 None 关闭")
    hwaccel_device: Optional[str] = Field(default=None,
                                          description="ffmpeg -hwaccel_device（如 cuda:0 或 /dev/dri/renderD128），默认 None")

    @property
    def grid_count(self) -> int:
        return self.grid_columns * self.grid_rows


class ExportConfig(BaseModel):
    """导出与目录配置。"""

    outputs_root: Path = Field(default=Path("outputs"), description="输出根目录")
    logs_root: Path = Field(default=Path("logs"), description="日志根目录")

    @field_validator("outputs_root", "logs_root")
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class NoteConfig(BaseModel):
    """
    笔记生成与展示配置。

    - mode：
      - "subtitle"：字幕模式（当前默认，逐行展示字幕行）
      - "optimized"：AI 优化模式（段落内容经过优化合并并标注重点）
    """

    mode: Literal["subtitle", "optimized"] = Field(default="subtitle", description="笔记模式")


class AppConfig(BaseModel):
    """
    应用总配置（支持 cache.json 记忆）。
    """

    text_llm: LLMConfig = Field(default_factory=LLMConfig)
    mm_llm: MultiModalConfig = Field(default_factory=MultiModalConfig)
    screenshot: ScreenshotConfig = Field(default_factory=ScreenshotConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    note: NoteConfig = Field(default_factory=NoteConfig)
