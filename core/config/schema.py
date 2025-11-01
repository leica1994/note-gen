from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    streaming: bool = Field(default=False, description="是否启用流式输出")
    # 放开并发上限约束：仅要求 >=1；具体风险由使用方自行控制
    concurrency: int = Field(default=10, ge=1, description="并发数（默认10，按需限流）")


class MultiModalConfig(LLMConfig):
    """多模态 LLM 配置（继承通用配置）。"""

    use_base64_image: bool = Field(default=True, description="是否使用 base64 方式传图")


class ScreenshotConfig(BaseModel):
    """截图与九宫格配置。"""

    ffmpeg_path: str = Field(default="ffmpeg", description="ffmpeg 可执行文件路径")
    ffprobe_path: str = Field(default="ffprobe", description="ffprobe 可执行文件路径")
    low_width: int = Field(default=960, description="九宫格单格低清宽度")
    low_height: int = Field(default=540, description="九宫格单格低清高度")
    grid_columns: int = Field(default=3, description="九宫格列数")
    grid_rows: int = Field(default=3, description="九宫格行数")
    grid_min_width: int = Field(
        default=960,
        ge=1,
        description="九宫格单格最小宽度，用于保障发给多模态模型的截图清晰度",
    )
    grid_min_height: int = Field(
        default=540,
        ge=1,
        description="九宫格单格最小高度，用于保障发给多模态模型的截图清晰度",
    )
    png_compression_level: int = Field(
        default=9,
        ge=0,
        le=9,
        description="PNG 压缩等级，0 为最快、9 为最小体积（仍保持无损）",
    )
    png_force_rgb24: bool = Field(
        default=True,
        description="PNG 输出时是否强制转换为 RGB24（去掉透明通道以减小体积）",
    )
    png_optimize: bool = Field(
        default=True,
        description="是否使用 pyoxipng 对 PNG 做无损优化（保持分辨率与画质）",
    )
    png_strip_metadata: bool = Field(
        default=True,
        description="pyoxipng 优化时是否剥离非必要元数据（保留解码所需字段）",
    )
    hi_quality: int = Field(default=2, ge=2, le=31, description="ffmpeg -q:v 值，数值越小质量越高")
    include_endpoints: bool = Field(
        default=True,
        description="九宫格是否包含区间两端点（当前逻辑固定包含首尾，参数保留向后兼容）",
    )
    # 历史参数：旧逻辑用于避开过渡帧，现阶段九宫格固定包含首尾时间点，保留该配置以兼容旧 cache
    edge_margin_sec: float = Field(default=0.5, ge=0.0, description="（兼容保留）时间点生成的首尾边缘留白（秒）")
    max_workers: int = Field(default=1, ge=1, le=64, description="截图/段落处理并发数（固定串行建议设为1）")
    # 硬件加速（可选）：例如 cuda/qsv/dxva2/d3d11va/vaapi 等；默认不启用
    hwaccel: Optional[str] = Field(default=None,
                                   description="ffmpeg -hwaccel 参数（如 cuda/qsv/dxva2/vaapi），默认 None 关闭")
    hwaccel_device: Optional[str] = Field(default=None,
                                          description="ffmpeg -hwaccel_device（如 cuda:0 或 /dev/dri/renderD128），默认 None")

    @model_validator(mode="after")
    def _ensure_minimum_resolution(self):
        min_w = max(1, self.grid_min_width)
        min_h = max(1, self.grid_min_height)
        object.__setattr__(self, "grid_min_width", min_w)
        object.__setattr__(self, "grid_min_height", min_h)
        if self.low_width < min_w:
            object.__setattr__(self, "low_width", min_w)
        if self.low_height < min_h:
            object.__setattr__(self, "low_height", min_h)
        return self

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
    - write_headings：导出 Markdown 时是否写入章节/段落标题。
    """

    mode: Literal["subtitle", "optimized"] = Field(default="subtitle", description="笔记模式")
    write_headings: bool = Field(default=True, description="导出 Markdown 时是否写入章节/段落标题")
    show_paragraph_time_range: bool = Field(
        default=False,
        description="导出 Markdown 时在段落首行写入时间戳范围",
    )
    note_dir: Optional[Path] = Field(default=None, description="笔记目录（Markdown 输出根目录优先覆盖）")
    screenshot_dir: Optional[Path] = Field(default=None, description="截图目录（高清重拍输出优先覆盖）")
    chapter_resegment_char_threshold: int = Field(
        default=1000,
        ge=0,
        description="单章字符阈值；超过该阈值时自动细分子章节，0 表示禁用",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data):
        """向后兼容旧字段：input_dir → note_dir，screenshot_input_dir → screenshot_dir。"""
        try:
            if isinstance(data, dict):
                if "input_dir" in data and "note_dir" not in data:
                    data["note_dir"] = data.get("input_dir")
                if "screenshot_input_dir" in data and "screenshot_dir" not in data:
                    data["screenshot_dir"] = data.get("screenshot_input_dir")
        except Exception:
            pass
        return data


class AppConfig(BaseModel):
    """
    应用总配置（支持 cache.json 记忆）。
    """

    text_llm: LLMConfig = Field(default_factory=LLMConfig)
    mm_llm: MultiModalConfig = Field(default_factory=MultiModalConfig)
    screenshot: ScreenshotConfig = Field(default_factory=ScreenshotConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    note: NoteConfig = Field(default_factory=NoteConfig)
