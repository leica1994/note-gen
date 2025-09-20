from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from core.config.schema import ScreenshotConfig
from core.utils.retry import RetryPolicy


class Screenshotter:
    """基于 ffmpeg 的截图工具。

    - 支持按指定时间点批量截帧
    - 生成九宫格（低清）与高清单帧
    - 若 ffmpeg 不存在或命令失败，抛出异常（不兜底）
    """

    def __init__(self, cfg: ScreenshotConfig) -> None:
        self.cfg = cfg
        self._retry = RetryPolicy(max_retries=3)

    def _run_ffmpeg(self, video: Path, out_path: Path, ts_sec: float, width: int | None = None, height: int | None = None, quality: int | None = None) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 为保证精确度，-ss 放在 -i 之后（精确寻址，速度略慢，质量优先）
        cmd = [
            self.cfg.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video),
            "-ss", f"{ts_sec:.3f}",
            "-frames:v", "1",
        ]
        if width and height:
            cmd += ["-vf", f"scale={width}:{height}"]
        if quality is not None:
            cmd += ["-q:v", str(quality)]
        cmd += ["-y", str(out_path)]

        def _invoke():
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg 截图失败：{res.stderr}")
            return None

        wrapped = self._retry(self._classify_ffmpeg)(_invoke)
        return wrapped()

    def _classify_ffmpeg(self, e: Exception) -> str | None:
        # 将所有异常按 5xx 类对待，触发 2s 间隔重试
        return "5xx"

    # 无需额外方法包装

    def capture_thumbs(self, video: Path, timestamps: Iterable[float], out_dir: Path) -> List[Path]:
        paths: List[Path] = []
        for idx, t in enumerate(timestamps, start=1):
            out = out_dir / f"thumb_{idx}.jpg"
            self._run_ffmpeg(video, out, t, width=self.cfg.low_width, height=self.cfg.low_height, quality=10)
            paths.append(out)
        return paths

    def compose_grid(self, thumbs: List[Path], out_path: Path) -> Path:
        cols, rows = self.cfg.grid_columns, self.cfg.grid_rows
        if len(thumbs) != cols * rows:
            raise ValueError("九宫格缩略图数量不匹配")
        w, h = self.cfg.low_width, self.cfg.low_height
        grid = Image.new("RGB", (w * cols, h * rows))
        for i, p in enumerate(thumbs):
            img = Image.open(p).convert("RGB")
            col = i % cols
            row = i // cols
            grid.paste(img, (col * w, row * h))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(out_path, format="JPEG", quality=85)
        return out_path

    def capture_high_quality(self, video: Path, timestamp: float, out_path: Path) -> Path:
        self._run_ffmpeg(video, out_path, timestamp, quality=self.cfg.hi_quality)
        return out_path
