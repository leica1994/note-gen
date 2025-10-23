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

    性能策略（变更说明）：
    - 九宫格缩略图：使用“快速寻址”（`-ss` 放在 `-i` 之前），显著降低靠后时间戳的解码成本；
    - 高清重拍：使用“快+准双 -ss”（先前置粗跳、再后置精确寻址）以兼顾性能与帧精度。
    """

    def __init__(self, cfg: ScreenshotConfig) -> None:
        self.cfg = cfg
        self._retry = RetryPolicy(max_retries=3)

    def _hwaccel_args(self) -> list[str]:
        args: list[str] = []
        # 硬件加速参数（可选）：只有在配置提供时才附加
        if getattr(self.cfg, "hwaccel", None):
            args += ["-hwaccel", str(self.cfg.hwaccel)]
            if getattr(self.cfg, "hwaccel_device", None):
                args += ["-hwaccel_device", str(self.cfg.hwaccel_device)]
        return args

    def _run_ffmpeg_fast_seek(self, video: Path, out_path: Path, ts_sec: float,
                              width: int | None = None, height: int | None = None,
                              quality: int | None = None) -> None:
        """快速寻址：`-ss` 放在 `-i` 之前，牺牲极小精度换取显著速度。

        适用：九宫格缩略图等“代表性帧”场景。
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.cfg.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            *self._hwaccel_args(),
            "-ss", f"{ts_sec:.3f}",
            "-i", str(video),
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

        # 修正：RetryPolicy 作为装饰器使用，不接受分类器参数；
        # 之前传入 _classify_ffmpeg 会导致返回字符串，被当作可调用引发 TypeError。
        wrapped = self._retry(_invoke)
        return wrapped()

    def _run_ffmpeg_dual_seek(
        self,
        video: Path,
        out_path: Path,
        ts_sec: float,
        pre_window: float = 5.0,
        width: int | None = None,
        height: int | None = None,
        quality: int | None = None,
    ) -> None:
        """双阶段寻址：先粗跳、再精确，可选缩放确保尺寸一致。

        - 先执行 `-ss pre_seek -i input` 进行快速粗跳；
        - 再执行 `-ss post_seek` 精确到目标时间；
        - 若提供 `width/height`，附加 scale 滤镜以生成固定尺寸缩略图。
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pre_seek = max(0.0, float(ts_sec) - float(pre_window))
        post_seek = float(ts_sec) - pre_seek
        cmd = [
            self.cfg.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            *self._hwaccel_args(),
            "-ss", f"{pre_seek:.3f}",
            "-i", str(video),
            "-ss", f"{post_seek:.3f}",
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

        wrapped = self._retry(_invoke)
        return wrapped()

    def _classify_ffmpeg(self, e: Exception) -> str | None:
        # 将所有异常按 5xx 类对待，触发 2s 间隔重试
        return "5xx"

    # 无需额外方法包装

    def capture_thumbs(self, video: Path, timestamps: Iterable[float], out_dir: Path) -> List[Path]:
        paths: List[Path] = []
        for idx, t in enumerate(timestamps, start=1):
            out = out_dir / f"thumb_{idx}.jpg"
            # 精确寻址后再缩放，确保九宫格缩略图与时间戳一一对应
            self._run_ffmpeg_dual_seek(
                video,
                out,
                t,
                pre_window=5.0,
                width=self.cfg.low_width,
                height=self.cfg.low_height,
                quality=10,
            )
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

    def compose_grid_one_shot(self, video: Path, timestamps: List[float], out_path: Path,
                              *, cols: int | None = None, rows: int | None = None,
                              width: int | None = None, height: int | None = None,
                              tol: float = 0.05) -> Path:
        """一次 ffmpeg 完成九宫格（无需中间缩略图）。

        - 通过 select + tile 过滤器从同一段时间窗口内选取 9 帧并铺成网格；
        - 为降低解码成本，使用前置粗跳，并以 -t 限制解码区间；
        - 注意：-ss 作为输入预跳后，filtergraph 中 t 通常以“预跳后起点”为 0；
          因此此处使用相对时间（ti - t_min）构造 between 窗口。
        - 若失败（帧数不足/命令异常）应由调用方决定是否降级为旧方案。
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(timestamps)
        if n == 0:
            raise ValueError("时间点列表为空")
        cols = cols or self.cfg.grid_columns
        rows = rows or self.cfg.grid_rows
        expected = cols * rows
        if n != expected:
            raise ValueError("时间点数量与网格大小不匹配")
        w = width or self.cfg.low_width
        h = height or self.cfg.low_height

        t_min = max(0.0, float(min(timestamps)) - 1.0)
        t_max = float(max(timestamps)) + 1.0
        clip_dur = max(0.5, t_max - t_min)

        # 构造 select 表达式：between(t, (ti_rel)-tol, (ti_rel)+tol) 的加和
        # 注意：ffmpeg filtergraph 中，select 的表达式内逗号需要使用反斜杠转义（避免被当作滤镜分隔符）。
        # 在 Python 字面量中需写成 "\\," 才能得到实际的 "\,"。
        parts = []
        for ti in timestamps:
            ti_rel = float(ti) - t_min
            parts.append(f"between(t\\,{ti_rel - tol:.3f}\\,{ti_rel + tol:.3f})")
        select_expr = "+".join(parts)
        vf = f"select='{select_expr}',scale={w}:{h},tile={cols}x{rows}:margin=0:padding=0"

        cmd = [
            self.cfg.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            *self._hwaccel_args(),
            "-ss", f"{t_min:.3f}",
            "-i", str(video),
            "-t", f"{clip_dur:.3f}",
            "-vf", vf,
            "-frames:v", "1",
            "-vsync", "0",
            "-q:v", "3",
            "-y", str(out_path),
        ]

        def _invoke():
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg 九宫格一把流失败：{res.stderr}")
            return None

        wrapped = self._retry(_invoke)
        wrapped()
        # 若未产生输出文件（常见于 select 未命中任何帧），显式视为失败，交由调用方走回退路径
        try:
            if not out_path.exists() or out_path.stat().st_size <= 0:
                raise RuntimeError(f"ffmpeg 九宫格一把流未产生输出：{out_path}")
        except Exception:
            # 统一抛出失败由调用方回退
            raise
        return out_path

    def capture_high_quality(self, video: Path, timestamp: float, out_path: Path) -> Path:
        # 使用“快+准双 -ss”以兼顾性能与精度
        self._run_ffmpeg_dual_seek(video, out_path, timestamp, pre_window=5.0, quality=self.cfg.hi_quality)
        return out_path
