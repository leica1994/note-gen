from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from core.config.schema import ScreenshotConfig
from core.utils.retry import RetryPolicy


import bisect
import json

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

    def _grid_tile_shape(
        self,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[int, int]:
        """计算九宫格单格尺寸，确保不低于配置的最小分辨率。"""

        base_w = width if width is not None else self.cfg.low_width
        base_h = height if height is not None else self.cfg.low_height
        min_w = getattr(self.cfg, "grid_min_width", base_w)
        min_h = getattr(self.cfg, "grid_min_height", base_h)
        tile_w = max(int(base_w), int(min_w))
        tile_h = max(int(base_h), int(min_h))
        return tile_w, tile_h

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
            "-loglevel",
            "error",
            *self._hwaccel_args(),
            "-ss",
            f"{pre_seek:.3f}",
            "-i",
            str(video),
            "-ss",
            f"{post_seek:.3f}",
            "-frames:v",
            "1",
        ]
        suffix = out_path.suffix.lower()
        if width and height:
            cmd += ["-vf", f"scale={width}:{height}"]
        if suffix == ".png":
            level_raw = getattr(self.cfg, "png_compression_level", 9)
            try:
                level = int(level_raw)
            except (TypeError, ValueError):
                level = 9
            # clamp 到 0-9，PNG 压缩仍为无损，仅影响编码耗时与文件体积
            level = max(0, min(9, level))
            cmd += [
                "-f",
                "image2",
                "-vcodec",
                "png",
                "-compression_level",
                str(level),
            ]
        elif quality is not None:
            cmd += ["-q:v", str(quality)]
        cmd += ["-y", str(out_path)]

        def _invoke():
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg 截图失败：{res.stderr}")
            return None

        wrapped = self._retry(_invoke)
        return wrapped()

    def align_timestamps_to_frames(
        self,
        video: Path,
        timestamps: Iterable[float],
        margin: float = 0.2,
    ) -> tuple[List[float], float] | None:
        """使用 ffprobe 将理想时间戳吸附到真实帧时间。

        - 读取 [min(ts)-margin, max(ts)+margin] 区间内的帧时间戳；
        - 逐个时间点选择距离最近且不回退的实际帧；
        - 返回调整后的时间列表与建议容差（约等于最小帧间隔的 1/3）。
        """
        ts_list = [float(t) for t in timestamps]
        if not ts_list:
            return None

        t_start = max(0.0, min(ts_list) - float(margin))
        t_stop = max(ts_list) + float(margin)
        read_span = max(0.5, t_stop - t_start)
        cmd = [
            self.cfg.ffprobe_path,
            "-hide_banner",
            "-loglevel", "error",
            "-select_streams", "v:0",
            "-read_intervals", f"{t_start:.3f}%+{read_span:.3f}",
            "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time,pts_time",
            "-of", "json",
            str(video),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            return None
        if res.returncode != 0:
            return None
        try:
            data = json.loads(res.stdout or "{}")
        except json.JSONDecodeError:
            return None

        frames = data.get("frames") or []
        pts_seq: List[float] = []
        for frame in frames:
            val = (
                frame.get("best_effort_timestamp_time")
                or frame.get("pkt_pts_time")
                or frame.get("pts_time")
            )
            if val is None:
                continue
            try:
                pts_seq.append(float(val))
            except (TypeError, ValueError):
                continue
        if not pts_seq:
            return None

        pts_seq = sorted(set(pts_seq))
        gaps = [pts_seq[i + 1] - pts_seq[i] for i in range(len(pts_seq) - 1) if pts_seq[i + 1] > pts_seq[i]]
        min_gap = min(gaps) if gaps else 0.0

        aligned: List[float] = []
        last_idx = 0
        max_idx = len(pts_seq) - 1
        for target in ts_list:
            idx = bisect.bisect_left(pts_seq, target, lo=last_idx)
            candidates: List[int] = []
            if idx <= max_idx:
                candidates.append(idx)
            if idx - 1 >= last_idx:
                candidates.append(idx - 1)
            if not candidates and last_idx <= max_idx:
                candidates.append(last_idx)
            if not candidates:
                candidates.append(max_idx)
            chosen = min(candidates, key=lambda i: abs(pts_seq[i] - target))
            if chosen < last_idx:
                chosen = last_idx
            last_idx = chosen
            aligned.append(pts_seq[chosen])
            if last_idx < max_idx and pts_seq[last_idx] < target and pts_seq[last_idx + 1] - pts_seq[last_idx] < max(0.5, min_gap * 1.5):
                last_idx = min(last_idx + 1, max_idx)

        tol_hint = min_gap / 3 if min_gap > 0 else 0.02
        return aligned, tol_hint

    def _classify_ffmpeg(self, e: Exception) -> str | None:
        # 将所有异常按 5xx 类对待，触发 2s 间隔重试
        return "5xx"

    # 无需额外方法包装

    def capture_thumbs(self, video: Path, timestamps: Iterable[float], out_dir: Path) -> List[Path]:
        paths: List[Path] = []
        tile_w, tile_h = self._grid_tile_shape()
        for idx, t in enumerate(timestamps, start=1):
            out = out_dir / f"thumb_{idx}.jpg"
            # 精确寻址后再缩放，确保九宫格缩略图与时间戳一一对应
            self._run_ffmpeg_dual_seek(
                video,
                out,
                t,
                pre_window=5.0,
                width=tile_w,
                height=tile_h,
                quality=3,
            )
            paths.append(out)
        return paths

    def compose_grid(self, thumbs: List[Path], out_path: Path) -> Path:
        cols, rows = self.cfg.grid_columns, self.cfg.grid_rows
        if len(thumbs) != cols * rows:
            raise ValueError("九宫格缩略图数量不匹配")
        tile_w, tile_h = self._grid_tile_shape()
        grid = Image.new("RGB", (tile_w * cols, tile_h * rows))
        for i, p in enumerate(thumbs):
            img = Image.open(p).convert("RGB")
            if img.size != (tile_w, tile_h):
                img = img.resize((tile_w, tile_h), Image.LANCZOS)
            col = i % cols
            row = i // cols
            grid.paste(img, (col * tile_w, row * tile_h))
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
        w, h = self._grid_tile_shape(width, height)

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
