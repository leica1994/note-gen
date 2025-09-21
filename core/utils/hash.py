from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_task(video_path: str | Path, subtitle_path: str | Path, params: dict[str, Any]) -> str:
    """基于视频、字幕与参数快照生成任务ID（sha1前12位）。"""
    h = hashlib.sha1()
    h.update(str(Path(video_path)).encode("utf-8"))
    h.update(b"|")
    h.update(str(Path(subtitle_path)).encode("utf-8"))
    h.update(b"|")
    h.update(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()[:12]
