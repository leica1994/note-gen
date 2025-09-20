from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvidenceWriter:
    """证据归档工具。

    - 将提示词、原始响应、错误日志写入 evidence/{task_id}/ 下
    - 文件均为 UTF-8 文本；二进制另存
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def for_task(self, task_id: str) -> "EvidenceWriter":
        return EvidenceWriter(self.root / task_id)

    def write_text(self, name: str, content: str) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def write_json(self, name: str, obj: Any) -> Path:
        return self.write_text(name, json.dumps(obj, ensure_ascii=False, indent=2))

    def write_bytes(self, name: str, data: bytes) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

