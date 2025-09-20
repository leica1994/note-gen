from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import AppConfig


class ConfigCache:
    """配置记忆（cache.json）读写工具。

    - 文件位置：项目根目录 `cache.json`
    - 读写均使用 UTF-8，无 BOM
    - 失败时返回默认配置，不兜底继续执行
    """

    def __init__(self, path: Path | str = "cache.json") -> None:
        self.path = Path(path)

    def load(self) -> AppConfig:
        if not self.path.exists():
            return AppConfig()
        try:
            data = json.loads(self.path.read_text("utf-8"))
            return AppConfig.model_validate(data)
        except Exception as e:
            raise RuntimeError(f"读取配置失败：{self.path}，错误：{e}")

    def save(self, config: AppConfig) -> None:
        # 使用 Pydantic v2 的 JSON 模式，自动将 Path 等类型转换为可序列化内容
        data: dict[str, Any] = config.model_dump(mode="json")
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
