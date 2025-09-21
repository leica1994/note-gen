from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """将日志格式化为 JSON 行，便于审计与检索。

    输出字段：时间、级别、logger 名称、消息、可选额外字段（如 task_id、chapter、para 等）。
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        # 使用东八区时间，格式：yyyy-MM-dd HH:mm:ss
        tz8 = timezone(timedelta(hours=8))
        dt = datetime.fromtimestamp(record.created, tz=tz8).strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "time": dt,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 合并额外字段（extra）
        for k, v in record.__dict__.items():
            # 过滤标准字段与 asyncio 注入的 taskName（避免恒为 null 干扰阅读）
            if k in {"args", "msg", "name", "levelno", "levelname", "created", "msecs", "relativeCreated", "pathname",
                     "filename", "module", "lineno", "funcName", "exc_info", "exc_text", "stack_info", "thread",
                     "threadName", "processName", "process", "taskName"}:
                continue
            # 跳过内部记录属性，仅保留用户 extra
            if k.startswith("_"):
                continue
            payload[k] = v

        # 若未显式提供 task_id，则尝试从 logger 名称中提取（note_gen.task.{task_id}[.child...]）
        try:
            if "task_id" not in payload and isinstance(record.name, str):
                prefix = "note_gen.task."
                if record.name.startswith(prefix):
                    rest = record.name[len(prefix):]
                    task_id = rest.split(".")[0] if rest else None
                    if task_id:
                        payload["task_id"] = task_id
        except Exception:
            # 提取失败不影响原始日志
            pass
        # 为兼容期望字段，若存在 task_id 则补充 taskName（避免外部读取到 null）
        if "task_id" in payload and "taskName" not in payload:
            payload["taskName"] = payload["task_id"]
        return json.dumps(payload, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """将日志格式化为简洁的人类可读文本（控制台）。

    规则：
    - 时间使用东八区：yyyy-MM-dd HH:mm:ss
    - 基本格式："<time> <level> <message> | k=v ..."
    - 仅展示常用关键字段（从 extra 提取），并对文件路径取文件名以避免过长。
    - 若未显式提供 task_id，则从 logger 名（note_gen.task.{task_id}）中提取并展示。
    """

    _IGNORE = {
        "args", "msg", "name", "levelno", "levelname", "created", "msecs",
        "relativeCreated", "pathname", "filename", "module", "lineno",
        "funcName", "exc_info", "exc_text", "stack_info", "thread",
        "threadName", "processName", "process", "taskName",
    }

    _KEY_ORDER = [
        "task_id", "chapter_index", "para_index", "title", "chosen_index",
        "timestamp_sec", "chapters", "paragraphs", "progress", "status",
        "cost_ms", "video", "subtitle", "grid", "hi_image", "error",
    ]

    @staticmethod
    def _basename(v: object) -> object:
        if isinstance(v, str) and ("/" in v or "\\" in v):
            try:
                return Path(v).name
            except Exception:
                return v
        return v

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        tz8 = timezone(timedelta(hours=8))
        dt = datetime.fromtimestamp(record.created, tz=tz8).strftime("%Y-%m-%d %H:%M:%S")
        base = f"{dt} {record.levelname} {record.getMessage()}"

        extras: dict[str, object] = {}
        for k, v in record.__dict__.items():
            if k in self._IGNORE:
                continue
            if k.startswith("_"):
                continue
            extras[k] = v

        # 若缺少 task_id，从 logger 名提取
        try:
            if "task_id" not in extras and isinstance(record.name, str):
                prefix = "note_gen.task."
                if record.name.startswith(prefix):
                    rest = record.name[len(prefix):]
                    task_id = rest.split(".")[0] if rest else None
                    if task_id:
                        extras["task_id"] = task_id
        except Exception:
            pass

        parts: list[str] = []
        for key in self._KEY_ORDER:
            if key in extras and extras[key] is not None and extras[key] != "":
                val = self._basename(extras[key])
                parts.append(f"{key}={val}")

        for k, v in extras.items():
            if k not in self._KEY_ORDER and v not in (None, ""):
                val = self._basename(v)
                parts.append(f"{k}={val}")

        if parts:
            return f"{base} | {' '.join(parts)}"
        return base


def init_task_logger(task_id: str, logs_root: Path, *, level: int = logging.INFO) -> logging.Logger:
    """初始化任务级别 logger。

    - 控制台与文件统一使用 PlainFormatter，便于人工阅读与统一查看；
    - 若需审计 JSON，可将文件 Handler 的 Formatter 切换为 JsonFormatter。
    """
    logger_name = f"note_gen.task.{task_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # 若已初始化则直接返回
    if logger.handlers:
        return logger

    # 控制台输出（简洁文本）
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(PlainFormatter())
    logger.addHandler(sh)

    # 文件输出（简洁文本）
    log_dir = Path(logs_root) / task_id
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(PlainFormatter())
    logger.addHandler(fh)

    return logger


def get_child(logger: logging.Logger, name: str) -> logging.Logger:
    """基于任务 logger 创建子 logger。"""
    return logging.getLogger(f"{logger.name}.{name}")
