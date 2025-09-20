from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


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
            if k in {"args", "msg", "name", "levelno", "levelname", "created", "msecs", "relativeCreated", "pathname", "filename", "module", "lineno", "funcName", "exc_info", "exc_text", "stack_info", "thread", "threadName", "processName", "process"}:
                continue
            # 跳过内部记录属性，仅保留用户 extra
            if k.startswith("_"):
                continue
            payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def init_task_logger(task_id: str, logs_root: Path, *, level: int = logging.INFO) -> logging.Logger:
    """初始化任务级别 logger，输出到控制台与 logs/{task_id}/run.log（JSON 行）。"""
    logger_name = f"note_gen.task.{task_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # 若已初始化则直接返回
    if logger.handlers:
        return logger

    fmt = JsonFormatter()

    # 控制台输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件输出
    log_dir = Path(logs_root) / task_id
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_child(logger: logging.Logger, name: str) -> logging.Logger:
    """基于任务 logger 创建子 logger。"""
    return logging.getLogger(f"{logger.name}.{name}")
