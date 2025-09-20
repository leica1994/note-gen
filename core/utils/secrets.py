from __future__ import annotations


def mask_secret(value: str | None, *, keep_start: int = 4, keep_end: int = 2) -> str:
    """对敏感字符串进行打码，保留头尾若干字符。"""
    if not value:
        return ""
    if len(value) <= keep_start + keep_end:
        return "*" * len(value)
    return value[:keep_start] + "*" * (len(value) - keep_start - keep_end) + value[-keep_end:]

