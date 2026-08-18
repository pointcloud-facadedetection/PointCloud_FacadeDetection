"""通用日志/跟踪辅助函数。"""
from __future__ import annotations


def trace(stage: str, **fields):
    """打印带前缀的结构化跟踪日志。"""
    msg = f"[PCFD] {stage}"
    if fields:
        msg += " " + " ".join(f"{k}={v}" for k, v in fields.items())
    print(msg, flush=True)
