"""通用日志/跟踪辅助函数。"""
from __future__ import annotations


def trace(stage: str, **fields):
    """打印带前缀的结构化跟踪日志。"""
    msg = f"[PCFD] {stage}"
    if fields:
        msg += " " + " ".join(f"{k}={v}" for k, v in fields.items())
    print(msg, flush=True)


def project_logger(project_uuid: str | None = None):
    import logging
    from config.storage import Storage
    logger = logging.getLogger(f"pcfd.project.{project_uuid or 'global'}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        if project_uuid:
            try:
                log_dir = Storage.ensure_project_dirs(project_uuid)["root"] / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_dir / "application.log", encoding="utf-8")
                file_handler.setFormatter(handler.formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass
    return logger


def log_event(project_uuid: str | None, event: str, **fields):
    values = " ".join(f"{key}={value!r}" for key, value in fields.items())
    project_logger(project_uuid).info("[%s] %s", event, values)
