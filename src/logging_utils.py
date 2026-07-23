"""Shared logging configuration so every module logs consistently to
console + a rotating file under logs/pipeline.log."""
from __future__ import annotations
import logging
import sys
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)

    root = logging.getLogger()
    if not _CONFIGURED:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "pipeline.log", mode="a")
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)

        _CONFIGURED = True

    return logger
