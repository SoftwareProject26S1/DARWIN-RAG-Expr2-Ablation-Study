"""Logging setup for the local API server."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
from typing import TextIO


DEFAULT_LOG_FILE = Path("runs/api/serve-api.log")
DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class ApiLoggingConfig:
    """Resolved logging options for serve-api."""

    log_file: Path
    level: str
    capture_stdio: bool = False


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.secondary.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()


def configure_api_logging(config: ApiLoggingConfig) -> None:
    """Configure console and file logs before Uvicorn imports the API app."""

    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    level = _parse_log_level(config.level)
    level_name = logging.getLevelName(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "darwin_rag_exp2"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = True

    if config.capture_stdio:
        capture = config.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.stdout, capture)  # type: ignore[assignment]
        sys.stderr = _TeeStream(sys.stderr, capture)  # type: ignore[assignment]

    vllm_config_path = _configure_vllm_logging(config.log_file, str(level_name))
    logging.getLogger(__name__).info(
        "serve-api logging configured log_file=%s level=%s vllm_logging_config=%s",
        config.log_file,
        level_name,
        vllm_config_path,
    )


def _configure_vllm_logging(log_file: Path, level_name: str) -> Path | str:
    if os.environ.get("VLLM_LOGGING_CONFIG_PATH"):
        return os.environ["VLLM_LOGGING_CONFIG_PATH"]

    config_path = log_file.parent / f"{log_file.stem}.vllm-logging.json"
    payload = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "stderr": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": str(log_file),
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "vllm": {
                "handlers": ["stderr", "file"],
                "level": level_name,
                "propagate": False,
            }
        },
    }
    config_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    os.environ["VLLM_LOGGING_CONFIG_PATH"] = str(config_path)
    os.environ["VLLM_LOGGING_LEVEL"] = level_name
    return config_path


def _parse_log_level(value: str) -> int:
    level = getattr(logging, value.strip().upper(), None)
    if not isinstance(level, int):
        raise ValueError("log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    return level
