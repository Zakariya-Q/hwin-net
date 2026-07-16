"""
HWIN-Net Structured Logging Utilities

Mathematical Purpose
--------------------
Provides structured, theorem-traceable logging for HWIN-Net training and inference.
Logs are linked to specific theorems/axioms for auditability.
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
import time


class TheoremFilter(logging.Filter):
    """Filter to add theorem_ref to log records."""
    def __init__(self, theorem_ref: str = "impl"):
        super().__init__()
        self.theorem_ref = theorem_ref

    def filter(self, record: logging.LogRecord) -> bool:
        record.theorem_ref = getattr(record, "theorem_ref", self.theorem_ref)
        return True


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for log entries."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "theorem_ref": getattr(record, "theorem_ref", "impl"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                           "levelname", "levelno", "lineno", "module", "msecs",
                           "message", "msg", "name", "pathname", "process",
                           "processName", "relativeCreated", "thread", "threadName",
                           "exc_info", "exc_text", "stack_info", "theorem_ref"]:
                log_entry[key] = value
        return json.dumps(log_entry)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_output: bool = False,
    theorem_ref: str = "impl"
) -> logging.Logger:
    """
    Configure structured logging for HWIN-Net.

    Theorem Traceability
    --------------------
    - Training: Logging for experiment tracking
    - All modules: Theorem traceability per spec

    Returns
    -------
    logging.Logger : Configured root logger
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Add theorem filter
    root_logger.addFilter(TheoremFilter(theorem_ref))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(theorem_ref)-12s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str, theorem_ref: str = "impl") -> logging.Logger:
    """
    Get a logger with theorem traceability.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    theorem_ref : str
        Theorem/axiom reference for this module

    Returns
    -------
    logging.Logger : Configured logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addFilter(TheoremFilter(theorem_ref))
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
    theorem_ref: str = "impl"
) -> None:
    """
    Log metrics in structured format.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    metrics : Dict[str, float]
        Metric name -> value mapping
    step : int
        Training step/epoch
    prefix : str
        Prefix for metric names (e.g., "train/", "val/")
    theorem_ref : str
        Theorem reference
    """
    for key, value in metrics.items():
        logger.info(
            f"{prefix}{key}={value:.6f}",
            extra={"theorem_ref": theorem_ref, "step": step, "metric": key, "value": value}
        )


def time_block(logger: logging.Logger, message: str, theorem_ref: str = "impl"):
    """
    Context manager for timing code blocks.

    Usage:
        with time_block(logger, "Forward pass", "A4, T3"):
            output = model(input)
    """
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            logger.debug(f"START: {message}", extra={"theorem_ref": theorem_ref})
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            logger.info(
                f"TIMING: {message} took {elapsed:.4f}s",
                extra={"theorem_ref": theorem_ref, "elapsed_s": elapsed}
            )
    return Timer()


def log_config(logger: logging.Logger, config: Any, theorem_ref: str = "impl") -> None:
    """Log configuration in structured format."""
    if hasattr(config, "__dataclass_fields__"):
        # Dataclass
        config_dict = {k: getattr(config, k) for k in config.__dataclass_fields__}
    elif hasattr(config, "__dict__"):
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    else:
        config_dict = dict(config)

    logger.info(
        "CONFIG:",
        extra={"theorem_ref": theorem_ref, "config": config_dict}
    )


if __name__ == "__main__":
    # Test logging
    logger = setup_logging("DEBUG", json_output=False)
    logger.info("Test message", extra={"theorem_ref": "A4, T3"})
    logger.debug("Debug message")

    # Test metrics logging
    log_metrics(logger, {"loss": 0.123456, "acc": 0.987654}, step=100, prefix="val/")

    # Test timing
    with time_block(logger, "Simulated work", "A4"):
        time.sleep(0.1)

    print("Logging module test PASSED")
