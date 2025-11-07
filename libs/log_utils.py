# libs/log_utils.py
import os, json, time, logging, tempfile
from typing import Any, Dict, List


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def setup_logger(name: str, log_dir: str = "artifacts/logs", level: str = "INFO") -> logging.Logger:
    """
    Create/get a logger that logs to both console and a file in log_dir/<name>.log.
    Prevents duplicate handlers if called multiple times.
    """
    _ensure_dir(log_dir)
    logger = logging.getLogger(name)

    # Normalize level string
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    logger.setLevel(level_map.get(str(level).upper(), logging.INFO))

    # If no handlers yet, attach
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File
        fh_path = os.path.join(log_dir, f"{name}.log")
        fh = logging.FileHandler(fh_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Avoid noisy propagation to root
        logger.propagate = False

    return logger


def now_slug() -> str:
    """Timestamp slug like 2025-10-12_15-48-33."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def safe_write_json(obj: Any, path: str):
    """
    Write JSON atomically:
      - write to tmp
      - replace target
    """
    _ensure_dir(os.path.dirname(path) or ".")
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="tmpjson_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        # Best effort cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def redact(data: Any, secrets: List[str]) -> Any:
    """
    Very simple redactor:
      - If data is a string: replace any secret substring with "***"
      - If dict/list: recurse
      - Else: return as-is
    """
    def _redact_str(s: str) -> str:
        if not secrets:
            return s
        out = s
        for sec in secrets:
            if isinstance(sec, str) and sec:
                out = out.replace(sec, "***")
        return out

    if isinstance(data, str):
        return _redact_str(data)

    if isinstance(data, dict):
        return {k: redact(v, secrets) for k, v in data.items()}

    if isinstance(data, list):
        return [redact(v, secrets) for v in data]

    return data
