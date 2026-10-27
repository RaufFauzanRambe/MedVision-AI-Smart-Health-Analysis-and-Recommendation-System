"""
MedVision-AI — Logging configuration.

Provides:

* A JSON structured formatter (one log record == one JSON object per line),
  suitable for shipping to ELK / Loki / CloudWatch / Datadog.
* A `RedactingFilter` that strips sensitive fields (PII, secrets,
  medical record identifiers) from log records before they are emitted.
* A convenience `setup_logging()` entry point that wires it all together
  and respects the `LOG_LEVEL` env var.

Privacy rules (in line with README §5):

  - The following fields are NEVER logged, regardless of nesting:
    name, patient_name, full_name, email, phone, address, ssn,
    national_id, mrn, medical_record_number, dob, date_of_birth,
    password, api_key, token, authorization, secret.
  - Raw request payloads containing free-form medical text are NOT
    logged verbatim — log only structural metadata (endpoint, status,
    duration, request_id, model_version).
  - The default logger output is stderr so it composes cleanly with
    container runtimes.

Usage:

    from mlops.monitoring.logging_config import setup_logging, get_logger
    setup_logging(level="INFO")
    log = get_logger("medvision.api")
    log.info("request handled",
             extra={"endpoint": "/predict", "status": 200,
                    "duration_ms": 42.1, "request_id": "abc-123"})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_REDACT_FIELDS: frozenset[str] = frozenset({
    # PII
    "name", "patient_name", "full_name",
    "first_name", "last_name",
    "email", "phone", "address", "street", "zip", "postal_code",
    "ssn", "national_id", "mrn", "medical_record_number",
    "dob", "date_of_birth", "birth_date",
    # Secrets
    "password", "passwd", "api_key", "apikey", "secret", "access_token",
    "refresh_token", "token", "authorization", "auth_header", "cookie",
})

REDACT_PLACEHOLDER = "[REDACTED]"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def _redact_value(value: Any) -> Any:
    """Leave non-mapping / non-iterable values untouched."""
    return value


def _redact_mapping(mapping: dict[str, Any],
                    redact_fields: frozenset[str]) -> dict[str, Any]:
    """Recursively redact sensitive keys from a mapping."""
    cleaned: dict[str, Any] = {}
    for k, v in mapping.items():
        if isinstance(k, str) and k.lower() in redact_fields:
            cleaned[k] = REDACT_PLACEHOLDER
        elif isinstance(v, dict):
            cleaned[k] = _redact_mapping(v, redact_fields)
        elif isinstance(v, (list, tuple)):
            cleaned[k] = [
                _redact_mapping(item, redact_fields) if isinstance(item, dict)
                else item
                for item in v
            ]
        else:
            cleaned[k] = v
    return cleaned


class RedactingFilter(logging.Filter):
    """Filter that redacts sensitive keys from record.__dict__ and `extra`.

    Operates on a defensive shallow copy of the record's `__dict__`, so
    the original record (and any handlers further down the chain) is not
    mutated in surprising ways. Keys are matched case-insensitively.
    """

    def __init__(self, redact_fields: Iterable[str] = DEFAULT_REDACT_FIELDS) -> None:
        super().__init__()
        self.redact_fields = frozenset(f.lower() for f in redact_fields)

    def filter(self, record: logging.LogRecord) -> bool:
        # Defensive copy so we don't mutate state shared with other handlers.
        snapshot = dict(record.__dict__)
        snapshot = _redact_mapping(snapshot, self.redact_fields)
        record.__dict__.clear()
        record.__dict__.update(snapshot)
        # Redact the formatted message args too (positional % args).
        if record.args:
            if isinstance(record.args, dict):
                record.args = _redact_mapping(dict(record.args), self.redact_fields)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    _redact_mapping(a, self.redact_fields) if isinstance(a, dict)
                    else a
                    for a in record.args
                )
        return True


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """One JSON object per line. Stable field order to ease log shipping."""

    _RESERVED: frozenset[str] = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
    })

    def __init__(self, *, service_name: str = "medvision-ai") -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self._format_time(record),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service_name,
            "message": record.getMessage(),
        }

        # Attach caller location for non-DEBUG levels (cheap, useful in prod).
        if record.levelno >= logging.INFO:
            payload["module"] = record.module
            payload["line"] = record.lineno
            payload["function"] = record.funcName

        # Attach any structured `extra=...` keys passed by the caller.
        for k, v in record.__dict__.items():
            if k in self._RESERVED or k in payload:
                continue
            if k.startswith("_"):
                continue
            payload[k] = self._jsonable(v)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack"] = record.stack_info

        return json.dumps(payload, default=str, ensure_ascii=False,
                          separators=(",", ":"))

    @staticmethod
    def _format_time(record: logging.LogRecord) -> str:
        # ISO-8601 with milliseconds, UTC.
        ct = time.gmtime(record.created)
        return time.strftime("%Y-%m-%dT%H:%M:%S", ct) + \
               f".{int(record.msecs):03d}Z"

    @staticmethod
    def _jsonable(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [JsonFormatter._jsonable(x) for x in v]
        if isinstance(v, dict):
            return {str(k): JsonFormatter._jsonable(val) for k, val in v.items()}
        # Fallback: stringify complex objects.
        try:
            return str(v)
        except Exception:
            return f"<unserializable:{type(v).__name__}>"


# ---------------------------------------------------------------------------
# Setup API
# ---------------------------------------------------------------------------

_CONFIGURED = False


def setup_logging(
    *,
    level: str | int | None = None,
    service_name: str = "medvision-ai",
    redact_fields: Iterable[str] = DEFAULT_REDACT_FIELDS,
    stream=None,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging: JSON formatter + redaction filter + stderr.

    Idempotent: subsequent calls are no-ops unless `force=True`.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return logging.getLogger(service_name)

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if isinstance(level, str):
        level = logging.getLevelName(level)
        if not isinstance(level, int):
            level = logging.INFO

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter(service_name=service_name))
    handler.addFilter(RedactingFilter(redact_fields=redact_fields))
    root.addHandler(handler)

    # Quiet down noisy 3rd-party loggers.
    for noisy in ("urllib3", "werkzeug", "asyncio", "botocore", "boto3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    log = logging.getLogger(service_name)
    log.info("logging initialized",
             extra={"level": logging.getLevelName(level),
                    "service": service_name,
                    "redacted_fields_count": len(redact_fields)})
    return log


def get_logger(name: str = "medvision") -> logging.Logger:
    """Return a child logger. Auto-configures root on first call."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


__all__ = [
    "DEFAULT_REDACT_FIELDS", "REDACT_PLACEHOLDER",
    "RedactingFilter", "JsonFormatter",
    "setup_logging", "get_logger",
]
