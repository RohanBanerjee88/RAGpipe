"""Lightweight opt-in JSONL tracing for retrieval and routing."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from config import TRACE_ENABLED, TRACE_PATH


def tracing_enabled() -> bool:
    override = os.getenv("FAQ_TRACE_ENABLED")
    if override is None:
        return TRACE_ENABLED
    return override.lower() in {"1", "true", "yes", "on"}


def trace_path() -> Path:
    return Path(os.getenv("FAQ_TRACE_PATH", TRACE_PATH))


def trace_event(event_type: str, payload: Dict) -> None:
    """Append one structured trace event when tracing is enabled."""
    if not tracing_enabled():
        return

    path = trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }

    with path.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(event, ensure_ascii=True) + "\n")
