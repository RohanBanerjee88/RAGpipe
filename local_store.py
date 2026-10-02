"""Atomic local state shared by model preparation and collection imports."""

import json
import os
import tempfile
from pathlib import Path

from config import PROJECT_ROOT


def data_root():
    return Path(os.getenv("FAQ_DATA_DIR", PROJECT_ROOT / ".ragpipe")).expanduser().resolve()


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
