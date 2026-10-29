#!/usr/bin/env python3
"""Exercise fresh application state and offline CLI using prepared Hub weights."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="ragpipe-smoke-") as temporary:
        root = Path(temporary)
        env = dict(os.environ, FAQ_DATA_DIR=str(root / "store"), HF_HUB_OFFLINE="1", FAQ_DEVICE="cpu")
        env.pop("FAQ_LLM_MODEL", None)
        env.pop("FAQ_MODELS_CONFIG", None)

        def run(*args, question=None, environment=None, success=True):
            process = subprocess.run([sys.executable, str(ROOT / "main.py"), *args],
                                     cwd=ROOT, env=environment or env,
                                     input=(question + "\nquit\n") if question else None,
                                     text=True, capture_output=True, timeout=180)
            if (process.returncode == 0) != success:
                raise AssertionError(process.stdout + process.stderr)
            return process.stdout

        assert "flan-small" in run("models", "list")
        assert "Ready:" in run("--offline", "models", "prepare", "retrieval-only")
        document = root / "protocol.txt"
        document.write_text("RNA concentration is measured with Qubit fluorometry.\n", encoding="utf-8")
        assert "1 records" in run("collections", "import", "smoke", str(document))
        assert "smoke: 1 records" in run("collections", "list")
        args = ("--offline", "--collection", "smoke", "--model", "retrieval-only")
        query = "How is RNA concentration measured?"
        answer = run(*args, question=query)
        assert "Qubit" in answer and "[S1]" in answer
        cache = root / "store" / "collections" / "smoke" / "embeddings.pt"
        timestamp = cache.stat().st_mtime_ns
        assert "Qubit" in run(*args, question=query)
        assert cache.stat().st_mtime_ns == timestamp, "Cached restart rebuilt unchanged embeddings"
        answer = run("--offline", "--collection", "smoke", "--model", str(root / "missing-generator"), question=query)
        assert "models prepare" in answer and "Qubit" in answer and "[S1]" in answer
        assert cache.stat().st_mtime_ns == timestamp, "Generator change rebuilt embeddings"
        empty_cache = str(root / "empty-hub")
        missing_env = dict(env, HF_HOME=empty_cache, HF_HUB_CACHE=empty_cache,
                           HUGGINGFACE_HUB_CACHE=empty_cache, TRANSFORMERS_CACHE=empty_cache)
        answer = run(*args, question=query, environment=missing_env, success=False)
        assert "prepare retrieval-only" in answer
    print("PASS: offline CLI, preparation from cache, import, cited answer, cached restart, missing generator, missing retriever")


if __name__ == "__main__":
    main()
