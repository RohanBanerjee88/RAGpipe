#!/usr/bin/env python3
"""Verify that the active PyTorch build can execute on the allocated GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from device_runtime import GPUUnavailableError, select_runtime_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit with an error instead of accepting a CPU fallback",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = "cuda" if args.require_cuda else "auto"
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'none (CPU build)'}")

    try:
        selection = select_runtime_device(request=request)
    except (GPUUnavailableError, ValueError) as exc:
        print(f"GPU check failed: {exc}")
        return 1

    print(f"Selected device: {selection.summary()}")
    if selection.compiled_architectures:
        print("Compiled architectures: " + ", ".join(selection.compiled_architectures))
    print(f"Preflight: {selection.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
