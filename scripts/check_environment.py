#!/usr/bin/env python3
"""Validate Python isolation and the project's exact direct dependencies."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import site
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
PACKAGE_MODULES = {
    "beautifulsoup4": "bs4",
    "huggingface-hub": "huggingface_hub",
    "numpy": "numpy",
    "requests": "requests",
    "sentence-transformers": "sentence_transformers",
    "sentencepiece": "sentencepiece",
    "torch": "torch",
    "transformers": "transformers",
}


def normalized_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_expected_versions() -> dict[str, str]:
    expected = {}
    for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"Requirements must use exact pins: {raw_line}")
        expected[normalized_package_name(match.group(1))] = match.group(2)
    return expected


def path_is_in_environment(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path(sys.prefix).resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    errors = []
    expected = load_expected_versions()
    conda_prefix = os.environ.get("CONDA_PREFIX")
    isolated = sys.prefix != sys.base_prefix or (
        conda_prefix and Path(conda_prefix).resolve() == Path(sys.prefix).resolve()
    )

    if sys.version_info[:2] != (3, 11):
        errors.append(
            f"Python 3.11 is required; running {sys.version_info.major}.{sys.version_info.minor}."
        )
    if not isolated:
        errors.append("No virtual environment or Conda environment is active.")
    using_conda_prefix = (
        conda_prefix and Path(conda_prefix).resolve() == Path(sys.prefix).resolve()
    )
    if os.environ.get("CONDA_DEFAULT_ENV") == "base" and using_conda_prefix:
        errors.append("The Conda base environment is active; use the dedicated ragpipe environment.")
    if site.ENABLE_USER_SITE:
        errors.append("User site-packages are enabled; export PYTHONNOUSERSITE=1.")
    if os.environ.get("PYTHONPATH"):
        errors.append("PYTHONPATH is set and may inject packages from outside this environment.")

    for package, module_name in PACKAGE_MODULES.items():
        expected_version = expected[package]
        try:
            actual_version = importlib.metadata.version(package)
            module = importlib.import_module(module_name)
        except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
            errors.append(f"{package} is unavailable: {exc}")
            continue

        if actual_version != expected_version:
            errors.append(
                f"{package}=={actual_version} is installed; expected {expected_version}."
            )

        module_path = getattr(module, "__file__", None)
        if module_path and not path_is_in_environment(Path(module_path)):
            errors.append(f"{package} was imported outside the environment: {module_path}")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    if pip_check.returncode:
        errors.append(pip_check.stdout.strip() or pip_check.stderr.strip())

    if errors:
        print("Environment check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Environment check passed: Python {sys.version.split()[0]} at {sys.prefix}")
    for package in sorted(PACKAGE_MODULES):
        print(f"  {package}=={expected[package]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
