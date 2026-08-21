#!/usr/bin/env python3
"""Validate Python isolation and one of the project's dependency profiles."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import site
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATHS = {
    "cpu": PROJECT_ROOT / "requirements.txt",
    "cuda126": PROJECT_ROOT / "requirements-cuda126.txt",
}
PACKAGE_MODULES = {
    "beautifulsoup4": "bs4",
    "huggingface-hub": "huggingface_hub",
    "numpy": "numpy",
    "packaging": "packaging",
    "requests": "requests",
    "sentence-transformers": "sentence_transformers",
    "sentencepiece": "sentencepiece",
    "torch": "torch",
    "transformers": "transformers",
}


def load_expected_versions(
    requirements_path: Path, marker_environment: dict[str, str] | None = None
) -> dict[str, str]:
    """Read exact pins, following nested requirement files and markers."""
    expected = {}
    visited = set()

    def visit(path: Path):
        resolved_path = path.resolve()
        if resolved_path in visited:
            return
        visited.add(resolved_path)

        for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(("-r ", "--requirement ")):
                include_path = line.split(maxsplit=1)[1]
                visit(resolved_path.parent / include_path)
                continue
            if line.startswith(("--index-url ", "--extra-index-url ")):
                continue

            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate(marker_environment):
                continue
            pins = [item for item in requirement.specifier if item.operator == "=="]
            if len(pins) != 1 or len(requirement.specifier) != 1 or pins[0].version.endswith(".*"):
                raise ValueError(f"Requirements must use exact pins: {raw_line}")
            package = requirement.name.lower().replace("_", "-")
            expected[package] = pins[0].version

    visit(requirements_path)
    return expected


def detect_profile() -> str:
    """Infer the profile from the installed PyTorch local version tag."""
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return "cpu"

    local_version = torch_version.partition("+")[2].lower()
    if local_version.startswith("cu126"):
        return "cuda126"
    if not local_version or local_version.startswith("cpu"):
        return "cpu"

    raise ValueError(
        f"Unsupported automatic profile for torch=={torch_version}. "
        "Install requirements.txt for CPU or requirements-cuda126.txt for HPCC GPUs."
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("auto", *PROFILE_PATHS),
        default="auto",
        help="dependency profile to validate (default: infer from installed torch)",
    )
    return parser.parse_args()


def path_is_in_environment(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path(sys.prefix).resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    errors = []
    args = parse_args()
    try:
        profile = detect_profile() if args.profile == "auto" else args.profile
        expected = load_expected_versions(PROFILE_PATHS[profile])
    except ValueError as exc:
        print(f"Environment check failed:\n  - {exc}")
        return 1

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
        errors.append(
            "The Conda base environment is active; use the dedicated ragpipe environment."
        )
    if site.ENABLE_USER_SITE:
        errors.append("User site-packages are enabled; export PYTHONNOUSERSITE=1.")
    if os.environ.get("PYTHONPATH"):
        errors.append("PYTHONPATH is set and may inject packages from outside this environment.")

    for package, expected_version in expected.items():
        module_name = PACKAGE_MODULES.get(package, package.replace("-", "_"))
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

    print(
        f"Environment check passed ({profile} profile): "
        f"Python {sys.version.split()[0]} at {sys.prefix}"
    )
    for package in sorted(expected):
        print(f"  {package}=={expected[package]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
