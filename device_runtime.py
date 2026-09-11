"""Select a usable PyTorch device before model initialization."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache

import torch


VALID_DEVICE_REQUESTS = {"auto", "cpu", "cuda"}


class GPUUnavailableError(RuntimeError):
    """Raised when CUDA was required but the installed stack cannot use it."""


@dataclass(frozen=True)
class DeviceSelection:
    device: str
    reason: str
    gpu_name: str | None = None
    capability: tuple[int, int] | None = None
    compiled_architectures: tuple[str, ...] = ()
    torch_cuda_version: str | None = None

    def summary(self) -> str:
        if self.device == "cpu":
            return f"CPU ({self.reason})"
        capability = ".".join(str(part) for part in self.capability or ())
        return (
            f"CUDA ({self.gpu_name}, capability {capability}, "
            f"PyTorch CUDA {self.torch_cuda_version})"
        )


def cuda_architecture_supported(
    capability: tuple[int, int], compiled_architectures: list[str] | tuple[str, ...]
) -> bool:
    """Return whether an sm binary or compatible PTX target can serve a GPU."""
    device_code = capability[0] * 10 + capability[1]
    ptx_codes = []
    for architecture in compiled_architectures:
        match = re.fullmatch(r"(sm|compute)_(\d+)[a-z]?", architecture.lower())
        if not match:
            continue
        target_type, target_code = match.groups()
        target_code = int(target_code)
        if target_type == "sm" and target_code == device_code:
            return True
        if target_type == "compute":
            ptx_codes.append(target_code)
    return any(target_code <= device_code for target_code in ptx_codes)


def _probe_cuda_subprocess(timeout_seconds: int = 30) -> tuple[bool, str]:
    """Exercise CUDA outside this process so fatal driver errors stay contained."""
    probe = (
        "import torch\n"
        "assert torch.cuda.is_available(), 'CUDA is unavailable'\n"
        "x = torch.arange(8, dtype=torch.float32, device='cuda')\n"
        "value = x.square().sum().item()\n"
        "torch.cuda.synchronize()\n"
        "assert value == 140.0, value\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return False, f"CUDA smoke test exceeded {timeout_seconds} seconds"

    if completed.returncode == 0:
        return True, "tensor smoke test passed"

    if completed.returncode < 0:
        signal_number = -completed.returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        return False, f"CUDA smoke test terminated by {signal_name}"

    output = (completed.stderr or completed.stdout).strip().splitlines()
    detail = " | ".join(output[-4:]) if output else "no diagnostic output"
    return False, f"CUDA smoke test failed: {detail}"


def _cuda_failure(message: str, required: bool) -> DeviceSelection:
    guidance = (
        f"{message} Install the HPCC profile with "
        "'python -m pip install -r requirements-cuda126.txt'."
    )
    if required:
        raise GPUUnavailableError(guidance)
    warnings.warn(f"{guidance} Falling back to CPU.", RuntimeWarning, stacklevel=3)
    return DeviceSelection(device="cpu", reason=message)


@lru_cache(maxsize=8)
def _select_runtime_device(
    use_gpu: bool, request: str, cuda_visible_devices: str | None
) -> DeviceSelection:
    if request not in VALID_DEVICE_REQUESTS:
        choices = ", ".join(sorted(VALID_DEVICE_REQUESTS))
        raise ValueError(f"FAQ_DEVICE must be one of {choices}; received {request!r}")

    if not use_gpu:
        return DeviceSelection(device="cpu", reason="GPU use is disabled in config.py")
    if request == "cpu":
        return DeviceSelection(device="cpu", reason="FAQ_DEVICE=cpu")

    required = request == "cuda"
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        return _cuda_failure(f"CUDA initialization failed: {exc}", required)
    if not cuda_available:
        return _cuda_failure("CUDA is not available to PyTorch", required)

    try:
        gpu_name = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))
        architectures = tuple(torch.cuda.get_arch_list())
    except Exception as exc:
        return _cuda_failure(f"CUDA device inspection failed: {exc}", required)

    if architectures and not cuda_architecture_supported(capability, architectures):
        architecture = f"sm_{capability[0]}{capability[1]}"
        compiled = ", ".join(architectures)
        return _cuda_failure(
            f"{gpu_name} requires {architecture}, but torch=={torch.__version__} "
            f"was compiled for [{compiled}]",
            required,
        )

    probe_ok, probe_detail = _probe_cuda_subprocess()
    if not probe_ok:
        return _cuda_failure(
            f"CUDA preflight failed on {gpu_name} (capability {capability[0]}.{capability[1]}): "
            f"{probe_detail}",
            required,
        )

    return DeviceSelection(
        device="cuda",
        reason=probe_detail,
        gpu_name=gpu_name,
        capability=capability,
        compiled_architectures=architectures,
        torch_cuda_version=torch.version.cuda,
    )


def select_runtime_device(use_gpu: bool = True, request: str | None = None) -> DeviceSelection:
    """Choose CPU or validated CUDA according to config and FAQ_DEVICE."""
    requested_device = (request or os.getenv("FAQ_DEVICE", "auto")).strip().lower()
    return _select_runtime_device(
        use_gpu,
        requested_device,
        os.getenv("CUDA_VISIBLE_DEVICES"),
    )


def clear_device_selection_cache():
    """Clear the process cache; intended for tests and changed allocations."""
    _select_runtime_device.cache_clear()
