"""Runtime device policy shared by local and ZeroGPU execution."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceAvailability:
    """Accelerators visible to the current process."""

    cuda: bool
    mps: bool


def detect_availability() -> DeviceAvailability:
    """Read accelerator availability from the active PyTorch runtime."""
    import torch

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available()
    )
    return DeviceAvailability(cuda=torch.cuda.is_available(), mps=mps_available)


def choose_device(preference: str, availability: DeviceAvailability) -> str:
    """Select the highest-priority accelerator for an automatic request."""
    if preference == "auto" and availability.cuda:
        return "cuda"
    if preference == "auto" and availability.mps:
        return "mps"
    if preference == "auto":
        return "cpu"
    if preference == "mps" and availability.mps:
        return "mps"
    if preference == "mps":
        raise RuntimeError("mps is not available in this process")
    if preference == "cpu":
        return "cpu"
    if preference == "cuda" and availability.cuda:
        return "cuda"
    if preference == "cuda":
        raise RuntimeError("cuda is not available in this process")
    raise ValueError(f"unknown device preference: {preference}")
