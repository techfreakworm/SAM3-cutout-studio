"""Cheap platform policy for local CUDA, Apple MPS, and Hugging Face ZeroGPU."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

SAM3_CHECKPOINT_REPO = "Comfy-Org/sam3.1"
SAM3_CHECKPOINT_FILENAME = "checkpoints/sam3.1_multiplex_fp16.safetensors"
SAM3_CHECKPOINT_REVISION = "ba901fbc9701054c359ed5240c4d76f83a178108"

_P = ParamSpec("_P")
_R = TypeVar("_R")
GpuDuration = int | Callable[..., int]
CheckpointDownloader = Callable[..., str | Path]


def setup_runtime(environ: MutableMapping[str, str] | None = None) -> None:
    """Set pre-Torch process defaults without overriding an operator choice."""
    environment = os.environ if environ is None else environ
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")


def on_zerogpu(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether this process belongs to a Hugging Face ZeroGPU Space."""
    environment = os.environ if environ is None else environ
    return bool(environment.get("SPACES_ZERO_GPU") or environment.get("SPACES_ZERO_GPU_V2"))


def target_device(
    *,
    environ: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
) -> str:
    """Return the eventual inference device, including pre-fork ZeroGPU processes."""
    if on_zerogpu(environ):
        return "cuda"

    runtime = torch_module if torch_module is not None else importlib.import_module("torch")
    if runtime.cuda.is_available():
        return "cuda"

    mps = getattr(getattr(runtime, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def gpu(
    duration: GpuDuration = 60,
    *,
    size: str | None = None,
    environ: Mapping[str, str] | None = None,
    spaces_module: Any | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Use ``spaces.GPU`` on ZeroGPU and preserve the function locally."""

    def decorator(function: Callable[_P, _R]) -> Callable[_P, _R]:
        if not on_zerogpu(environ):
            return function

        active_spaces = spaces_module
        if active_spaces is None:
            try:
                active_spaces = importlib.import_module("spaces")
            except ModuleNotFoundError as exc:
                raise RuntimeError("the spaces package is required inside a ZeroGPU Space") from exc

        gpu_factory = getattr(active_spaces, "GPU", None)
        if not callable(gpu_factory):
            raise RuntimeError("the spaces package does not expose a callable GPU decorator")
        gpu_options: dict[str, object] = {"duration": duration}
        if size is not None:
            gpu_options["size"] = size
        wrapped = gpu_factory(**gpu_options)(function)
        return cast(Callable[_P, _R], wrapped)

    return decorator


def resolve_sam_checkpoint(
    *,
    environ: Mapping[str, str] | None = None,
    downloader: CheckpointDownloader | None = None,
) -> Path:
    """Resolve the operator checkpoint or download the pinned public artifact."""
    environment = os.environ if environ is None else environ
    explicit_checkpoint = environment.get("SAM3_CHECKPOINT")
    if explicit_checkpoint:
        checkpoint = Path(explicit_checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3_CHECKPOINT does not point to a file: {checkpoint}")
        return checkpoint.resolve()

    active_downloader = downloader
    if active_downloader is None:
        hub = importlib.import_module("huggingface_hub")
        active_downloader = hub.hf_hub_download

    checkpoint = Path(
        active_downloader(
            repo_id=SAM3_CHECKPOINT_REPO,
            filename=SAM3_CHECKPOINT_FILENAME,
            revision=SAM3_CHECKPOINT_REVISION,
        )
    ).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint download did not produce a file: {checkpoint}")
    return checkpoint.resolve()


setup_runtime()
