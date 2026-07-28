"""Hugging Face Spaces and local-CUDA entrypoint."""

from __future__ import annotations

import importlib
import os

if os.environ.get("SPACES_ZERO_GPU") or os.environ.get("SPACES_ZERO_GPU_V2"):
    importlib.import_module("spaces")

import sys
from pathlib import Path

# Hugging Face Spaces install requirements.txt before copying the application
# source, so the src-layout package cannot be pip-installed there. When running
# from a source checkout, make the package importable directly.
_SOURCE_PACKAGE = Path(__file__).resolve().parent / "src"
if (_SOURCE_PACKAGE / "sam3_matting").is_dir() and str(_SOURCE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_SOURCE_PACKAGE))

from collections.abc import Callable, Mapping  # noqa: E402
from typing import Any  # noqa: E402

from sam3_matting.application import (  # noqa: E402
    ApplicationResources,
    build_resources,
    create_process_callback,
    create_request_validator,
)
from sam3_matting.runtime import gpu, on_zerogpu, resolve_sam_checkpoint, target_device  # noqa: E402
from sam3_matting.ui import build_ui, studio_launch_kwargs  # noqa: E402

RESOURCES: ApplicationResources | None = None
DEMO: Any | None = None


def bootstrap_application(
    *,
    resolve_checkpoint_fn: Callable[[], Path] = resolve_sam_checkpoint,
    target_device_fn: Callable[[], str] = target_device,
    zerogpu_detector: Callable[[], bool] = on_zerogpu,
    build_resources_fn: Callable[..., Any] = build_resources,
    create_callback_fn: Callable[..., Any] = create_process_callback,
    create_validator_fn: Callable[..., Any] = create_request_validator,
    gpu_factory: Callable[..., Any] = gpu,
    build_ui_fn: Callable[..., Any] = build_ui,
) -> tuple[Any, Any]:
    """Preload global resources, decorate inference, and build the studio."""
    checkpoint = resolve_checkpoint_fn()
    device = target_device_fn()
    active_zerogpu = zerogpu_detector()
    resources = build_resources_fn(checkpoint, device=device, preload=True)
    process = create_callback_fn(resources, zerogpu=active_zerogpu)
    validator = create_validator_fn(zerogpu=active_zerogpu)
    accelerated_process = gpu_factory(duration=90, size="xlarge")(process)
    runtime_status = {
        "device": device.upper(),
        "cuda": "Active",
        "mps": "Next phase",
        "zerogpu": "96 GB xlarge" if active_zerogpu else "Available",
    }
    demo = build_ui_fn(
        accelerated_process,
        validator_fn=validator,
        hosted=active_zerogpu,
        runtime_status=runtime_status,
    )
    return resources, demo


def launch_application(
    demo: Any,
    *,
    environ: Mapping[str, str] | None = None,
    launch_kwargs_fn: Callable[[], dict[str, Any]] = studio_launch_kwargs,
) -> None:
    """Launch on the host and port selected by the runtime environment."""
    environment = os.environ if environ is None else environ
    server_name = environment.get("GRADIO_SERVER_NAME") or environment.get("HOST") or "0.0.0.0"
    raw_port = environment.get("GRADIO_SERVER_PORT") or environment.get("PORT") or "7860"
    server_port = int(raw_port)
    if not 1 <= server_port <= 65535:
        raise ValueError("Gradio server port must be between 1 and 65535")

    active_zerogpu = bool(environment.get("SPACES_ZERO_GPU") or environment.get("SPACES_ZERO_GPU_V2"))
    max_file_size = 100 * 1024 * 1024 if active_zerogpu else 2 * 1024 * 1024 * 1024
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        max_file_size=max_file_size,
        show_error=False,
        **launch_kwargs_fn(),
    )


def main() -> None:
    """Start the production application while retaining models at module scope."""
    global DEMO, RESOURCES
    RESOURCES, DEMO = bootstrap_application()
    launch_application(DEMO)


if __name__ == "__main__":
    main()
