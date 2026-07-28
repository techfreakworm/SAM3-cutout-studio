"""ZeroGPU worker-resident model registry.

GPU-decorated tasks execute in one persistent worker process per decorated
function, separate from the main application process. Only pickled values cross
the boundary in either direction, so built models can never leave the worker.
This module holds the process-wide inference resources inside the worker and
builds them lazily on the first request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sam3_matting.application import ApplicationResources

_RESOURCES: ApplicationResources | None = None


def get_or_build_resources(checkpoint: str, device: str) -> Any:
    """Build the process-wide resources once inside this worker, then reuse."""
    global _RESOURCES
    if _RESOURCES is None:
        from sam3_matting.application import build_resources

        _RESOURCES = build_resources(checkpoint, device=device, preload=True)
    return _RESOURCES
