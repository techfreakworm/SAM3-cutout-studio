"""Backend-neutral contracts for text-guided video segmentation."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class BackendProtocolError(RuntimeError):
    """Raised when an inference backend violates its documented response contract."""


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """SAM tracking result for one source-video frame."""

    frame_index: int
    object_ids: NDArray[np.int64]
    scores: NDArray[np.float32]
    boxes_xywh: NDArray[np.float32]
    object_masks: NDArray[np.bool_]
    union_mask: NDArray[np.bool_]
    frame_stats: Mapping[str, object] | None


class SamVideoBackend(ABC):
    """Streaming interface implemented by SAM video-tracking runtimes."""

    @abstractmethod
    def track(
        self,
        video_path: str,
        *,
        prompt: str,
        detection_threshold: float = 0.5,
    ) -> Iterator[TrackedFrame]:
        """Yield masks in source-frame order and release backend state on completion."""
