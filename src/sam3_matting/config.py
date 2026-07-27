"""Typed configuration for tracking and alpha refinement."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """SAM 3.1 parameters matching the finalized workflow."""

    detection_threshold: float = 0.5
    max_objects: int = 8
    detect_interval: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be between 0 and 1")
        if self.max_objects < 1:
            raise ValueError("max_objects must be positive")
        if self.detect_interval < 1:
            raise ValueError("detect_interval must be positive")


@dataclass(frozen=True, slots=True)
class MatteConfig:
    """ViTMatte parameters matching the finalized workflow."""

    erode_kernel: int = 6
    dilate_kernel: int = 6
    black_point: float = 0.15
    white_point: float = 0.99
    max_megapixels: float = 2.0

    def __post_init__(self) -> None:
        if self.erode_kernel < 1:
            raise ValueError("erode_kernel must be positive")
        if self.dilate_kernel < 1:
            raise ValueError("dilate_kernel must be positive")
        if self.max_megapixels <= 0:
            raise ValueError("max_megapixels must be positive")
