"""Typed configuration for tracking and alpha refinement."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _bounded_real(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number between {minimum:g} and {maximum:g}")
    normalized = float(value)
    if not isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return normalized


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer between {minimum} and {maximum}")
    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return normalized


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """SAM 3.1 parameters matching the finalized workflow."""

    detection_threshold: float = 0.5
    max_objects: int = 8
    detect_interval: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "detection_threshold",
            _bounded_real(
                "detection_threshold",
                self.detection_threshold,
                minimum=0.05,
                maximum=0.95,
            ),
        )
        object.__setattr__(
            self,
            "max_objects",
            _bounded_integer("max_objects", self.max_objects, minimum=1, maximum=8),
        )
        object.__setattr__(
            self,
            "detect_interval",
            _bounded_integer("detect_interval", self.detect_interval, minimum=1, maximum=30),
        )


@dataclass(frozen=True, slots=True)
class MatteConfig:
    """ViTMatte parameters matching the finalized workflow."""

    erode_kernel: int = 6
    dilate_kernel: int = 6
    black_point: float = 0.15
    white_point: float = 0.99
    max_megapixels: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "erode_kernel",
            _bounded_integer("erode_kernel", self.erode_kernel, minimum=1, maximum=31),
        )
        object.__setattr__(
            self,
            "dilate_kernel",
            _bounded_integer("dilate_kernel", self.dilate_kernel, minimum=1, maximum=31),
        )
        object.__setattr__(
            self,
            "black_point",
            _bounded_real("black_point", self.black_point, minimum=0.0, maximum=0.9),
        )
        object.__setattr__(
            self,
            "white_point",
            _bounded_real("white_point", self.white_point, minimum=0.1, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_megapixels",
            _bounded_real(
                "max_megapixels",
                self.max_megapixels,
                minimum=0.25,
                maximum=4.0,
            ),
        )
