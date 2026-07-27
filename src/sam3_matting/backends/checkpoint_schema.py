"""Read-only tensor-schema inspection for SAM 3.1 checkpoints."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Shape and serialized dtype stored for one checkpoint tensor."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Structural differences between a checkpoint and a model state dict."""

    checkpoint_key_count: int
    expected_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_compatible(self) -> bool:
        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)


def read_safetensors_schema(path: str | Path) -> dict[str, TensorSpec]:
    """Read keys, shapes and dtypes without materializing tensor payloads."""
    schema: dict[str, TensorSpec] = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():  # noqa: SIM118 - safe_open is not iterable
            tensor_slice = checkpoint.get_slice(key)
            schema[key] = TensorSpec(
                shape=tuple(tensor_slice.get_shape()),
                dtype=tensor_slice.get_dtype(),
            )
    return schema


def compare_schemas(
    checkpoint_schema: Mapping[str, TensorSpec],
    expected_schema: Mapping[str, TensorSpec],
) -> CompatibilityReport:
    """Compare state-dict names and shapes while allowing precision conversion."""
    checkpoint_keys = set(checkpoint_schema)
    expected_keys = set(expected_schema)
    shared_keys = checkpoint_keys.intersection(expected_keys)
    shape_mismatches = tuple(
        (
            key,
            checkpoint_schema[key].shape,
            expected_schema[key].shape,
        )
        for key in sorted(shared_keys)
        if checkpoint_schema[key].shape != expected_schema[key].shape
    )
    return CompatibilityReport(
        checkpoint_key_count=len(checkpoint_keys),
        expected_key_count=len(expected_keys),
        missing_keys=tuple(sorted(expected_keys.difference(checkpoint_keys))),
        unexpected_keys=tuple(sorted(checkpoint_keys.difference(expected_keys))),
        shape_mismatches=shape_mismatches,
    )


_SAM31_UNUSED_TEXT_PROJECTION = "detector.backbone.language_backbone.encoder.text_projection"
_SAM31_ROPE_BUFFER_PREFIX = "detector.backbone.vision_backbone.trunk.blocks."
_SAM31_ROPE_BUFFER_NAMES = frozenset({"freqs_cis", "freqs_cis_real", "freqs_cis_imag"})


def partition_sam31_missing_keys(
    missing_keys: tuple[str, ...] | list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Separate constructor-derived/unused SAM 3.1 state from learned weights."""
    allowed: list[str] = []
    blocking: list[str] = []
    for key in missing_keys:
        is_rope_buffer = (
            key.startswith(_SAM31_ROPE_BUFFER_PREFIX)
            and ".attn." in key
            and key.rsplit(".", 1)[-1] in _SAM31_ROPE_BUFFER_NAMES
        )
        if key == _SAM31_UNUSED_TEXT_PROJECTION or is_rope_buffer:
            allowed.append(key)
        else:
            blocking.append(key)
    return tuple(sorted(allowed)), tuple(sorted(blocking))
