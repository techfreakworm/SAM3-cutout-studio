#!/usr/bin/env python3
"""Compare a safetensors SAM 3.1 checkpoint with Meta's pinned model builder."""

import argparse
import contextlib
import io
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sam3_matting.backends.checkpoint_schema import (  # noqa: E402
    TensorSpec,
    compare_schemas,
    partition_sam31_missing_keys,
    read_safetensors_schema,
)


class _NoAutocast(contextlib.ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _torch_dtype_code(dtype: object) -> str:
    import torch

    names = {
        torch.bool: "BOOL",
        torch.uint8: "U8",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.complex64: "C64",
        torch.complex128: "C128",
    }
    return names.get(dtype, str(dtype))


def _build_expected_schema(checkpoint_path: Path) -> dict[str, TensorSpec]:
    """Instantiate Meta's architecture on the meta device without loading weights."""
    import torch

    warnings.filterwarnings(
        "ignore",
        message="Importing from timm.models.layers is deprecated",
        category=FutureWarning,
    )
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    original_linspace = torch.linspace

    def cpu_linspace(*args: object, **kwargs: object):
        # ViT initialization calls .item(); that one tiny tensor cannot live on meta.
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)

    quiet_output = io.StringIO()
    with (
        contextlib.redirect_stdout(quiet_output),
        contextlib.redirect_stderr(quiet_output),
        torch.device("meta"),
        patch.object(torch, "load", return_value={}),
        patch.object(torch, "linspace", cpu_linspace),
        patch.object(torch.nn.Module, "to", lambda module, *args, **kwargs: module),
        patch.object(torch.nn.Module, "cuda", lambda module, *args, **kwargs: module),
        patch.object(
            torch.nn.Module,
            "load_state_dict",
            lambda module, *args, **kwargs: ([], []),
        ),
        patch.object(torch, "autocast", lambda *args, **kwargs: _NoAutocast()),
    ):
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            max_num_objects=8,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
        )

    return {
        key: TensorSpec(shape=tuple(tensor.shape), dtype=_torch_dtype_code(tensor.dtype))
        for key, tensor in predictor.model.state_dict().items()
    }


def _prefix_counts(schema: dict[str, TensorSpec]) -> dict[str, int]:
    return dict(sorted(Counter(key.split(".", 1)[0] for key in schema).items()))


def inspect_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_schema = read_safetensors_schema(checkpoint_path)
    expected_schema = _build_expected_schema(checkpoint_path)
    comparison = compare_schemas(checkpoint_schema, expected_schema)
    allowed_missing, blocking_missing = partition_sam31_missing_keys(comparison.missing_keys)
    compatible = not (blocking_missing or comparison.unexpected_keys or comparison.shape_mismatches)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_key_count": comparison.checkpoint_key_count,
        "expected_key_count": comparison.expected_key_count,
        "checkpoint_prefix_counts": _prefix_counts(checkpoint_schema),
        "expected_prefix_counts": _prefix_counts(expected_schema),
        "allowed_missing_keys": list(allowed_missing),
        "blocking_missing_keys": list(blocking_missing),
        "unexpected_keys": list(comparison.unexpected_keys),
        "shape_mismatches": [
            {
                "key": key,
                "checkpoint_shape": list(checkpoint_shape),
                "expected_shape": list(expected_shape),
            }
            for key, checkpoint_shape, expected_shape in comparison.shape_mismatches
        ],
        "compatible_with_runtime_initializers": compatible,
        "notes": [
            "RoPE freqs_cis buffers are deterministically created by Meta's constructor.",
            (
                "The absent text_projection is an unused pooled-output projection "
                "and must be initialized safely at runtime."
            ),
            (
                "Serialized F16/BF16 weights may load into the builder's parameter dtype; "
                "dtype differences are informational."
            ),
        ],
    }


def _print_human_report(report: dict[str, object]) -> None:
    print(f"Checkpoint: {report['checkpoint']}")
    print(
        "Tensor keys: "
        f"{report['checkpoint_key_count']} checkpoint / "
        f"{report['expected_key_count']} Meta expected"
    )
    print(f"Checkpoint prefixes: {report['checkpoint_prefix_counts']}")
    print(f"Meta prefixes: {report['expected_prefix_counts']}")
    print(f"Allowed missing: {len(report['allowed_missing_keys'])}")
    print(f"Blocking missing: {len(report['blocking_missing_keys'])}")
    print(f"Unexpected: {len(report['unexpected_keys'])}")
    print(f"Shape mismatches: {len(report['shape_mismatches'])}")
    verdict = (
        "COMPATIBLE WITH RUNTIME INITIALIZERS"
        if report["compatible_with_runtime_initializers"]
        else "INCOMPATIBLE"
    )
    print(f"Verdict: {verdict}")
    for note in report["notes"]:
        print(f"- {note}")

    for label in ("blocking_missing_keys", "unexpected_keys"):
        values = report[label]
        if values:
            print(f"{label}:")
            for value in values:
                print(f"  {value}")
    if report["shape_mismatches"]:
        print("shape_mismatches:")
        for mismatch in report["shape_mismatches"]:
            print(f"  {mismatch['key']}: {mismatch['checkpoint_shape']} != {mismatch['expected_shape']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = inspect_checkpoint(args.checkpoint)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 0 if report["compatible_with_runtime_initializers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
