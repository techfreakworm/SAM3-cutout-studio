import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class _AvailabilityFlag:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _fake_torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=_AvailabilityFlag(cuda),
        backends=SimpleNamespace(mps=_AvailabilityFlag(mps)),
    )


def test_import_sets_pre_torch_environment_without_heavy_imports() -> None:
    source_root = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    environment.pop("TOKENIZERS_PARALLELISM", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_root), environment.get("PYTHONPATH")) if value
    )
    script = """
import os
import sys
import sam3_matting.runtime

print(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])
print(os.environ["TOKENIZERS_PARALLELISM"])
print(",".join(name for name in ("torch", "spaces", "huggingface_hub") if name in sys.modules))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.stdout.splitlines() == ["1", "false", ""]


def test_setup_runtime_is_idempotent_and_preserves_explicit_values() -> None:
    from sam3_matting.runtime import setup_runtime

    environment = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "custom",
    }

    setup_runtime(environment)
    setup_runtime(environment)

    assert environment == {
        "PYTORCH_ENABLE_MPS_FALLBACK": "custom",
        "TOKENIZERS_PARALLELISM": "false",
    }


@pytest.mark.parametrize("flag", ["SPACES_ZERO_GPU", "SPACES_ZERO_GPU_V2"])
def test_on_zerogpu_recognizes_both_space_flags(flag: str) -> None:
    from sam3_matting.runtime import on_zerogpu

    assert on_zerogpu({flag: "1"}) is True
    assert on_zerogpu({}) is False


def test_target_device_is_cuda_for_zerogpu_before_cuda_is_visible() -> None:
    from sam3_matting.runtime import target_device

    unavailable_torch = _fake_torch(cuda=False, mps=False)

    assert (
        target_device(
            environ={"SPACES_ZERO_GPU": "1"},
            torch_module=unavailable_torch,
        )
        == "cuda"
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_target_device_uses_local_accelerator_visibility(
    cuda: bool,
    mps: bool,
    expected: str,
) -> None:
    from sam3_matting.runtime import target_device

    assert target_device(environ={}, torch_module=_fake_torch(cuda=cuda, mps=mps)) == expected


def test_gpu_decorator_is_identity_away_from_zerogpu() -> None:
    from sam3_matting.runtime import gpu

    class SpacesMustNotBeUsed:
        def GPU(self, **_: object) -> object:
            raise AssertionError("spaces.GPU must not be touched for a local process")

    def add_one(value: int) -> int:
        return value + 1

    decorated = gpu(duration=75, environ={}, spaces_module=SpacesMustNotBeUsed())(add_one)

    assert decorated is add_one
    assert decorated(2) == 3


def test_gpu_decorator_delegates_duration_and_size_inside_zerogpu() -> None:
    from sam3_matting.runtime import gpu

    observed_options: list[dict[str, object]] = []

    class FakeSpaces:
        def GPU(self, **options: object):
            observed_options.append(options)

            def decorator(function):
                def wrapped(*args: object, **kwargs: object):
                    return "gpu", function(*args, **kwargs)

                return wrapped

            return decorator

    def dynamic_duration(*_: object) -> int:
        return 90

    def add_one(value: int) -> int:
        return value + 1

    decorated = gpu(
        duration=dynamic_duration,
        size="xlarge",
        environ={"SPACES_ZERO_GPU_V2": "1"},
        spaces_module=FakeSpaces(),
    )(add_one)

    assert observed_options == [{"duration": dynamic_duration, "size": "xlarge"}]
    assert decorated(2) == ("gpu", 3)


def test_resolve_checkpoint_prefers_valid_explicit_path(tmp_path: Path) -> None:
    from sam3_matting.runtime import resolve_sam_checkpoint

    checkpoint = tmp_path / "local.safetensors"
    checkpoint.write_bytes(b"checkpoint")

    def forbidden_download(**_: object) -> str:
        raise AssertionError("an explicit checkpoint must not call Hugging Face")

    resolved = resolve_sam_checkpoint(
        environ={"SAM3_CHECKPOINT": str(checkpoint)},
        downloader=forbidden_download,
    )

    assert resolved == checkpoint.resolve()


def test_resolve_checkpoint_rejects_missing_explicit_path(tmp_path: Path) -> None:
    from sam3_matting.runtime import resolve_sam_checkpoint

    missing = tmp_path / "missing.safetensors"

    with pytest.raises(FileNotFoundError, match="SAM3_CHECKPOINT"):
        resolve_sam_checkpoint(environ={"SAM3_CHECKPOINT": str(missing)})


def test_resolve_checkpoint_downloads_the_pinned_public_model(tmp_path: Path) -> None:
    from sam3_matting.runtime import resolve_sam_checkpoint

    checkpoint = tmp_path / "sam3.1_multiplex_fp16.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    observed_kwargs: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        observed_kwargs.append(kwargs)
        return str(checkpoint)

    resolved = resolve_sam_checkpoint(environ={}, downloader=fake_download)

    assert resolved == checkpoint.resolve()
    assert observed_kwargs == [
        {
            "repo_id": "Comfy-Org/sam3.1",
            "filename": "checkpoints/sam3.1_multiplex_fp16.safetensors",
            "revision": "ba901fbc9701054c359ed5240c4d76f83a178108",
        }
    ]
