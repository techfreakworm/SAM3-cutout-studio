from __future__ import annotations

import inspect
import logging
import tempfile
from fractions import Fraction
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import gradio as gr
import pytest

from sam3_matting.application import (
    LOCAL_LIMITS,
    ZERO_GPU_LIMITS,
    ApplicationResources,
    UnsupportedDeviceError,
    _output_root,
    build_resources,
    create_process_callback,
    select_input_limits,
    validate_request_input,
)
from sam3_matting.config import MatteConfig
from sam3_matting.media import MediaValidationError, VideoMetadata


class FriendlyError(Exception):
    """Small stand-in for gr.Error in callback unit tests."""


class PlannedCancellation(BaseException):
    """Stand-in for task cancellation exceptions that do not inherit Exception."""


class FakeRefiner:
    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self.erode_kernel = 6
        self.dilate_kernel = 6
        self.black_point = 0.15
        self.white_point = 0.99
        self.max_megapixels = 2.0
        self.load_calls = 0

    def _load_model(self) -> None:
        self.load_calls += 1


class ProgressRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[float, str]] = []

    def __call__(self, value: float, *, desc: str) -> None:
        self.calls.append((value, desc))


def _metadata(*, width: int = 1920, height: int = 1080) -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=2.0,
        fps=Fraction(30, 1),
        width=width,
        height=height,
        frame_count=60,
        has_audio=True,
    )


def test_callback_has_exact_contract_and_wires_validated_job_config(tmp_path: Path) -> None:
    backend = object()
    refiner = FakeRefiner()
    resources = ApplicationResources(backend=backend, refiner=refiner, device="cuda")
    validations: list[tuple[Path, object]] = []
    pipeline_calls: list[dict[str, object]] = []
    progress = ProgressRecorder()

    def validate(source: str | Path, limits: object) -> VideoMetadata:
        validations.append((Path(source), limits))
        return _metadata()

    def pipeline(source: str | Path, **kwargs: object) -> SimpleNamespace:
        progress_callback = kwargs["progress_callback"]
        assert callable(progress_callback)
        progress_callback("tracking", 30, 60)
        progress_callback("matting", 15, 60)
        pipeline_calls.append({"source": Path(source), **kwargs})
        active_refiner = kwargs["refiner"]
        assert active_refiner is refiner
        assert (
            refiner.erode_kernel,
            refiner.dilate_kernel,
            refiner.black_point,
            refiner.white_point,
            refiner.max_megapixels,
        ) == (3, 5, 0.2, 0.9, 1.5)
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            preview_path=output_dir / "preview.mp4",
            master_path=output_dir / "master.mov",
            matte_path=output_dir / "matte.mp4",
            processed_frame_count=60,
            effective_sam_prompt=("person", "hair"),
            elapsed_seconds=1.25,
        )

    callback = create_process_callback(
        resources,
        zerogpu=True,
        pipeline_fn=pipeline,
        input_validator=validate,
        output_root=tmp_path,
        progress_factory=lambda: progress,
        error_factory=FriendlyError,
    )

    assert list(inspect.signature(callback).parameters) == [
        "source_video",
        "prompt",
        "detection_threshold",
        "max_objects",
        "detect_interval",
        "erode_kernel",
        "dilate_kernel",
        "black_point",
        "white_point",
        "max_megapixels",
    ]

    preview, master, matte, status = callback(
        "/uploads/shot.mp4",
        "person,hair",
        0.55,
        6,
        2,
        3,
        5,
        0.2,
        0.9,
        1.5,
    )

    assert Path(preview).name == "preview.mp4"
    assert Path(master).name == "master.mov"
    assert Path(matte).name == "matte.mp4"
    assert "**COMPLETE**" in status
    assert "60 frames" in status
    assert "2 clauses" in status
    assert "1.25s" in status
    assert "CUDA" in status
    assert validations == [(Path("/uploads/shot.mp4"), ZERO_GPU_LIMITS)]

    assert len(pipeline_calls) == 1
    call = pipeline_calls[0]
    assert call["source"] == Path("/uploads/shot.mp4")
    assert call["prompt"] == "person,hair"
    assert call["detection_threshold"] == pytest.approx(0.55)
    assert call["max_objects"] == 6
    assert call["detect_interval"] == 2
    assert call["backend"] is backend
    assert call["refiner"] is refiner
    assert call["limits"] is ZERO_GPU_LIMITS
    assert Path(call["output_dir"]).parent == tmp_path
    assert progress.calls == [
        (pytest.approx(0.35), "Tracking subjects"),
        (pytest.approx(0.775), "Refining alpha"),
        (pytest.approx(1.0), "Complete"),
    ]

    assert (
        refiner.erode_kernel,
        refiner.dilate_kernel,
        refiner.black_point,
        refiner.white_point,
        refiner.max_megapixels,
    ) == (6, 6, 0.15, 0.99, 2.0)


def test_cached_refiner_settings_are_restored_between_jobs_and_after_failure(tmp_path: Path) -> None:
    refiner = FakeRefiner()
    resources = ApplicationResources(backend=object(), refiner=refiner, device="cuda")
    observed: list[tuple[int, int, float, float, float]] = []

    def pipeline(_source: str | Path, **kwargs: object) -> SimpleNamespace:
        active = kwargs["refiner"]
        observed.append(
            (
                active.erode_kernel,
                active.dilate_kernel,
                active.black_point,
                active.white_point,
                active.max_megapixels,
            )
        )
        if len(observed) == 2:
            raise RuntimeError("CUDA worker ran out of memory")
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            preview_path=output_dir / "preview.mp4",
            master_path=output_dir / "master.mov",
            matte_path=output_dir / "matte.mp4",
            processed_frame_count=1,
            effective_sam_prompt=("person",),
            elapsed_seconds=0.5,
        )

    callback = create_process_callback(
        resources,
        zerogpu=True,
        pipeline_fn=pipeline,
        input_validator=lambda *_args: _metadata(),
        output_root=tmp_path,
        progress_factory=ProgressRecorder,
        error_factory=FriendlyError,
        incident_id_factory=lambda: "C0FFEE12",
    )

    callback("first.mp4", "person", 0.5, 8, 1, 3, 5, 0.2, 0.9, 1.0)
    with pytest.raises(FriendlyError, match="Reference: C0FFEE12") as error:
        callback("second.mp4", "person", 0.5, 8, 1, 9, 11, 0.1, 0.8, 3.0)
    assert "CUDA worker ran out of memory" not in str(error.value)

    assert observed == [
        (3, 5, 0.2, 0.9, 1.0),
        (9, 11, 0.1, 0.8, 3.0),
    ]
    assert (
        refiner.erode_kernel,
        refiner.dilate_kernel,
        refiner.black_point,
        refiner.white_point,
        refiner.max_megapixels,
    ) == (6, 6, 0.15, 0.99, 2.0)
    assert len(list(tmp_path.iterdir())) == 1


def test_invalid_callback_config_becomes_a_friendly_gradio_error(tmp_path: Path) -> None:
    called = False

    def pipeline(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("pipeline must not run")

    callback = create_process_callback(
        ApplicationResources(backend=object(), refiner=FakeRefiner(), device="cuda"),
        zerogpu=True,
        pipeline_fn=pipeline,
        input_validator=lambda *_args: _metadata(),
        output_root=tmp_path,
        progress_factory=ProgressRecorder,
        error_factory=FriendlyError,
        incident_id_factory=lambda: "BADF00D1",
    )

    with pytest.raises(FriendlyError, match="Reference: BADF00D1") as error:
        callback("shot.mp4", "person", 0.5, 0, 1, 6, 6, 0.15, 0.99, 2.0)
    assert "max_objects" not in str(error.value)
    assert not called


def test_hosted_validator_has_required_oauth_injection_and_returns_one_result_per_input() -> None:
    from sam3_matting.application import create_request_validator

    validated: list[tuple[Path, object]] = []

    def validate(source: str | Path, limits: object) -> VideoMetadata:
        validated.append((Path(source), limits))
        return _metadata()

    validator = create_request_validator(
        zerogpu=True,
        input_validator=validate,
    )
    signature = inspect.signature(validator)
    assert list(signature.parameters) == [
        "source_video",
        "prompt",
        "detection_threshold",
        "max_objects",
        "detect_interval",
        "erode_kernel",
        "dilate_kernel",
        "black_point",
        "white_point",
        "max_megapixels",
        "oauth_profile",
    ]
    assert inspect.get_annotations(validator, eval_str=True)["oauth_profile"] is gr.OAuthProfile

    from gradio.helpers import special_args

    with pytest.raises(gr.Error, match="requires a logged in user"):
        special_args(
            validator,
            inputs=[
                "/uploads/shot.mp4",
                "person",
                0.5,
                8,
                1,
                6,
                6,
                0.15,
                0.99,
                2.0,
            ],
        )

    profile = gr.OAuthProfile(
        {
            "name": "Studio User",
            "preferred_username": "studio-user",
            "profile": "https://huggingface.co/studio-user",
            "picture": "https://huggingface.co/avatar.png",
        }
    )
    results = validator(
        "/uploads/shot.mp4",
        "person,hair,collar mic",
        0.5,
        8,
        1,
        6,
        6,
        0.15,
        0.99,
        2.0,
        profile,
    )

    assert len(results) == 10
    assert all(result == gr.validate(True, "") for result in results)
    assert validated == [(Path("/uploads/shot.mp4"), ZERO_GPU_LIMITS)]


def test_hosted_validator_rejects_four_unique_clauses_before_inference() -> None:
    from sam3_matting.application import create_request_validator

    validator = create_request_validator(
        zerogpu=True,
        input_validator=lambda *_args: _metadata(),
    )
    profile = gr.OAuthProfile(
        {
            "name": "Studio User",
            "preferred_username": "studio-user",
            "profile": "https://huggingface.co/studio-user",
            "picture": "https://huggingface.co/avatar.png",
        }
    )

    results = validator(
        "shot.mp4",
        "person,hair,collar microphone,jacket",
        0.5,
        8,
        1,
        6,
        6,
        0.15,
        0.99,
        2.0,
        profile,
    )

    assert len(results) == 10
    assert results[1] == gr.validate(False, "Use 1 to 3 comma-separated subject clauses.")
    assert all(result["is_valid"] for index, result in enumerate(results) if index != 1)


def test_local_validator_allows_four_clauses_without_oauth_and_rejects_five() -> None:
    from sam3_matting.application import create_request_validator

    validator = create_request_validator(
        zerogpu=False,
        input_validator=lambda *_args: _metadata(),
    )
    assert list(inspect.signature(validator).parameters)[-1] == "max_megapixels"

    valid = validator("shot.mp4", "a,b,c,d", 0.5, 8, 1, 6, 6, 0.15, 0.99, 2.0)
    invalid = validator("shot.mp4", "a,b,c,d,e", 0.5, 8, 1, 6, 6, 0.15, 0.99, 2.0)

    assert all(result["is_valid"] for result in valid)
    assert invalid[1] == gr.validate(False, "Use 1 to 4 comma-separated subject clauses.")


def test_validator_uses_controlled_messages_for_media_and_hostile_api_values() -> None:
    from sam3_matting.application import create_request_validator

    secret = "/private/uploads/customer-name.mp4"

    def broken_media(*_args: object) -> VideoMetadata:
        raise RuntimeError(f"ffprobe exploded while reading {secret}")

    validator = create_request_validator(
        zerogpu=False,
        input_validator=broken_media,
    )
    results = validator(secret, "person", "0.5", True, 1.0, 6.0, "6", float("nan"), 0.99, 5.0)

    assert len(results) == 10
    assert results[0] == gr.validate(
        False,
        "We could not read this video. Try another MP4, MOV, or WebM file.",
    )
    assert secret not in str(results)
    assert "ffprobe exploded" not in str(results)
    for index in (2, 3, 4, 5, 6, 7, 9):
        assert results[index]["is_valid"] is False


def test_default_output_root_is_nested_inside_the_actual_gradio_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gradio_cache = tmp_path / "custom-gradio"
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(gradio_cache))
    monkeypatch.delenv("SAM3_OUTPUT_DIR", raising=False)

    assert _output_root(None) == gradio_cache / "sam3-cutout-studio"

    monkeypatch.delenv("GRADIO_TEMP_DIR")
    assert _output_root(None) == Path(tempfile.gettempdir()) / "gradio" / "sam3-cutout-studio"


def test_explicit_output_root_and_operator_environment_override_gradio_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator_root = tmp_path / "operator"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    monkeypatch.setenv("SAM3_OUTPUT_DIR", str(operator_root))

    assert _output_root(None) == operator_root
    assert _output_root(explicit_root) == explicit_root


def test_successful_default_outputs_are_published_in_gradio_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio-cache"))
    monkeypatch.delenv("SAM3_OUTPUT_DIR", raising=False)

    def pipeline(_source: str | Path, **kwargs: object) -> SimpleNamespace:
        output_dir = Path(kwargs["output_dir"])
        paths = [output_dir / name for name in ("preview.mp4", "master.mov", "matte.mp4")]
        for path in paths:
            path.write_bytes(b"render")
        return SimpleNamespace(
            preview_path=paths[0],
            master_path=paths[1],
            matte_path=paths[2],
            processed_frame_count=1,
            effective_sam_prompt=("person",),
            elapsed_seconds=0.5,
        )

    callback = create_process_callback(
        ApplicationResources(backend=object(), refiner=FakeRefiner(), device="cuda"),
        zerogpu=False,
        pipeline_fn=pipeline,
        input_validator=lambda *_args: _metadata(),
        progress_factory=ProgressRecorder,
        error_factory=FriendlyError,
    )
    preview, master, matte, _status = callback(
        "shot.mp4",
        "person",
        0.5,
        8,
        1,
        6,
        6,
        0.15,
        0.99,
        2.0,
    )

    cache_root = tmp_path / "gradio-cache" / "sam3-cutout-studio"
    for output in (preview, master, matte):
        output_path = Path(output)
        assert output_path.is_relative_to(cache_root)
        assert output_path.is_file()


def test_process_logs_full_failure_but_only_exposes_an_incident_reference(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "/private/models/checkpoint.safetensors"

    def pipeline(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"failed to load {secret}")

    callback = create_process_callback(
        ApplicationResources(backend=object(), refiner=FakeRefiner(), device="cuda"),
        zerogpu=False,
        pipeline_fn=pipeline,
        input_validator=lambda *_args: _metadata(),
        output_root=tmp_path,
        progress_factory=ProgressRecorder,
        error_factory=FriendlyError,
        incident_id_factory=lambda: "1A2B3C4D",
    )

    with (
        caplog.at_level(logging.ERROR, logger="sam3_matting.application"),
        pytest.raises(FriendlyError, match="Reference: 1A2B3C4D") as error,
    ):
        callback("shot.mp4", "person", 0.5, 8, 1, 6, 6, 0.15, 0.99, 2.0)

    assert secret not in str(error.value)
    assert secret in caplog.text
    assert "1A2B3C4D" in caplog.text


@pytest.mark.parametrize(
    "primary",
    [
        pytest.param(KeyboardInterrupt("cancelled"), id="keyboard-interrupt"),
        pytest.param(PlannedCancellation("cancelled"), id="base-exception-cancellation"),
    ],
)
def test_non_exception_cancellation_is_cleaned_and_reraised_unchanged(
    tmp_path: Path,
    primary: BaseException,
) -> None:
    def pipeline(*_args: object, **kwargs: object) -> object:
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "partial-output").write_bytes(b"partial")
        raise primary

    def forbidden_error(_message: str) -> Exception:
        raise AssertionError("non-Exception cancellation must not become a public Gradio error")

    callback = create_process_callback(
        ApplicationResources(backend=object(), refiner=FakeRefiner(), device="cuda"),
        zerogpu=False,
        pipeline_fn=pipeline,
        input_validator=lambda *_args: _metadata(),
        output_root=tmp_path,
        progress_factory=ProgressRecorder,
        error_factory=forbidden_error,
    )

    with pytest.raises(type(primary)) as caught:
        callback("shot.mp4", "person", 0.5, 8, 1, 6, 6, 0.15, 0.99, 2.0)

    assert caught.value is primary
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "cleanup_failure",
    [
        pytest.param(OSError("permission denied"), id="ordinary-cleanup-error"),
        pytest.param(PlannedCancellation("cleanup cancelled"), id="base-cleanup-error"),
    ],
)
def test_cleanup_failure_never_masks_an_ordinary_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cleanup_failure: BaseException,
) -> None:
    import sam3_matting.application as application

    primary = RuntimeError("primary inference failure")
    original_rmtree = application.shutil.rmtree

    def pipeline(*_args: object, **kwargs: object) -> object:
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "partial-output").write_bytes(b"partial")
        raise primary

    def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        raise cleanup_failure

    monkeypatch.setattr(application.shutil, "rmtree", fail_cleanup)
    callback = create_process_callback(
        ApplicationResources(backend=object(), refiner=FakeRefiner(), device="cuda"),
        zerogpu=False,
        pipeline_fn=pipeline,
        input_validator=lambda *_args: _metadata(),
        output_root=tmp_path,
        progress_factory=ProgressRecorder,
        error_factory=FriendlyError,
        incident_id_factory=lambda: "CLEAN123",
    )

    try:
        with (
            caplog.at_level(logging.ERROR, logger="sam3_matting.application"),
            pytest.raises(FriendlyError, match="Reference: CLEAN123") as caught,
        ):
            callback("shot.mp4", "person", 0.5, 8, 1, 6, 6, 0.15, 0.99, 2.0)
    finally:
        monkeypatch.setattr(application.shutil, "rmtree", original_rmtree)
        for child in tmp_path.iterdir():
            original_rmtree(child)

    assert caught.value.__cause__ is primary
    assert "primary inference failure" in caplog.text
    assert str(cleanup_failure) in caplog.text


def test_cleanup_failure_never_masks_a_non_exception_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sam3_matting.application as application

    primary = PlannedCancellation("primary cancellation")
    cleanup_calls = 0
    original_rmtree = application.shutil.rmtree

    def pipeline(*_args: object, **kwargs: object) -> object:
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "partial-output").write_bytes(b"partial")
        raise primary

    def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise OSError("cleanup failed")

    monkeypatch.setattr(application.shutil, "rmtree", fail_cleanup)
    callback = create_process_callback(
        ApplicationResources(backend=object(), refiner=FakeRefiner(), device="cuda"),
        zerogpu=False,
        pipeline_fn=pipeline,
        input_validator=lambda *_args: _metadata(),
        output_root=tmp_path,
        progress_factory=ProgressRecorder,
        error_factory=FriendlyError,
    )

    try:
        with pytest.raises(PlannedCancellation) as caught:
            callback("shot.mp4", "person", 0.5, 8, 1, 6, 6, 0.15, 0.99, 2.0)
    finally:
        monkeypatch.setattr(application.shutil, "rmtree", original_rmtree)
        for child in tmp_path.iterdir():
            original_rmtree(child)

    assert caught.value is primary
    assert cleanup_calls == 1


def test_hosted_process_defensively_rejects_four_clauses(tmp_path: Path) -> None:
    called = False

    def pipeline(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("pipeline must not run")

    callback = create_process_callback(
        ApplicationResources(backend=object(), refiner=FakeRefiner(), device="cuda"),
        zerogpu=True,
        pipeline_fn=pipeline,
        input_validator=lambda *_args: _metadata(),
        output_root=tmp_path,
        progress_factory=ProgressRecorder,
        error_factory=FriendlyError,
        incident_id_factory=lambda: "3CLAUSE0",
    )

    with pytest.raises(FriendlyError, match="Reference: 3CLAUSE0"):
        callback(
            "shot.mp4",
            "person,hair,microphone,jacket",
            0.5,
            8,
            1,
            6,
            6,
            0.15,
            0.99,
            2.0,
        )
    assert not called


def test_zerogpu_policy_enforces_two_second_sixty_frame_1080p_canvas() -> None:
    seen: list[object] = []

    def landscape(_source: str | Path, limits: object) -> VideoMetadata:
        seen.append(limits)
        return _metadata(width=1920, height=1080)

    def portrait(_source: str | Path, _limits: object) -> VideoMetadata:
        return _metadata(width=1080, height=1920)

    assert (
        validate_request_input(
            "landscape.mp4",
            ZERO_GPU_LIMITS,
            zerogpu=True,
            validator=landscape,
        ).width
        == 1920
    )
    assert (
        validate_request_input(
            "portrait.mp4",
            ZERO_GPU_LIMITS,
            zerogpu=True,
            validator=portrait,
        ).height
        == 1920
    )
    assert seen == [ZERO_GPU_LIMITS]

    with pytest.raises(MediaValidationError, match="1080x1920"):
        validate_request_input(
            "oversized-square.mp4",
            ZERO_GPU_LIMITS,
            zerogpu=True,
            validator=lambda *_args: _metadata(width=1081, height=1081),
        )

    assert ZERO_GPU_LIMITS.max_duration_seconds == 2.0
    assert ZERO_GPU_LIMITS.max_frames == 60
    assert ZERO_GPU_LIMITS.max_width == 1920
    assert ZERO_GPU_LIMITS.max_height == 1920
    assert ZERO_GPU_LIMITS.max_fps == 30
    assert ZERO_GPU_LIMITS.max_file_size_bytes == 100 * 1024 * 1024
    assert select_input_limits(True) is ZERO_GPU_LIMITS
    assert select_input_limits(False) is LOCAL_LIMITS
    assert LOCAL_LIMITS.max_duration_seconds > ZERO_GPU_LIMITS.max_duration_seconds
    assert LOCAL_LIMITS.max_frames > ZERO_GPU_LIMITS.max_frames


def test_resources_are_constructed_and_preloaded_once_and_reject_non_cuda() -> None:
    factory_calls: list[tuple[str, object]] = []

    class FakeBackend:
        def __init__(self) -> None:
            self.load_calls = 0

        def _get_predictor(self) -> object:
            self.load_calls += 1
            return object()

    backend = FakeBackend()
    refiner = FakeRefiner()

    def backend_factory(checkpoint_path: str, **kwargs: object) -> FakeBackend:
        factory_calls.append(("backend", (checkpoint_path, kwargs)))
        return backend

    def refiner_factory(**kwargs: object) -> FakeRefiner:
        factory_calls.append(("refiner", kwargs))
        return refiner

    resources = build_resources(
        Path("/models/sam.safetensors"),
        device="cuda",
        backend_factory=backend_factory,
        refiner_factory=refiner_factory,
        preload=True,
    )
    resources.preload()

    assert resources.backend is backend
    assert resources.refiner is refiner
    assert resources.device == "cuda"
    assert factory_calls == [
        (
            "backend",
            (
                "/models/sam.safetensors",
                {"max_objects": 8, "device": "cuda"},
            ),
        ),
        ("refiner", {"device": "cuda"}),
    ]
    assert backend.load_calls == 1
    assert refiner.load_calls == 1

    with pytest.raises(UnsupportedDeviceError, match="CUDA"):
        build_resources(
            "/models/sam.safetensors",
            device="mps",
            backend_factory=backend_factory,
            refiner_factory=refiner_factory,
        )
    assert len(factory_calls) == 2


def test_configured_refiner_serializes_overlapping_jobs_and_restores_after_error() -> None:
    class PlannedJobFailure(RuntimeError):
        pass

    refiner = FakeRefiner()
    resources = ApplicationResources(backend=object(), refiner=refiner, device="cuda")
    first_config = MatteConfig(
        erode_kernel=3,
        dilate_kernel=5,
        black_point=0.2,
        white_point=0.9,
        max_megapixels=1.0,
    )
    second_config = MatteConfig(
        erode_kernel=9,
        dilate_kernel=11,
        black_point=0.1,
        white_point=0.8,
        max_megapixels=3.0,
    )
    first_entered = Event()
    release_first = Event()
    first_failed = Event()
    second_attempting = Event()
    second_acquired = Event()
    observations: dict[str, tuple[int, int, float, float, float]] = {}

    def snapshot() -> tuple[int, int, float, float, float]:
        return (
            refiner.erode_kernel,
            refiner.dilate_kernel,
            refiner.black_point,
            refiner.white_point,
            refiner.max_megapixels,
        )

    def first_job() -> None:
        try:
            with resources.configured_refiner(first_config):
                observations["first_entered"] = snapshot()
                first_entered.set()
                assert release_first.wait(timeout=2)
                observations["first_before_error"] = snapshot()
                raise PlannedJobFailure
        except PlannedJobFailure:
            first_failed.set()

    def second_job() -> None:
        second_attempting.set()
        with resources.configured_refiner(second_config):
            observations["second_entered"] = snapshot()
            second_acquired.set()

    first_thread = Thread(target=first_job)
    second_thread = Thread(target=second_job)
    first_thread.start()
    assert first_entered.wait(timeout=2)
    second_thread.start()
    assert second_attempting.wait(timeout=2)

    assert not second_acquired.wait(timeout=0.1)
    assert observations["first_entered"] == (3, 5, 0.2, 0.9, 1.0)
    assert snapshot() == (3, 5, 0.2, 0.9, 1.0)

    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert first_failed.is_set()
    assert second_acquired.is_set()
    assert observations == {
        "first_entered": (3, 5, 0.2, 0.9, 1.0),
        "first_before_error": (3, 5, 0.2, 0.9, 1.0),
        "second_entered": (9, 11, 0.1, 0.8, 3.0),
    }
    assert snapshot() == (6, 6, 0.15, 0.99, 2.0)
