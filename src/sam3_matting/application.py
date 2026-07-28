"""Production inference wiring for the Gradio studio."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from typing import Any

import gradio as gr

from .backends.base import SamVideoBackend
from .backends.meta_sam31 import MetaSam31Backend
from .config import MatteConfig, TrackingConfig
from .matte import VitMatteRefiner
from .media import InputLimits, MediaValidationError, VideoMetadata, validate_input
from .pipeline import PipelineResult, parse_sam_prompts, run_pipeline

ZERO_GPU_LIMITS = InputLimits(
    max_duration_seconds=2.0,
    max_width=1920,
    max_height=1920,
    max_frames=60,
    max_fps=30,
    max_file_size_bytes=100 * 1024 * 1024,
)
LOCAL_LIMITS = InputLimits(
    max_duration_seconds=120.0,
    max_width=4096,
    max_height=4096,
    max_frames=7200,
    max_fps=60,
    max_file_size_bytes=2 * 1024 * 1024 * 1024,
)

_REFINER_SETTING_NAMES = (
    "erode_kernel",
    "dilate_kernel",
    "black_point",
    "white_point",
    "max_megapixels",
)

PipelineFn = Callable[..., PipelineResult]
InputValidator = Callable[[str | Path, InputLimits], VideoMetadata]
ProgressFactory = Callable[[], Any]
ErrorFactory = Callable[[str], Exception]
IncidentIdFactory = Callable[[], str]

_LOGGER = logging.getLogger(__name__)
_HOSTED_MAX_PROMPT_CLAUSES = 3
_LOCAL_MAX_PROMPT_CLAUSES = 4


class UnsupportedDeviceError(RuntimeError):
    """Raised when the CUDA-first production backend cannot use a device."""


def _require_cuda(device: str) -> None:
    if device != "cuda":
        raise UnsupportedDeviceError(
            f"SAM3 Cutout Studio currently requires CUDA; {device.upper()} support is not implemented yet"
        )


@dataclass(slots=True)
class ApplicationResources:
    """One process-wide SAM backend and ViTMatte model allocation."""

    backend: SamVideoBackend
    refiner: VitMatteRefiner
    device: str
    _settings_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _preload_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _preloaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _require_cuda(self.device)

    def preload(self) -> None:
        """Materialize both lazy models once during application startup."""
        with self._preload_lock:
            if self._preloaded:
                return

            backend_loader = getattr(self.backend, "_get_predictor", None)
            refiner_loader = getattr(self.refiner, "_load_model", None)
            if not callable(backend_loader) or not callable(refiner_loader):
                raise RuntimeError("cached inference resources do not expose eager preload hooks")

            backend_loader()
            refiner_loader()
            self._preloaded = True

    @contextmanager
    def configured_refiner(self, config: MatteConfig) -> Iterator[VitMatteRefiner]:
        """Apply one job's settings while holding the shared refiner lease."""
        requested = {
            "erode_kernel": config.erode_kernel,
            "dilate_kernel": config.dilate_kernel,
            "black_point": config.black_point,
            "white_point": config.white_point,
            "max_megapixels": config.max_megapixels,
        }
        with self._settings_lock:
            original = {setting: getattr(self.refiner, setting) for setting in _REFINER_SETTING_NAMES}
            try:
                for setting, value in requested.items():
                    setattr(self.refiner, setting, value)
                yield self.refiner
            finally:
                for setting, value in original.items():
                    setattr(self.refiner, setting, value)


def build_resources(
    checkpoint_path: str | Path,
    *,
    device: str,
    backend_factory: Callable[..., SamVideoBackend] = MetaSam31Backend,
    refiner_factory: Callable[..., VitMatteRefiner] = VitMatteRefiner,
    preload: bool = False,
) -> ApplicationResources:
    """Construct the process-wide CUDA resources, optionally loading immediately."""
    _require_cuda(device)
    resources = ApplicationResources(
        backend=backend_factory(str(checkpoint_path), max_objects=8, device=device),
        refiner=refiner_factory(device=device),
        device=device,
    )
    if preload:
        resources.preload()
    return resources


def select_input_limits(zerogpu: bool) -> InputLimits:
    """Choose strict hosted limits or the broader local-CUDA policy."""
    return ZERO_GPU_LIMITS if zerogpu else LOCAL_LIMITS


def validate_request_input(
    source: str | Path,
    limits: InputLimits,
    *,
    zerogpu: bool,
    validator: InputValidator = validate_input,
) -> VideoMetadata:
    """Validate an upload, including the orientation-neutral ZeroGPU canvas."""
    metadata = validator(Path(source), limits)
    if zerogpu:
        long_edge = max(metadata.width, metadata.height)
        short_edge = min(metadata.width, metadata.height)
        if long_edge > 1920 or short_edge > 1080:
            raise MediaValidationError(
                (
                    (
                        f"dimensions {metadata.width}x{metadata.height}px exceed the "
                        "ZeroGPU 1080x1920 portrait/landscape canvas"
                    ),
                )
            )
    return metadata


def _default_progress_factory() -> gr.Progress:
    return gr.Progress(track_tqdm=False)


def _progress_bridge(progress: Any) -> Callable[[str, int, int], None]:
    def report(phase: str, completed: int, total: int) -> None:
        if total <= 0:
            return
        ratio = min(max(completed / total, 0.0), 1.0)
        if phase == "tracking":
            progress(0.7 * ratio, desc="Tracking subjects")
        elif phase == "matting":
            progress(0.7 + 0.3 * ratio, desc="Refining alpha")

    return report


def _output_root(explicit_root: str | Path | None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)

    configured_root = os.environ.get("SAM3_OUTPUT_DIR")
    if configured_root:
        return Path(configured_root).expanduser()

    gradio_cache = os.environ.get("GRADIO_TEMP_DIR")
    base = Path(gradio_cache).expanduser() if gradio_cache else Path(tempfile.gettempdir()) / "gradio"
    return base / "sam3-cutout-studio"


def _prompt_clauses(prompt: object, *, zerogpu: bool) -> tuple[str, ...]:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be text")
    clauses = parse_sam_prompts(prompt)
    maximum = _HOSTED_MAX_PROMPT_CLAUSES if zerogpu else _LOCAL_MAX_PROMPT_CLAUSES
    if len(clauses) > maximum:
        raise ValueError(f"prompt supports at most {maximum} unique clauses")
    return clauses


def _validated_configuration(
    *,
    prompt: object,
    detection_threshold: object,
    max_objects: object,
    detect_interval: object,
    erode_kernel: object,
    dilate_kernel: object,
    black_point: object,
    white_point: object,
    max_megapixels: object,
    zerogpu: bool,
) -> tuple[TrackingConfig, MatteConfig]:
    _prompt_clauses(prompt, zerogpu=zerogpu)
    tracking = TrackingConfig(
        detection_threshold=detection_threshold,  # type: ignore[arg-type]
        max_objects=max_objects,  # type: ignore[arg-type]
        detect_interval=detect_interval,  # type: ignore[arg-type]
    )
    matte = MatteConfig(
        erode_kernel=erode_kernel,  # type: ignore[arg-type]
        dilate_kernel=dilate_kernel,  # type: ignore[arg-type]
        black_point=black_point,  # type: ignore[arg-type]
        white_point=white_point,  # type: ignore[arg-type]
        max_megapixels=max_megapixels,  # type: ignore[arg-type]
    )
    return tracking, matte


def _validation(check: Callable[[], object], message: str) -> dict[str, object]:
    try:
        check()
    except Exception:
        return gr.validate(False, message)
    return gr.validate(True, "")


def _request_validation_results(
    source_video: object,
    prompt: object,
    detection_threshold: object,
    max_objects: object,
    detect_interval: object,
    erode_kernel: object,
    dilate_kernel: object,
    black_point: object,
    white_point: object,
    max_megapixels: object,
    *,
    zerogpu: bool,
    input_validator: InputValidator,
) -> tuple[dict[str, object], ...]:
    limits = select_input_limits(zerogpu)
    maximum_clauses = _HOSTED_MAX_PROMPT_CLAUSES if zerogpu else _LOCAL_MAX_PROMPT_CLAUSES

    def validate_source() -> None:
        if not source_video or not isinstance(source_video, (str, Path)):
            raise ValueError("source is missing")
        validate_request_input(
            source_video,
            limits,
            zerogpu=zerogpu,
            validator=input_validator,
        )

    source_message = (
        "Upload a video within 2 seconds, 60 frames, 30 FPS, 1080p, and 100 MiB."
        if zerogpu
        else "We could not read this video. Try another MP4, MOV, or WebM file."
    )

    checks: tuple[tuple[Callable[[], object], str], ...] = (
        (validate_source, source_message),
        (
            lambda: _prompt_clauses(prompt, zerogpu=zerogpu),
            f"Use 1 to {maximum_clauses} comma-separated subject clauses.",
        ),
        (
            lambda: TrackingConfig(detection_threshold=detection_threshold),  # type: ignore[arg-type]
            "Use a finite detection threshold from 0.05 to 0.95.",
        ),
        (
            lambda: TrackingConfig(max_objects=max_objects),  # type: ignore[arg-type]
            "Use a whole-number maximum object count from 1 to 8.",
        ),
        (
            lambda: TrackingConfig(detect_interval=detect_interval),  # type: ignore[arg-type]
            "Use a whole-number detection interval from 1 to 30.",
        ),
        (
            lambda: MatteConfig(erode_kernel=erode_kernel),  # type: ignore[arg-type]
            "Use a whole-number erode kernel from 1 to 31.",
        ),
        (
            lambda: MatteConfig(dilate_kernel=dilate_kernel),  # type: ignore[arg-type]
            "Use a whole-number dilate kernel from 1 to 31.",
        ),
        (
            lambda: MatteConfig(black_point=black_point),  # type: ignore[arg-type]
            "Use a finite black point from 0 to 0.9.",
        ),
        (
            lambda: MatteConfig(white_point=white_point),  # type: ignore[arg-type]
            "Use a finite white point from 0.1 to 1.",
        ),
        (
            lambda: MatteConfig(max_megapixels=max_megapixels),  # type: ignore[arg-type]
            "Use a finite ViTMatte budget from 0.25 to 4 megapixels.",
        ),
    )
    return tuple(_validation(check, message) for check, message in checks)


def create_request_validator(
    *,
    zerogpu: bool,
    input_validator: InputValidator = validate_input,
) -> Callable[..., tuple[dict[str, object], ...]]:
    """Build Gradio's non-queued preflight validator for the ten UI inputs."""
    if zerogpu:

        def hosted_validator(
            source_video: str,
            prompt: str,
            detection_threshold: float,
            max_objects: int,
            detect_interval: int,
            erode_kernel: int,
            dilate_kernel: int,
            black_point: float,
            white_point: float,
            max_megapixels: float,
            oauth_profile: gr.OAuthProfile,
        ) -> tuple[dict[str, object], ...]:
            del oauth_profile
            return _request_validation_results(
                source_video,
                prompt,
                detection_threshold,
                max_objects,
                detect_interval,
                erode_kernel,
                dilate_kernel,
                black_point,
                white_point,
                max_megapixels,
                zerogpu=True,
                input_validator=input_validator,
            )

        return hosted_validator

    def local_validator(
        source_video: str,
        prompt: str,
        detection_threshold: float,
        max_objects: int,
        detect_interval: int,
        erode_kernel: int,
        dilate_kernel: int,
        black_point: float,
        white_point: float,
        max_megapixels: float,
    ) -> tuple[dict[str, object], ...]:
        return _request_validation_results(
            source_video,
            prompt,
            detection_threshold,
            max_objects,
            detect_interval,
            erode_kernel,
            dilate_kernel,
            black_point,
            white_point,
            max_megapixels,
            zerogpu=False,
            input_validator=input_validator,
        )

    return local_validator


def _effective_clause_count(effective_prompt: object) -> int:
    if isinstance(effective_prompt, str):
        return len([clause for clause in effective_prompt.split(",") if clause.strip()])
    try:
        return len(effective_prompt)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _status(result: PipelineResult, *, device: str) -> str:
    frame_label = "frame" if result.processed_frame_count == 1 else "frames"
    clause_count = _effective_clause_count(result.effective_sam_prompt)
    clause_label = "clause" if clause_count == 1 else "clauses"
    return (
        f"**COMPLETE** · {result.processed_frame_count} {frame_label} · "
        f"{clause_count} {clause_label} · {result.elapsed_seconds:.2f}s · {device.upper()}"
    )


def _cleanup_failed_request(request_directory: Path | None) -> None:
    """Best-effort removal of one incomplete directory owned by this request."""
    if request_directory is None:
        return
    try:
        shutil.rmtree(request_directory)
    except BaseException:
        _LOGGER.exception("Failed to remove an incomplete request directory")


def create_process_callback(
    resources: ApplicationResources | Callable[[], ApplicationResources],
    *,
    zerogpu: bool,
    pipeline_fn: PipelineFn = run_pipeline,
    input_validator: InputValidator = validate_input,
    output_root: str | Path | None = None,
    progress_factory: ProgressFactory = _default_progress_factory,
    error_factory: ErrorFactory = gr.Error,
    incident_id_factory: IncidentIdFactory = lambda: uuid.uuid4().hex[:8].upper(),
) -> Callable[
    [str, str, float, int, int, int, int, float, float, float],
    tuple[str, str, str, str],
]:
    """Create the exact ten-input/four-output Gradio inference callback.

    ``resources`` may be the eagerly built resources (local CUDA) or a provider
    invoked on each call (ZeroGPU), so models can be resolved lazily inside the
    persistent GPU worker instead of crossing its pickle boundary.
    """
    limits = select_input_limits(zerogpu)
    cache_root = _output_root(output_root)

    def process(
        source_video: str,
        prompt: str,
        detection_threshold: float,
        max_objects: int,
        detect_interval: int,
        erode_kernel: int,
        dilate_kernel: int,
        black_point: float,
        white_point: float,
        max_megapixels: float,
    ) -> tuple[str, str, str, str]:
        request_directory: Path | None = None
        try:
            if not source_video:
                raise ValueError("upload a source video before starting")

            tracking, matte = _validated_configuration(
                prompt=prompt,
                detection_threshold=detection_threshold,
                max_objects=max_objects,
                detect_interval=detect_interval,
                erode_kernel=erode_kernel,
                dilate_kernel=dilate_kernel,
                black_point=black_point,
                white_point=white_point,
                max_megapixels=max_megapixels,
                zerogpu=zerogpu,
            )
            source = Path(source_video)
            validate_request_input(
                source,
                limits,
                zerogpu=zerogpu,
                validator=input_validator,
            )

            cache_root.mkdir(parents=True, exist_ok=True)
            request_directory = Path(tempfile.mkdtemp(prefix="job-", dir=cache_root))
            progress = progress_factory()
            progress_callback = _progress_bridge(progress)

            active_resources = resources() if callable(resources) else resources
            with active_resources.configured_refiner(matte) as refiner:
                result = pipeline_fn(
                    source,
                    prompt=prompt,
                    detection_threshold=tracking.detection_threshold,
                    detect_interval=tracking.detect_interval,
                    max_objects=tracking.max_objects,
                    backend=active_resources.backend,
                    refiner=refiner,
                    output_dir=request_directory,
                    limits=limits,
                    progress_callback=progress_callback,
                )

            progress(1.0, desc="Complete")
            return (
                str(result.preview_path),
                str(result.master_path),
                str(result.matte_path),
                _status(result, device=active_resources.device),
            )
        except BaseException as exc:
            _cleanup_failed_request(request_directory)
            if not isinstance(exc, Exception):
                raise
            incident_id = incident_id_factory()
            _LOGGER.exception("Inference failed [incident %s]", incident_id)
            message = f"Could not process this video. Reference: {incident_id}"
            if os.environ.get("SAM3_DEBUG_ERRORS"):
                message += f" (debug: {type(exc).__name__}: {str(exc)[:180]})"
            raise error_factory(message) from exc

    return process
