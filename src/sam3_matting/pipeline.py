"""Bounded-memory orchestration for multi-clause SAM tracking and video matting."""

from __future__ import annotations

import re
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Protocol

import numpy as np

from .backends.base import SamVideoBackend
from .matte import VitMatteRefiner
from .media import (
    H264Mp4Sink,
    InputLimits,
    ProRes4444MovSink,
    VideoMetadata,
    decode_video_frames,
    validate_input,
)

MAX_PROMPT_CLAUSES = 4
MAX_OUTPUT_STEM_LENGTH = 96
_MIC_WORD = re.compile(r"\bmic\b", flags=re.IGNORECASE)
_UNSAFE_OUTPUT_STEM = re.compile(r"[^A-Za-z0-9._-]+")


class PipelineError(RuntimeError):
    """Base exception for orchestration failures."""


class PipelineProtocolError(PipelineError):
    """Raised when frame, mask, or alpha streams violate their contracts."""


class NoSubjectDetectedError(PipelineError):
    """Raised when the union of every SAM clause is empty."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Published outputs and timing from one completed pipeline run."""

    preview_path: Path
    master_path: Path
    matte_path: Path
    processed_frame_count: int
    effective_sam_prompt: tuple[str, ...]
    elapsed_seconds: float


class FrameSink(Protocol):
    """Small sink contract shared by real encoders and unit-test doubles."""

    output_path: Path

    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> Path: ...

    def abort(self) -> None: ...


SinkFactory = Callable[..., FrameSink]
ValidateInput = Callable[[str | Path, InputLimits], VideoMetadata]
DecodeFrames = Callable[[str | Path], Iterator[np.ndarray]]
ProgressCallback = Callable[[str, int, int], None]
Clock = Callable[[], float]


def parse_sam_prompts(prompt: str) -> tuple[str, ...]:
    """Parse, normalize, and deduplicate the historical comma-clause prompt."""

    raw_clauses = prompt.split(",")
    if not raw_clauses or any(not clause.strip() for clause in raw_clauses):
        raise ValueError("SAM prompt contains a blank clause")

    clauses: list[str] = []
    seen: set[str] = set()
    for raw_clause in raw_clauses:
        clause = _MIC_WORD.sub("microphone", raw_clause.strip())
        dedupe_key = clause.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        clauses.append(clause)

    if len(clauses) > MAX_PROMPT_CLAUSES:
        raise ValueError(f"SAM prompt supports at most {MAX_PROMPT_CLAUSES} unique clauses")
    return tuple(clauses)


def _validate_metadata(metadata: VideoMetadata) -> None:
    if metadata.frame_count < 1:
        raise PipelineProtocolError("video metadata must report at least one frame")
    if metadata.width < 1 or metadata.height < 1:
        raise PipelineProtocolError("video metadata must report positive dimensions")
    if metadata.fps <= 0:
        raise PipelineProtocolError("video metadata must report a positive frame rate")


def _safe_output_stem(source: Path) -> str:
    sanitized = _UNSAFE_OUTPUT_STEM.sub("-", source.stem).strip("._-")
    bounded = sanitized[:MAX_OUTPUT_STEM_LENGTH].rstrip("._-")
    return bounded or "video"


def _claim_run_directory(destination: Path, output_stem: str) -> Path:
    """Atomically claim a namespace that no earlier or concurrent run owns."""

    collision_index = 0
    while True:
        suffix = "" if collision_index == 0 else f"-{collision_index}"
        run_directory = destination / f"{output_stem}{suffix}"
        claimed = False
        try:
            run_directory.mkdir()
            claimed = True
            return run_directory
        except FileExistsError:
            collision_index += 1
            continue
        except BaseException:
            if claimed:
                with suppress(BaseException):
                    run_directory.rmdir()
            raise


@contextmanager
def _packed_mask_store(
    metadata: VideoMetadata,
    *,
    directory: str | Path | None,
):
    store_directory = Path(directory) if directory is not None else None
    if store_directory is not None:
        store_directory.mkdir(parents=True, exist_ok=True)

    packed_bytes = (metadata.width * metadata.height + 7) // 8
    handle = None
    store_path: Path | None = None
    store = None
    primary_error: BaseException | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".sam3-mask-",
            suffix=".bin",
            dir=store_directory,
            delete=False,
        ) as handle:
            store_path = Path(handle.name)
        handle = None

        store = np.memmap(
            store_path,
            dtype=np.uint8,
            mode="w+",
            shape=(metadata.frame_count, packed_bytes),
        )
        store[:] = 0
        store.flush()
        yield store
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None

        def attempt_cleanup(action: Callable[[], None]) -> None:
            nonlocal cleanup_error
            try:
                action()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc

        if store is not None:
            attempt_cleanup(store.flush)
            mapped_file = getattr(store, "_mmap", None)
            if mapped_file is not None:
                attempt_cleanup(mapped_file.close)
        if handle is not None:
            attempt_cleanup(handle.close)
        if store_path is not None:
            attempt_cleanup(lambda: store_path.unlink(missing_ok=True))

        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


def _tracking_mask(frame: object, expected_index: int, metadata: VideoMetadata) -> np.ndarray:
    frame_index = getattr(frame, "frame_index", None)
    if type(frame_index) is not int or frame_index != expected_index:
        raise PipelineProtocolError(
            f"SAM tracking expected frame_index {expected_index}, received {frame_index!r}"
        )

    mask = np.asarray(getattr(frame, "union_mask", None))
    expected_shape = (metadata.height, metadata.width)
    if mask.shape != expected_shape:
        raise PipelineProtocolError(
            f"SAM mask shape {mask.shape} does not match source frame shape {expected_shape}"
        )
    if mask.dtype != np.bool_:
        if not np.issubdtype(mask.dtype, np.number):
            raise PipelineProtocolError("SAM mask must contain boolean or numeric values")
        if np.iscomplexobj(mask):
            raise PipelineProtocolError("numeric SAM mask must be real-valued")
        if not np.all(np.isfinite(mask)):
            raise PipelineProtocolError("SAM mask must contain only finite values")
        if not np.all((mask == 0) | (mask == 1)):
            raise PipelineProtocolError("numeric SAM mask must contain only 0 and 1")
    return mask.astype(np.bool_, copy=False)


def _accumulate_masks(
    *,
    source: Path,
    prompts: tuple[str, ...],
    backend: SamVideoBackend,
    detection_threshold: float,
    detect_interval: int,
    max_objects: int,
    metadata: VideoMetadata,
    store: np.memmap,
    progress_callback: ProgressCallback | None,
) -> bool:
    detected_subject = False
    total_tracking_steps = len(prompts) * metadata.frame_count

    for prompt_index, prompt in enumerate(prompts):
        produced = 0
        stream = backend.track(
            str(source),
            prompt=prompt,
            detection_threshold=detection_threshold,
            detect_interval=detect_interval,
            max_objects=max_objects,
        )
        stream_succeeded = False
        try:
            for frame in stream:
                if produced >= metadata.frame_count:
                    raise PipelineProtocolError(
                        f"SAM prompt {prompt!r} produced more than {metadata.frame_count} frames"
                    )
                mask = _tracking_mask(frame, produced, metadata)
                packed = np.packbits(mask.reshape(-1), bitorder="little")
                store[produced] |= packed
                detected_subject = detected_subject or bool(np.any(mask))
                produced += 1
                if progress_callback is not None:
                    completed = prompt_index * metadata.frame_count + produced
                    progress_callback("tracking", completed, total_tracking_steps)

            if produced != metadata.frame_count:
                raise PipelineProtocolError(
                    f"SAM prompt {prompt!r} produced {produced} frames; expected {metadata.frame_count}"
                )
            stream_succeeded = True
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                try:
                    close_stream()
                except BaseException:
                    if stream_succeeded:
                        raise
    store.flush()
    return detected_subject


def _decoded_rgb(frame: object, index: int, metadata: VideoMetadata) -> np.ndarray:
    rgb = np.asarray(frame)
    expected_shape = (metadata.height, metadata.width, 3)
    if rgb.shape != expected_shape:
        raise PipelineProtocolError(
            f"decoded RGB frame {index} shape {rgb.shape} does not match {expected_shape}"
        )
    if rgb.dtype != np.uint8:
        raise PipelineProtocolError(f"decoded RGB frame {index} must use uint8 pixels")
    return np.ascontiguousarray(rgb)


def _validated_alpha(alpha: object, index: int, expected_shape: tuple[int, int]) -> np.ndarray:
    raw_alpha = np.asarray(alpha)
    if np.iscomplexobj(raw_alpha):
        raise PipelineProtocolError(f"alpha for frame {index} must be real-valued")
    try:
        normalized = np.asarray(raw_alpha, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise PipelineProtocolError(f"alpha for frame {index} must be numeric") from exc
    if normalized.shape != expected_shape:
        raise PipelineProtocolError(
            f"alpha shape {normalized.shape} for frame {index} does not match {expected_shape}"
        )
    if not np.all(np.isfinite(normalized)):
        raise PipelineProtocolError(f"alpha for frame {index} must contain only finite values")
    if np.any(normalized < 0.0) or np.any(normalized > 1.0):
        raise PipelineProtocolError(f"alpha for frame {index} must remain within [0, 1]")
    return normalized


def _frame_outputs(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_u8 = np.rint(alpha * 255.0).astype(np.uint8)
    master_rgba = np.dstack((rgb, alpha_u8))
    preview_rgb = np.rint(rgb.astype(np.float32) * alpha[..., None]).astype(np.uint8)
    matte_rgb = np.repeat(alpha_u8[..., None], 3, axis=2)
    return preview_rgb, master_rgba, matte_rgb


def _abort_sinks(
    sinks: tuple[FrameSink, ...],
    output_paths: tuple[Path, ...],
    run_directory: Path,
) -> None:
    for sink in sinks:
        with suppress(BaseException):
            sink.abort()
    for output_path in output_paths:
        with suppress(BaseException):
            output_path.unlink(missing_ok=True)
    with suppress(BaseException):
        run_directory.rmdir()


def _process_frames(
    *,
    source: Path,
    metadata: VideoMetadata,
    store: np.memmap,
    refiner: VitMatteRefiner,
    preview_sink: FrameSink,
    master_sink: FrameSink,
    matte_sink: FrameSink,
    decode_fn: DecodeFrames,
    progress_callback: ProgressCallback | None,
) -> int:
    decoder = iter(decode_fn(source))
    processed = 0
    processing_succeeded = False
    try:
        for frame_index in range(metadata.frame_count):
            try:
                decoded = next(decoder)
            except StopIteration as exc:
                raise PipelineProtocolError(
                    f"decoded {processed} frames; expected {metadata.frame_count}"
                ) from exc

            rgb = _decoded_rgb(decoded, frame_index, metadata)
            unpacked = np.unpackbits(
                store[frame_index],
                bitorder="little",
                count=metadata.width * metadata.height,
            )
            union_mask = unpacked.reshape(metadata.height, metadata.width).astype(np.bool_)
            if np.any(union_mask):
                alpha = _validated_alpha(
                    refiner.refine(rgb, union_mask),
                    frame_index,
                    union_mask.shape,
                )
            else:
                alpha = np.zeros(union_mask.shape, dtype=np.float32)

            preview_rgb, master_rgba, matte_rgb = _frame_outputs(rgb, alpha)
            preview_sink.write(preview_rgb)
            master_sink.write(master_rgba)
            matte_sink.write(matte_rgb)
            processed += 1
            if progress_callback is not None:
                progress_callback("matting", processed, metadata.frame_count)

        try:
            next(decoder)
        except StopIteration:
            pass
        else:
            raise PipelineProtocolError(f"decoded more than the expected {metadata.frame_count} frames")
        processing_succeeded = True
        return processed
    finally:
        close_decoder = getattr(decoder, "close", None)
        if callable(close_decoder):
            try:
                close_decoder()
            except BaseException:
                if processing_succeeded:
                    raise


def run_pipeline(
    video_path: str | Path,
    *,
    prompt: str,
    detection_threshold: float,
    detect_interval: int,
    max_objects: int = 8,
    backend: SamVideoBackend,
    refiner: VitMatteRefiner,
    output_dir: str | Path,
    limits: InputLimits | None = None,
    progress_callback: ProgressCallback | None = None,
    validate_fn: ValidateInput = validate_input,
    decode_fn: DecodeFrames = decode_video_frames,
    preview_sink_factory: SinkFactory = H264Mp4Sink,
    master_sink_factory: SinkFactory = ProRes4444MovSink,
    matte_sink_factory: SinkFactory = H264Mp4Sink,
    mask_store_dir: str | Path | None = None,
    clock: Clock = time.perf_counter,
) -> PipelineResult:
    """Run multi-clause SAM union and ViTMatte refinement with bounded memory."""

    if (
        isinstance(detection_threshold, bool)
        or not isinstance(detection_threshold, Real)
        or not isfinite(detection_threshold)
        or not 0.0 <= detection_threshold <= 1.0
    ):
        raise ValueError("detection_threshold must be a finite real number between 0 and 1")
    if isinstance(detect_interval, bool) or not isinstance(detect_interval, Integral) or detect_interval < 1:
        raise ValueError("detect_interval must be an integer at least 1")
    if isinstance(max_objects, bool) or not isinstance(max_objects, Integral) or not 1 <= max_objects <= 8:
        raise ValueError("max_objects must be an integer between 1 and 8")

    started_at = clock()
    source = Path(video_path)
    safe_source_stem = _safe_output_stem(source)
    prompts = parse_sam_prompts(prompt)
    active_limits = limits if limits is not None else InputLimits()
    metadata = validate_fn(source, active_limits)
    _validate_metadata(metadata)

    with _packed_mask_store(metadata, directory=mask_store_dir) as mask_store:
        detected_subject = _accumulate_masks(
            source=source,
            prompts=prompts,
            backend=backend,
            detection_threshold=detection_threshold,
            detect_interval=detect_interval,
            max_objects=max_objects,
            metadata=metadata,
            store=mask_store,
            progress_callback=progress_callback,
        )
        if not detected_subject:
            raise NoSubjectDetectedError("No subject was detected for any normalized SAM prompt clause")

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        run_token = uuid.uuid4().hex[:12]
        requested_output_stem = f"{safe_source_stem}-{run_token}"
        run_directory = _claim_run_directory(destination, requested_output_stem)
        output_stem = run_directory.name
        preview_path = run_directory / f"{output_stem}-preview.mp4"
        master_path = run_directory / f"{output_stem}-master.mov"
        matte_path = run_directory / f"{output_stem}-matte.mp4"
        output_paths = (preview_path, master_path, matte_path)

        created_sinks: list[FrameSink] = []
        try:
            source_audio_path = source if metadata.has_audio else None
            preview_sink = preview_sink_factory(
                preview_path,
                fps=metadata.fps,
                source_audio_path=source_audio_path,
            )
            created_sinks.append(preview_sink)
            master_sink = master_sink_factory(
                master_path,
                fps=metadata.fps,
                source_audio_path=source_audio_path,
            )
            created_sinks.append(master_sink)
            matte_sink = matte_sink_factory(
                matte_path,
                fps=metadata.fps,
                source_audio_path=None,
            )
            created_sinks.append(matte_sink)

            processed = _process_frames(
                source=source,
                metadata=metadata,
                store=mask_store,
                refiner=refiner,
                preview_sink=preview_sink,
                master_sink=master_sink,
                matte_sink=matte_sink,
                decode_fn=decode_fn,
                progress_callback=progress_callback,
            )
            published_preview = Path(preview_sink.close())
            published_master = Path(master_sink.close())
            published_matte = Path(matte_sink.close())
        except BaseException:
            _abort_sinks(tuple(created_sinks), output_paths, run_directory)
            raise

        return PipelineResult(
            preview_path=published_preview,
            master_path=published_master,
            matte_path=published_matte,
            processed_frame_count=processed,
            effective_sam_prompt=prompts,
            elapsed_seconds=clock() - started_at,
        )
