"""Platform-neutral video I/O helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import ClassVar

import av
import numpy as np

DEFAULT_FFPROBE_TIMEOUT_SECONDS = 30.0
DEFAULT_FFMPEG_TIMEOUT_SECONDS = 120.0
_AUDIO_CODECS_SAFE_FOR_MP4_AND_MOV = frozenset({"aac"})


class MediaError(RuntimeError):
    """Base exception for media processing failures."""


class MediaProbeError(MediaError):
    """Raised when video metadata cannot be read."""


class MediaEncodeError(MediaError):
    """Raised when frames cannot be encoded into a video."""


class MediaValidationError(MediaError, ValueError):
    """Raised when an input exceeds one or more configured limits."""

    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


@dataclass(frozen=True, slots=True)
class InputLimits:
    """Optional resource limits applied before GPU work starts."""

    max_duration_seconds: float | None = None
    max_width: int | None = None
    max_height: int | None = None
    max_frames: int | None = None
    max_fps: float | Fraction | None = None
    max_file_size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata needed to validate and process an uploaded video."""

    duration_seconds: float
    fps: Fraction
    width: int
    height: int
    frame_count: int
    has_audio: bool


def _resolve_binary(name: str, explicit: str | os.PathLike[str] | None = None) -> str:
    candidate = str(explicit) if explicit is not None else os.environ.get(f"{name.upper()}_BINARY")
    resolved = shutil.which(candidate or name)
    if resolved is None:
        raise MediaProbeError(
            f"{name} was not found. Install FFmpeg or set {name.upper()}_BINARY to its executable."
        )
    return resolved


def _validate_timeout_seconds(value: object, *, tool: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{tool} timeout must be a positive finite number")
    timeout_seconds = float(value)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(f"{tool} timeout must be a positive finite number")
    return timeout_seconds


def _run_cleanups(
    cleanups: Iterable[Callable[[], object]],
    *,
    primary: BaseException | None = None,
) -> None:
    """Attempt every cleanup without replacing an initiating failure."""

    first_failure = primary
    for cleanup in cleanups:
        try:
            cleanup()
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
    if primary is None and first_failure is not None:
        raise first_failure


class _ExclusivePublisher:
    """Publish a complete temporary file without replacing an existing path."""

    def __init__(self, source: Path, destination: Path) -> None:
        self.source = source
        self.destination = destination
        self._destination_preexisted = destination.exists()
        self.owns_destination = False

    def _recover_ambiguous_link_ownership(self) -> None:
        if self._destination_preexisted:
            return
        try:
            self.owns_destination = self.source.samefile(self.destination)
        except OSError:
            self.owns_destination = False

    def _remove_owned_destination(self) -> None:
        if not self.owns_destination:
            return
        self.destination.unlink(missing_ok=True)
        self.owns_destination = False

    def rollback(self, *, primary: BaseException | None = None) -> None:
        """Remove only a destination proven to have been linked by this publisher."""

        _run_cleanups((self._remove_owned_destination,), primary=primary)

    def publish(self) -> None:
        try:
            try:
                os.link(self.source, self.destination)
            except FileExistsError as exc:
                raise MediaEncodeError(
                    f"Could not publish {self.destination.name}: destination already exists"
                ) from exc
            self.owns_destination = True
            self.source.unlink()
        except BaseException as exc:
            if not self.owns_destination:
                self._recover_ambiguous_link_ownership()
            self.rollback(primary=exc)
            raise


def _parse_fraction(value: object) -> Fraction:
    if not isinstance(value, str) or value in {"", "0/0", "N/A"}:
        return Fraction(0, 1)
    return Fraction(value)


def _parse_positive_int(value: object) -> int | None:
    if not isinstance(value, str) or not value.isdigit():
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def probe_video(
    path: str | os.PathLike[str],
    *,
    ffprobe_binary: str | os.PathLike[str] | None = None,
    ffprobe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
) -> VideoMetadata:
    """Read stable video metadata with ffprobe."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    timeout_seconds = _validate_timeout_seconds(
        ffprobe_timeout_seconds,
        tool="ffprobe",
    )

    command = [
        _resolve_binary("ffprobe", ffprobe_binary),
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "format=duration:"
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,"
            "duration,nb_frames,nb_read_frames"
        ),
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        payload = json.loads(completed.stdout)
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(
            f"Could not inspect {source.name}: ffprobe timed out after {timeout_seconds:g} seconds"
        ) from exc
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise MediaProbeError(f"Could not inspect {source.name}: {detail.strip()}") from exc

    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise MediaProbeError(f"{source.name} does not contain a video stream")

    fps = _parse_fraction(video.get("avg_frame_rate")) or _parse_fraction(video.get("r_frame_rate"))
    if fps <= 0:
        raise MediaProbeError(f"{source.name} reports an invalid frame rate")

    raw_duration = video.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration_seconds = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise MediaProbeError(f"{source.name} reports an invalid duration") from exc

    frame_count = _parse_positive_int(video.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _parse_positive_int(video.get("nb_frames"))
    if frame_count is None:
        frame_count = round(duration_seconds * fps)

    return VideoMetadata(
        duration_seconds=duration_seconds,
        fps=fps,
        width=int(video["width"]),
        height=int(video["height"]),
        frame_count=frame_count,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def validate_input(
    path: str | os.PathLike[str],
    limits: InputLimits,
    *,
    ffprobe_binary: str | os.PathLike[str] | None = None,
    ffprobe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
) -> VideoMetadata:
    """Probe an upload and reject it before expensive inference if it exceeds limits."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    file_size_bytes = source.stat().st_size
    if limits.max_file_size_bytes is not None and file_size_bytes > limits.max_file_size_bytes:
        raise MediaValidationError(
            (f"file size {file_size_bytes} bytes exceeds {limits.max_file_size_bytes} bytes",)
        )

    metadata = probe_video(
        source,
        ffprobe_binary=ffprobe_binary,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
    )
    violations: list[str] = []
    if limits.max_duration_seconds is not None and metadata.duration_seconds > limits.max_duration_seconds:
        violations.append(
            f"duration {metadata.duration_seconds:.3f}s exceeds {limits.max_duration_seconds:.3f}s"
        )
    if limits.max_width is not None and metadata.width > limits.max_width:
        violations.append(f"width {metadata.width}px exceeds {limits.max_width}px")
    if limits.max_height is not None and metadata.height > limits.max_height:
        violations.append(f"height {metadata.height}px exceeds {limits.max_height}px")
    if limits.max_frames is not None and metadata.frame_count > limits.max_frames:
        violations.append(f"frame count {metadata.frame_count} exceeds {limits.max_frames}")
    if limits.max_fps is not None and metadata.fps > limits.max_fps:
        violations.append(f"frame rate {float(metadata.fps):.3f}fps exceeds {float(limits.max_fps):.3f}fps")

    if violations:
        raise MediaValidationError(tuple(violations))
    return metadata


def decode_video_frames(path: str | os.PathLike[str]) -> Iterator[np.ndarray]:
    """Yield decoded RGB uint8 frames in presentation order."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    with av.open(str(source), mode="r") as container:
        if not container.streams.video:
            raise MediaError(f"{source.name} does not contain a video stream")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield np.ascontiguousarray(frame.to_ndarray(format="rgb24"))


def _probe_source_audio_codec(
    source_audio: Path,
    ffprobe_binary: str | os.PathLike[str] | None,
    timeout_seconds: float,
) -> str | None:
    try:
        resolved_ffprobe = _resolve_binary("ffprobe", ffprobe_binary)
    except MediaProbeError as exc:
        raise MediaEncodeError(str(exc)) from exc

    command = [
        resolved_ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(source_audio),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        payload = json.loads(completed.stdout)
    except subprocess.TimeoutExpired as exc:
        raise MediaEncodeError(
            f"Could not inspect source audio: ffprobe timed out after {timeout_seconds:g} seconds"
        ) from exc
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise MediaEncodeError(
            f"Could not inspect source audio in {source_audio.name}: {detail.strip()}"
        ) from exc

    streams = payload.get("streams", [])
    codec_name = streams[0].get("codec_name") if streams else None
    return codec_name if isinstance(codec_name, str) and codec_name else None


def _remux_source_audio(
    video_only: Path,
    source_audio: Path,
    output: Path,
    ffmpeg_binary: str | os.PathLike[str] | None,
    *,
    ffprobe_binary: str | os.PathLike[str] | None = None,
    ffprobe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_timeout_seconds: float = DEFAULT_FFMPEG_TIMEOUT_SECONDS,
    publications: list[_ExclusivePublisher] | None = None,
) -> None:
    if not source_audio.is_file():
        raise FileNotFoundError(source_audio)
    probe_timeout_seconds = _validate_timeout_seconds(
        ffprobe_timeout_seconds,
        tool="ffprobe",
    )
    remux_timeout_seconds = _validate_timeout_seconds(
        ffmpeg_timeout_seconds,
        tool="ffmpeg",
    )
    resolved_ffmpeg = _resolve_binary("ffmpeg", ffmpeg_binary)
    source_audio_codec = _probe_source_audio_codec(
        source_audio,
        ffprobe_binary,
        probe_timeout_seconds,
    )
    audio_mode = "copy" if source_audio_codec in _AUDIO_CODECS_SAFE_FOR_MP4_AND_MOV else "aac"

    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-audio-", suffix=output.suffix or ".mp4", dir=output.parent, delete=False
    ) as remux_handle:
        remuxed = Path(remux_handle.name)
    command = [
        resolved_ffmpeg,
        "-v",
        "error",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(source_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        audio_mode,
        "-movflags",
        "+faststart",
        str(remuxed),
    ]
    active_publication: _ExclusivePublisher | None = None
    try:
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=remux_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaEncodeError(
                f"Could not preserve source audio: ffmpeg timed out after {remux_timeout_seconds:g} seconds"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise MediaEncodeError(f"Could not preserve source audio: {exc.stderr.strip()}") from exc
        active_publication = _ExclusivePublisher(remuxed, output)
        if publications is not None:
            publications.append(active_publication)
        active_publication.publish()
    except BaseException as exc:
        cleanups: list[Callable[[], object]] = []
        if active_publication is not None:
            cleanups.append(active_publication.rollback)
        cleanups.append(lambda: remuxed.unlink(missing_ok=True))
        _run_cleanups(cleanups, primary=exc)
        raise


class _IncrementalVideoSink:
    """Encode one frame at a time and atomically publish only a complete video."""

    _codec_name: ClassVar[str]
    _pixel_format: ClassVar[str]
    _input_format: ClassVar[str]
    _channels: ClassVar[int]
    _codec_options: ClassVar[dict[str, str]]

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        fps: Fraction | int | float,
        source_audio_path: str | os.PathLike[str] | None = None,
        ffmpeg_binary: str | os.PathLike[str] | None = None,
        ffprobe_binary: str | os.PathLike[str] | None = None,
        ffprobe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
        ffmpeg_timeout_seconds: float = DEFAULT_FFMPEG_TIMEOUT_SECONDS,
    ) -> None:
        frame_rate = Fraction(fps).limit_denominator(100_000)
        if frame_rate <= 0:
            raise MediaEncodeError("fps must be greater than zero")

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = frame_rate
        self.source_audio_path = Path(source_audio_path) if source_audio_path is not None else None
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.ffprobe_timeout_seconds = _validate_timeout_seconds(
            ffprobe_timeout_seconds,
            tool="ffprobe",
        )
        self.ffmpeg_timeout_seconds = _validate_timeout_seconds(
            ffmpeg_timeout_seconds,
            tool="ffmpeg",
        )
        self._container = None
        self._stream = None
        self._temporary: Path | None = None
        self._expected_shape: tuple[int, int, int] | None = None
        self._frame_count = 0
        self._closed = False
        self._succeeded = False
        self._publications: list[_ExclusivePublisher] = []

    @property
    def frame_count(self) -> int:
        """Number of frames accepted by the encoder."""

        return self._frame_count

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != self._channels:
            layout = "RGB" if self._channels == 3 else "RGBA"
            raise MediaEncodeError(f"frames must be HxWx{self._channels} {layout} uint8 arrays")
        if self._expected_shape is not None and array.shape != self._expected_shape:
            raise MediaEncodeError(f"frame shape {array.shape} does not match {self._expected_shape}")
        return np.ascontiguousarray(array)

    def _open(self, first_frame: np.ndarray) -> None:
        height, width, _ = first_frame.shape
        with tempfile.NamedTemporaryFile(
            prefix=f".{self.output_path.stem}-",
            suffix=self.output_path.suffix,
            dir=self.output_path.parent,
            delete=False,
        ) as temporary_handle:
            self._temporary = Path(temporary_handle.name)

        self._container = av.open(str(self._temporary), mode="w", options={"movflags": "+faststart"})
        self._stream = self._container.add_stream(
            self._codec_name,
            rate=self.fps,
            options=self._codec_options,
        )
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = self._pixel_format
        self._stream.codec_context.thread_count = 1
        self._expected_shape = first_frame.shape

    def write(self, frame: np.ndarray) -> None:
        """Encode one frame immediately without retaining prior frames."""

        if self._closed:
            raise MediaEncodeError("cannot write to a closed video sink")
        try:
            normalized = self._normalize(frame)
            if self._container is None:
                self._open(normalized)
            assert self._container is not None
            assert self._stream is not None

            video_frame = av.VideoFrame.from_ndarray(normalized, format=self._input_format)
            video_frame.pts = self._frame_count
            video_frame.time_base = Fraction(self.fps.denominator, self.fps.numerator)
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)
            self._frame_count += 1
        except BaseException as exc:
            self.abort(primary=exc)
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, MediaEncodeError):
                raise
            raise MediaEncodeError(f"Could not encode frame {self._frame_count}: {exc}") from exc

    def close(self) -> Path:
        """Flush, finalize, and atomically publish the encoded video."""

        if self._closed:
            if self._succeeded:
                return self.output_path
            raise MediaEncodeError("video sink is closed after a failed encode")
        if self._frame_count == 0 or self._container is None or self._stream is None:
            error = MediaEncodeError("at least one frame is required")
            self.abort(primary=error)
            raise error

        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._container = None

            assert self._temporary is not None
            if self.source_audio_path is None:
                publication = _ExclusivePublisher(self._temporary, self.output_path)
                self._publications.append(publication)
                publication.publish()
            else:
                _remux_source_audio(
                    self._temporary,
                    self.source_audio_path,
                    self.output_path,
                    self.ffmpeg_binary,
                    ffprobe_binary=self.ffprobe_binary,
                    ffprobe_timeout_seconds=self.ffprobe_timeout_seconds,
                    ffmpeg_timeout_seconds=self.ffmpeg_timeout_seconds,
                    publications=self._publications,
                )
                self._temporary.unlink(missing_ok=True)
            self._temporary = None
            self._closed = True
            self._succeeded = True
            return self.output_path
        except BaseException as exc:
            self.abort(primary=exc, discard_output=True)
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, (MediaEncodeError, FileNotFoundError)):
                raise
            raise MediaEncodeError(f"Could not finalize {self.output_path.name}: {exc}") from exc

    def abort(
        self,
        *,
        primary: BaseException | None = None,
        discard_output: bool = False,
    ) -> None:
        """Discard partial artifacts while preserving any pre-existing destination."""

        container = self._container
        temporary = self._temporary
        self._container = None
        self._stream = None
        self._temporary = None
        self._closed = True

        cleanups: list[Callable[[], object]] = []
        if container is not None:
            cleanups.append(container.close)
        if temporary is not None:
            cleanups.append(lambda: temporary.unlink(missing_ok=True))
        if discard_output:
            cleanups.extend(publication.rollback for publication in self._publications)
        _run_cleanups(cleanups, primary=primary)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort(primary=exc)
        return False


class H264Mp4Sink(_IncrementalVideoSink):
    """Incremental CRF19 H.264/yuv420p MP4 encoder."""

    _codec_name = "libx264"
    _pixel_format = "yuv420p"
    _input_format = "rgb24"
    _channels = 3
    _codec_options: ClassVar[dict[str, str]] = {"crf": "19", "preset": "medium"}


class ProRes4444MovSink(_IncrementalVideoSink):
    """Incremental alpha-preserving ProRes 4444 MOV encoder."""

    _codec_name = "prores_ks"
    _pixel_format = "yuva444p10le"
    _input_format = "rgba"
    _channels = 4
    _codec_options: ClassVar[dict[str, str]] = {"profile": "4444", "alpha_bits": "16"}


def encode_video_frames(
    frames: Iterable[np.ndarray],
    output_path: str | os.PathLike[str],
    *,
    fps: Fraction | int | float,
    source_audio_path: str | os.PathLike[str] | None = None,
    ffmpeg_binary: str | os.PathLike[str] | None = None,
    ffprobe_binary: str | os.PathLike[str] | None = None,
    ffprobe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_timeout_seconds: float = DEFAULT_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Encode RGB frames to a deterministic-timing H.264 MP4."""

    sink = H264Mp4Sink(
        output_path,
        fps=fps,
        source_audio_path=source_audio_path,
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffmpeg_timeout_seconds=ffmpeg_timeout_seconds,
    )
    with sink:
        for frame in frames:
            sink.write(frame)
    return sink.output_path
