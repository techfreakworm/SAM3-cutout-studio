from __future__ import annotations

import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        pytest.fail(f"{name} is required for media integration tests")
    return path


def _audio_packet_hash(path: Path) -> str:
    completed = subprocess.run(
        [
            _require_tool("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _audio_codec_name(path: Path) -> str:
    completed = subprocess.run(
        [
            _require_tool("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_audio(path: Path, *, codec: str, duration_seconds: float) -> None:
    subprocess.run(
        [
            _require_tool("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=660:sample_rate=48000:duration={duration_seconds}",
            "-c:a",
            codec,
            str(path),
        ],
        check=True,
    )


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    output = tmp_path / "sample.mp4"
    subprocess.run(
        [
            _require_tool("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x48:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ],
        check=True,
    )
    return output


def test_probe_video_reports_video_and_audio_metadata(sample_video: Path) -> None:
    try:
        from sam3_matting.media import probe_video
    except ModuleNotFoundError:
        pytest.fail("media probing has not been implemented")

    metadata = probe_video(sample_video)

    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.fps == Fraction(10, 1)
    assert metadata.frame_count == 10
    assert metadata.duration_seconds == pytest.approx(1.0, abs=0.05)
    assert metadata.has_audio is True


def test_validate_input_reports_every_exceeded_probed_limit(sample_video: Path) -> None:
    try:
        from sam3_matting.media import InputLimits, MediaValidationError, validate_input
    except ImportError:
        pytest.fail("input-limit validation has not been implemented")

    limits = InputLimits(
        max_duration_seconds=0.5,
        max_width=32,
        max_height=24,
        max_frames=5,
        max_fps=5,
    )

    with pytest.raises(MediaValidationError) as caught:
        validate_input(sample_video, limits)

    violations = caught.value.violations
    assert len(violations) == 5
    assert any("duration" in item for item in violations)
    assert any("width" in item for item in violations)
    assert any("height" in item for item in violations)
    assert any("frame count" in item for item in violations)
    assert any("frame rate" in item for item in violations)


def test_validate_input_rejects_oversized_file_without_starting_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import InputLimits, MediaValidationError, validate_input

    source = tmp_path / "oversized.mp4"
    source.write_bytes(b"too large")

    def fail_run(*args, **kwargs):
        del args, kwargs
        pytest.fail("ffprobe must not run for an input already over the file-size limit")

    monkeypatch.setattr("sam3_matting.media.subprocess.run", fail_run)

    with pytest.raises(MediaValidationError) as caught:
        validate_input(
            source,
            InputLimits(max_file_size_bytes=1),
            ffprobe_binary=_require_tool("true"),
        )

    assert caught.value.violations == ("file size 9 bytes exceeds 1 bytes",)


def test_probe_video_uses_finite_default_and_configurable_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import DEFAULT_FFPROBE_TIMEOUT_SECONDS, probe_video

    source = tmp_path / "input.mp4"
    source.touch()
    payload = json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 64,
                    "height": 48,
                    "avg_frame_rate": "10/1",
                    "duration": "1.0",
                    "nb_read_frames": "10",
                }
            ],
            "format": {"duration": "1.0"},
        }
    )
    timeouts: list[float] = []

    def complete_probe(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr="")

    monkeypatch.setattr("sam3_matting.media.subprocess.run", complete_probe)

    probe_video(source, ffprobe_binary=_require_tool("true"))
    probe_video(
        source,
        ffprobe_binary=_require_tool("true"),
        ffprobe_timeout_seconds=0.125,
    )

    assert math.isfinite(DEFAULT_FFPROBE_TIMEOUT_SECONDS)
    assert DEFAULT_FFPROBE_TIMEOUT_SECONDS > 0
    assert timeouts == [DEFAULT_FFPROBE_TIMEOUT_SECONDS, 0.125]


def test_probe_video_reports_ffprobe_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import MediaProbeError, probe_video

    source = tmp_path / "input.mp4"
    source.touch()

    def time_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("sam3_matting.media.subprocess.run", time_out)

    with pytest.raises(MediaProbeError, match=r"ffprobe timed out after 0\.25 seconds"):
        probe_video(
            source,
            ffprobe_binary=_require_tool("true"),
            ffprobe_timeout_seconds=0.25,
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [None, True, 0, -1, math.inf, math.nan],
)
def test_probe_video_rejects_non_finite_or_non_positive_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: object,
) -> None:
    from sam3_matting.media import probe_video

    source = tmp_path / "input.mp4"
    source.touch()

    def fail_run(*args, **kwargs):
        del args, kwargs
        pytest.fail("ffprobe must not run with an invalid timeout")

    monkeypatch.setattr("sam3_matting.media.subprocess.run", fail_run)

    with pytest.raises(ValueError, match="ffprobe timeout must be a positive finite number"):
        probe_video(
            source,
            ffprobe_binary=_require_tool("true"),
            ffprobe_timeout_seconds=timeout_seconds,
        )


def test_decode_video_frames_returns_repeatable_rgb_arrays(sample_video: Path) -> None:
    try:
        from sam3_matting.media import decode_video_frames
    except ImportError:
        pytest.fail("frame decoding has not been implemented")

    first_pass = list(decode_video_frames(sample_video))
    second_pass = list(decode_video_frames(sample_video))

    assert len(first_pass) == 10
    assert first_pass[0].shape == (48, 64, 3)
    assert first_pass[0].dtype == np.uint8
    assert float(first_pass[0][..., 0].mean()) > 240
    assert float(first_pass[0][..., 1:].mean()) < 15
    for first, second in zip(first_pass, second_pass, strict=True):
        np.testing.assert_array_equal(first, second)


def test_encode_video_frames_writes_requested_timing(sample_video: Path, tmp_path: Path) -> None:
    try:
        from sam3_matting.media import decode_video_frames, encode_video_frames, probe_video
    except ImportError:
        pytest.fail("frame encoding has not been implemented")

    frames = list(decode_video_frames(sample_video))[:4]
    output = tmp_path / "encoded.mp4"

    result = encode_video_frames(frames, output, fps=Fraction(5, 1))

    assert result == output
    metadata = probe_video(output)
    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.fps == Fraction(5, 1)
    assert metadata.frame_count == 4
    assert metadata.duration_seconds == pytest.approx(0.8, abs=0.05)
    assert metadata.has_audio is False


def test_encode_video_frames_remuxes_source_audio(sample_video: Path, tmp_path: Path) -> None:
    from sam3_matting.media import decode_video_frames, encode_video_frames, probe_video

    source_metadata = probe_video(sample_video)
    frames = decode_video_frames(sample_video)
    output = tmp_path / "with-audio.mp4"

    encode_video_frames(
        frames,
        output,
        fps=source_metadata.fps,
        source_audio_path=sample_video,
    )

    result_metadata = probe_video(output)
    assert result_metadata.frame_count == source_metadata.frame_count
    assert result_metadata.duration_seconds == pytest.approx(source_metadata.duration_seconds, abs=0.05)
    assert result_metadata.has_audio is True
    assert _audio_packet_hash(output) == _audio_packet_hash(sample_video)


def test_encode_video_frames_forwards_configured_remux_tools_and_timeouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import encode_video_frames

    source_audio = tmp_path / "source.mp4"
    source_audio.touch()
    output = tmp_path / "output.mp4"
    captured: dict[str, object] = {}

    def capture_remux(
        video_only,
        source,
        destination,
        ffmpeg_binary,
        *,
        ffprobe_binary,
        ffprobe_timeout_seconds,
        ffmpeg_timeout_seconds,
        publications,
    ):
        assert publications == []
        captured.update(
            {
                "source": source,
                "ffmpeg_binary": ffmpeg_binary,
                "ffprobe_binary": ffprobe_binary,
                "ffprobe_timeout_seconds": ffprobe_timeout_seconds,
                "ffmpeg_timeout_seconds": ffmpeg_timeout_seconds,
            }
        )
        video_only.replace(destination)

    monkeypatch.setattr("sam3_matting.media._remux_source_audio", capture_remux)

    encode_video_frames(
        [np.zeros((32, 48, 3), dtype=np.uint8)],
        output,
        fps=Fraction(24, 1),
        source_audio_path=source_audio,
        ffmpeg_binary="/custom/ffmpeg",
        ffprobe_binary="/custom/ffprobe",
        ffprobe_timeout_seconds=1.25,
        ffmpeg_timeout_seconds=2.5,
    )

    assert output.is_file()
    assert captured == {
        "source": source_audio,
        "ffmpeg_binary": "/custom/ffmpeg",
        "ffprobe_binary": "/custom/ffprobe",
        "ffprobe_timeout_seconds": 1.25,
        "ffmpeg_timeout_seconds": 2.5,
    }


def test_remux_does_not_truncate_video_when_source_audio_is_short(
    sample_video: Path,
    tmp_path: Path,
) -> None:
    from sam3_matting.media import _remux_source_audio, probe_video

    short_audio = tmp_path / "short.m4a"
    _write_audio(short_audio, codec="aac", duration_seconds=0.15)
    output = tmp_path / "full-length.mp4"

    _remux_source_audio(
        sample_video,
        short_audio,
        output,
        ffmpeg_binary=_require_tool("ffmpeg"),
    )

    metadata = probe_video(output)
    assert metadata.frame_count == 10
    assert metadata.duration_seconds == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(
    ("source_codec", "source_suffix", "output_suffix"),
    [
        ("libopus", ".webm", ".mp4"),
        ("libvorbis", ".ogg", ".mov"),
    ],
)
def test_remux_transcodes_incompatible_audio_to_aac(
    sample_video: Path,
    tmp_path: Path,
    source_codec: str,
    source_suffix: str,
    output_suffix: str,
) -> None:
    from sam3_matting.media import _remux_source_audio

    source_audio = tmp_path / f"source{source_suffix}"
    _write_audio(source_audio, codec=source_codec, duration_seconds=0.5)
    output = tmp_path / f"output{output_suffix}"

    _remux_source_audio(
        sample_video,
        source_audio,
        output,
        ffmpeg_binary=_require_tool("ffmpeg"),
    )

    assert _audio_codec_name(output) == "aac"


def test_h264_sink_accepts_incremental_single_pass_writes(sample_video: Path, tmp_path: Path) -> None:
    try:
        from sam3_matting.media import H264Mp4Sink, decode_video_frames, probe_video
    except ImportError:
        pytest.fail("incremental H.264 sink has not been implemented")

    output = tmp_path / "incremental.mp4"
    decoder = decode_video_frames(sample_video)

    with H264Mp4Sink(output, fps=Fraction(5, 1)) as sink:
        for index in range(4):
            sink.write(next(decoder))
            assert sink.frame_count == index + 1
            assert not output.exists()

    metadata = probe_video(output)
    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.fps == Fraction(5, 1)
    assert metadata.frame_count == 4
    assert metadata.duration_seconds == pytest.approx(0.8, abs=0.05)
    assert metadata.has_audio is False


def test_prores_4444_sink_roundtrips_alpha_and_remuxes_audio(sample_video: Path, tmp_path: Path) -> None:
    try:
        from sam3_matting.media import ProRes4444MovSink, probe_video
    except ImportError:
        pytest.fail("incremental transparent ProRes sink has not been implemented")

    width, height = 48, 32
    expected_alpha = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))

    def rgba_frames():
        for index in range(3):
            frame = np.zeros((height, width, 4), dtype=np.uint8)
            frame[..., 0] = 40 + index * 20
            frame[..., 1] = 100
            frame[..., 2] = 180
            frame[..., 3] = expected_alpha
            yield frame

    output = tmp_path / "transparent.mov"
    with ProRes4444MovSink(
        output,
        fps=Fraction(12, 1),
        source_audio_path=sample_video,
    ) as sink:
        for frame in rgba_frames():
            sink.write(frame)

    metadata = probe_video(output)
    assert metadata.width == width
    assert metadata.height == height
    assert metadata.fps == Fraction(12, 1)
    assert metadata.frame_count == 3
    assert metadata.duration_seconds == pytest.approx(0.25, abs=0.05)
    assert metadata.has_audio is True

    with av.open(str(output), mode="r") as container:
        decoded = [frame.to_ndarray(format="rgba") for frame in container.decode(video=0)]
    assert len(decoded) == 3
    assert decoded[0].shape == (height, width, 4)
    np.testing.assert_allclose(decoded[0][..., 3], expected_alpha, atol=1)


def test_sink_failure_preserves_existing_output_and_removes_temporary_files(tmp_path: Path) -> None:
    try:
        from sam3_matting.media import H264Mp4Sink, MediaEncodeError
    except ImportError:
        pytest.fail("atomic incremental sink cleanup has not been implemented")

    output = tmp_path / "existing.mp4"
    original = b"do-not-replace-on-failure"
    output.write_bytes(original)

    with (
        pytest.raises(MediaEncodeError, match="frame shape"),
        H264Mp4Sink(output, fps=Fraction(24, 1)) as sink,
    ):
        sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
        sink.write(np.zeros((30, 48, 3), dtype=np.uint8))

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".existing-*.mp4")) == []


def test_sink_close_cancellation_during_encoder_flush_aborts_all_artifacts(
    tmp_path: Path,
) -> None:
    from sam3_matting.media import H264Mp4Sink

    output = tmp_path / "cancelled-flush.mp4"
    sink = H264Mp4Sink(output, fps=Fraction(24, 1))
    sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
    cancellation = KeyboardInterrupt("cancel during encoder flush")

    class CancellingStream:
        def encode(self):
            raise cancellation

    sink._stream = CancellingStream()

    with pytest.raises(KeyboardInterrupt) as caught:
        sink.close()

    assert caught.value is cancellation
    assert not output.exists()
    assert list(tmp_path.glob(".cancelled-flush-*.mp4")) == []


def test_sink_close_cancellation_during_container_close_aborts_all_artifacts(
    tmp_path: Path,
) -> None:
    from sam3_matting.media import H264Mp4Sink

    output = tmp_path / "cancelled-close.mp4"
    sink = H264Mp4Sink(output, fps=Fraction(24, 1))
    sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
    cancellation = KeyboardInterrupt("cancel during container close")
    wrapped_container = sink._container
    assert wrapped_container is not None

    class CancellingContainer:
        def __init__(self) -> None:
            self.closed = False

        def mux(self, packet) -> None:
            wrapped_container.mux(packet)

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                wrapped_container.close()
            raise cancellation

    sink._container = CancellingContainer()

    with pytest.raises(KeyboardInterrupt) as caught:
        sink.close()

    assert caught.value is cancellation
    assert not output.exists()
    assert list(tmp_path.glob(".cancelled-close-*.mp4")) == []


def test_sink_close_cancellation_during_audio_remux_aborts_all_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import H264Mp4Sink

    output = tmp_path / "cancelled-remux.mp4"
    source_audio = tmp_path / "source.mp4"
    source_audio.touch()
    sink = H264Mp4Sink(
        output,
        fps=Fraction(24, 1),
        source_audio_path=source_audio,
    )
    sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
    cancellation = KeyboardInterrupt("cancel during audio remux")

    def cancel_remux(*args, **kwargs):
        del args, kwargs
        raise cancellation

    monkeypatch.setattr("sam3_matting.media._remux_source_audio", cancel_remux)

    with pytest.raises(KeyboardInterrupt) as caught:
        sink.close()

    assert caught.value is cancellation
    assert not output.exists()
    assert list(tmp_path.glob(".cancelled-remux-*.mp4")) == []


def test_sink_close_cancellation_after_atomic_publish_removes_published_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import H264Mp4Sink

    output = tmp_path / "cancelled-publish.mp4"
    sink = H264Mp4Sink(output, fps=Fraction(24, 1))
    sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
    temporary = sink._temporary
    assert temporary is not None
    real_link = os.link
    cancellation = KeyboardInterrupt("cancel after atomic publish")

    def publish_then_cancel(source: Path, target: Path, *args, **kwargs) -> None:
        real_link(source, target, *args, **kwargs)
        if Path(source) == temporary:
            raise cancellation

    monkeypatch.setattr("sam3_matting.media.os.link", publish_then_cancel)

    with pytest.raises(KeyboardInterrupt) as caught:
        sink.close()

    assert caught.value is cancellation
    assert not output.exists()
    assert list(tmp_path.glob(".cancelled-publish-*.mp4")) == []


def test_sink_post_link_line_cancellation_recovers_ownership_before_temp_cleanup(
    tmp_path: Path,
) -> None:
    from sam3_matting.media import H264Mp4Sink, _ExclusivePublisher

    output = tmp_path / "post-link-cancelled.mp4"
    sink = H264Mp4Sink(output, fps=Fraction(24, 1))
    sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
    temporary = sink._temporary
    assert temporary is not None
    cancellation = KeyboardInterrupt("cancel after os.link returned")
    source_lines, first_line = inspect.getsourcelines(_ExclusivePublisher.publish)
    ownership_line = next(
        first_line + offset
        for offset, source_line in enumerate(source_lines)
        if source_line.strip() == "self.owns_destination = True"
    )
    observed_link_state: list[tuple[bool, bool]] = []

    def interrupt_before_ownership_assignment(frame, event, arg):
        del arg
        if (
            event == "line"
            and frame.f_code is _ExclusivePublisher.publish.__code__
            and frame.f_lineno == ownership_line
        ):
            observed_link_state.append((temporary.exists(), output.exists()))
            sys.settrace(None)
            raise cancellation
        return interrupt_before_ownership_assignment

    sys.settrace(interrupt_before_ownership_assignment)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            sink.close()
    finally:
        sys.settrace(None)

    assert caught.value is cancellation
    assert observed_link_state == [(True, True)]
    assert not temporary.exists()
    assert not output.exists()


def test_sink_refuses_to_overwrite_preexisting_destination(tmp_path: Path) -> None:
    from sam3_matting.media import H264Mp4Sink, MediaEncodeError

    output = tmp_path / "sentinel.mp4"
    sentinel = b"pre-existing output must survive"
    output.write_bytes(sentinel)
    sink = H264Mp4Sink(output, fps=Fraction(24, 1))
    sink.write(np.zeros((32, 48, 3), dtype=np.uint8))

    with pytest.raises(MediaEncodeError, match="already exists"):
        sink.close()

    assert output.read_bytes() == sentinel
    assert list(tmp_path.glob(".sentinel-*.mp4")) == []


def test_sink_context_preserves_primary_cancellation_when_abort_cleanup_fails(
    tmp_path: Path,
) -> None:
    from sam3_matting.media import H264Mp4Sink

    output = tmp_path / "primary-cancellation.mp4"
    sink = H264Mp4Sink(output, fps=Fraction(24, 1))
    primary = KeyboardInterrupt("primary cancellation")
    cleanup_failure = SystemExit("container cleanup failed")

    class FailingContainer:
        def close(self) -> None:
            raise cleanup_failure

    with pytest.raises(KeyboardInterrupt) as caught, sink:
        sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
        sink._container = FailingContainer()
        raise primary

    assert caught.value is primary
    assert not output.exists()
    assert list(tmp_path.glob(".primary-cancellation-*.mp4")) == []


def test_sink_abort_surfaces_first_cleanup_failure_after_later_cleanup_succeeds(
    tmp_path: Path,
) -> None:
    from sam3_matting.media import H264Mp4Sink

    output = tmp_path / "standalone-abort.mp4"
    sink = H264Mp4Sink(output, fps=Fraction(24, 1))
    sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
    cleanup_failure = KeyboardInterrupt("container cleanup failed")

    class FailingContainer:
        def close(self) -> None:
            raise cleanup_failure

    sink._container = FailingContainer()

    with pytest.raises(KeyboardInterrupt) as caught:
        sink.abort()

    assert caught.value is cleanup_failure
    assert not output.exists()
    assert list(tmp_path.glob(".standalone-abort-*.mp4")) == []


def test_sink_abort_never_removes_unowned_destination_after_temporary_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import H264Mp4Sink

    output = tmp_path / "partial-publish.mp4"
    sink = H264Mp4Sink(output, fps=Fraction(24, 1))
    sink.write(np.zeros((32, 48, 3), dtype=np.uint8))
    temporary = sink._temporary
    assert temporary is not None
    sentinel = b"output created outside this sink"
    output.write_bytes(sentinel)
    cleanup_failure = KeyboardInterrupt("temporary cleanup failed")
    real_unlink = Path.unlink

    def fail_temporary_unlink(self: Path, *args, **kwargs) -> None:
        if self == temporary:
            raise cleanup_failure
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)

    with pytest.raises(KeyboardInterrupt) as caught:
        sink.abort(discard_output=True)

    assert caught.value is cleanup_failure
    assert temporary.exists()
    assert output.read_bytes() == sentinel


def test_remux_source_audio_removes_temporary_file_on_unexpected_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import _remux_source_audio

    video_only = tmp_path / "video-only.mp4"
    source_audio = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    video_only.write_bytes(b"video")
    source_audio.write_bytes(b"audio")

    def fail_run(*args, **kwargs):
        del args, kwargs
        raise OSError("ffmpeg could not start")

    monkeypatch.setattr("sam3_matting.media.subprocess.run", fail_run)

    with pytest.raises(OSError, match="could not start"):
        _remux_source_audio(
            video_only,
            source_audio,
            output,
            ffmpeg_binary=_require_tool("true"),
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output-audio-*.mp4")) == []


def test_remux_preserves_primary_cancellation_when_temporary_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import _remux_source_audio

    video_only = tmp_path / "video-only.mp4"
    source_audio = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    video_only.write_bytes(b"video")
    source_audio.write_bytes(b"audio")
    primary = KeyboardInterrupt("remux cancelled")
    cleanup_failure = SystemExit("temporary cleanup failed")

    def cancel_remux(command, **kwargs):
        if "-show_entries" in command:
            stdout = json.dumps({"streams": [{"codec_name": "aac"}]})
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        del kwargs
        raise primary

    def fail_unlink(self: Path, *args, **kwargs) -> None:
        del self, args, kwargs
        raise cleanup_failure

    monkeypatch.setattr("sam3_matting.media.subprocess.run", cancel_remux)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(KeyboardInterrupt) as caught:
        _remux_source_audio(
            video_only,
            source_audio,
            output,
            ffmpeg_binary=_require_tool("true"),
            ffprobe_binary=_require_tool("true"),
        )

    assert caught.value is primary
    assert not output.exists()
    assert len(list(tmp_path.glob(".output-audio-*.mp4"))) == 1


def test_remux_surfaces_cleanup_only_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import _remux_source_audio

    video_only = tmp_path / "video-only.mp4"
    source_audio = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    video_only.write_bytes(b"video")
    source_audio.write_bytes(b"audio")
    cleanup_failure = KeyboardInterrupt("temporary cleanup failed")

    def complete_run(command, **kwargs):
        del kwargs
        stdout = json.dumps({"streams": [{"codec_name": "aac"}]}) if "-show_entries" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    def fail_unlink(self: Path, *args, **kwargs) -> None:
        del self, args, kwargs
        raise cleanup_failure

    monkeypatch.setattr("sam3_matting.media.subprocess.run", complete_run)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(KeyboardInterrupt) as caught:
        _remux_source_audio(
            video_only,
            source_audio,
            output,
            ffmpeg_binary=_require_tool("true"),
            ffprobe_binary=_require_tool("true"),
        )

    assert caught.value is cleanup_failure
    assert output.exists()


def test_remux_source_audio_removes_temporary_file_when_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import _remux_source_audio

    video_only = tmp_path / "video-only.mp4"
    source_audio = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    video_only.write_bytes(b"video")
    source_audio.write_bytes(b"audio")

    def succeed_run(command, **kwargs):
        del kwargs
        stdout = json.dumps({"streams": [{"codec_name": "aac"}]})
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("sam3_matting.media.subprocess.run", succeed_run)

    def fail_link(source: Path, target: Path, *args, **kwargs) -> None:
        del source, target, args, kwargs
        raise OSError("atomic publish failed")

    monkeypatch.setattr("sam3_matting.media.os.link", fail_link)

    with pytest.raises(OSError, match="publish failed"):
        _remux_source_audio(
            video_only,
            source_audio,
            output,
            ffmpeg_binary=_require_tool("true"),
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output-audio-*.mp4")) == []


@pytest.mark.parametrize(
    ("source_codec", "expected_audio_mode"),
    [
        ("aac", "copy"),
        ("opus", "aac"),
        ("vorbis", "aac"),
    ],
)
def test_remux_selects_container_safe_audio_mode_and_finite_default_timeouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_codec: str,
    expected_audio_mode: str,
) -> None:
    from sam3_matting.media import (
        DEFAULT_FFMPEG_TIMEOUT_SECONDS,
        DEFAULT_FFPROBE_TIMEOUT_SECONDS,
        _remux_source_audio,
    )

    video_only = tmp_path / "video-only.mp4"
    source_audio = tmp_path / "source-audio"
    output = tmp_path / "output.mp4"
    video_only.write_bytes(b"video")
    source_audio.write_bytes(b"audio")
    calls: list[tuple[list[str], float]] = []

    def complete_run(command, **kwargs):
        calls.append((command, kwargs["timeout"]))
        stdout = json.dumps({"streams": [{"codec_name": source_codec}]}) if "-show_entries" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("sam3_matting.media.subprocess.run", complete_run)

    _remux_source_audio(
        video_only,
        source_audio,
        output,
        ffmpeg_binary=_require_tool("true"),
        ffprobe_binary=_require_tool("true"),
    )

    assert len(calls) == 2
    probe_command, probe_timeout = calls[0]
    remux_command, remux_timeout = calls[1]
    assert "-show_entries" in probe_command
    assert probe_timeout == DEFAULT_FFPROBE_TIMEOUT_SECONDS
    assert math.isfinite(probe_timeout)
    assert remux_command[remux_command.index("-c:a") + 1] == expected_audio_mode
    assert "-shortest" not in remux_command
    assert remux_timeout == DEFAULT_FFMPEG_TIMEOUT_SECONDS
    assert math.isfinite(remux_timeout)


def test_remux_reports_ffmpeg_timeout_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import MediaEncodeError, _remux_source_audio

    video_only = tmp_path / "video-only.mp4"
    source_audio = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    video_only.write_bytes(b"video")
    source_audio.write_bytes(b"audio")

    def run_with_timeout(command, **kwargs):
        if "-show_entries" in command:
            stdout = json.dumps({"streams": [{"codec_name": "aac"}]})
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("sam3_matting.media.subprocess.run", run_with_timeout)

    with pytest.raises(MediaEncodeError, match=r"ffmpeg timed out after 0\.25 seconds"):
        _remux_source_audio(
            video_only,
            source_audio,
            output,
            ffmpeg_binary=_require_tool("true"),
            ffprobe_binary=_require_tool("true"),
            ffmpeg_timeout_seconds=0.25,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output-audio-*.mp4")) == []


def test_remux_rejects_non_finite_timeout_before_starting_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.media import _remux_source_audio

    video_only = tmp_path / "video-only.mp4"
    source_audio = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    video_only.write_bytes(b"video")
    source_audio.write_bytes(b"audio")

    def fail_run(*args, **kwargs):
        del args, kwargs
        pytest.fail("no subprocess may start with an invalid timeout")

    monkeypatch.setattr("sam3_matting.media.subprocess.run", fail_run)

    with pytest.raises(ValueError, match="ffmpeg timeout must be a positive finite number"):
        _remux_source_audio(
            video_only,
            source_audio,
            output,
            ffmpeg_binary=_require_tool("true"),
            ffprobe_binary=_require_tool("true"),
            ffmpeg_timeout_seconds=math.inf,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output-audio-*.mp4")) == []
