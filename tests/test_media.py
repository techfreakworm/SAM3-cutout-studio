from __future__ import annotations

import shutil
import subprocess
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


def test_validate_input_reports_every_exceeded_limit(sample_video: Path) -> None:
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
        max_file_size_bytes=1,
    )

    with pytest.raises(MediaValidationError) as caught:
        validate_input(sample_video, limits)

    violations = caught.value.violations
    assert len(violations) == 6
    assert any("duration" in item for item in violations)
    assert any("width" in item for item in violations)
    assert any("height" in item for item in violations)
    assert any("frame count" in item for item in violations)
    assert any("frame rate" in item for item in violations)
    assert any("file size" in item for item in violations)


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
