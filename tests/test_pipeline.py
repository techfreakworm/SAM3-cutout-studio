from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest


def _pipeline_api():
    try:
        from sam3_matting.pipeline import (
            NoSubjectDetectedError,
            PipelineProtocolError,
            PipelineResult,
            parse_sam_prompts,
            run_pipeline,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.fail("streaming union pipeline has not been implemented")
    return (
        NoSubjectDetectedError,
        PipelineProtocolError,
        PipelineResult,
        parse_sam_prompts,
        run_pipeline,
    )


def test_pipeline_contract_exposes_max_objects_as_keyword_only() -> None:
    *_, run_pipeline = _pipeline_api()

    parameter = inspect.signature(run_pipeline).parameters["max_objects"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 8


def test_claim_run_directory_cancellation_at_return_removes_only_interrupted_claim(
    tmp_path: Path,
) -> None:
    from sam3_matting.pipeline import _claim_run_directory

    destination = tmp_path / "outputs"
    destination.mkdir()
    cancellation = KeyboardInterrupt("cancel at claimed-directory return")
    source_lines, first_line = inspect.getsourcelines(_claim_run_directory)
    return_line = next(
        first_line + offset
        for offset, source_line in enumerate(source_lines)
        if source_line.strip() == "return run_directory"
    )
    observed_claim: list[bool] = []

    def interrupt_before_return(frame, event, arg):
        del arg
        if (
            event == "line"
            and frame.f_code is _claim_run_directory.__code__
            and frame.f_lineno == return_line
        ):
            observed_claim.append((destination / "source-token").is_dir())
            sys.settrace(None)
            raise cancellation
        return interrupt_before_return

    sys.settrace(interrupt_before_return)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            _claim_run_directory(destination, "source-token")
    finally:
        sys.settrace(None)

    assert caught.value is cancellation
    assert observed_claim == [True]
    assert list(destination.iterdir()) == []

    successful_claim = _claim_run_directory(destination, "source-token")
    assert successful_claim == destination / "source-token"
    assert successful_claim.is_dir()


def test_parse_sam_prompts_preserves_all_clauses_and_normalizes_standalone_mic() -> None:
    *_, parse_sam_prompts, _ = _pipeline_api()

    assert parse_sam_prompts("man,hair,collar mic") == (
        "man",
        "hair",
        "collar microphone",
    )
    assert parse_sam_prompts("MIC, microscope, dynamic") == (
        "microphone",
        "microscope",
        "dynamic",
    )


def test_parse_sam_prompts_trims_and_deduplicates_case_insensitively() -> None:
    *_, parse_sam_prompts, _ = _pipeline_api()

    assert parse_sam_prompts(" man , MAN, hair, collar mic, HAIR ") == (
        "man",
        "hair",
        "collar microphone",
    )


@pytest.mark.parametrize("prompt", ["", "   ", " , ", "man,,hair"])
def test_parse_sam_prompts_rejects_blank_clauses(prompt: str) -> None:
    *_, parse_sam_prompts, _ = _pipeline_api()

    with pytest.raises(ValueError, match="blank"):
        parse_sam_prompts(prompt)


def test_parse_sam_prompts_caps_unique_clauses_at_four() -> None:
    *_, parse_sam_prompts, _ = _pipeline_api()

    with pytest.raises(ValueError, match="at most 4"):
        parse_sam_prompts("one,two,three,four,five")


@dataclass(slots=True)
class FakeTrackedFrame:
    frame_index: int
    union_mask: np.ndarray


class FakeBackend:
    def __init__(
        self,
        streams: dict[str, list[FakeTrackedFrame]],
        events: list[object],
        *,
        mask_store_dir: Path | None = None,
    ) -> None:
        self.streams = streams
        self.events = events
        self.mask_store_dir = mask_store_dir
        self.calls: list[dict[str, object]] = []
        self.mask_store_sizes: list[int] = []

    def track(
        self,
        video_path: str,
        *,
        prompt: str,
        detection_threshold: float,
        detect_interval: int,
        max_objects: int,
    ):
        self.calls.append(
            {
                "video_path": video_path,
                "prompt": prompt,
                "detection_threshold": detection_threshold,
                "detect_interval": detect_interval,
                "max_objects": max_objects,
            }
        )
        self.events.append(("track-start", prompt))
        if self.mask_store_dir is not None:
            stores = list(self.mask_store_dir.glob(".sam3-mask-*.bin"))
            assert len(stores) == 1
            self.mask_store_sizes.append(stores[0].stat().st_size)
        for frame in self.streams[prompt]:
            self.events.append(("track-yield", prompt, frame.frame_index))
            yield frame


class ManualCloseTrackingStream:
    """Iterator whose cleanup runs only when the consumer explicitly closes it."""

    def __init__(
        self,
        frames: list[FakeTrackedFrame],
        events: list[object],
        prompt: str,
        *,
        close_failure: BaseException | None = None,
    ) -> None:
        self._frames = iter(frames)
        self.events = events
        self.prompt = prompt
        self.close_failure = close_failure
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> FakeTrackedFrame:
        return next(self._frames)

    def close(self) -> None:
        if not self.closed:
            self.events.append(("track-close", self.prompt))
            self.closed = True
            if self.close_failure is not None:
                raise self.close_failure


class RetainingBackend:
    """Keep manual-close streams alive so refcounting cannot hide missing cleanup."""

    def __init__(
        self,
        frames: list[FakeTrackedFrame],
        events: list[object],
        *,
        close_failure: BaseException | None = None,
    ) -> None:
        self.frames = frames
        self.events = events
        self.close_failure = close_failure
        self.streams: list[ManualCloseTrackingStream] = []

    def track(
        self,
        video_path: str,
        *,
        prompt: str,
        detection_threshold: float,
        detect_interval: int,
        max_objects: int,
    ):
        del video_path, detection_threshold, detect_interval, max_objects
        stream = ManualCloseTrackingStream(
            self.frames,
            self.events,
            prompt,
            close_failure=self.close_failure,
        )
        self.streams.append(stream)
        self.events.append(("track-start", prompt))
        return stream


class FakeRefiner:
    def __init__(self, alphas: list[np.ndarray], events: list[object]) -> None:
        self.alphas = alphas
        self.events = events
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.events.append(("refine", len(self.calls)))
        self.calls.append((np.array(image, copy=True), np.array(mask, copy=True)))
        return self.alphas[len(self.calls) - 1]


class InterruptingRefiner:
    def __init__(self, failure: BaseException, events: list[object]) -> None:
        self.failure = failure
        self.events = events

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del image, mask
        self.events.append(("refine-interrupt", 0))
        raise self.failure


class CloseFailingDecoder:
    def __init__(
        self,
        frames: list[np.ndarray],
        events: list[object],
        close_failure: BaseException,
    ) -> None:
        self._frames = iter(frames)
        self.events = events
        self.close_failure = close_failure
        self.closed = False

    def __call__(self, path: str | Path):
        del path
        return self

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = next(self._frames)
        self.events.append(("decode-yield", 0))
        return frame

    def close(self) -> None:
        self.closed = True
        self.events.append(("decode-close", 0))
        raise self.close_failure


class FakeSink:
    def __init__(
        self,
        output_path: Path,
        *,
        label: str,
        events: list[object],
        options: dict[str, object],
    ) -> None:
        self.output_path = output_path
        self.label = label
        self.events = events
        self.options = options
        self.frames: list[np.ndarray] = []
        self.closed = False
        self.aborted = False

    def write(self, frame: np.ndarray) -> None:
        self.events.append((f"{self.label}-write", len(self.frames)))
        self.frames.append(np.array(frame, copy=True))

    def close(self) -> Path:
        self.events.append((f"{self.label}-close", len(self.frames)))
        self.closed = True
        return self.output_path

    def abort(self) -> None:
        self.events.append((f"{self.label}-abort", len(self.frames)))
        self.aborted = True


class FakeSinkFactory:
    def __init__(self, label: str, events: list[object]) -> None:
        self.label = label
        self.events = events
        self.instances: list[FakeSink] = []

    def __call__(self, output_path: Path, **options: object) -> FakeSink:
        sink = FakeSink(
            output_path,
            label=self.label,
            events=self.events,
            options=options,
        )
        self.instances.append(sink)
        self.events.append((f"{self.label}-create", output_path.suffix))
        return sink


class PublishingFakeSink(FakeSink):
    def __init__(self, *args, fail_on_close: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_on_close = fail_on_close

    def close(self) -> Path:
        self.events.append((f"{self.label}-close", len(self.frames)))
        self.output_path.write_bytes(b"published-before-close-returned")
        self.closed = True
        if self.fail_on_close:
            raise RuntimeError(f"{self.label} close failed")
        return self.output_path


class PublishingSinkFactory:
    def __init__(
        self,
        label: str,
        events: list[object],
        *,
        fail_on_close: bool,
    ) -> None:
        self.label = label
        self.events = events
        self.fail_on_close = fail_on_close
        self.instances: list[PublishingFakeSink] = []

    def __call__(self, output_path: Path, **options: object) -> PublishingFakeSink:
        sink = PublishingFakeSink(
            output_path,
            label=self.label,
            events=self.events,
            options=options,
            fail_on_close=self.fail_on_close,
        )
        self.instances.append(sink)
        self.events.append((f"{self.label}-create", output_path.suffix))
        return sink


class ArtifactSink(FakeSink):
    def __init__(
        self,
        *args,
        failure_stage: str | None,
        failure: BaseException | None,
        abort_failure: BaseException | None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.failure_stage = failure_stage
        self.failure = failure
        self.abort_failure = abort_failure
        self.temporary_path = self.output_path.with_name(f".{self.output_path.name}.partial")
        self.output_path.write_bytes(b"output-artifact")
        self.temporary_path.write_bytes(b"temporary-artifact")

    def _raise_if_requested(self, stage: str) -> None:
        if self.failure_stage == stage:
            assert self.failure is not None
            raise self.failure

    def write(self, frame: np.ndarray) -> None:
        self.events.append((f"{self.label}-write", len(self.frames)))
        self._raise_if_requested(f"{self.label}-write")
        self.frames.append(np.array(frame, copy=True))

    def close(self) -> Path:
        self.events.append((f"{self.label}-close", len(self.frames)))
        self._raise_if_requested(f"{self.label}-close")
        self.closed = True
        return self.output_path

    def abort(self) -> None:
        self.events.append((f"{self.label}-abort", len(self.frames)))
        self.aborted = True
        self.temporary_path.unlink(missing_ok=True)
        if self.abort_failure is not None:
            raise self.abort_failure


class ArtifactSinkFactory:
    def __init__(
        self,
        label: str,
        events: list[object],
        *,
        failure_stage: str | None = None,
        failure: BaseException | None = None,
        abort_failure: BaseException | None = None,
    ) -> None:
        self.label = label
        self.events = events
        self.failure_stage = failure_stage
        self.failure = failure
        self.abort_failure = abort_failure
        self.instances: list[ArtifactSink] = []

    def __call__(self, output_path: Path, **options: object) -> ArtifactSink:
        sink = ArtifactSink(
            output_path,
            label=self.label,
            events=self.events,
            options=options,
            failure_stage=self.failure_stage,
            failure=self.failure,
            abort_failure=self.abort_failure,
        )
        self.instances.append(sink)
        self.events.append((f"{self.label}-create", output_path.suffix))
        return sink


class FakeMappedFile:
    def __init__(self, close_failure: BaseException | None = None) -> None:
        self.close_failure = close_failure
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


class FailingMaskStore:
    def __init__(
        self,
        *,
        init_failure: BaseException | None = None,
        flush_failures: dict[int, BaseException] | None = None,
        close_failure: BaseException | None = None,
    ) -> None:
        self.init_failure = init_failure
        self.flush_failures = flush_failures or {}
        self.flush_calls = 0
        self._mmap = FakeMappedFile(close_failure)

    def __setitem__(self, key, value) -> None:
        del key, value
        if self.init_failure is not None:
            raise self.init_failure

    def flush(self) -> None:
        self.flush_calls += 1
        failure = self.flush_failures.get(self.flush_calls)
        if failure is not None:
            raise failure


def _metadata(
    *,
    frame_count: int = 3,
    width: int = 3,
    height: int = 2,
    has_audio: bool = True,
):
    from sam3_matting.media import VideoMetadata

    return VideoMetadata(
        duration_seconds=frame_count / 2,
        fps=Fraction(2, 1),
        width=width,
        height=height,
        frame_count=frame_count,
        has_audio=has_audio,
    )


def _validator(metadata, events: list[object]):
    def validate(path, limits):
        events.append(("validate", Path(path), limits))
        return metadata

    return validate


def _decoder(frames: list[np.ndarray], events: list[object]):
    def decode(path):
        for index, frame in enumerate(frames):
            events.append(("decode-yield", index))
            yield frame

    return decode


def _clock(*values: float):
    iterator = iter(values)
    return lambda: next(iterator)


def _factories(events: list[object]):
    return (
        FakeSinkFactory("preview", events),
        FakeSinkFactory("master", events),
        FakeSinkFactory("matte", events),
    )


def _collision_sentinels(output_dir: Path, output_stem: str) -> dict[Path, bytes]:
    output_dir.mkdir(parents=True)
    colliding_run_directory = output_dir / output_stem
    colliding_run_directory.mkdir()
    sentinels: dict[Path, bytes] = {}
    for parent in (output_dir, colliding_run_directory):
        for label, suffix in (
            ("preview", ".mp4"),
            ("master", ".mov"),
            ("matte", ".mp4"),
        ):
            path = parent / f"{output_stem}-{label}{suffix}"
            payload = f"sentinel:{parent.name}:{label}".encode()
            path.write_bytes(payload)
            sentinels[path] = payload
    return sentinels


def _assert_sentinels_unchanged(sentinels: dict[Path, bytes]) -> None:
    for path, payload in sentinels.items():
        assert path.read_bytes() == payload


def _mask(*coordinates: tuple[int, int], shape: tuple[int, int] = (2, 3)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.bool_)
    for row, column in coordinates:
        mask[row, column] = True
    return mask


def test_pipeline_unions_clauses_in_packed_store_and_streams_exact_outputs(tmp_path: Path) -> None:
    _, _, PipelineResult, _, run_pipeline = _pipeline_api()
    from sam3_matting.media import InputLimits

    events: list[object] = []
    source = tmp_path / "source.mp4"
    metadata = _metadata()
    backend = FakeBackend(
        {
            "man": [
                FakeTrackedFrame(0, _mask((0, 0))),
                FakeTrackedFrame(1, _mask((0, 0))),
                FakeTrackedFrame(2, _mask()),
            ],
            "hair": [
                FakeTrackedFrame(0, _mask((0, 1))),
                FakeTrackedFrame(1, _mask()),
                FakeTrackedFrame(2, _mask()),
            ],
            "collar microphone": [
                FakeTrackedFrame(0, _mask((1, 2))),
                FakeTrackedFrame(1, _mask()),
                FakeTrackedFrame(2, _mask()),
            ],
        },
        events,
        mask_store_dir=tmp_path,
    )
    rgb0 = np.array(
        [
            [[100, 50, 20], [20, 40, 60], [9, 18, 27]],
            [[255, 128, 64], [10, 20, 30], [80, 40, 20]],
        ],
        dtype=np.uint8,
    )
    rgb1 = np.full((2, 3, 3), 50, dtype=np.uint8)
    rgb2 = np.full((2, 3, 3), 77, dtype=np.uint8)
    alpha0 = np.array([[1.0, 0.5, 0.0], [0.25, 0.0, 0.75]], dtype=np.float32)
    alpha1 = np.ones((2, 3), dtype=np.float32)
    refiner = FakeRefiner([alpha0, alpha1], events)
    preview_factory, master_factory, matte_factory = _factories(events)
    progress: list[tuple[str, int, int]] = []
    limits = InputLimits(max_frames=10)

    result = run_pipeline(
        source,
        prompt="man,hair,collar mic",
        detection_threshold=0.6,
        detect_interval=3,
        max_objects=3,
        backend=backend,
        refiner=refiner,
        output_dir=tmp_path,
        limits=limits,
        progress_callback=lambda phase, completed, total: progress.append((phase, completed, total)),
        validate_fn=_validator(metadata, events),
        decode_fn=_decoder([rgb0, rgb1, rgb2], events),
        preview_sink_factory=preview_factory,
        master_sink_factory=master_factory,
        matte_sink_factory=matte_factory,
        mask_store_dir=tmp_path,
        clock=_clock(10.0, 12.5),
    )

    assert isinstance(result, PipelineResult)
    assert result.processed_frame_count == 3
    assert result.effective_sam_prompt == ("man", "hair", "collar microphone")
    assert result.elapsed_seconds == pytest.approx(2.5)
    assert [call["prompt"] for call in backend.calls] == [
        "man",
        "hair",
        "collar microphone",
    ]
    assert all(call["detection_threshold"] == 0.6 for call in backend.calls)
    assert all(call["detect_interval"] == 3 for call in backend.calls)
    assert all(call["max_objects"] == 3 for call in backend.calls)
    assert events[0][0] == "validate"
    assert backend.mask_store_sizes == [3, 3, 3]
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []
    assert progress[0] == ("tracking", 1, 9)
    assert progress[8] == ("tracking", 9, 9)
    assert progress[9:] == [
        ("matting", 1, 3),
        ("matting", 2, 3),
        ("matting", 3, 3),
    ]
    assert events.index(("track-yield", "collar microphone", 2)) < events.index(("decode-yield", 0))
    assert events.index(("preview-write", 0)) < events.index(("decode-yield", 1))

    expected_union0 = _mask((0, 0), (0, 1), (1, 2))
    expected_union1 = _mask((0, 0))
    assert len(refiner.calls) == 2
    np.testing.assert_array_equal(refiner.calls[0][1], expected_union0)
    np.testing.assert_array_equal(refiner.calls[1][1], expected_union1)
    assert refiner.calls[0][1][1, 2]

    preview = preview_factory.instances[0]
    master = master_factory.instances[0]
    matte = matte_factory.instances[0]
    assert result.preview_path == preview.output_path
    assert result.master_path == master.output_path
    assert result.matte_path == matte.output_path
    assert preview.options == {"fps": Fraction(2, 1), "source_audio_path": source}
    assert master.options == {"fps": Fraction(2, 1), "source_audio_path": source}
    assert matte.options == {"fps": Fraction(2, 1), "source_audio_path": None}
    assert all(sink.closed and not sink.aborted for sink in (preview, master, matte))

    expected_alpha_u8 = np.array([[255, 128, 0], [64, 0, 191]], dtype=np.uint8)
    expected_preview0 = np.array(
        [
            [[100, 50, 20], [10, 20, 30], [0, 0, 0]],
            [[64, 32, 16], [0, 0, 0], [60, 30, 15]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(preview.frames[0], expected_preview0)
    np.testing.assert_array_equal(preview.frames[1], rgb1)
    np.testing.assert_array_equal(preview.frames[2], np.zeros_like(rgb2))
    np.testing.assert_array_equal(master.frames[0], np.dstack((rgb0, expected_alpha_u8)))
    np.testing.assert_array_equal(
        master.frames[2],
        np.dstack((rgb2, np.zeros((2, 3), dtype=np.uint8))),
    )
    np.testing.assert_array_equal(
        matte.frames[0],
        np.repeat(expected_alpha_u8[..., None], 3, axis=2),
    )
    np.testing.assert_array_equal(matte.frames[2], np.zeros_like(rgb2))


def test_pipeline_validates_before_backend_or_sink_creation(tmp_path: Path) -> None:
    *_, run_pipeline = _pipeline_api()
    from sam3_matting.media import InputLimits

    events: list[object] = []
    backend = FakeBackend({"man": []}, events)
    refiner = FakeRefiner([], events)
    factories = _factories(events)

    def reject(path, limits):
        events.append("validate")
        raise ValueError("input is too long")

    with pytest.raises(ValueError, match="too long"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=refiner,
            output_dir=tmp_path,
            limits=InputLimits(),
            validate_fn=reject,
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert events == ["validate"]
    assert backend.calls == []
    assert all(factory.instances == [] for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


@pytest.mark.parametrize("max_objects", [0, 9])
def test_pipeline_rejects_invalid_max_objects_before_input_validation(
    tmp_path: Path,
    max_objects: int,
) -> None:
    *_, run_pipeline = _pipeline_api()

    events: list[object] = []
    backend = FakeBackend({"man": []}, events)
    factories = _factories(events)
    mask_store_dir = tmp_path / "mask-store"

    with pytest.raises(ValueError, match="between 1 and 8"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            max_objects=max_objects,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=mask_store_dir,
        )

    assert events == []
    assert backend.calls == []
    assert all(factory.instances == [] for factory in factories)
    assert not mask_store_dir.exists()


@pytest.mark.parametrize(
    ("option", "invalid_value", "message"),
    [
        ("detection_threshold", False, "detection_threshold"),
        ("detection_threshold", True, "detection_threshold"),
        ("detection_threshold", "0.5", "detection_threshold"),
        ("detection_threshold", 0.5 + 0j, "detection_threshold"),
        ("detection_threshold", -0.01, "detection_threshold"),
        ("detection_threshold", 1.01, "detection_threshold"),
        ("detection_threshold", float("-inf"), "detection_threshold"),
        ("detection_threshold", float("inf"), "detection_threshold"),
        ("detection_threshold", float("nan"), "detection_threshold"),
        ("detect_interval", True, "detect_interval"),
        ("detect_interval", 1.0, "detect_interval"),
        ("detect_interval", "1", "detect_interval"),
        ("detect_interval", 0, "detect_interval"),
        ("detect_interval", float("inf"), "detect_interval"),
        ("detect_interval", float("nan"), "detect_interval"),
        ("max_objects", True, "max_objects"),
        ("max_objects", 1.0, "max_objects"),
        ("max_objects", "1", "max_objects"),
        ("max_objects", float("inf"), "max_objects"),
        ("max_objects", float("nan"), "max_objects"),
    ],
)
def test_pipeline_rejects_invalid_tracking_controls_before_input_validation(
    tmp_path: Path,
    option: str,
    invalid_value: object,
    message: str,
) -> None:
    *_, run_pipeline = _pipeline_api()

    events: list[object] = []
    backend = FakeBackend({"man": []}, events)
    factories = _factories(events)
    mask_store_dir = tmp_path / "mask-store"
    tracking_options = {
        "detection_threshold": 0.5,
        "detect_interval": 1,
        "max_objects": 8,
    }
    tracking_options[option] = invalid_value

    with pytest.raises(ValueError, match=message):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            **tracking_options,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=mask_store_dir,
        )

    assert events == []
    assert backend.calls == []
    assert all(factory.instances == [] for factory in factories)
    assert not mask_store_dir.exists()


@pytest.mark.parametrize(
    ("detection_threshold", "max_objects"),
    [(0.0, 1), (1.0, 8)],
)
def test_pipeline_accepts_tracking_control_boundaries_before_input_validation(
    tmp_path: Path,
    detection_threshold: float,
    max_objects: int,
) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []

    def stop_at_validation(path, limits):
        del path, limits
        events.append("validate")
        raise RuntimeError("validation reached")

    with pytest.raises(RuntimeError, match="validation reached"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=detection_threshold,
            detect_interval=1,
            max_objects=max_objects,
            backend=FakeBackend({"man": []}, events),
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=stop_at_validation,
            mask_store_dir=tmp_path / "mask-store",
        )

    assert events == ["validate"]
    assert not (tmp_path / "mask-store").exists()


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (
            [FakeTrackedFrame(0, _mask()), FakeTrackedFrame(2, _mask())],
            "expected frame_index 1",
        ),
        ([FakeTrackedFrame(0, _mask())], "produced 1 frames; expected 2"),
        (
            [
                FakeTrackedFrame(0, _mask(shape=(1, 3))),
                FakeTrackedFrame(1, _mask(shape=(1, 3))),
            ],
            "mask shape",
        ),
    ],
)
def test_pipeline_rejects_tracking_alignment_protocol_errors_and_cleans_store(
    tmp_path: Path,
    frames: list[FakeTrackedFrame],
    message: str,
) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend({"man": frames}, events)
    factories = _factories(events)

    with pytest.raises(PipelineProtocolError, match=message):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=2), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert all(factory.instances == [] for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_rejects_tracking_stream_overrun_and_cleans_store(tmp_path: Path) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend(
        {
            "man": [
                FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2))),
                FakeTrackedFrame(1, _mask((0, 0), shape=(2, 2))),
                FakeTrackedFrame(2, _mask((0, 0), shape=(2, 2))),
            ]
        },
        events,
    )
    factories = _factories(events)

    with pytest.raises(PipelineProtocolError, match="produced more than 2 frames"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=2, width=2, height=2), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert all(factory.instances == [] for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_closes_retained_tracking_stream_after_protocol_failure(tmp_path: Path) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = RetainingBackend(
        [
            FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2))),
            FakeTrackedFrame(2, _mask((0, 0), shape=(2, 2))),
        ],
        events,
    )
    factories = _factories(events)

    with pytest.raises(PipelineProtocolError, match="expected frame_index 1"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=2, width=2, height=2), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert events.count(("track-close", "man")) == 1
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_closes_retained_tracking_stream_when_cancelled(tmp_path: Path) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = RetainingBackend(
        [
            FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2))),
            FakeTrackedFrame(1, _mask((0, 0), shape=(2, 2))),
        ],
        events,
    )
    factories = _factories(events)

    def cancel_tracking(phase: str, completed: int, total: int) -> None:
        del phase, completed, total
        raise KeyboardInterrupt("tracking cancelled")

    with pytest.raises(KeyboardInterrupt, match="tracking cancelled"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            progress_callback=cancel_tracking,
            validate_fn=_validator(_metadata(frame_count=2, width=2, height=2), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert events.count(("track-close", "man")) == 1
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_closes_retained_tracking_stream_after_success(tmp_path: Path) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = RetainingBackend(
        [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))],
        events,
    )
    factories = _factories(events)

    run_pipeline(
        tmp_path / "source.mp4",
        prompt="man",
        detection_threshold=0.5,
        detect_interval=1,
        backend=backend,
        refiner=FakeRefiner([np.ones((2, 2), dtype=np.float32)], events),
        output_dir=tmp_path,
        validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
        decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
        preview_sink_factory=factories[0],
        master_sink_factory=factories[1],
        matte_sink_factory=factories[2],
        mask_store_dir=tmp_path,
    )

    assert backend.streams[0].closed is True
    assert events.count(("track-close", "man")) == 1
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.array([[0, 2], [1, 0]], dtype=np.int8), "only 0 and 1"),
        (np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32), "finite"),
        (np.array([["0", "1"], ["1", "0"]]), "boolean or numeric"),
        (np.array([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]]), "real"),
    ],
)
def test_pipeline_rejects_non_binary_tracking_masks(
    tmp_path: Path,
    mask: np.ndarray,
    message: str,
) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend({"man": [FakeTrackedFrame(0, mask)]}, events)
    factories = _factories(events)

    with pytest.raises(PipelineProtocolError, match=message):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert all(factory.instances == [] for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_accepts_numeric_binary_tracking_masks(tmp_path: Path) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    numeric_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    backend = FakeBackend({"man": [FakeTrackedFrame(0, numeric_mask)]}, events)
    refiner = FakeRefiner([np.ones((2, 2), dtype=np.float32)], events)
    factories = _factories(events)

    run_pipeline(
        tmp_path / "source.mp4",
        prompt="man",
        detection_threshold=0.5,
        detect_interval=1,
        backend=backend,
        refiner=refiner,
        output_dir=tmp_path,
        validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
        decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
        preview_sink_factory=factories[0],
        master_sink_factory=factories[1],
        matte_sink_factory=factories[2],
        mask_store_dir=tmp_path,
    )

    assert refiner.calls[0][1].dtype == np.bool_
    np.testing.assert_array_equal(refiner.calls[0][1], numeric_mask.astype(np.bool_))
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


@pytest.mark.parametrize(
    ("alpha", "message"),
    [
        (np.ones((1, 2), dtype=np.float32), "alpha shape"),
        (np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32), "finite"),
        (np.array([[1.1, 0.0], [0.0, 1.0]], dtype=np.float32), r"\[0, 1\]"),
    ],
)
def test_pipeline_rejects_invalid_alpha_and_aborts_every_sink(
    tmp_path: Path,
    alpha: np.ndarray,
    message: str,
) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend({"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]}, events)
    refiner = FakeRefiner([alpha], events)
    factories = _factories(events)

    with pytest.raises(PipelineProtocolError, match=message):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=refiner,
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([np.full((2, 2, 3), 100, dtype=np.uint8)], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    sinks = [factory.instances[0] for factory in factories]
    assert all(sink.aborted and not sink.closed for sink in sinks)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_rejects_decode_count_mismatch_and_aborts_sinks(tmp_path: Path) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend(
        {
            "man": [
                FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2))),
                FakeTrackedFrame(1, _mask((0, 0), shape=(2, 2))),
            ]
        },
        events,
    )
    factories = _factories(events)

    with pytest.raises(PipelineProtocolError, match="decoded 1 frames; expected 2"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([np.ones((2, 2), dtype=np.float32)], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=2, width=2, height=2), events),
            decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert all(factory.instances[0].aborted for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_rejects_decoder_overrun_and_aborts_sinks(tmp_path: Path) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend(
        {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
        events,
    )
    factories = _factories(events)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(PipelineProtocolError, match="decoded more than the expected 1 frames"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([np.ones((2, 2), dtype=np.float32)], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([rgb, rgb], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert all(factory.instances[0].aborted for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


@pytest.mark.parametrize(
    ("decoded_frame", "message"),
    [
        pytest.param(np.zeros((1, 2, 3), dtype=np.uint8), "shape", id="shape"),
        pytest.param(np.zeros((2, 2, 3), dtype=np.float32), "uint8 pixels", id="dtype"),
    ],
)
def test_pipeline_rejects_invalid_decoded_rgb_frames_and_aborts_sinks(
    tmp_path: Path,
    decoded_frame: np.ndarray,
    message: str,
) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend(
        {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
        events,
    )
    refiner = FakeRefiner([], events)
    factories = _factories(events)

    with pytest.raises(PipelineProtocolError, match=message):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=refiner,
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([decoded_frame], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert refiner.calls == []
    assert all(factory.instances[0].aborted for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_fails_without_any_subject_before_publishing_outputs(tmp_path: Path) -> None:
    NoSubjectDetectedError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    empty_frames = [FakeTrackedFrame(0, _mask()), FakeTrackedFrame(1, _mask())]
    backend = FakeBackend({"man": empty_frames, "hair": empty_frames}, events)
    refiner = FakeRefiner([], events)
    factories = _factories(events)

    with pytest.raises(NoSubjectDetectedError, match="No subject"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man,hair",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=refiner,
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=2), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert [call["prompt"] for call in backend.calls] == ["man", "hair"]
    assert refiner.calls == []
    assert all(factory.instances == [] for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_aborts_already_created_sinks_if_later_factory_fails(tmp_path: Path) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend(
        {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
        events,
    )
    preview_factory = FakeSinkFactory("preview", events)
    matte_factory = FakeSinkFactory("matte", events)

    def fail_master_factory(output_path: Path, **options: object):
        raise RuntimeError("master sink construction failed")

    with pytest.raises(RuntimeError, match="construction failed"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=preview_factory,
            master_sink_factory=fail_master_factory,
            matte_sink_factory=matte_factory,
            mask_store_dir=tmp_path,
        )

    assert preview_factory.instances[0].aborted is True
    assert matte_factory.instances == []
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_factory_failure_never_removes_fixed_token_collision_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    token = "feedfacecafe"
    output_stem = f"source-{token}"
    output_dir = tmp_path / "outputs"
    sentinels = _collision_sentinels(output_dir, output_stem)
    preview_factory = FakeSinkFactory("preview", events)
    matte_factory = FakeSinkFactory("matte", events)

    class FixedUuid:
        hex = token

    monkeypatch.setattr("sam3_matting.pipeline.uuid.uuid4", lambda: FixedUuid())

    def fail_master_factory(output_path: Path, **options: object):
        del output_path, options
        raise RuntimeError("master sink construction failed")

    with pytest.raises(RuntimeError, match="construction failed"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=FakeBackend(
                {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
                events,
            ),
            refiner=FakeRefiner([], events),
            output_dir=output_dir,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([], events),
            preview_sink_factory=preview_factory,
            master_sink_factory=fail_master_factory,
            matte_sink_factory=matte_factory,
            mask_store_dir=tmp_path,
        )

    _assert_sentinels_unchanged(sentinels)
    assert preview_factory.instances[0].aborted is True
    assert matte_factory.instances == []
    assert set(output_dir.rglob("*")) == set(sentinels) | {output_dir / output_stem}


def test_pipeline_close_cancellation_never_removes_fixed_token_collision_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    token = "feedfacecafe"
    output_stem = f"source-{token}"
    output_dir = tmp_path / "outputs"
    sentinels = _collision_sentinels(output_dir, output_stem)
    cancellation = KeyboardInterrupt("master close cancelled")
    factories = tuple(
        ArtifactSinkFactory(
            label,
            events,
            failure_stage="master-close" if label == "master" else None,
            failure=cancellation if label == "master" else None,
        )
        for label in ("preview", "master", "matte")
    )

    class FixedUuid:
        hex = token

    monkeypatch.setattr("sam3_matting.pipeline.uuid.uuid4", lambda: FixedUuid())

    with pytest.raises(KeyboardInterrupt) as caught:
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=FakeBackend(
                {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
                events,
            ),
            refiner=FakeRefiner([np.ones((2, 2), dtype=np.float32)], events),
            output_dir=output_dir,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert caught.value is cancellation
    _assert_sentinels_unchanged(sentinels)
    assert all(factory.instances[0].aborted for factory in factories)
    assert set(output_dir.rglob("*")) == set(sentinels) | {output_dir / output_stem}


@pytest.mark.parametrize("failing_label", ["preview", "master", "matte"])
def test_pipeline_close_failure_removes_every_published_output_and_mask_store(
    tmp_path: Path,
    failing_label: str,
) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = FakeBackend(
        {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
        events,
    )
    output_dir = tmp_path / "outputs"
    factories = tuple(
        PublishingSinkFactory(
            label,
            events,
            fail_on_close=label == failing_label,
        )
        for label in ("preview", "master", "matte")
    )

    with pytest.raises(RuntimeError, match=rf"{failing_label} close failed"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([np.ones((2, 2), dtype=np.float32)], events),
            output_dir=output_dir,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    sinks = [factory.instances[0] for factory in factories]
    assert all(sink.aborted for sink in sinks)
    assert all(not sink.output_path.exists() for sink in sinks)
    assert list(output_dir.iterdir()) == []
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


@pytest.mark.parametrize(
    "failure_stage",
    ["refine", "progress", "master-write", "master-close"],
)
def test_pipeline_base_exception_aborts_every_sink_and_removes_artifacts(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    cancellation = KeyboardInterrupt(f"cancelled during {failure_stage}")
    sink_failure_stage = failure_stage if "-" in failure_stage else None
    factories = tuple(
        ArtifactSinkFactory(
            label,
            events,
            failure_stage=sink_failure_stage,
            failure=cancellation,
        )
        for label in ("preview", "master", "matte")
    )
    refiner = (
        InterruptingRefiner(cancellation, events)
        if failure_stage == "refine"
        else FakeRefiner([np.ones((2, 2), dtype=np.float32)], events)
    )

    def interrupt_progress(phase: str, completed: int, total: int) -> None:
        del completed, total
        if failure_stage == "progress" and phase == "matting":
            raise cancellation

    with pytest.raises(KeyboardInterrupt) as caught:
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=FakeBackend(
                {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
                events,
            ),
            refiner=refiner,
            output_dir=tmp_path / "outputs",
            progress_callback=interrupt_progress,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert caught.value is cancellation
    sinks = [factory.instances[0] for factory in factories]
    assert all(sink.aborted for sink in sinks)
    assert all(not sink.output_path.exists() for sink in sinks)
    assert all(not sink.temporary_path.exists() for sink in sinks)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_cleanup_failures_do_not_mask_cancellation_or_stop_other_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    cancellation = KeyboardInterrupt("matting cancelled")
    factories = (
        ArtifactSinkFactory(
            "preview",
            events,
            abort_failure=SystemExit("preview abort failed"),
        ),
        ArtifactSinkFactory("master", events),
        ArtifactSinkFactory("matte", events),
    )
    original_unlink = Path.unlink
    unlink_attempts: list[Path] = []

    def fail_preview_output_unlink(path: Path, missing_ok: bool = False) -> None:
        unlink_attempts.append(path)
        if path.suffix == ".mp4" and "-preview" in path.stem and not path.name.startswith("."):
            raise OSError("preview output unlink failed")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_preview_output_unlink)

    def interrupt_matting(phase: str, completed: int, total: int) -> None:
        del completed, total
        if phase == "matting":
            raise cancellation

    with pytest.raises(KeyboardInterrupt) as caught:
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=FakeBackend(
                {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
                events,
            ),
            refiner=FakeRefiner([np.ones((2, 2), dtype=np.float32)], events),
            output_dir=tmp_path / "outputs",
            progress_callback=interrupt_matting,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert caught.value is cancellation
    sinks = [factory.instances[0] for factory in factories]
    assert all(sink.aborted for sink in sinks)
    assert all(sink.output_path in unlink_attempts for sink in sinks)
    assert sinks[0].output_path.exists()
    assert all(not sink.output_path.exists() for sink in sinks[1:])
    assert all(not sink.temporary_path.exists() for sink in sinks)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []
    original_unlink(sinks[0].output_path, missing_ok=True)


def test_tracking_close_failure_does_not_mask_primary_protocol_error(tmp_path: Path) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    backend = RetainingBackend(
        [
            FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2))),
            FakeTrackedFrame(2, _mask((0, 0), shape=(2, 2))),
        ],
        events,
        close_failure=SystemExit("tracking close failed"),
    )

    with pytest.raises(PipelineProtocolError, match="expected frame_index 1"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=backend,
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=2, width=2, height=2), events),
            mask_store_dir=tmp_path,
        )

    assert backend.streams[0].closed is True
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_decoder_close_failure_does_not_mask_primary_protocol_error(tmp_path: Path) -> None:
    _, PipelineProtocolError, *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    decoder = CloseFailingDecoder(
        [np.zeros((1, 2, 3), dtype=np.uint8)],
        events,
        SystemExit("decoder close failed"),
    )
    factories = _factories(events)

    with pytest.raises(PipelineProtocolError, match="decoded RGB frame 0 shape"):
        run_pipeline(
            tmp_path / "source.mp4",
            prompt="man",
            detection_threshold=0.5,
            detect_interval=1,
            backend=FakeBackend(
                {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
                events,
            ),
            refiner=FakeRefiner([], events),
            output_dir=tmp_path,
            validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
            decode_fn=decoder,
            preview_sink_factory=factories[0],
            master_sink_factory=factories[1],
            matte_sink_factory=factories[2],
            mask_store_dir=tmp_path,
        )

    assert decoder.closed is True
    assert all(factory.instances[0].aborted for factory in factories)
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_mask_store_unlinks_file_when_temporary_handle_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.pipeline import _packed_mask_store

    close_failure = OSError("temporary handle close failed")
    store_path = tmp_path / ".sam3-mask-created.bin"
    store_path.write_bytes(b"")

    class CloseFailingTemporary:
        name = str(store_path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            self.close()

        def close(self) -> None:
            raise close_failure

    monkeypatch.setattr(
        "sam3_matting.pipeline.tempfile.NamedTemporaryFile",
        lambda *args, **kwargs: CloseFailingTemporary(),
    )

    with (
        pytest.raises(OSError) as caught,
        _packed_mask_store(_metadata(frame_count=1, width=2, height=2), directory=tmp_path),
    ):
        pytest.fail("mask store setup unexpectedly succeeded")

    assert caught.value is close_failure
    assert not store_path.exists()


def test_mask_store_unlinks_file_when_memmap_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.pipeline import _packed_mask_store

    creation_failure = MemoryError("memmap creation failed")

    def fail_memmap(*args, **kwargs):
        del args, kwargs
        raise creation_failure

    monkeypatch.setattr("sam3_matting.pipeline.np.memmap", fail_memmap)

    with (
        pytest.raises(MemoryError) as caught,
        _packed_mask_store(_metadata(frame_count=1, width=2, height=2), directory=tmp_path),
    ):
        pytest.fail("mask store setup unexpectedly succeeded")

    assert caught.value is creation_failure
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


@pytest.mark.parametrize("failure_stage", ["initialize", "initial-flush"])
def test_mask_store_unlinks_file_when_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    from sam3_matting.pipeline import _packed_mask_store

    setup_failure = OSError(f"{failure_stage} failed")
    store = FailingMaskStore(
        init_failure=setup_failure if failure_stage == "initialize" else None,
        flush_failures={1: setup_failure} if failure_stage == "initial-flush" else None,
    )
    monkeypatch.setattr("sam3_matting.pipeline.np.memmap", lambda *args, **kwargs: store)

    with (
        pytest.raises(OSError) as caught,
        _packed_mask_store(_metadata(frame_count=1, width=2, height=2), directory=tmp_path),
    ):
        pytest.fail("mask store setup unexpectedly succeeded")

    assert caught.value is setup_failure
    assert store._mmap.close_calls == 1
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_mask_store_cleanup_failures_do_not_mask_primary_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.pipeline import _packed_mask_store

    primary = KeyboardInterrupt("pipeline cancelled")
    store = FailingMaskStore(
        flush_failures={2: OSError("teardown flush failed")},
        close_failure=SystemExit("mapped-file close failed"),
    )
    monkeypatch.setattr("sam3_matting.pipeline.np.memmap", lambda *args, **kwargs: store)

    with (
        pytest.raises(KeyboardInterrupt) as caught,
        _packed_mask_store(_metadata(frame_count=1, width=2, height=2), directory=tmp_path),
    ):
        raise primary

    assert caught.value is primary
    assert store.flush_calls == 2
    assert store._mmap.close_calls == 1
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_mask_store_attempts_all_cleanup_before_raising_first_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.pipeline import _packed_mask_store

    flush_failure = OSError("teardown flush failed")
    store = FailingMaskStore(
        flush_failures={2: flush_failure},
        close_failure=SystemExit("mapped-file close failed"),
    )
    monkeypatch.setattr("sam3_matting.pipeline.np.memmap", lambda *args, **kwargs: store)

    with (
        pytest.raises(OSError) as caught,
        _packed_mask_store(_metadata(frame_count=1, width=2, height=2), directory=tmp_path),
    ):
        pass

    assert caught.value is flush_failure
    assert store.flush_calls == 2
    assert store._mmap.close_calls == 1
    assert list(tmp_path.glob(".sam3-mask-*.bin")) == []


def test_pipeline_omits_source_audio_for_silent_input(tmp_path: Path) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    factories = _factories(events)

    run_pipeline(
        tmp_path / "silent.mp4",
        prompt="man",
        detection_threshold=0.5,
        detect_interval=1,
        backend=FakeBackend(
            {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
            events,
        ),
        refiner=FakeRefiner([np.ones((2, 2), dtype=np.float32)], events),
        output_dir=tmp_path,
        validate_fn=_validator(
            _metadata(frame_count=1, width=2, height=2, has_audio=False),
            events,
        ),
        decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
        preview_sink_factory=factories[0],
        master_sink_factory=factories[1],
        matte_sink_factory=factories[2],
        mask_store_dir=tmp_path,
    )

    preview, master, matte = (factory.instances[0] for factory in factories)
    assert preview.options["source_audio_path"] is None
    assert master.options["source_audio_path"] is None
    assert matte.options["source_audio_path"] is None


def test_pipeline_sanitizes_and_bounds_output_stem(tmp_path: Path) -> None:
    *_, run_pipeline = _pipeline_api()
    events: list[object] = []
    factories = _factories(events)
    source = tmp_path / ((" Weird name[]🔥.." * 40) + ".mp4")

    result = run_pipeline(
        source,
        prompt="man",
        detection_threshold=0.5,
        detect_interval=1,
        backend=FakeBackend(
            {"man": [FakeTrackedFrame(0, _mask((0, 0), shape=(2, 2)))]},
            events,
        ),
        refiner=FakeRefiner([np.ones((2, 2), dtype=np.float32)], events),
        output_dir=tmp_path / "outputs",
        validate_fn=_validator(_metadata(frame_count=1, width=2, height=2), events),
        decode_fn=_decoder([np.zeros((2, 2, 3), dtype=np.uint8)], events),
        preview_sink_factory=factories[0],
        master_sink_factory=factories[1],
        matte_sink_factory=factories[2],
        mask_store_dir=tmp_path,
    )

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    for output_path in (result.preview_path, result.master_path, result.matte_path):
        assert len(output_path.name.encode()) <= 160
        assert set(output_path.name) <= allowed
