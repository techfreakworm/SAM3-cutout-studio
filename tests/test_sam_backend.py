import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from threading import Barrier, BrokenBarrierError, Lock, get_ident, local
from types import SimpleNamespace

import numpy as np
import pytest


def _backend_api():
    try:
        from sam3_matting.backends import (
            BackendProtocolError,
            MetaSam31Backend,
            SamVideoBackend,
            TrackedFrame,
        )
    except ModuleNotFoundError:
        pytest.fail("SAM backend contract has not been implemented")
    return BackendProtocolError, MetaSam31Backend, SamVideoBackend, TrackedFrame


def _frame_outputs(frame_index: int) -> dict[str, object]:
    if frame_index == 0:
        masks = np.array(
            [
                [
                    [False, True, True, False],
                    [False, True, False, False],
                    [False, False, False, False],
                ],
                [
                    [False, False, False, False],
                    [False, False, True, True],
                    [False, False, True, False],
                ],
            ],
            dtype=np.bool_,
        )
        object_ids = np.array([12, 21], dtype=np.int64)
        probabilities = np.array([0.91, 0.77], dtype=np.float32)
        boxes = np.array(
            [[0.125, 0.0, 0.5, 0.667], [0.5, 0.333, 0.5, 0.667]],
            dtype=np.float32,
        )
    else:
        masks = np.array(
            [
                [
                    [False, False, True, False],
                    [False, True, True, False],
                    [False, True, False, False],
                ]
            ],
            dtype=np.bool_,
        )
        object_ids = np.array([12], dtype=np.int64)
        probabilities = np.array([0.88], dtype=np.float32)
        boxes = np.array([[0.25, 0.0, 0.5, 1.0]], dtype=np.float32)

    return {
        "out_obj_ids": object_ids,
        "out_probs": probabilities,
        "out_boxes_xywh": boxes,
        "out_binary_masks": masks,
        "frame_stats": {
            "frame_index": frame_index,
            "num_total_objects": int(object_ids.size),
            "num_new_objects": int(object_ids.size if frame_index == 0 else 0),
            "num_suppressed_objects": 0,
            "num_removed_objects": 0,
        },
    }


class RealisticPredictorDouble:
    """Small boundary double matching Meta's complete predictor responses."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.model = SimpleNamespace(
            new_det_thresh=0.65,
            score_threshold_detection=0.4,
            recondition_every_nth_frame=16,
            max_num_objects=8,
            num_obj_for_compile=1,
        )
        self.fail_on = fail_on
        self.requests: list[dict[str, object]] = []
        self.runtime_settings_seen: list[tuple[str, float, float, int, int, int]] = []

    def _record_runtime_settings(self, phase: str) -> None:
        self.runtime_settings_seen.append(
            (
                phase,
                self.model.new_det_thresh,
                self.model.score_threshold_detection,
                self.model.recondition_every_nth_frame,
                self.model.max_num_objects,
                self.model.num_obj_for_compile,
            )
        )

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        request_type = str(request["type"])
        self._record_runtime_settings(request_type)
        if self.fail_on == request_type:
            raise RuntimeError(f"synthetic {request_type} failure")
        if request_type == "start_session":
            return {"session_id": "session-8c2f4a"}
        if request_type == "add_prompt":
            return {"frame_index": 0, "outputs": _frame_outputs(0)}
        if request_type == "close_session":
            return {
                "is_success": True,
                "gpu_mem": {
                    "free_bytes": 86_000_000_000,
                    "total_bytes": 102_000_000_000,
                    "allocated_bytes": 0,
                    "reserved_bytes": 0,
                    "free_pct": 84.313725,
                    "active_session_count": 0,
                },
            }
        raise AssertionError(f"unexpected request: {request}")

    def handle_stream_request(self, request: dict[str, object]):
        self.requests.append(request)
        request_type = str(request["type"])
        self._record_runtime_settings(request_type)
        if self.fail_on == request_type:
            raise RuntimeError(f"synthetic {request_type} failure")
        # Meta propagation includes the prompted frame. The backend must not emit it twice.
        yield {"frame_index": 0, "outputs": _frame_outputs(0)}
        yield {"frame_index": 1, "outputs": _frame_outputs(1)}


class RecordingBuilder:
    def __init__(self, predictor: RealisticPredictorDouble) -> None:
        self.predictor = predictor
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> RealisticPredictorDouble:
        self.calls.append(kwargs)
        return self.predictor


class AutocastProbe:
    """Thread-local stand-in for CUDA autocast used by cross-thread tests."""

    def __init__(self) -> None:
        self._state = local()
        self.events: list[tuple[str, int, int]] = []

    def depth(self) -> int:
        return int(getattr(self._state, "depth", 0))

    def is_active(self) -> bool:
        return self.depth() > 0

    @contextmanager
    def context(self):
        thread_id = get_ident()
        previous_depth = self.depth()
        self._state.depth = previous_depth + 1
        self.events.append(("enter", thread_id, self.depth()))
        try:
            yield
        finally:
            assert self.depth() == previous_depth + 1
            self._state.depth = previous_depth
            self.events.append(("exit", thread_id, self.depth()))


class ExitFailingContext:
    """Context wrapper that restores state before reporting an exit failure."""

    def __init__(self, wrapped_context) -> None:
        self._wrapped_context = wrapped_context

    def __enter__(self):
        return self._wrapped_context.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self._wrapped_context.__exit__(exc_type, exc_value, traceback)
        raise RuntimeError("synthetic autocast unwind failure")


def _entered_predictor_graph(
    probe: AutocastProbe,
    *,
    fail_predictor_exit: bool = False,
    include_legacy_tracker_model_context: bool = False,
):
    """Build the pinned two-owner graph, optionally with its legacy third owner."""
    tracker_model = SimpleNamespace()
    if include_legacy_tracker_model_context:
        tracker_model_context = probe.context()
        tracker_model_context.__enter__()
        tracker_model.bf16_context = tracker_model_context

    tracker_context = probe.context()
    tracker_context.__enter__()
    tracker = SimpleNamespace(
        bf16_context=tracker_context,
        model=tracker_model,
    )

    predictor_context = probe.context()
    if fail_predictor_exit:
        predictor_context = ExitFailingContext(predictor_context)
    predictor_context.__enter__()
    return SimpleNamespace(
        bf16_context=predictor_context,
        model=SimpleNamespace(tracker=tracker),
    )


class AutocastCheckingStream:
    def __init__(
        self,
        predictor: "AutocastCheckingPredictor",
        *,
        fail_on_next: int | None,
    ) -> None:
        self._predictor = predictor
        self._fail_on_next = fail_on_next
        self._next_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, object]:
        next_index = self._next_index
        self._predictor.assert_autocast_active(f"stream_next:{next_index}")
        if self._fail_on_next == next_index:
            raise RuntimeError("synthetic propagation next failure")
        if next_index >= 3:
            raise StopIteration
        self._next_index += 1
        return {
            "frame_index": next_index,
            "outputs": _frame_outputs(next_index),
        }


class AutocastCheckingPredictor(RealisticPredictorDouble):
    def __init__(
        self,
        probe: AutocastProbe,
        *,
        fail_on_next: int | None = None,
        fail_on_close: bool = False,
    ) -> None:
        super().__init__()
        self._probe = probe
        self._fail_on_next = fail_on_next
        self._fail_on_close = fail_on_close
        self.lifecycle: list[tuple[str, int, int]] = []

    def assert_autocast_active(self, phase: str) -> None:
        assert self._probe.is_active(), f"{phase} ran outside request-thread autocast"
        self.lifecycle.append((phase, get_ident(), self._probe.depth()))

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        request_type = str(request["type"])
        self.assert_autocast_active(request_type)
        if request_type == "close_session" and self._fail_on_close:
            raise RuntimeError("synthetic close_session failure")
        return super().handle_request(request)

    def handle_stream_request(self, request: dict[str, object]):
        request_type = str(request["type"])
        self.lifecycle.append((f"{request_type}:create", get_ident(), self._probe.depth()))
        self.requests.append(request)
        self._record_runtime_settings(request_type)
        return AutocastCheckingStream(
            self,
            fail_on_next=self._fail_on_next,
        )


def _preload_backend_for_autocast_test(
    MetaSam31Backend,
    predictor: AutocastCheckingPredictor,
):
    builder_thread_ids: list[int] = []

    def builder(**kwargs: object) -> AutocastCheckingPredictor:
        builder_thread_ids.append(get_ident())
        return predictor

    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=builder,
    )
    preload_thread_id = get_ident()
    assert backend._get_predictor() is predictor
    assert builder_thread_ids == [preload_thread_id]
    return backend, preload_thread_id


def _install_autocast_probe(
    monkeypatch: pytest.MonkeyPatch,
    probe: AutocastProbe,
) -> None:
    import sam3_matting.backends.meta_sam31 as backend_module

    monkeypatch.setattr(
        backend_module,
        "_cuda_bfloat16_autocast",
        probe.context,
        raising=False,
    )


def _consume_with_autocast_depths(
    backend,
    probe: AutocastProbe,
    *,
    expected_depth: int,
) -> tuple[list[int], list[int], int, int]:
    frame_indices: list[int] = []
    depths_after_yield: list[int] = []
    for frame in backend.track("/videos/intro.mp4", prompt="person"):
        frame_indices.append(frame.frame_index)
        depths_after_yield.append(probe.depth())
        assert probe.depth() == expected_depth
    return frame_indices, depths_after_yield, probe.depth(), get_ident()


def _assert_fresh_phase_contexts(
    probe: AutocastProbe,
    *,
    worker_thread_id: int,
    baseline_depth: int,
    phase_count: int,
) -> None:
    assert probe.events == [
        event
        for _ in range(phase_count)
        for event in (
            ("enter", worker_thread_id, baseline_depth + 1),
            ("exit", worker_thread_id, baseline_depth),
        )
    ]


def test_meta_backend_builds_predictor_once_when_lazy_loads_race() -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    build_barrier = Barrier(2)
    calls_lock = Lock()
    built_predictors: list[object] = []

    def racing_builder(**kwargs: object) -> object:
        predictor = SimpleNamespace(model=SimpleNamespace(tracker=None))
        with calls_lock:
            built_predictors.append(predictor)
        with suppress(BrokenBarrierError):
            build_barrier.wait(timeout=0.25)
        return predictor

    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=racing_builder,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(backend._get_predictor) for _ in range(2)]
        results = [future.result() for future in futures]

    assert len(built_predictors) == 1
    assert results[0] is results[1]
    assert results[0] is backend._predictor


def test_meta_backend_exits_bfloat16_autocast_before_every_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    probe = AutocastProbe()
    predictor = AutocastCheckingPredictor(probe)
    backend, preload_thread_id = _preload_backend_for_autocast_test(
        MetaSam31Backend,
        predictor,
    )
    _install_autocast_probe(monkeypatch, probe)

    with ThreadPoolExecutor(max_workers=1) as executor:
        frame_indices, depths_after_yield, final_depth, worker_thread_id = executor.submit(
            _consume_with_autocast_depths,
            backend,
            probe,
            expected_depth=0,
        ).result()

    assert worker_thread_id != preload_thread_id
    assert frame_indices == [0, 1, 2]
    assert depths_after_yield == [0, 0, 0]
    assert final_depth == 0
    _assert_fresh_phase_contexts(
        probe,
        worker_thread_id=worker_thread_id,
        baseline_depth=0,
        phase_count=7,
    )
    assert predictor.lifecycle == [
        ("start_session", worker_thread_id, 1),
        ("add_prompt", worker_thread_id, 1),
        ("propagate_in_video:create", worker_thread_id, 0),
        ("stream_next:0", worker_thread_id, 1),
        ("stream_next:1", worker_thread_id, 1),
        ("stream_next:2", worker_thread_id, 1),
        ("stream_next:3", worker_thread_id, 1),
        ("close_session", worker_thread_id, 1),
    ]


def test_meta_backend_restores_nested_autocast_depth_after_every_phase_and_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    probe = AutocastProbe()
    predictor = AutocastCheckingPredictor(probe)
    backend, preload_thread_id = _preload_backend_for_autocast_test(
        MetaSam31Backend,
        predictor,
    )
    _install_autocast_probe(monkeypatch, probe)

    def consume_nested() -> tuple[list[int], list[int], int, int]:
        with probe.context():
            result = _consume_with_autocast_depths(
                backend,
                probe,
                expected_depth=1,
            )
            assert result[2] == 1
        return result[0], result[1], probe.depth(), result[3]

    with ThreadPoolExecutor(max_workers=1) as executor:
        frame_indices, depths_after_yield, final_depth, worker_thread_id = executor.submit(
            consume_nested
        ).result()

    assert worker_thread_id != preload_thread_id
    assert frame_indices == [0, 1, 2]
    assert depths_after_yield == [1, 1, 1]
    assert final_depth == 0
    assert probe.events == [
        ("enter", worker_thread_id, 1),
        *[
            event
            for _ in range(7)
            for event in (
                ("enter", worker_thread_id, 2),
                ("exit", worker_thread_id, 1),
            )
        ],
        ("exit", worker_thread_id, 0),
    ]


def test_meta_backend_leaves_autocast_inactive_between_sequential_jobs_on_same_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    probe = AutocastProbe()
    predictor = AutocastCheckingPredictor(probe)
    backend, preload_thread_id = _preload_backend_for_autocast_test(
        MetaSam31Backend,
        predictor,
    )
    _install_autocast_probe(monkeypatch, probe)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            _consume_with_autocast_depths,
            backend,
            probe,
            expected_depth=0,
        ).result()
        second = executor.submit(
            _consume_with_autocast_depths,
            backend,
            probe,
            expected_depth=0,
        ).result()

    assert first[3] == second[3]
    assert first[3] != preload_thread_id
    assert first[:3] == ([0, 1, 2], [0, 0, 0], 0)
    assert second[:3] == ([0, 1, 2], [0, 0, 0], 0)
    _assert_fresh_phase_contexts(
        probe,
        worker_thread_id=first[3],
        baseline_depth=0,
        phase_count=14,
    )


def test_meta_backend_preserves_propagation_error_when_close_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    probe = AutocastProbe()
    predictor = AutocastCheckingPredictor(
        probe,
        fail_on_next=1,
        fail_on_close=True,
    )
    backend, _ = _preload_backend_for_autocast_test(
        MetaSam31Backend,
        predictor,
    )
    _install_autocast_probe(monkeypatch, probe)

    with (
        caplog.at_level("ERROR"),
        pytest.raises(RuntimeError, match="synthetic propagation next failure"),
    ):
        list(backend.track("/videos/intro.mp4", prompt="person"))

    assert probe.depth() == 0
    assert predictor.lifecycle[-1][0] == "close_session"
    assert "close_session failed while preserving the primary inference error" in caplog.text


def test_meta_backend_surfaces_close_failure_when_iterator_closes_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    probe = AutocastProbe()
    predictor = AutocastCheckingPredictor(probe, fail_on_close=True)
    backend, _ = _preload_backend_for_autocast_test(
        MetaSam31Backend,
        predictor,
    )
    _install_autocast_probe(monkeypatch, probe)

    frames = backend.track("/videos/intro.mp4", prompt="person")
    assert next(frames).frame_index == 0
    assert probe.depth() == 0

    with pytest.raises(RuntimeError, match="synthetic close_session failure"):
        frames.close()

    assert probe.depth() == 0


def test_meta_backend_unwinds_pinned_two_owner_context_graph_once_in_reverse_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    probe = AutocastProbe()
    _install_autocast_probe(monkeypatch, probe)

    def pinned_builder(**kwargs: object) -> object:
        return _entered_predictor_graph(probe)

    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=pinned_builder,
    )

    with probe.context():
        assert probe.depth() == 1
        predictor = backend._get_predictor()
        assert backend._get_predictor() is predictor
        assert probe.depth() == 1

    assert probe.depth() == 0
    thread_id = get_ident()
    assert probe.events == [
        ("enter", thread_id, 1),
        ("enter", thread_id, 2),
        ("enter", thread_id, 3),
        ("exit", thread_id, 2),
        ("exit", thread_id, 1),
        ("exit", thread_id, 0),
    ]


def test_meta_backend_unwinds_supported_legacy_tracker_model_context() -> None:
    from sam3_matting.backends.meta_sam31 import _exit_upstream_autocast_contexts

    probe = AutocastProbe()
    predictor = _entered_predictor_graph(
        probe,
        include_legacy_tracker_model_context=True,
    )

    _exit_upstream_autocast_contexts(predictor)

    assert probe.depth() == 0
    thread_id = get_ident()
    assert probe.events == [
        ("enter", thread_id, 1),
        ("enter", thread_id, 2),
        ("enter", thread_id, 3),
        ("exit", thread_id, 2),
        ("exit", thread_id, 1),
        ("exit", thread_id, 0),
    ]


def test_safetensors_builder_unwinds_constructor_autocast_when_loading_fails() -> None:
    from sam3_matting.backends.meta_sam31 import _build_safetensors_predictor

    probe = AutocastProbe()
    predictor = _entered_predictor_graph(probe)

    def builder(**kwargs: object) -> object:
        return predictor

    def failing_load_model(*args: object, **kwargs: object):
        raise RuntimeError("synthetic safetensors load failure")

    with pytest.raises(RuntimeError, match="synthetic safetensors load failure"):
        _build_safetensors_predictor(
            "/models/sam3.1_multiplex_fp16.safetensors",
            predictor_builder=builder,
            load_model=failing_load_model,
            builder_kwargs={},
        )

    assert probe.depth() == 0


def test_default_builder_unwinds_constructor_autocast_when_adaptation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sam3.model_builder as upstream_builder

    import sam3_matting.backends.meta_sam31 as backend_module

    probe = AutocastProbe()
    predictor = _entered_predictor_graph(probe)
    monkeypatch.setattr(
        upstream_builder,
        "build_sam3_multiplex_video_predictor",
        lambda **kwargs: predictor,
    )

    def failing_adaptation(predictor_to_adapt: object) -> object:
        assert predictor_to_adapt is predictor
        raise RuntimeError("synthetic predictor adaptation failure")

    monkeypatch.setattr(backend_module, "_adapt_legacy_state_offload", failing_adaptation)

    with pytest.raises(RuntimeError, match="synthetic predictor adaptation failure"):
        backend_module._default_predictor_builder(checkpoint_path="sam3.1_multiplex.pt")

    assert probe.depth() == 0


@pytest.mark.parametrize("failure_path", ["safetensors", "default"])
def test_predictor_build_preserves_primary_error_when_autocast_unwind_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_path: str,
) -> None:
    import sam3.model_builder as upstream_builder

    import sam3_matting.backends.meta_sam31 as backend_module

    probe = AutocastProbe()
    predictor = _entered_predictor_graph(probe, fail_predictor_exit=True)
    primary_message = f"synthetic {failure_path} primary failure"

    with caplog.at_level("ERROR"):
        if failure_path == "safetensors":

            def failing_load_model(*args: object, **kwargs: object):
                raise RuntimeError(primary_message)

            with pytest.raises(RuntimeError, match=primary_message):
                backend_module._build_safetensors_predictor(
                    "/models/sam3.1_multiplex_fp16.safetensors",
                    predictor_builder=lambda **kwargs: predictor,
                    load_model=failing_load_model,
                    builder_kwargs={},
                )
        else:
            monkeypatch.setattr(
                upstream_builder,
                "build_sam3_multiplex_video_predictor",
                lambda **kwargs: predictor,
            )

            def failing_adaptation(predictor_to_adapt: object) -> object:
                assert predictor_to_adapt is predictor
                raise RuntimeError(primary_message)

            monkeypatch.setattr(
                backend_module,
                "_adapt_legacy_state_offload",
                failing_adaptation,
            )
            with pytest.raises(RuntimeError, match=primary_message):
                backend_module._default_predictor_builder(checkpoint_path="sam3.1_multiplex.pt")

    assert probe.depth() == 0
    assert "autocast cleanup failed while preserving predictor build error" in caplog.text


@pytest.mark.skipif(
    os.environ.get("SAM3_CUDA_THREAD_SMOKE") != "1",
    reason="set SAM3_CUDA_THREAD_SMOKE=1 for the real CUDA worker-thread smoke",
)
def test_real_cuda_backend_tracks_after_main_thread_preload() -> None:
    import torch

    checkpoint_path = os.environ.get("SAM3_CUDA_THREAD_CHECKPOINT")
    video_path = os.environ.get("SAM3_CUDA_THREAD_VIDEO")
    if not checkpoint_path or not video_path:
        pytest.skip(
            "set SAM3_CUDA_THREAD_CHECKPOINT and SAM3_CUDA_THREAD_VIDEO for the real CUDA worker-thread smoke"
        )

    _, MetaSam31Backend, _, _ = _backend_api()
    backend = MetaSam31Backend(checkpoint_path=checkpoint_path)
    preload_thread_id = get_ident()
    preload_autocast_enabled = torch.is_autocast_enabled("cuda")
    backend._get_predictor()
    assert torch.is_autocast_enabled("cuda") is preload_autocast_enabled

    def consume() -> tuple[int, int, list[bool], bool]:
        autocast_after_yield: list[bool] = []
        frame_count = 0
        for _frame in backend.track(
            video_path,
            prompt="person",
            detection_threshold=0.5,
            detect_interval=1,
            max_objects=2,
        ):
            frame_count += 1
            autocast_after_yield.append(torch.is_autocast_enabled("cuda"))
        return (
            frame_count,
            get_ident(),
            autocast_after_yield,
            torch.is_autocast_enabled("cuda"),
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(consume).result()
        second = executor.submit(consume).result()

    assert first[1] == second[1]
    assert first[1] != preload_thread_id
    assert first[0] > 0
    assert second[0] == first[0]
    assert first[2] == [False] * first[0]
    assert second[2] == [False] * second[0]
    assert first[3] is False
    assert second[3] is False


@pytest.mark.parametrize(
    ("parameter_name", "default"),
    [("detect_interval", 1), ("max_objects", 8)],
)
def test_backend_contract_exposes_tracking_controls_as_keyword_only(
    parameter_name: str,
    default: int,
) -> None:
    _, _, SamVideoBackend, _ = _backend_api()

    parameter = inspect.signature(SamVideoBackend.track).parameters[parameter_name]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == default


def test_cuda_meta_backend_streams_complete_frame_contract_and_unions_objects() -> None:
    _, MetaSam31Backend, SamVideoBackend, TrackedFrame = _backend_api()
    predictor = RealisticPredictorDouble()
    builder = RecordingBuilder(predictor)
    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        max_objects=8,
        predictor_builder=builder,
    )

    frames = list(
        backend.track(
            "/videos/intro.mp4",
            prompt="man,hair,collar mic",
            detection_threshold=0.5,
            detect_interval=1,
            max_objects=3,
        )
    )

    assert isinstance(backend, SamVideoBackend)
    assert len(frames) == 2
    assert all(isinstance(frame, TrackedFrame) for frame in frames)
    assert frames[0].frame_index == 0
    np.testing.assert_array_equal(frames[0].object_ids, np.array([12, 21]))
    np.testing.assert_allclose(frames[0].scores, np.array([0.91, 0.77]))
    np.testing.assert_allclose(
        frames[0].boxes_xywh,
        np.array([[0.125, 0.0, 0.5, 0.667], [0.5, 0.333, 0.5, 0.667]]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        frames[0].union_mask,
        np.array(
            [
                [False, True, True, False],
                [False, True, True, True],
                [False, False, True, False],
            ]
        ),
    )
    assert frames[0].frame_stats == {
        "frame_index": 0,
        "num_total_objects": 2,
        "num_new_objects": 2,
        "num_suppressed_objects": 0,
        "num_removed_objects": 0,
    }

    assert builder.calls == [
        {
            "checkpoint_path": "/models/sam3.1_multiplex.pt",
            "max_num_objects": 8,
            "multiplex_count": 16,
            "use_fa3": False,
            "use_rope_real": True,
            "compile": False,
            "warm_up": False,
            "session_expiration_sec": 1200,
            "default_output_prob_thresh": 0.5,
            "async_loading_frames": True,
        }
    ]
    assert predictor.requests == [
        {
            "type": "start_session",
            "resource_path": "/videos/intro.mp4",
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
        },
        {
            "type": "add_prompt",
            "session_id": "session-8c2f4a",
            "frame_index": 0,
            "text": "man,hair,collar mic",
            "output_prob_thresh": 0.5,
        },
        {
            "type": "propagate_in_video",
            "session_id": "session-8c2f4a",
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "output_prob_thresh": 0.5,
        },
        {
            "type": "close_session",
            "session_id": "session-8c2f4a",
            "run_gc_collect": True,
        },
    ]
    assert predictor.runtime_settings_seen == [
        ("start_session", 0.5, 0.5, 1, 3, 1),
        ("add_prompt", 0.5, 0.5, 1, 3, 1),
        ("propagate_in_video", 0.5, 0.5, 1, 3, 1),
        ("close_session", 0.5, 0.5, 1, 3, 1),
    ]
    assert predictor.model.new_det_thresh == 0.65
    assert predictor.model.score_threshold_detection == 0.4
    assert predictor.model.recondition_every_nth_frame == 16
    assert predictor.model.max_num_objects == 8
    assert predictor.model.num_obj_for_compile == 1


@pytest.mark.parametrize(
    "failure_phase",
    ["start_session", "add_prompt", "propagate_in_video", "close_session"],
)
def test_meta_backend_restores_runtime_settings_when_predictor_phase_fails(
    failure_phase: str,
) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    predictor = RealisticPredictorDouble(fail_on=failure_phase)
    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=RecordingBuilder(predictor),
    )

    with pytest.raises(RuntimeError, match=f"synthetic {failure_phase} failure"):
        list(
            backend.track(
                "/videos/intro.mp4",
                prompt="person",
                detection_threshold=0.55,
                detect_interval=2,
                max_objects=4,
            )
        )

    observed_phases = [settings[0] for settings in predictor.runtime_settings_seen]
    if failure_phase == "start_session":
        assert observed_phases == ["start_session"]
    elif failure_phase == "add_prompt":
        assert observed_phases == ["start_session", "add_prompt", "close_session"]
    else:
        assert observed_phases == [
            "start_session",
            "add_prompt",
            "propagate_in_video",
            "close_session",
        ]
    assert all(settings[1:] == (0.55, 0.55, 2, 4, 1) for settings in predictor.runtime_settings_seen)
    assert predictor.model.new_det_thresh == 0.65
    assert predictor.model.score_threshold_detection == 0.4
    assert predictor.model.recondition_every_nth_frame == 16
    assert predictor.model.max_num_objects == 8
    assert predictor.model.num_obj_for_compile == 1


def test_meta_backend_reuses_predictor_across_request_max_objects_values() -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    predictor = RealisticPredictorDouble()
    builder = RecordingBuilder(predictor)
    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=builder,
    )

    list(backend.track("/videos/intro.mp4", prompt="person", max_objects=2))
    list(backend.track("/videos/intro.mp4", prompt="hair", max_objects=6))

    assert len(builder.calls) == 1
    assert [settings[4] for settings in predictor.runtime_settings_seen] == [2] * 4 + [6] * 4
    assert predictor.model.max_num_objects == 8


def test_meta_backend_rejects_incomplete_meta_output_and_still_closes_session() -> None:
    BackendProtocolError, MetaSam31Backend, _, _ = _backend_api()
    predictor = RealisticPredictorDouble()
    incomplete = _frame_outputs(0)
    del incomplete["out_boxes_xywh"]

    def incomplete_add_prompt(request: dict[str, object]) -> dict[str, object]:
        predictor.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session-8c2f4a"}
        if request["type"] == "add_prompt":
            return {"frame_index": 0, "outputs": incomplete}
        if request["type"] == "close_session":
            return {"is_success": True}
        raise AssertionError(f"unexpected request: {request}")

    predictor.handle_request = incomplete_add_prompt
    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=RecordingBuilder(predictor),
    )

    with pytest.raises(BackendProtocolError, match="out_boxes_xywh"):
        list(backend.track("/videos/intro.mp4", prompt="person"))

    assert predictor.requests[-1]["type"] == "close_session"


def test_meta_sam31_adapter_is_explicitly_cuda_only() -> None:
    _, MetaSam31Backend, _, _ = _backend_api()

    with pytest.raises(ValueError, match="CUDA-only"):
        MetaSam31Backend(
            checkpoint_path="/models/sam3.1_multiplex.pt",
            device="mps",
            predictor_builder=RecordingBuilder(RealisticPredictorDouble()),
        )


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_meta_backend_rejects_invalid_detection_threshold(threshold: float) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=RecordingBuilder(RealisticPredictorDouble()),
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        list(backend.track("/videos/intro.mp4", prompt="person", detection_threshold=threshold))


@pytest.mark.parametrize("detect_interval", [0, -1])
def test_meta_backend_rejects_invalid_detect_interval(detect_interval: int) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    builder = RecordingBuilder(RealisticPredictorDouble())
    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=builder,
    )

    with pytest.raises(ValueError, match="at least 1"):
        list(backend.track("/videos/intro.mp4", prompt="person", detect_interval=detect_interval))

    assert builder.calls == []


@pytest.mark.parametrize("max_objects", [0, 9])
def test_meta_backend_rejects_invalid_request_max_objects_before_predictor_loading(
    max_objects: int,
) -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    builder = RecordingBuilder(RealisticPredictorDouble())
    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=builder,
    )

    with pytest.raises(ValueError, match="between 1 and 8"):
        list(backend.track("/videos/intro.mp4", prompt="person", max_objects=max_objects))

    assert builder.calls == []


def _checkpoint_api():
    try:
        from sam3_matting.backends.checkpoint_schema import (
            TensorSpec,
            compare_schemas,
            read_safetensors_schema,
        )
    except ModuleNotFoundError:
        pytest.fail("checkpoint schema inspector has not been implemented")
    return TensorSpec, compare_schemas, read_safetensors_schema


def test_safetensors_inspector_reads_tensor_schema_from_header(tmp_path) -> None:
    import torch
    from safetensors.torch import save_file

    TensorSpec, _, read_safetensors_schema = _checkpoint_api()
    checkpoint_path = tmp_path / "fixture.safetensors"
    save_file(
        {
            "detector.encoder.weight": torch.zeros((2, 3), dtype=torch.float16),
            "tracker.model.bias": torch.ones((4,), dtype=torch.bfloat16),
        },
        checkpoint_path,
    )

    schema = read_safetensors_schema(checkpoint_path)

    assert schema == {
        "detector.encoder.weight": TensorSpec(shape=(2, 3), dtype="F16"),
        "tracker.model.bias": TensorSpec(shape=(4,), dtype="BF16"),
    }


def test_checkpoint_schema_comparison_reports_all_key_and_shape_differences() -> None:
    TensorSpec, compare_schemas, _ = _checkpoint_api()
    checkpoint_schema = {
        "detector.good": TensorSpec(shape=(2, 2), dtype="F16"),
        "tracker.model.wrong_shape": TensorSpec(shape=(3, 4), dtype="F16"),
        "detector.unexpected": TensorSpec(shape=(1,), dtype="F16"),
    }
    expected_schema = {
        "detector.good": TensorSpec(shape=(2, 2), dtype="F32"),
        "tracker.model.wrong_shape": TensorSpec(shape=(4, 3), dtype="F32"),
        "tracker.model.missing": TensorSpec(shape=(8,), dtype="F32"),
    }

    report = compare_schemas(checkpoint_schema, expected_schema)

    assert report.is_compatible is False
    assert report.missing_keys == ("tracker.model.missing",)
    assert report.unexpected_keys == ("detector.unexpected",)
    assert report.shape_mismatches == (("tracker.model.wrong_shape", (3, 4), (4, 3)),)


def test_checkpoint_schema_comparison_allows_half_precision_weights() -> None:
    TensorSpec, compare_schemas, _ = _checkpoint_api()
    checkpoint_schema = {
        "detector.weight": TensorSpec(shape=(2, 2), dtype="F16"),
        "tracker.model.weight": TensorSpec(shape=(2, 2), dtype="BF16"),
    }
    expected_schema = {
        "detector.weight": TensorSpec(shape=(2, 2), dtype="F32"),
        "tracker.model.weight": TensorSpec(shape=(2, 2), dtype="F32"),
    }

    report = compare_schemas(checkpoint_schema, expected_schema)

    assert report.is_compatible is True
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert report.shape_mismatches == ()


def test_sam31_missing_key_classification_allows_generated_buffers_and_unused_projection() -> None:
    try:
        from sam3_matting.backends.checkpoint_schema import partition_sam31_missing_keys
    except ImportError:
        pytest.fail("SAM 3.1 missing-key classification has not been implemented")

    allowed, blocking = partition_sam31_missing_keys(
        (
            "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis",
            "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis_real",
            "detector.backbone.language_backbone.encoder.text_projection",
            "detector.transformer.encoder.layers.0.linear1.weight",
        )
    )

    assert allowed == (
        "detector.backbone.language_backbone.encoder.text_projection",
        "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis",
        "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis_real",
    )
    assert blocking == ("detector.transformer.encoder.layers.0.linear1.weight",)


def test_public_safetensors_builder_loads_meta_model_and_initializes_allowed_projection() -> None:
    import os
    from types import SimpleNamespace

    import torch

    try:
        from sam3_matting.backends.meta_sam31 import _build_safetensors_predictor
    except ImportError:
        pytest.fail("public safetensors loading has not been implemented")

    projection = torch.nn.Parameter(torch.ones((2, 2)))
    model = SimpleNamespace(
        detector=SimpleNamespace(
            backbone=SimpleNamespace(
                language_backbone=SimpleNamespace(encoder=SimpleNamespace(text_projection=projection))
            )
        )
    )
    predictor = SimpleNamespace(model=model)
    builder_calls: list[dict[str, object]] = []
    load_calls: list[tuple[object, str, bool, str]] = []
    temporary_checkpoint: list[str] = []

    def builder(**kwargs: object) -> object:
        builder_calls.append(kwargs)
        temporary_checkpoint.append(str(kwargs["checkpoint_path"]))
        assert os.path.exists(temporary_checkpoint[-1])
        assert temporary_checkpoint[-1].endswith(".pt")
        return predictor

    def load_model(
        target_model: object,
        filename: str,
        *,
        strict: bool,
        device: str,
    ) -> tuple[list[str], list[str]]:
        load_calls.append((target_model, filename, strict, device))
        return (
            [
                "detector.backbone.language_backbone.encoder.text_projection",
                "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis",
            ],
            [],
        )

    result = _build_safetensors_predictor(
        "/models/sam3.1_multiplex_fp16.safetensors",
        predictor_builder=builder,
        load_model=load_model,
        builder_kwargs={"max_num_objects": 8, "compile": False},
    )

    assert result is predictor
    assert builder_calls == [
        {
            "checkpoint_path": temporary_checkpoint[0],
            "max_num_objects": 8,
            "compile": False,
        }
    ]
    assert os.path.exists(temporary_checkpoint[0]) is False
    assert load_calls == [
        (
            model,
            "/models/sam3.1_multiplex_fp16.safetensors",
            False,
            "cpu",
        )
    ]
    torch.testing.assert_close(projection, torch.zeros_like(projection))


def test_public_safetensors_builder_rejects_missing_learned_weights() -> None:
    from types import SimpleNamespace

    try:
        from sam3_matting.backends.meta_sam31 import (
            CheckpointCompatibilityError,
            _build_safetensors_predictor,
        )
    except ImportError:
        pytest.fail("safetensors compatibility validation has not been implemented")

    predictor = SimpleNamespace(model=SimpleNamespace())

    def builder(**kwargs: object) -> object:
        return predictor

    def incompatible_load_model(
        model: object,
        filename: str,
        *,
        strict: bool,
        device: str,
    ) -> tuple[list[str], list[str]]:
        return (["detector.transformer.encoder.layers.0.linear1.weight"], [])

    with pytest.raises(CheckpointCompatibilityError, match="missing learned keys"):
        _build_safetensors_predictor(
            "/models/sam3.1_multiplex_fp16.safetensors",
            predictor_builder=builder,
            load_model=incompatible_load_model,
            builder_kwargs={},
        )


def _upstream_session_predictor_without_state_offload():
    from sam3.model.sam3_base_predictor import Sam3BasePredictor

    class UpstreamMultiplexModel:
        def __init__(self) -> None:
            self.init_calls: list[dict[str, object]] = []

        def init_state(
            self,
            resource_path: str,
            offload_video_to_cpu: bool = False,
            async_loading_frames: bool = False,
            use_torchcodec: bool = False,
            use_cv2: bool = False,
            input_is_mp4: bool = False,
        ) -> dict[str, object]:
            self.init_calls.append(
                {
                    "resource_path": resource_path,
                    "offload_video_to_cpu": offload_video_to_cpu,
                    "async_loading_frames": async_loading_frames,
                    "use_torchcodec": use_torchcodec,
                    "use_cv2": use_cv2,
                    "input_is_mp4": input_is_mp4,
                }
            )
            return {"resource_path": resource_path, "initialized": True}

    predictor = Sam3BasePredictor()
    model = UpstreamMultiplexModel()
    predictor.model = model
    predictor.async_loading_frames = False
    return predictor, model


def _patch_default_meta_builder(monkeypatch: pytest.MonkeyPatch, checkpoint_name: str):
    import sam3.model_builder as upstream_builder

    import sam3_matting.backends.meta_sam31 as backend_module

    predictor, model = _upstream_session_predictor_without_state_offload()
    monkeypatch.setattr(
        upstream_builder,
        "build_sam3_multiplex_video_predictor",
        lambda **kwargs: predictor,
    )
    monkeypatch.setattr(
        backend_module,
        "_build_safetensors_predictor",
        lambda *args, **kwargs: predictor,
    )
    built = backend_module._default_predictor_builder(checkpoint_path=checkpoint_name)
    return built, model


@pytest.mark.parametrize(
    "checkpoint_name",
    ["sam3.1_multiplex.pt", "sam3.1_multiplex_fp16.safetensors"],
)
def test_default_predictor_drops_false_legacy_state_offload_and_preserves_session_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_name: str,
) -> None:
    predictor, model = _patch_default_meta_builder(monkeypatch, checkpoint_name)

    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": "/videos/input.mp4",
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
        }
    )

    session_id = response["session_id"]
    assert isinstance(session_id, str)
    assert session_id in predictor._all_inference_states
    assert model.init_calls == [
        {
            "resource_path": "/videos/input.mp4",
            "offload_video_to_cpu": True,
            "async_loading_frames": False,
            "use_torchcodec": False,
            "use_cv2": False,
            "input_is_mp4": False,
        }
    ]

    close_response = predictor.handle_request(
        {
            "type": "close_session",
            "session_id": session_id,
            "run_gc_collect": False,
        }
    )
    assert close_response == {"is_success": True}
    assert predictor._all_inference_states == {}


@pytest.mark.parametrize(
    "checkpoint_name",
    ["sam3.1_multiplex.pt", "sam3.1_multiplex_fp16.safetensors"],
)
def test_default_predictor_never_silently_accepts_true_legacy_state_offload(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_name: str,
) -> None:
    predictor, model = _patch_default_meta_builder(monkeypatch, checkpoint_name)

    with pytest.raises(
        ValueError,
        match="offload_state_to_cpu=True is not supported",
    ):
        predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": "/videos/input.mp4",
                "offload_state_to_cpu": True,
            }
        )

    assert model.init_calls == []
    assert predictor._all_inference_states == {}
