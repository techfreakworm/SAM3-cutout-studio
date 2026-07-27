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

    def __init__(self, *, fail_during_propagation: bool = False) -> None:
        self.model = SimpleNamespace(
            new_det_thresh=0.65,
            score_threshold_detection=0.4,
        )
        self.fail_during_propagation = fail_during_propagation
        self.requests: list[dict[str, object]] = []
        self.thresholds_seen: list[tuple[float, float]] = []

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        request_type = request["type"]
        if request_type == "start_session":
            return {"session_id": "session-8c2f4a"}
        if request_type == "add_prompt":
            self.thresholds_seen.append((self.model.new_det_thresh, self.model.score_threshold_detection))
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
        self.thresholds_seen.append((self.model.new_det_thresh, self.model.score_threshold_detection))
        if self.fail_during_propagation:
            raise RuntimeError("synthetic CUDA propagation failure")
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
    assert predictor.thresholds_seen == [(0.5, 0.5), (0.5, 0.5)]
    assert predictor.model.new_det_thresh == 0.65
    assert predictor.model.score_threshold_detection == 0.4


def test_meta_backend_closes_session_when_cuda_propagation_fails() -> None:
    _, MetaSam31Backend, _, _ = _backend_api()
    predictor = RealisticPredictorDouble(fail_during_propagation=True)
    backend = MetaSam31Backend(
        checkpoint_path="/models/sam3.1_multiplex.pt",
        predictor_builder=RecordingBuilder(predictor),
    )

    with pytest.raises(RuntimeError, match="synthetic CUDA propagation failure"):
        list(backend.track("/videos/intro.mp4", prompt="person"))

    assert predictor.requests[-1] == {
        "type": "close_session",
        "session_id": "session-8c2f4a",
        "run_gc_collect": True,
    }
    assert predictor.model.new_det_thresh == 0.65
    assert predictor.model.score_threshold_detection == 0.4


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
