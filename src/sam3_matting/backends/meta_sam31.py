"""CUDA adapter for Meta's official SAM 3.1 multiplex video predictor."""

import contextlib
import inspect
import io
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import numpy as np

from .base import BackendProtocolError, SamVideoBackend, TrackedFrame
from .checkpoint_schema import partition_sam31_missing_keys

_REQUIRED_OUTPUT_FIELDS = frozenset(
    {
        "out_obj_ids",
        "out_probs",
        "out_boxes_xywh",
        "out_binary_masks",
        "frame_stats",
    }
)


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a public checkpoint omits learned state required by Meta's model."""


def _build_safetensors_predictor(
    checkpoint_path: str,
    *,
    predictor_builder: Callable[..., object],
    load_model: Callable[..., tuple[list[str], list[str]]],
    builder_kwargs: dict[str, object],
) -> object:
    """Build Meta's architecture with an empty PT then load safetensors directly."""
    import torch

    with tempfile.NamedTemporaryFile(suffix=".pt") as empty_checkpoint:
        torch.save({}, empty_checkpoint.name)
        quiet_builder_output = io.StringIO()
        with contextlib.redirect_stdout(quiet_builder_output):
            predictor = predictor_builder(
                checkpoint_path=empty_checkpoint.name,
                **builder_kwargs,
            )

    missing_keys, unexpected_keys = load_model(
        predictor.model,
        checkpoint_path,
        strict=False,
        device="cpu",
    )
    allowed_missing, blocking_missing = partition_sam31_missing_keys(missing_keys)
    if blocking_missing:
        preview = ", ".join(blocking_missing[:5])
        raise CheckpointCompatibilityError(f"checkpoint is missing learned keys: {preview}")
    if unexpected_keys:
        preview = ", ".join(sorted(unexpected_keys)[:5])
        raise CheckpointCompatibilityError(f"checkpoint has unexpected keys: {preview}")

    text_projection_key = "detector.backbone.language_backbone.encoder.text_projection"
    if text_projection_key in allowed_missing:
        try:
            text_projection = predictor.model.detector.backbone.language_backbone.encoder.text_projection
        except AttributeError as exc:
            raise CheckpointCompatibilityError(
                "could not initialize the absent Meta text_projection"
            ) from exc
        with torch.no_grad():
            text_projection.zero_()

    return predictor


def _adapt_legacy_state_offload(predictor: object) -> object:
    """Bridge the predictor's legacy session kwarg to the multiplex model API."""
    model = predictor.model
    original_init_state = model.init_state
    if "offload_state_to_cpu" in inspect.signature(original_init_state).parameters:
        return predictor

    def compatible_init_state(
        *args: object,
        offload_state_to_cpu: bool = False,
        **kwargs: object,
    ) -> object:
        if offload_state_to_cpu:
            raise ValueError(
                "offload_state_to_cpu=True is not supported by the pinned Meta SAM 3.1 multiplex model"
            )
        return original_init_state(*args, **kwargs)

    model.init_state = compatible_init_state
    return predictor


def _default_predictor_builder(**kwargs: object) -> object:
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
    except ImportError as exc:
        raise RuntimeError("Meta SAM 3.1 is not installed; install the pinned sam3 dependency") from exc

    checkpoint_path = str(kwargs["checkpoint_path"])
    if Path(checkpoint_path).suffix == ".safetensors":
        from safetensors.torch import load_model

        builder_kwargs = dict(kwargs)
        del builder_kwargs["checkpoint_path"]
        predictor = _build_safetensors_predictor(
            checkpoint_path,
            predictor_builder=build_sam3_multiplex_video_predictor,
            load_model=load_model,
            builder_kwargs=builder_kwargs,
        )
    else:
        predictor = build_sam3_multiplex_video_predictor(**kwargs)
    return _adapt_legacy_state_offload(predictor)


@contextmanager
def _temporary_detection_threshold(predictor: object, threshold: float) -> Iterator[None]:
    model = getattr(predictor, "model", None)
    original_values: dict[str, object] = {}
    if model is not None:
        for attribute in ("new_det_thresh", "score_threshold_detection"):
            if hasattr(model, attribute):
                original_values[attribute] = getattr(model, attribute)
                setattr(model, attribute, threshold)
    try:
        yield
    finally:
        if model is not None:
            for attribute, value in original_values.items():
                setattr(model, attribute, value)


def _tracked_frame(response: object) -> TrackedFrame:
    if not isinstance(response, Mapping):
        raise BackendProtocolError("Meta predictor frame response must be a mapping")
    if "frame_index" not in response or "outputs" not in response:
        raise BackendProtocolError("Meta predictor response requires frame_index and outputs")

    outputs = response["outputs"]
    if not isinstance(outputs, Mapping):
        raise BackendProtocolError("Meta predictor outputs must be a mapping")
    missing_fields = sorted(_REQUIRED_OUTPUT_FIELDS.difference(outputs))
    if missing_fields:
        raise BackendProtocolError(
            "Meta predictor outputs missing required fields: " + ", ".join(missing_fields)
        )

    frame_index = response["frame_index"]
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise BackendProtocolError("Meta predictor frame_index must be a non-negative integer")

    object_ids = np.asarray(outputs["out_obj_ids"], dtype=np.int64)
    scores = np.asarray(outputs["out_probs"], dtype=np.float32)
    boxes_xywh = np.asarray(outputs["out_boxes_xywh"], dtype=np.float32)
    object_masks = np.asarray(outputs["out_binary_masks"], dtype=np.bool_)

    if object_ids.ndim != 1:
        raise BackendProtocolError("out_obj_ids must have shape [objects]")
    object_count = object_ids.shape[0]
    if scores.shape != (object_count,):
        raise BackendProtocolError("out_probs must have shape [objects]")
    if boxes_xywh.shape != (object_count, 4):
        raise BackendProtocolError("out_boxes_xywh must have shape [objects, 4]")
    if object_masks.ndim != 3 or object_masks.shape[0] != object_count:
        raise BackendProtocolError("out_binary_masks must have shape [objects, height, width]")

    frame_stats = outputs["frame_stats"]
    if frame_stats is not None and not isinstance(frame_stats, Mapping):
        raise BackendProtocolError("frame_stats must be a mapping or None")

    union_mask = np.any(object_masks, axis=0)
    return TrackedFrame(
        frame_index=frame_index,
        object_ids=np.array(object_ids, copy=True),
        scores=np.array(scores, copy=True),
        boxes_xywh=np.array(boxes_xywh, copy=True),
        object_masks=np.array(object_masks, copy=True),
        union_mask=np.array(union_mask, copy=True),
        frame_stats=dict(frame_stats) if frame_stats is not None else None,
    )


class MetaSam31Backend(SamVideoBackend):
    """Run the official stateful SAM 3.1 multiplex predictor on one CUDA worker."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        max_objects: int = 8,
        device: str = "cuda",
        predictor_builder: Callable[..., object] | None = None,
    ) -> None:
        if device != "cuda":
            raise ValueError("MetaSam31Backend is CUDA-only; use a separate MPS-compatible backend")
        if max_objects < 1:
            raise ValueError("max_objects must be at least 1")

        self.checkpoint_path = str(checkpoint_path)
        self.max_objects = max_objects
        self.device = device
        self._predictor_builder = predictor_builder or _default_predictor_builder
        self._predictor: object | None = None
        self._inference_lock = Lock()

    def _get_predictor(self) -> object:
        if self._predictor is None:
            self._predictor = self._predictor_builder(
                checkpoint_path=self.checkpoint_path,
                max_num_objects=self.max_objects,
                multiplex_count=16,
                use_fa3=False,
                use_rope_real=True,
                compile=False,
                warm_up=False,
                session_expiration_sec=1200,
                default_output_prob_thresh=0.5,
                async_loading_frames=True,
            )
        return self._predictor

    def track(
        self,
        video_path: str,
        *,
        prompt: str,
        detection_threshold: float = 0.5,
    ) -> Iterator[TrackedFrame]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 0.0 <= detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be between 0 and 1")

        with self._inference_lock:
            predictor = self._get_predictor()
            start_response = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_path),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False,
                }
            )
            if not isinstance(start_response, Mapping) or not isinstance(
                start_response.get("session_id"), str
            ):
                raise BackendProtocolError("Meta predictor start_session requires a session_id")
            session_id = start_response["session_id"]

            try:
                with _temporary_detection_threshold(predictor, detection_threshold):
                    prompt_response = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": prompt,
                            "output_prob_thresh": detection_threshold,
                        }
                    )
                    prompted_frame = _tracked_frame(prompt_response)
                    seen_frame_indices = {prompted_frame.frame_index}
                    yield prompted_frame

                    stream = predictor.handle_stream_request(
                        {
                            "type": "propagate_in_video",
                            "session_id": session_id,
                            "propagation_direction": "forward",
                            "start_frame_index": 0,
                            "output_prob_thresh": detection_threshold,
                        }
                    )
                    for response in stream:
                        frame = _tracked_frame(response)
                        if frame.frame_index in seen_frame_indices:
                            continue
                        seen_frame_indices.add(frame.frame_index)
                        yield frame
            finally:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )
