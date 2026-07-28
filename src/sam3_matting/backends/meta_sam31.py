"""CUDA adapter for Meta's official SAM 3.1 multiplex video predictor."""

import contextlib
import inspect
import io
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from threading import Lock

import numpy as np

from .base import BackendProtocolError, SamVideoBackend, TrackedFrame
from .checkpoint_schema import partition_sam31_missing_keys

logger = logging.getLogger(__name__)

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
    """Build Meta's architecture with a stubbed torch.load, then load safetensors."""
    from unittest import mock

    import torch

    # Meta's builder only consumes .pt checkpoints through torch.load, but the
    # real weights arrive through safetensors below, so construction runs against
    # an empty state dict. A torch.save/torch.load roundtrip is unreliable under
    # ZeroGPU's torch patching, so torch.load is stubbed for the construction
    # call instead of writing an empty checkpoint to disk. The string path is
    # still forwarded so builder diagnostics stay truthful.
    quiet_builder_output = io.StringIO()
    with contextlib.redirect_stdout(quiet_builder_output), mock.patch.object(torch, "load", return_value={}):
        predictor = predictor_builder(
            checkpoint_path=checkpoint_path,
            **builder_kwargs,
        )

    try:
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
    except BaseException:
        _cleanup_upstream_autocast_after_build_failure(predictor)
        raise

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


def _is_safetensors_checkpoint(checkpoint_path: str) -> bool:
    """Detect the safetensors container by filename or content.

    Xet-backed Hub downloads resolve to content-addressed blob paths that drop
    the filename suffix, so the header is sniffed when the suffix is absent.
    """
    if checkpoint_path.endswith(".safetensors"):
        return True
    try:
        with open(checkpoint_path, "rb") as handle:
            prefix = handle.read(9)
    except OSError:
        return False
    # safetensors: 8-byte little-endian JSON header length, then '{'.
    return len(prefix) == 9 and prefix[8:] == b"{" and not prefix.startswith(b"PK\x03\x04")


def _default_predictor_builder(**kwargs: object) -> object:
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
    except ImportError as exc:
        raise RuntimeError("Meta SAM 3.1 is not installed; install the pinned sam3 dependency") from exc

    checkpoint_path = str(kwargs["checkpoint_path"])
    if _is_safetensors_checkpoint(checkpoint_path):
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
    try:
        return _adapt_legacy_state_offload(predictor)
    except BaseException:
        _cleanup_upstream_autocast_after_build_failure(predictor)
        raise


@contextmanager
def _cuda_bfloat16_autocast() -> Iterator[None]:
    import torch

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yield


def _exit_upstream_autocast_contexts(predictor: object) -> None:
    """Undo lifetime autocast contexts entered by the pinned Meta constructors."""
    model = getattr(predictor, "model", None)
    tracker = getattr(model, "tracker", None)
    tracker_model = getattr(tracker, "model", None)

    # The pinned multiplex graph owns contexts on predictor and tracker. Keep
    # tracker_model support for the older three-owner graph used by legacy builds.
    seen_contexts: set[int] = set()
    exit_errors: list[BaseException] = []
    for owner in (predictor, tracker, tracker_model):
        context = getattr(owner, "bf16_context", None)
        context_id = id(context)
        if context is None or context_id in seen_contexts:
            continue
        seen_contexts.add(context_id)
        exit_context = getattr(context, "__exit__", None)
        if not callable(exit_context):
            continue
        try:
            exit_context(None, None, None)
        except BaseException as exc:
            exit_errors.append(exc)

    if exit_errors:
        for secondary_error in exit_errors[1:]:
            logger.error(
                "additional upstream autocast context failed to exit",
                exc_info=(
                    type(secondary_error),
                    secondary_error,
                    secondary_error.__traceback__,
                ),
            )
        raise exit_errors[0]


def _cleanup_upstream_autocast_after_build_failure(predictor: object) -> None:
    """Best-effort unwind without replacing the active predictor-build error."""
    try:
        _exit_upstream_autocast_contexts(predictor)
    except BaseException:
        logger.exception("autocast cleanup failed while preserving predictor build error")


@contextmanager
def _temporary_tracking_settings(
    predictor: object,
    *,
    detection_threshold: float,
    detect_interval: int,
    max_objects: int,
) -> Iterator[None]:
    model = getattr(predictor, "model", None)
    original_values: dict[str, object] = {}
    if model is not None:
        requested_values = {
            "new_det_thresh": detection_threshold,
            "score_threshold_detection": detection_threshold,
            "recondition_every_nth_frame": detect_interval,
            "max_num_objects": max_objects,
        }
        for attribute, requested_value in requested_values.items():
            if hasattr(model, attribute):
                original_values[attribute] = getattr(model, attribute)
                setattr(model, attribute, requested_value)
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
        self._predictor_load_lock = Lock()
        self._inference_lock = Lock()

    def _get_predictor(self) -> object:
        if self._predictor is None:
            with self._predictor_load_lock:
                if self._predictor is None:
                    predictor = self._predictor_builder(
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
                    _exit_upstream_autocast_contexts(predictor)
                    self._predictor = predictor
        return self._predictor

    def track(
        self,
        video_path: str,
        *,
        prompt: str,
        detection_threshold: float = 0.5,
        detect_interval: int = 1,
        max_objects: int = 8,
    ) -> Iterator[TrackedFrame]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 0.0 <= detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be between 0 and 1")
        if detect_interval < 1:
            raise ValueError("detect_interval must be at least 1")
        if not 1 <= max_objects <= 8:
            raise ValueError("max_objects must be between 1 and 8")

        # Meta's constructors enter lifetime autocast contexts on their construction
        # thread. _get_predictor unwinds those; each CUDA operation below receives a
        # fresh request-thread context that exits before control is yielded.
        with self._inference_lock:
            predictor = self._get_predictor()
            with _temporary_tracking_settings(
                predictor,
                detection_threshold=detection_threshold,
                detect_interval=detect_interval,
                max_objects=max_objects,
            ):
                with _cuda_bfloat16_autocast():
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

                def close_session() -> None:
                    with _cuda_bfloat16_autocast():
                        predictor.handle_request(
                            {
                                "type": "close_session",
                                "session_id": session_id,
                                "run_gc_collect": True,
                            }
                        )

                try:
                    with _cuda_bfloat16_autocast():
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

                    stream = iter(
                        predictor.handle_stream_request(
                            {
                                "type": "propagate_in_video",
                                "session_id": session_id,
                                "propagation_direction": "forward",
                                "start_frame_index": 0,
                                "output_prob_thresh": detection_threshold,
                            }
                        )
                    )
                    while True:
                        try:
                            with _cuda_bfloat16_autocast():
                                response = next(stream)
                        except StopIteration:
                            break
                        frame = _tracked_frame(response)
                        if frame.frame_index in seen_frame_indices:
                            continue
                        seen_frame_indices.add(frame.frame_index)
                        yield frame
                except GeneratorExit:
                    close_session()
                    raise
                except BaseException:
                    try:
                        close_session()
                    except BaseException:
                        logger.exception("close_session failed while preserving the primary inference error")
                    raise
                else:
                    close_session()
