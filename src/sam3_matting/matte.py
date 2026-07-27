"""ViTMatte preprocessing and alpha refinement."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

MORPHOLOGY_ITERATIONS = 5
PIXELS_PER_MEGAPIXEL = 1_048_576


def generate_trimap(
    mask: np.ndarray,
    *,
    erode_kernel: int = 6,
    dilate_kernel: int = 6,
) -> np.ndarray:
    """Convert a normalized mask to the 0/128/255 trimap used by the final workflow."""
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    erode_structuring_element = np.ones((erode_kernel, erode_kernel), dtype=np.uint8)
    dilate_structuring_element = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)

    eroded = cv2.erode(
        mask_binary,
        erode_structuring_element,
        iterations=MORPHOLOGY_ITERATIONS,
    )
    dilated = cv2.dilate(
        mask_binary,
        dilate_structuring_element,
        iterations=MORPHOLOGY_ITERATIONS,
    )

    trimap = np.zeros_like(mask_binary)
    trimap[dilated == 255] = 128
    trimap[eroded == 255] = 255
    return trimap


def resize_to_megapixel_budget(
    image: Image.Image,
    trimap: Image.Image,
    *,
    max_megapixels: float,
) -> tuple[Image.Image, Image.Image]:
    """Resize an image and trimap using the finalized workflow's pixel-budget formula."""
    width, height = image.size
    max_pixels = max_megapixels * PIXELS_PER_MEGAPIXEL
    if width * height <= max_pixels:
        return image, trimap

    aspect_ratio = width / height
    target_width = int(math.sqrt(aspect_ratio * max_pixels))
    target_height = int(target_width / aspect_ratio)
    target_size = (target_width, target_height)
    return (
        image.resize(target_size, Image.Resampling.BILINEAR),
        trimap.resize(target_size, Image.Resampling.BILINEAR),
    )


def histogram_remap(
    alpha: np.ndarray,
    *,
    black_point: float,
    white_point: float,
) -> np.ndarray:
    """Map alpha values between black and white points and clip them to [0, 1]."""
    effective_black_point = min(black_point, white_point - 0.001)
    scale = 1.0 / (white_point - effective_black_point)
    return np.clip((alpha - effective_black_point) * scale, 0.0, 1.0)


class VitMatteRefiner:
    """Lazy ViTMatte model wrapper bound to one explicit execution device."""

    MODEL_ID = "hustvl/vitmatte-small-composition-1k"
    MODEL_REVISION = "6a58ad7646403c1df626fbd746900aec7361ea1d"

    def __init__(
        self,
        *,
        device: str,
        erode_kernel: int = 6,
        dilate_kernel: int = 6,
        black_point: float = 0.15,
        white_point: float = 0.99,
        max_megapixels: float = 2.0,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
    ) -> None:
        self.device = device
        self.erode_kernel = erode_kernel
        self.dilate_kernel = dilate_kernel
        self.black_point = black_point
        self.white_point = white_point
        self.max_megapixels = max_megapixels
        self.model_id = model_id
        self.model_revision = model_revision
        self.model: object | None = None
        self.processor: object | None = None

    def refine(self, image: Image.Image | np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Refine one binary mask into a soft alpha matte."""
        import torch

        self._load_model()

        image_pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        mask_array = np.asarray(mask)
        if mask_array.max() > 1.0:
            mask_array = mask_array / 255.0
        trimap = generate_trimap(
            mask_array,
            erode_kernel=self.erode_kernel,
            dilate_kernel=self.dilate_kernel,
        )
        trimap_pil = Image.fromarray(trimap).convert("L")

        original_width, original_height = image_pil.size
        image_resized, trimap_resized = resize_to_megapixel_budget(
            image_pil,
            trimap_pil,
            max_megapixels=self.max_megapixels,
        )
        inputs = self.processor(
            images=image_resized,
            trimaps=trimap_resized,
            return_tensors="pt",
        )
        torch_device = torch.device(self.device)
        inputs = {name: value.to(torch_device) for name, value in inputs.items()}

        with torch.no_grad():
            predictions = self.model(**inputs).alphas

        alpha = predictions[0, 0].detach().float().cpu().numpy()
        alpha = alpha[: image_resized.height, : image_resized.width]

        if image_resized.size != image_pil.size:
            alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8))
            alpha = (
                np.asarray(
                    alpha_pil.resize(
                        (original_width, original_height),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )

        return histogram_remap(
            alpha,
            black_point=self.black_point,
            white_point=self.white_point,
        )

    def _load_model(self) -> None:
        if self.model is not None:
            return

        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self.processor = VitMatteImageProcessor.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        self.model = VitMatteForImageMatting.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        self.model.to(torch.device(self.device))
        self.model.eval()

    def cleanup(self) -> None:
        """Release model resources and accelerator caches."""
        import gc

        import torch

        model = self.model
        processor = self.processor
        self.model = None
        self.processor = None
        del model, processor

        torch_device = torch.device(self.device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch_device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
