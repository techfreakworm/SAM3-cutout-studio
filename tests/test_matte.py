import cv2
import numpy as np
import pytest
import torch
from PIL import Image


def matte_module():
    try:
        from sam3_matting import matte
    except ModuleNotFoundError:
        pytest.fail("matte preprocessing has not been implemented")
    return matte


def test_generate_trimap_uses_a_strict_half_threshold() -> None:
    mask = np.array([[0.49, 0.5, 0.50001, 1.0]], dtype=np.float32)

    trimap = matte_module().generate_trimap(
        mask,
        erode_kernel=1,
        dilate_kernel=1,
    )

    np.testing.assert_array_equal(
        trimap,
        np.array([[0, 0, 255, 255]], dtype=np.uint8),
    )


def test_generate_trimap_applies_five_morphology_iterations() -> None:
    mask = np.zeros((31, 31), dtype=np.float32)
    mask[10:21, 10:21] = 1.0
    expected_binary = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    expected_eroded = cv2.erode(expected_binary, kernel, iterations=5)
    expected_dilated = cv2.dilate(expected_binary, kernel, iterations=5)
    expected = np.zeros_like(expected_binary)
    expected[expected_dilated == 255] = 128
    expected[expected_eroded == 255] = 255

    trimap = matte_module().generate_trimap(
        mask,
        erode_kernel=3,
        dilate_kernel=3,
    )

    np.testing.assert_array_equal(trimap, expected)
    assert trimap[15, 15] == 255
    assert trimap[5, 5] == 128
    assert trimap[4, 4] == 0


def test_resize_to_megapixel_budget_preserves_aspect_ratio() -> None:
    image_np = np.zeros((100, 200, 3), dtype=np.uint8)
    image_np[:, 100:] = 255
    trimap_np = np.zeros((100, 200), dtype=np.uint8)
    trimap_np[:, 100:] = 255
    image = Image.fromarray(image_np, mode="RGB")
    trimap = Image.fromarray(trimap_np, mode="L")
    ten_thousand_pixels_in_megapixels = 10_000 / 1_048_576

    resized_image, resized_trimap = matte_module().resize_to_megapixel_budget(
        image,
        trimap,
        max_megapixels=ten_thousand_pixels_in_megapixels,
    )

    assert resized_image.size == (141, 70)
    assert resized_trimap.size == (141, 70)
    assert resized_image.size[0] * resized_image.size[1] <= 10_000


def test_resize_to_megapixel_budget_keeps_images_within_budget() -> None:
    image = Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8), mode="RGB")
    trimap = Image.fromarray(np.zeros((8, 12), dtype=np.uint8), mode="L")

    resized_image, resized_trimap = matte_module().resize_to_megapixel_budget(
        image,
        trimap,
        max_megapixels=1.0,
    )

    assert resized_image is image
    assert resized_trimap is trimap


def test_histogram_remap_applies_black_and_white_points() -> None:
    alpha = np.array([0.0, 0.15, 0.57, 0.99, 1.0], dtype=np.float32)

    remapped = matte_module().histogram_remap(
        alpha,
        black_point=0.15,
        white_point=0.99,
    )

    np.testing.assert_allclose(
        remapped,
        np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_histogram_remap_clamps_black_point_below_white_point() -> None:
    alpha = np.array([0.799, 0.7995, 0.8], dtype=np.float32)

    remapped = matte_module().histogram_remap(
        alpha,
        black_point=0.9,
        white_point=0.8,
    )

    np.testing.assert_allclose(remapped, np.array([0.0, 0.5, 1.0], dtype=np.float32), atol=5e-5)


def test_refiner_construction_is_lazy_and_keeps_explicit_device() -> None:
    refiner = matte_module().VitMatteRefiner(device="mps")

    assert refiner.device == "mps"
    assert refiner.model is None
    assert refiner.processor is None


def test_refine_lazily_loads_once_and_normalizes_uint8_masks(monkeypatch) -> None:
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    record = {"model_loads": [], "processor_loads": []}

    class FakeProcessor:
        def __call__(self, *, images, trimaps, return_tensors):
            record["image_mode"] = images.mode
            record["trimap"] = np.asarray(trimaps).copy()
            height, width = np.asarray(images).shape[:2]
            return {"pixel_values": torch.zeros((1, 3, height, width))}

    class FakeModel:
        def __init__(self) -> None:
            self.device = None
            self.eval_called = False
            self.call_count = 0

        def to(self, device):
            self.device = str(device)
            return self

        def eval(self) -> None:
            self.eval_called = True

        def __call__(self, **inputs):
            self.call_count += 1
            height, width = next(iter(inputs.values())).shape[-2:]
            alphas = torch.linspace(
                0.0,
                1.0,
                height * width,
                dtype=torch.bfloat16,
            ).reshape(1, 1, height, width)
            return type("Prediction", (), {"alphas": alphas})()

    processor = FakeProcessor()
    model = FakeModel()

    def load_processor(model_id):
        record["processor_loads"].append(model_id)
        return processor

    def load_model(model_id):
        record["model_loads"].append(model_id)
        return model

    monkeypatch.setattr(VitMatteImageProcessor, "from_pretrained", staticmethod(load_processor))
    monkeypatch.setattr(VitMatteForImageMatting, "from_pretrained", staticmethod(load_model))

    refiner = matte_module().VitMatteRefiner(
        device="cpu",
        erode_kernel=1,
        dilate_kernel=1,
        black_point=0.0,
        white_point=1.0,
        max_megapixels=1.0,
    )
    image = Image.fromarray(np.full((2, 3, 3), 127, dtype=np.uint8))
    mask = np.array([[0, 127, 128], [255, 0, 255]], dtype=np.uint8)

    first_alpha = refiner.refine(image, mask)
    second_alpha = refiner.refine(image, mask)

    model_id = "hustvl/vitmatte-small-composition-1k"
    assert record["processor_loads"] == [model_id]
    assert record["model_loads"] == [model_id]
    assert record["image_mode"] == "RGB"
    np.testing.assert_array_equal(
        record["trimap"],
        np.array([[0, 0, 255], [255, 0, 255]], dtype=np.uint8),
    )
    expected_alpha = torch.linspace(0.0, 1.0, 6, dtype=torch.bfloat16).float().numpy().reshape(2, 3)
    np.testing.assert_allclose(first_alpha, expected_alpha)
    np.testing.assert_allclose(second_alpha, expected_alpha)
    assert model.device == "cpu"
    assert model.eval_called
    assert model.call_count == 2


def test_cleanup_releases_loaded_resources() -> None:
    refiner = matte_module().VitMatteRefiner(device="cpu")
    refiner.model = object()
    refiner.processor = object()

    refiner.cleanup()

    assert refiner.model is None
    assert refiner.processor is None
