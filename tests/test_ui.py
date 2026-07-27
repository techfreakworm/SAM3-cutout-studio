import asyncio
from collections.abc import Sequence
from pathlib import Path

import gradio as gr
import pytest


def _component_by_elem_id(config: dict, elem_id: str) -> dict:
    for component in config["components"]:
        if component.get("props", {}).get("elem_id") == elem_id:
            return component
    pytest.fail(f"Gradio component #{elem_id} was not rendered")


def test_product_name_has_migrated_to_sam3_cutout_studio() -> None:
    source = (Path(__file__).parents[1] / "src" / "sam3_matting" / "ui.py").read_text()

    assert "SAM3 Cutout Studio" in source
    assert "SAM3 Pro Matte Studio" not in source


def test_studio_blocks_expose_the_workflow_controls_and_outputs() -> None:
    try:
        from sam3_matting.ui import build_ui
    except ModuleNotFoundError:
        pytest.fail("the Gradio studio shell has not been implemented")

    def process_fn(*values: object) -> Sequence[str]:
        assert len(values) == 10
        return "preview.mp4", "master.mov", "matte.mp4", "Render complete"

    demo = build_ui(
        process_fn,
        runtime_status={
            "device": "CUDA · NVIDIA RTX PRO 6000",
            "cuda": "Ready",
            "mps": "Portable",
            "zerogpu": "Compatible",
        },
    )

    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    assert config["title"] == "SAM3 Cutout Studio"
    assert "SAM3 Pro Matte Studio" not in str(config)

    expected_components = {
        "studio-video": ("video", "Source footage", None),
        "studio-prompt": ("textbox", "Subject prompt", None),
        "detection-threshold": ("slider", "Detection threshold", 0.5),
        "max-objects": ("slider", "Maximum objects", 8),
        "detect-interval": ("slider", "Detection interval", 1),
        "erode-kernel": ("slider", "Erode kernel", 6),
        "dilate-kernel": ("slider", "Dilate kernel", 6),
        "black-point": ("slider", "Black point", 0.15),
        "white-point": ("slider", "White point", 0.99),
        "max-megapixels": ("slider", "ViTMatte megapixel budget", 2.0),
        "result-preview": ("video", "Composite preview", None),
        "result-master": ("file", "Transparent master", None),
        "result-matte": ("file", "Alpha matte", None),
        "job-status": ("markdown", None, None),
    }
    for elem_id, (component_type, label, value) in expected_components.items():
        component = _component_by_elem_id(config, elem_id)
        assert component["type"] == component_type
        if label is not None:
            assert component["props"]["label"] == label
        if value is not None:
            assert component["props"]["value"] == value

    advanced = _component_by_elem_id(config, "advanced-controls")
    assert advanced["type"] == "accordion"
    assert advanced["props"]["open"] is False

    status_chrome = _component_by_elem_id(config, "runtime-status")
    status_html = status_chrome["props"]["value"]
    assert "CUDA" in status_html
    assert "MPS" in status_html
    assert "ZeroGPU" in status_html
    assert "NVIDIA RTX PRO 6000" in status_html

    run_button = _component_by_elem_id(config, "studio-run")
    dependency = next(
        item for item in config["dependencies"] if (run_button["id"], "click") in item["targets"]
    )
    expected_input_ids = [
        _component_by_elem_id(config, elem_id)["id"]
        for elem_id in (
            "studio-video",
            "studio-prompt",
            "detection-threshold",
            "max-objects",
            "detect-interval",
            "erode-kernel",
            "dilate-kernel",
            "black-point",
            "white-point",
            "max-megapixels",
        )
    ]
    assert dependency["inputs"] == expected_input_ids
    assert dependency["outputs"] == [
        _component_by_elem_id(config, elem_id)["id"]
        for elem_id in ("result-preview", "result-master", "result-matte", "job-status")
    ]
    assert dependency["queue"] is True


def test_run_event_invokes_the_injected_real_handler(tmp_path: Path) -> None:
    from sam3_matting.ui import build_ui

    observed: list[object] = []
    outputs = tuple(str(tmp_path / name) for name in ("preview.mp4", "master.mov", "matte.mp4"))

    def process_fn(*values: object) -> tuple[str, str, str, str]:
        observed.extend(values)
        return *outputs, "Render complete · 72 frames"

    demo = build_ui(process_fn)
    values = ["source.mp4", "person, hair", 0.5, 8, 1, 6, 6, 0.15, 0.99, 2.0]
    result = asyncio.run(demo.call_function(0, values))

    assert observed == values
    assert result["prediction"] == (*outputs, "Render complete · 72 frames")


def test_launch_assets_include_responsive_and_reduced_motion_treatment() -> None:
    from sam3_matting.ui import studio_launch_kwargs

    kwargs = studio_launch_kwargs()
    css_path = Path(kwargs["css_paths"])

    assert css_path.name == "studio.css"
    assert css_path.is_file()
    css = css_path.read_text()
    assert "@media (max-width: 760px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "Bricolage Grotesque" in kwargs["head"]
