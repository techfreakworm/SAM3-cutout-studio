import asyncio
from collections.abc import Sequence
from pathlib import Path

import gradio as gr
import httpx
import pytest


def _component_by_elem_id(config: dict, elem_id: str) -> dict:
    for component in config["components"]:
        if component.get("props", {}).get("elem_id") == elem_id:
            return component
    pytest.fail(f"Gradio component #{elem_id} was not rendered")


def _always_valid(*values: object) -> tuple[dict[str, object], ...]:
    return tuple(gr.validate(True, "") for _ in values)


def test_product_name_has_migrated_to_sam3_cutout_studio() -> None:
    source = (Path(__file__).parents[1] / "src" / "sam3_matting" / "ui.py").read_text()

    assert "SAM3 Cutout Studio" in source
    assert "SAM3 Pro Matte Studio" not in source
    assert "ALPHA / 16-BIT" not in source
    assert "MPS PLANNED" in source


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
        validator_fn=_always_valid,
        hosted=False,
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
        "max-objects": ("slider", "Maximum objects per prompt clause", 8),
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
    assert demo.fns[dependency["id"]].validator is _always_valid
    assert demo.delete_cache == (600, 3600)
    assert demo.api_open is False


def test_run_event_invokes_the_injected_real_handler(tmp_path: Path) -> None:
    from sam3_matting.ui import build_ui

    observed: list[object] = []
    outputs = tuple(str(tmp_path / name) for name in ("preview.mp4", "master.mov", "matte.mp4"))

    def process_fn(*values: object) -> tuple[str, str, str, str]:
        observed.extend(values)
        return *outputs, "Render complete · 72 frames"

    demo = build_ui(process_fn, validator_fn=_always_valid, hosted=False)
    values = ["source.mp4", "person, hair", 0.5, 8, 1, 6, 6, 0.15, 0.99, 2.0]
    result = asyncio.run(demo.call_function(0, values))

    assert observed == values
    assert result["prediction"] == (*outputs, "Render complete · 72 frames")


def test_direct_named_api_cannot_bypass_hosted_oauth_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gradio.oauth

    from sam3_matting.ui import build_ui

    calls: list[str] = []
    gradio_cache = tmp_path / "gradio"
    source_video = gradio_cache / "source.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"test upload")
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(gradio_cache))

    def process_fn(*_values: object) -> tuple[str, str, str, str]:
        calls.append("inference")
        return "preview.mp4", "master.mov", "matte.mp4", "Complete"

    def validator_fn(
        source_video: str,
        prompt: str,
        detection_threshold: float,
        max_objects: int,
        detect_interval: int,
        erode_kernel: int,
        dilate_kernel: int,
        black_point: float,
        white_point: float,
        max_megapixels: float,
        oauth_profile: gr.OAuthProfile,
    ) -> tuple[dict[str, object], ...]:
        del (
            source_video,
            prompt,
            detection_threshold,
            max_objects,
            detect_interval,
            erode_kernel,
            dilate_kernel,
            black_point,
            white_point,
            max_megapixels,
            oauth_profile,
        )
        calls.append("validator")
        return tuple(gr.validate(True, "") for _ in range(10))

    monkeypatch.setattr(
        gradio.oauth,
        "_get_mocked_oauth_info",
        lambda: {"userinfo": {"preferred_username": "test-user"}},
    )
    with pytest.warns(UserWarning, match="Gradio does not support OAuth"):
        demo = build_ui(
            process_fn,
            validator_fn=validator_fn,
            hosted=True,
        )

    async def post_directly() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=demo.app),
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/gradio_api/api/remove_background",
                json={
                    "data": [
                        {
                            "path": str(source_video),
                            "meta": {"_type": "gradio.FileData"},
                        },
                        "person",
                        0.5,
                        8,
                        1,
                        6,
                        6,
                        0.15,
                        0.99,
                        2.0,
                    ]
                },
            )

    response = asyncio.run(post_directly())

    assert response.status_code == 404
    assert "does not accept direct HTTP POST requests" in response.text
    assert calls == []


def test_hosted_studio_requires_login_and_explains_three_clause_xlarge_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gradio.oauth

    from sam3_matting.ui import build_ui

    monkeypatch.setattr(
        gradio.oauth,
        "_get_mocked_oauth_info",
        lambda: {"userinfo": {"preferred_username": "test-user"}},
    )
    with pytest.warns(UserWarning, match="Gradio does not support OAuth"):
        demo = build_ui(
            lambda *_values: ("preview.mp4", "master.mov", "matte.mp4", "Complete"),
            validator_fn=_always_valid,
            hosted=True,
        )
    config = demo.get_config_file()

    login = _component_by_elem_id(config, "studio-login")
    assert login["type"] == "button"
    login_component = next(
        block for block in demo.blocks.values() if getattr(block, "elem_id", None) == "studio-login"
    )
    assert isinstance(login_component, gr.LoginButton)
    hosted_cue = _component_by_elem_id(config, "hosted-cue")
    hosted_copy = hosted_cue["props"]["value"]
    assert "Sign in" in hosted_copy
    assert "3 prompt clauses" in hosted_copy
    assert "96 GB" in hosted_copy
    assert "xlarge" in hosted_copy

    prompt = _component_by_elem_id(config, "studio-prompt")
    assert "3" in prompt["props"]["info"]


def test_local_studio_has_no_login_control_and_allows_four_clauses() -> None:
    from sam3_matting.ui import build_ui

    demo = build_ui(
        lambda *_values: ("preview.mp4", "master.mov", "matte.mp4", "Complete"),
        validator_fn=_always_valid,
        hosted=False,
    )
    config = demo.get_config_file()

    assert all(
        component.get("props", {}).get("elem_id") != "studio-login" for component in config["components"]
    )
    prompt = _component_by_elem_id(config, "studio-prompt")
    assert "4" in prompt["props"]["info"]


def test_outputs_already_in_gradio_cache_are_registered_without_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_matting.ui import build_ui

    gradio_cache = tmp_path / "gradio"
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(gradio_cache))
    output_dir = gradio_cache / "sam3-cutout-studio" / "job-test"
    output_dir.mkdir(parents=True)
    outputs = tuple(output_dir / name for name in ("preview.mp4", "master.mov", "matte.mp4"))
    for output in outputs:
        output.write_bytes(b"render")

    demo = build_ui(
        lambda *_values: (*map(str, outputs), "Complete"),
        validator_fn=_always_valid,
        hosted=False,
    )
    block_fn = demo.fns[0]
    predictions = asyncio.run(
        demo.postprocess_data(
            block_fn,
            [*map(str, outputs), "Complete"],
            None,
        )
    )

    predicted_paths = [prediction["path"] for prediction in predictions[:3]]
    assert predicted_paths == [str(output.resolve()) for output in outputs]
    for component, output in zip(block_fn.outputs[:3], outputs, strict=True):
        assert str(output.resolve()) in component.temp_files


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
    assert ".studio-masthead-grid" in css
    assert ".darkroom-masthead" not in css
