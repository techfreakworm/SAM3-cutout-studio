import pytest


def test_pipeline_defaults_match_finalized_workflow() -> None:
    try:
        from sam3_matting.config import MatteConfig, TrackingConfig
    except ModuleNotFoundError:
        pytest.fail("pipeline configuration has not been implemented")

    tracking = TrackingConfig()
    matte = MatteConfig()

    assert tracking.detection_threshold == 0.5
    assert tracking.max_objects == 8
    assert tracking.detect_interval == 1
    assert matte.erode_kernel == 6
    assert matte.dilate_kernel == 6
    assert matte.black_point == 0.15
    assert matte.white_point == 0.99
    assert matte.max_megapixels == 2.0


def test_tracking_threshold_must_be_a_probability() -> None:
    try:
        from sam3_matting.config import TrackingConfig
    except ModuleNotFoundError:
        pytest.fail("pipeline configuration has not been implemented")

    with pytest.raises(ValueError, match="detection_threshold"):
        TrackingConfig(detection_threshold=1.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_objects", 0), ("detect_interval", 0)],
)
def test_tracking_counts_must_be_positive(field: str, value: int) -> None:
    try:
        from sam3_matting.config import TrackingConfig
    except ModuleNotFoundError:
        pytest.fail("pipeline configuration has not been implemented")

    with pytest.raises(ValueError, match=field):
        TrackingConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("erode_kernel", 0),
        ("dilate_kernel", 0),
        ("max_megapixels", 0.0),
    ],
)
def test_matte_sizes_must_be_positive(field: str, value: int | float) -> None:
    try:
        from sam3_matting.config import MatteConfig
    except ModuleNotFoundError:
        pytest.fail("pipeline configuration has not been implemented")

    with pytest.raises(ValueError, match=field):
        MatteConfig(**{field: value})
