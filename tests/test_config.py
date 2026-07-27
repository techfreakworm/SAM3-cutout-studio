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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detection_threshold", True),
        ("detection_threshold", "0.5"),
        ("detection_threshold", float("nan")),
        ("detection_threshold", float("inf")),
        ("detection_threshold", 0.049),
        ("detection_threshold", 0.951),
        ("max_objects", True),
        ("max_objects", 1.0),
        ("max_objects", "8"),
        ("max_objects", 0),
        ("max_objects", 9),
        ("detect_interval", False),
        ("detect_interval", 1.0),
        ("detect_interval", "1"),
        ("detect_interval", 0),
        ("detect_interval", 31),
    ],
)
def test_tracking_config_rejects_hostile_api_values(field: str, value: object) -> None:
    from sam3_matting.config import TrackingConfig

    with pytest.raises((TypeError, ValueError), match=field):
        TrackingConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("erode_kernel", True),
        ("erode_kernel", 1.0),
        ("erode_kernel", "6"),
        ("erode_kernel", 0),
        ("erode_kernel", 32),
        ("dilate_kernel", False),
        ("dilate_kernel", 1.0),
        ("dilate_kernel", "6"),
        ("dilate_kernel", 0),
        ("dilate_kernel", 32),
        ("black_point", True),
        ("black_point", "0.15"),
        ("black_point", float("nan")),
        ("black_point", float("inf")),
        ("black_point", -0.001),
        ("black_point", 0.901),
        ("white_point", False),
        ("white_point", "0.99"),
        ("white_point", float("nan")),
        ("white_point", float("inf")),
        ("white_point", 0.099),
        ("white_point", 1.001),
        ("max_megapixels", True),
        ("max_megapixels", "2"),
        ("max_megapixels", float("nan")),
        ("max_megapixels", float("inf")),
        ("max_megapixels", 0.249),
        ("max_megapixels", 4.001),
    ],
)
def test_matte_config_rejects_hostile_api_values(field: str, value: object) -> None:
    from sam3_matting.config import MatteConfig

    with pytest.raises((TypeError, ValueError), match=field):
        MatteConfig(**{field: value})
