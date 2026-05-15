from math import isclose

from semantic_map.geometry import CameraIntrinsics, Quaternion, Transform3D, pixel_to_camera
from semantic_map.projection import Detection2D, SemanticProjector, median_depth_around


def test_pixel_to_camera_uses_pinhole_model() -> None:
    intrinsics = CameraIntrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0)

    point = pixel_to_camera(u=15.0, v=30.0, depth_m=2.0, intrinsics=intrinsics)

    assert point == (0.1, 0.1, 2.0)


def test_transform_applies_translation_and_rotation() -> None:
    transform = Transform3D(
        translation=(1.0, 2.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.70710678118, 0.70710678118),
    )

    x, y, z = transform.apply((1.0, 0.0, 0.0))

    assert isclose(x, 1.0, abs_tol=1e-6)
    assert isclose(y, 3.0, abs_tol=1e-6)
    assert isclose(z, 0.0, abs_tol=1e-6)


def test_median_depth_ignores_invalid_values() -> None:
    depth = [
        [0.0, None, 3.0],
        [2.0, 2.2, 2.4],
        [10.0, 0.0, 2.6],
    ]

    assert median_depth_around(depth, center_u=1.0, center_v=1.0, window_px=3) == 2.5


def test_projector_maps_supported_detection_to_semantic_rectangle() -> None:
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.0, cy=1.0)
    projector = SemanticProjector(intrinsics=intrinsics, depth_window_px=1)
    transform = Transform3D(translation=(10.0, 20.0, 0.0), source_frame="front_camera_link")
    detection = Detection2D(
        label="couch",
        confidence=0.9,
        center_u=2.0,
        center_v=1.0,
        width=20.0,
        height=10.0,
    )

    projected = projector.project_detection(detection, [[2.0, 2.0, 2.0]], transform)

    assert projected is not None
    assert projected.label == "sofa"
    assert projected.display_label == "沙发"
    assert isclose(projected.center_x, 10.02)
    assert isclose(projected.center_y, 20.0)
    assert projected.size_x == 1.8
    assert projected.size_y == 0.9
    assert projected.source_frame == "front_camera_link"


def test_projector_filters_low_confidence_and_unknown_classes() -> None:
    projector = SemanticProjector(CameraIntrinsics(100.0, 100.0, 1.0, 1.0))
    transform = Transform3D(translation=(0.0, 0.0, 0.0))
    depth = [[1.0]]

    assert projector.project_detection(Detection2D("toilet", 0.1, 0.0, 0.0, 1.0, 1.0), depth, transform) is None
    assert projector.project_detection(Detection2D("unknown", 0.9, 0.0, 0.0, 1.0, 1.0), depth, transform) is None
