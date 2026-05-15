from semantic_map.fusion import SemanticObject
from semantic_map.visualization import object_to_rectangle, rectangle_corners


def test_rectangle_corners_closes_polyline() -> None:
    corners = rectangle_corners(center_x=1.0, center_y=2.0, size_x=2.0, size_y=1.0)

    assert corners == ((0.0, 1.5), (2.0, 1.5), (2.0, 2.5), (0.0, 2.5), (0.0, 1.5))


def test_object_to_rectangle_uses_semantic_color_and_display_label() -> None:
    semantic_object = SemanticObject(
        id="toilet-1",
        label="toilet",
        display_label="马桶",
        confidence=0.9,
        center_x=1.0,
        center_y=2.0,
        yaw=0.0,
        size_x=0.8,
        size_y=0.8,
        source_frame="front_camera_link",
        first_seen=1.0,
        last_seen=1.0,
        observations=1,
    )

    rectangle = object_to_rectangle(semantic_object)

    assert rectangle.label == "马桶"
    assert rectangle.color_rgba == (0.1, 0.9, 0.3, 0.85)
    assert rectangle.points[0] == (0.6, 1.6)
