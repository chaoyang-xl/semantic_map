from semantic_map.fusion import SemanticObjectTracker
from semantic_map.projection import ProjectedObject


def projected(label: str, x: float, y: float, confidence: float = 0.8) -> ProjectedObject:
    return ProjectedObject(
        label=label,
        display_label=label,
        confidence=confidence,
        center_x=x,
        center_y=y,
        yaw=0.0,
        size_x=1.0,
        size_y=1.0,
        source_frame="camera_link",
    )


def test_tracker_merges_same_label_nearby_observations() -> None:
    tracker = SemanticObjectTracker(association_distance_m=0.5, smoothing_alpha=0.5)

    objects = tracker.update([projected("toilet", 1.0, 2.0)], now=1.0)
    first_id = objects[0].id
    objects = tracker.update([projected("toilet", 1.2, 2.2, confidence=0.9)], now=2.0)

    assert len(objects) == 1
    assert objects[0].id == first_id
    assert objects[0].center_x == 1.1
    assert objects[0].center_y == 2.1
    assert objects[0].confidence == 0.9
    assert objects[0].observations == 2


def test_tracker_keeps_different_labels_or_far_objects_separate() -> None:
    tracker = SemanticObjectTracker(association_distance_m=0.5)

    objects = tracker.update(
        [projected("toilet", 1.0, 1.0), projected("sofa", 1.1, 1.1), projected("toilet", 3.0, 3.0)],
        now=1.0,
    )

    assert len(objects) == 3


def test_tracker_prunes_expired_objects() -> None:
    tracker = SemanticObjectTracker(max_age_s=5.0)

    tracker.update([projected("toilet", 1.0, 1.0)], now=10.0)
    tracker.prune(now=16.0)

    assert tracker.objects == ()
