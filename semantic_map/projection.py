"""Project 2D detector outputs into map-frame semantic rectangles."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

from semantic_map.config import DEFAULT_SEMANTIC_CLASSES, SemanticClass, resolve_semantic_class
from semantic_map.geometry import CameraIntrinsics, Transform3D, pixel_to_camera


DepthImage = Sequence[Sequence[float | int | None]]


@dataclass(frozen=True)
class Detection2D:
    """A detector result in image coordinates."""

    label: str
    confidence: float
    center_u: float
    center_v: float
    width: float
    height: float
    track_id: str | None = None


@dataclass(frozen=True)
class ProjectedObject:
    """A semantic object rectangle already expressed in the map frame."""

    label: str
    display_label: str
    confidence: float
    center_x: float
    center_y: float
    yaw: float
    size_x: float
    size_y: float
    source_frame: str
    track_id: str | None = None


class SemanticProjector:
    """Convert RGB-D detections into 2D semantic map objects."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        classes: Mapping[str, SemanticClass] | None = None,
        depth_window_px: int = 5,
        min_confidence: float = 0.35,
    ) -> None:
        if depth_window_px < 1:
            raise ValueError("depth_window_px must be at least 1")
        self.intrinsics = intrinsics
        self.classes = classes or DEFAULT_SEMANTIC_CLASSES
        self.depth_window_px = depth_window_px
        self.min_confidence = min_confidence

    def project_detection(
        self,
        detection: Detection2D,
        depth_image_m: DepthImage,
        camera_to_map: Transform3D,
    ) -> ProjectedObject | None:
        """Project a single detection into the map frame.

        Returns None when the class is unsupported, confidence is too low, or no
        valid depth exists near the detection center.
        """

        if detection.confidence < self.min_confidence:
            return None

        semantic_class = resolve_semantic_class(detection.label, self.classes)
        if semantic_class is None:
            return None

        depth = median_depth_around(
            depth_image_m,
            center_u=detection.center_u,
            center_v=detection.center_v,
            window_px=self.depth_window_px,
        )
        if depth is None:
            return None

        camera_point = pixel_to_camera(
            detection.center_u,
            detection.center_v,
            depth,
            self.intrinsics,
        )
        map_x, map_y, _ = camera_to_map.apply(camera_point)

        return ProjectedObject(
            label=semantic_class.canonical_label,
            display_label=semantic_class.display_label,
            confidence=float(detection.confidence),
            center_x=map_x,
            center_y=map_y,
            yaw=0.0,
            size_x=semantic_class.size_x,
            size_y=semantic_class.size_y,
            source_frame=camera_to_map.source_frame,
            track_id=detection.track_id,
        )

    def project_detections(
        self,
        detections: Sequence[Detection2D],
        depth_image_m: DepthImage,
        camera_to_map: Transform3D,
    ) -> list[ProjectedObject]:
        """Project all usable detections into the map frame."""

        projected: list[ProjectedObject] = []
        for detection in detections:
            item = self.project_detection(detection, depth_image_m, camera_to_map)
            if item is not None:
                projected.append(item)
        return projected


def median_depth_around(
    depth_image_m: DepthImage,
    center_u: float,
    center_v: float,
    window_px: int,
) -> float | None:
    """Return median positive depth around a pixel center."""

    if not depth_image_m:
        return None

    height = len(depth_image_m)
    width = len(depth_image_m[0]) if height else 0
    if width == 0:
        return None

    half = window_px // 2
    center_u_px = min(width - 1, max(0, int(round(center_u))))
    center_v_px = min(height - 1, max(0, int(round(center_v))))
    u0 = max(0, center_u_px - half)
    u1 = min(width - 1, center_u_px + half)
    v0 = max(0, center_v_px - half)
    v1 = min(height - 1, center_v_px + half)

    values: list[float] = []
    for v in range(v0, v1 + 1):
        row = depth_image_m[v]
        for u in range(u0, u1 + 1):
            value = row[u]
            if value is None:
                continue
            depth = float(value)
            if depth > 0.0:
                values.append(depth)

    if not values:
        return None
    return float(median(values))
