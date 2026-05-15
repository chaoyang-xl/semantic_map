"""Multi-frame semantic object fusion and lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from time import time
from typing import Iterable
from uuid import uuid4

from semantic_map.projection import ProjectedObject


@dataclass(frozen=True)
class SemanticObject:
    """A stable semantic map object after multi-frame fusion."""

    id: str
    label: str
    display_label: str
    confidence: float
    center_x: float
    center_y: float
    yaw: float
    size_x: float
    size_y: float
    source_frame: str
    first_seen: float
    last_seen: float
    observations: int

    def distance_to(self, projected: ProjectedObject) -> float:
        return hypot(self.center_x - projected.center_x, self.center_y - projected.center_y)


class SemanticObjectTracker:
    """Fuse projected objects into stable map-frame semantic objects."""

    def __init__(
        self,
        association_distance_m: float = 0.8,
        smoothing_alpha: float = 0.35,
        max_age_s: float = 30.0,
    ) -> None:
        if association_distance_m <= 0.0:
            raise ValueError("association_distance_m must be positive")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be positive")
        self.association_distance_m = association_distance_m
        self.smoothing_alpha = smoothing_alpha
        self.max_age_s = max_age_s
        self._objects: dict[str, SemanticObject] = {}

    @property
    def objects(self) -> tuple[SemanticObject, ...]:
        """Current tracked objects sorted by last update time."""

        return tuple(sorted(self._objects.values(), key=lambda item: item.last_seen, reverse=True))

    def update(
        self,
        projected_objects: Iterable[ProjectedObject],
        now: float | None = None,
    ) -> tuple[SemanticObject, ...]:
        """Associate a batch of projected detections and return live objects."""

        timestamp = time() if now is None else float(now)
        for projected in projected_objects:
            match = self._find_match(projected)
            if match is None:
                self._add(projected, timestamp)
            else:
                self._merge(match, projected, timestamp)
        self.prune(now=timestamp)
        return self.objects

    def prune(self, now: float | None = None) -> None:
        """Remove objects that have not been observed within max_age_s."""

        timestamp = time() if now is None else float(now)
        expired = [
            object_id
            for object_id, semantic_object in self._objects.items()
            if timestamp - semantic_object.last_seen > self.max_age_s
        ]
        for object_id in expired:
            del self._objects[object_id]

    def query(self, label_or_alias: str) -> tuple[SemanticObject, ...]:
        """Return objects whose canonical or display label matches the query text."""

        text = label_or_alias.strip().lower()
        return tuple(
            item
            for item in self.objects
            if item.label.lower() == text or item.display_label.lower() == text
        )

    def _find_match(self, projected: ProjectedObject) -> SemanticObject | None:
        candidates = [
            item
            for item in self._objects.values()
            if item.label == projected.label and item.distance_to(projected) <= self.association_distance_m
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.distance_to(projected))

    def _add(self, projected: ProjectedObject, timestamp: float) -> None:
        object_id = projected.track_id or f"{projected.label}-{uuid4().hex[:8]}"
        self._objects[object_id] = SemanticObject(
            id=object_id,
            label=projected.label,
            display_label=projected.display_label,
            confidence=projected.confidence,
            center_x=projected.center_x,
            center_y=projected.center_y,
            yaw=projected.yaw,
            size_x=projected.size_x,
            size_y=projected.size_y,
            source_frame=projected.source_frame,
            first_seen=timestamp,
            last_seen=timestamp,
            observations=1,
        )

    def _merge(self, match: SemanticObject, projected: ProjectedObject, timestamp: float) -> None:
        alpha = self.smoothing_alpha
        beta = 1.0 - alpha
        self._objects[match.id] = replace(
            match,
            confidence=max(match.confidence, projected.confidence),
            center_x=beta * match.center_x + alpha * projected.center_x,
            center_y=beta * match.center_y + alpha * projected.center_y,
            yaw=beta * match.yaw + alpha * projected.yaw,
            size_x=beta * match.size_x + alpha * projected.size_x,
            size_y=beta * match.size_y + alpha * projected.size_y,
            source_frame=projected.source_frame,
            last_seen=timestamp,
            observations=match.observations + 1,
        )
