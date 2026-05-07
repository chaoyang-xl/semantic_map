#!/usr/bin/env python3
"""Classic RGB-D semantic fusion for Replica-style reconstruction results.

The implementation is deliberately simple and reproducible: it back-projects
RGB-D frames, transforms them with Twc poses, voxelizes world points, and fuses
semantic labels with per-voxel histograms.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr"}
DEPTH_SUFFIXES = IMAGE_SUFFIXES | {".npy"}


@dataclass
class VoxelAccumulator:
    """Running statistics for a semantic voxel."""

    count: int = 0
    point_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    labels: dict[int, int] = field(default_factory=dict)

    def add(self, point: np.ndarray, color: np.ndarray, label: int) -> None:
        self.count += 1
        self.point_sum += point
        self.color_sum += color
        self.labels[label] = self.labels.get(label, 0) + 1

    @property
    def point(self) -> np.ndarray:
        return self.point_sum / max(self.count, 1)

    @property
    def color(self) -> np.ndarray:
        return np.clip(self.color_sum / max(self.count, 1), 0.0, 1.0)

    @property
    def label(self) -> int:
        return max(self.labels.items(), key=lambda item: (item[1], -item[0]))[0]


class SemanticVoxelMap:
    """Sparse semantic voxel map with count-vote label fusion."""

    def __init__(self, voxel_size: float) -> None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = voxel_size
        self._voxels: dict[tuple[int, int, int], VoxelAccumulator] = {}

    def integrate(self, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int64)
        for point, color, label, voxel_index in zip(points, colors, labels, voxel_indices, strict=True):
            key = tuple(int(v) for v in voxel_index)
            accumulator = self._voxels.setdefault(key, VoxelAccumulator())
            accumulator.add(point.astype(np.float64), color.astype(np.float64), int(label))

    def to_arrays(self, label_color_lut: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = []
        colors = []
        labels = []
        for accumulator in self._voxels.values():
            label = accumulator.label
            points.append(accumulator.point)
            labels.append(label)
            if label_color_lut is None:
                colors.append(label_to_color(label))
            else:
                colors.append(label_color_lut[label % len(label_color_lut)])
        if not points:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,), dtype=np.int64)
        return np.vstack(points), np.vstack(colors), np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._voxels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"), help="Root result directory.")
    parser.add_argument("--rgb-dir", default="rgb", help="RGB subdirectory under result-dir.")
    parser.add_argument("--depth-dir", default="depth", help="Depth subdirectory under result-dir.")
    parser.add_argument("--mask-dir", default="masks", help="Semantic mask subdirectory under result-dir.")
    parser.add_argument("--intrinsic", type=Path, default=Path("intrinsic.txt"), help="Intrinsic file path or path relative to result-dir.")
    parser.add_argument("--trajectory", type=Path, default=Path("trajectory.txt"), help="Twc trajectory file path or path relative to result-dir.")
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic_map.ply"), help="Output semantic point cloud PLY.")
    parser.add_argument("--label-json", type=Path, default=None, help="Optional output JSON with point labels.")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth divisor that converts raw depth to meters.")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="Voxel size in meters.")
    parser.add_argument("--max-depth", type=float, default=8.0, help="Discard depth values beyond this many meters.")
    parser.add_argument("--min-depth", type=float, default=0.05, help="Discard depth values below this many meters.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--pixel-stride", type=int, default=2, help="Use every Nth pixel in x and y for speed.")
    parser.add_argument("--keep-background", action="store_true", help="Keep label 0 instead of dropping it.")
    return parser.parse_args()


def resolve_under_result(result_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else result_dir / path


def sorted_files(directory: Path, suffixes: set[str]) -> list[Path]:
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    return sorted(files, key=lambda path: path.name)


def load_intrinsic(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    flat = values.reshape(-1)
    if values.shape == (3, 3):
        intrinsic = values
    elif values.shape == (4, 4):
        intrinsic = values[:3, :3]
    elif flat.size >= 4:
        fx, fy, cx, cy = flat[:4]
        intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported intrinsic format in {path}")
    return intrinsic


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.linalg.norm([qx, qy, qz, qw])
    if norm == 0:
        raise ValueError("Quaternion norm is zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def load_trajectory(path: Path) -> list[np.ndarray]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Trajectory file is empty: {path}")

    poses: list[np.ndarray] = []
    blocks = [block for block in text.split("\n\n") if block.strip()]
    if all(len(block.splitlines()) == 4 for block in blocks):
        for block in blocks:
            pose = np.loadtxt(block.splitlines(), dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"Invalid 4x4 pose block in {path}")
            poses.append(pose)
        return poses

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        numbers = [float(value) for value in stripped.replace(",", " ").split()]
        if len(numbers) == 16:
            poses.append(np.asarray(numbers, dtype=np.float64).reshape(4, 4))
        elif len(numbers) == 8:
            _, tx, ty, tz, qx, qy, qz, qw = numbers
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = quaternion_to_rotation(qx, qy, qz, qw)
            pose[:3, 3] = [tx, ty, tz]
            poses.append(pose)
        else:
            raise ValueError(f"Unsupported trajectory line with {len(numbers)} values: {line}")
    return poses


def read_depth(path: Path, depth_scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Unable to read depth image: {path}")
        depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / depth_scale


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_label_map(path: Path) -> np.ndarray:
    label_map = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label_map is None:
        raise ValueError(f"Unable to read label map: {path}")
    if label_map.ndim == 3:
        label_map = label_map[..., 0]
    return label_map.astype(np.int64)


def back_project_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray,
    intrinsic: np.ndarray,
    twc: np.ndarray,
    min_depth: float,
    max_depth: float,
    pixel_stride: int,
    keep_background: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth.shape[:2]
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    if labels.shape[:2] != (height, width):
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    z = depth[ys, xs]
    semantic = labels[ys, xs]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if not keep_background:
        valid &= semantic > 0
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,), dtype=np.int64)

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    semantic = semantic[valid].astype(np.int64)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points_camera = np.column_stack([x, y, z, np.ones_like(z)])
    points_world = (twc @ points_camera.T).T[:, :3]
    colors = rgb[ys.astype(np.int64), xs.astype(np.int64)]
    return points_world, colors, semantic


def label_to_color(label: int) -> np.ndarray:
    if label == 0:
        return np.array([0.35, 0.35, 0.35], dtype=np.float64)
    value = int(label) * 2654435761 % 2**32
    r = ((value >> 0) & 255) / 255.0
    g = ((value >> 8) & 255) / 255.0
    b = ((value >> 16) & 255) / 255.0
    return np.array([r, g, b], dtype=np.float64)


def write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), point_cloud):
        raise RuntimeError(f"Failed to write point cloud: {path}")


def write_label_json(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": labels.tolist()}, ensure_ascii=False), encoding="utf-8")


def validate_frame_counts(rgb_files: list[Path], depth_files: list[Path], mask_files: list[Path], poses: list[np.ndarray]) -> int:
    frame_count = min(len(rgb_files), len(depth_files), len(mask_files), len(poses))
    if frame_count == 0:
        raise ValueError("No aligned RGB/depth/mask/pose frames found")
    counts = {"rgb": len(rgb_files), "depth": len(depth_files), "mask": len(mask_files), "pose": len(poses)}
    if len(set(counts.values())) != 1:
        print(f"Warning: frame counts differ; using first {frame_count} frames: {counts}")
    return frame_count


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    intrinsic_path = resolve_under_result(result_dir, args.intrinsic)
    trajectory_path = resolve_under_result(result_dir, args.trajectory)

    rgb_files = sorted_files(result_dir / args.rgb_dir, IMAGE_SUFFIXES)
    depth_files = sorted_files(result_dir / args.depth_dir, DEPTH_SUFFIXES)
    mask_files = sorted_files(result_dir / args.mask_dir, IMAGE_SUFFIXES)
    intrinsic = load_intrinsic(intrinsic_path)
    poses = load_trajectory(trajectory_path)
    frame_count = validate_frame_counts(rgb_files, depth_files, mask_files, poses)

    semantic_map = SemanticVoxelMap(args.voxel_size)
    frame_indices = range(0, frame_count, args.frame_stride)
    for index in tqdm(frame_indices, desc="Fusing RGB-D semantic frames"):
        rgb = read_rgb(rgb_files[index])
        depth = read_depth(depth_files[index], args.depth_scale)
        labels = read_label_map(mask_files[index])
        points, colors, semantic = back_project_frame(
            rgb=rgb,
            depth=depth,
            labels=labels,
            intrinsic=intrinsic,
            twc=poses[index],
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            pixel_stride=args.pixel_stride,
            keep_background=args.keep_background,
        )
        if len(points):
            semantic_map.integrate(points, colors, semantic)

    points, semantic_colors, labels = semantic_map.to_arrays()
    if len(points) == 0:
        raise RuntimeError("Fusion produced no points; check depth scale, poses, masks, and depth thresholds")
    write_point_cloud(args.output, points, semantic_colors)
    if args.label_json is not None:
        write_label_json(args.label_json, labels)
    print(f"Wrote {len(points)} semantic voxels to {args.output}")


if __name__ == "__main__":
    main()
