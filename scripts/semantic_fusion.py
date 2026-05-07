#!/usr/bin/env python3
"""Replica 风格 RGB-D 语义融合脚本。

脚本采用经典 SemanticFusion 风格流程：逐帧反投影 RGB-D，使用 Twc
变换到世界坐标系，并同时增量维护 3D 语义体素地图与 2D BEV 语义地图。
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
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class VoxelAccumulator:
    """单个 3D voxel 的增量统计量。"""

    count: int = 0
    point_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    labels: dict[int, int] = field(default_factory=dict)

    def add(self, point: np.ndarray, color: np.ndarray, label: int) -> None:
        """向当前 voxel 增量加入一个点观测。"""
        self.count += 1
        self.point_sum += point
        self.color_sum += color
        self.labels[label] = self.labels.get(label, 0) + 1

    @property
    def point(self) -> np.ndarray:
        """返回该 voxel 中点坐标的运行均值。"""
        return self.point_sum / max(self.count, 1)

    @property
    def color(self) -> np.ndarray:
        """返回该 voxel 中 RGB 颜色的运行均值。"""
        return np.clip(self.color_sum / max(self.count, 1), 0.0, 1.0)

    @property
    def label(self) -> int:
        """返回投票数最多的语义类别。"""
        return max(self.labels.items(), key=lambda item: (item[1], -item[0]))[0]


class SemanticVoxelMap:
    """稀疏 3D 语义体素地图，使用标签直方图做增量语义融合。"""

    def __init__(self, voxel_size: float) -> None:
        if voxel_size <= 0:
            raise ValueError("voxel_size 必须大于 0")
        self.voxel_size = voxel_size
        self._voxels: dict[tuple[int, int, int], VoxelAccumulator] = {}

    def integrate(self, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
        """把一帧或一批世界坐标点融合进 3D voxel map。"""
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int64)
        for point, color, label, voxel_index in zip(points, colors, labels, voxel_indices, strict=True):
            key = tuple(int(v) for v in voxel_index)
            accumulator = self._voxels.setdefault(key, VoxelAccumulator())
            accumulator.add(point.astype(np.float64), color.astype(np.float64), int(label))

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """导出点、语义颜色和标签数组，供 PLY/JSON 保存。"""
        points = []
        colors = []
        labels = []
        for accumulator in self._voxels.values():
            label = accumulator.label
            points.append(accumulator.point)
            colors.append(label_to_color(label))
            labels.append(label)
        if not points:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,), dtype=np.int64)
        return np.vstack(points), np.vstack(colors), np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._voxels)


@dataclass
class BevCellAccumulator:
    """单个 BEV grid cell 的语义投票统计。"""

    count: int = 0
    labels: dict[int, int] = field(default_factory=dict)

    def add(self, label: int) -> None:
        """向 BEV cell 增量加入一个语义观测。"""
        self.count += 1
        self.labels[label] = self.labels.get(label, 0) + 1

    @property
    def label(self) -> int:
        """返回该 BEV cell 的最大投票语义类别。"""
        return max(self.labels.items(), key=lambda item: (item[1], -item[0]))[0]


class SemanticBevMap:
    """稀疏 BEV 语义地图。

    该类在每一帧 3D 点生成后立即更新，所以可以观察随时间增长的
    BEV 语义地图。内部使用 sparse dict，最终导出时再转成 dense image。
    """

    def __init__(self, resolution: float, up_axis: str, min_height: float | None, max_height: float | None) -> None:
        if resolution <= 0:
            raise ValueError("bev_resolution 必须大于 0")
        if up_axis not in AXIS_TO_INDEX:
            raise ValueError("bev_up_axis 只能是 x、y 或 z")
        self.resolution = resolution
        self.up_axis = up_axis
        self.up_index = AXIS_TO_INDEX[up_axis]
        self.plane_indices = tuple(index for index in range(3) if index != self.up_index)
        self.min_height = min_height
        self.max_height = max_height
        self._cells: dict[tuple[int, int], BevCellAccumulator] = {}

    def integrate(self, points: np.ndarray, labels: np.ndarray) -> None:
        """把一批世界坐标点按水平面投影后融合到 BEV grid。"""
        if len(points) == 0:
            return
        heights = points[:, self.up_index]
        valid = np.ones(len(points), dtype=bool)
        if self.min_height is not None:
            valid &= heights >= self.min_height
        if self.max_height is not None:
            valid &= heights <= self.max_height
        if not np.any(valid):
            return

        plane_points = points[valid][:, self.plane_indices]
        bev_labels = labels[valid]
        cell_indices = np.floor(plane_points / self.resolution).astype(np.int64)
        for cell_index, label in zip(cell_indices, bev_labels, strict=True):
            key = (int(cell_index[0]), int(cell_index[1]))
            accumulator = self._cells.setdefault(key, BevCellAccumulator())
            accumulator.add(int(label))

    def to_label_grid(self) -> tuple[np.ndarray, tuple[int, int]]:
        """导出 dense BEV label grid 和 grid 左上角对应的 sparse cell 原点。"""
        if not self._cells:
            return np.zeros((0, 0), dtype=np.int64), (0, 0)

        keys = np.asarray(list(self._cells.keys()), dtype=np.int64)
        min_col, min_row = keys.min(axis=0)
        max_col, max_row = keys.max(axis=0)
        width = int(max_col - min_col + 1)
        height = int(max_row - min_row + 1)
        grid = np.zeros((height, width), dtype=np.int64)

        for (col, row), accumulator in self._cells.items():
            image_row = int(max_row - row)
            image_col = int(col - min_col)
            grid[image_row, image_col] = accumulator.label
        return grid, (int(min_col), int(max_row))

    def metadata(self, origin_cell: tuple[int, int]) -> dict[str, object]:
        """返回 BEV 栅格转世界坐标所需的元信息。"""
        plane_axes = [axis for axis in ("x", "y", "z") if axis != self.up_axis]
        return {
            "resolution": self.resolution,
            "up_axis": self.up_axis,
            "plane_axes": plane_axes,
            "origin_cell_top_left": list(origin_cell),
            "origin_world_top_left": [origin_cell[0] * self.resolution, origin_cell[1] * self.resolution],
            "min_height": self.min_height,
            "max_height": self.max_height,
        }

    def __len__(self) -> int:
        return len(self._cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"), help="result 根目录。")
    parser.add_argument("--rgb-dir", default="rgb", help="result-dir 下的 RGB 子目录。")
    parser.add_argument("--depth-dir", default="depth", help="result-dir 下的深度图子目录。")
    parser.add_argument("--mask-dir", default="masks", help="result-dir 下的语义 mask 子目录。")
    parser.add_argument("--intrinsic", type=Path, default=Path("intrinsic.txt"), help="内参文件路径，或相对 result-dir 的路径。")
    parser.add_argument("--trajectory", type=Path, default=Path("trajectory.txt"), help="Twc 轨迹文件路径，或相对 result-dir 的路径。")
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic_map.ply"), help="输出 3D 语义点云 PLY。")
    parser.add_argument("--label-json", type=Path, default=None, help="可选：输出 3D 点云每个点的语义标签 JSON。")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="深度值除以该系数后转成米。Replica PNG 常用 1000。")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="3D voxel 边长，单位米。")
    parser.add_argument("--max-depth", type=float, default=8.0, help="丢弃超过该距离的深度点，单位米。")
    parser.add_argument("--min-depth", type=float, default=0.05, help="丢弃小于该距离的深度点，单位米。")
    parser.add_argument("--frame-stride", type=int, default=1, help="每隔多少帧融合一次。")
    parser.add_argument("--pixel-stride", type=int, default=2, help="像素采样步长；越大越快，但地图更稀疏。")
    parser.add_argument("--keep-background", action="store_true", help="保留 label 0，否则默认丢弃 unknown/background。")
    parser.add_argument("--bev-output", type=Path, default=Path("outputs/bev_semantic.png"), help="输出 BEV 语义彩色图。")
    parser.add_argument("--bev-label-npy", type=Path, default=None, help="可选：输出 BEV label id 数组 .npy。")
    parser.add_argument("--bev-resolution", type=float, default=0.05, help="BEV 栅格分辨率，单位米。")
    parser.add_argument("--bev-up-axis", choices=("x", "y", "z"), default="y", help="世界坐标中的竖直轴；Replica/Habitat 常用 y。")
    parser.add_argument("--bev-min-height", type=float, default=None, help="可选：BEV 只融合高于该值的点。")
    parser.add_argument("--bev-max-height", type=float, default=None, help="可选：BEV 只融合低于该值的点。")
    parser.add_argument("--disable-bev", action="store_true", help="只构建 3D 语义地图，不构建 BEV 地图。")
    parser.add_argument("--bev-snapshot-dir", type=Path, default=None, help="可选：保存 BEV 增量更新中间结果的目录。")
    parser.add_argument("--bev-snapshot-every", type=int, default=0, help="每融合多少帧保存一次 BEV 快照；0 表示不保存。")
    return parser.parse_args()


def resolve_under_result(result_dir: Path, path: Path) -> Path:
    """把相对路径解析到 result-dir 下，绝对路径保持不变。"""
    return path if path.is_absolute() else result_dir / path


def sorted_files(directory: Path, suffixes: set[str]) -> list[Path]:
    """按文件名排序读取目录中的指定后缀文件。"""
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    return sorted(files, key=lambda path: path.name)


def load_intrinsic(path: Path) -> np.ndarray:
    """读取相机内参，支持 3x3、4x4 或 fx fy cx cy。"""
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
        raise ValueError(f"不支持的内参格式：{path}")
    return intrinsic


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """把四元数转换为旋转矩阵。"""
    norm = np.linalg.norm([qx, qy, qz, qw])
    if norm == 0:
        raise ValueError("四元数范数为 0")
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
    """读取 Twc 轨迹，支持 16 数、TUM 和空行分隔 4x4 矩阵。"""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"轨迹文件为空：{path}")

    poses: list[np.ndarray] = []
    blocks = [block for block in text.split("\n\n") if block.strip()]
    if all(len(block.splitlines()) == 4 for block in blocks):
        for block in blocks:
            pose = np.loadtxt(block.splitlines(), dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"4x4 位姿块格式错误：{path}")
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
            raise ValueError(f"不支持的轨迹行格式，共 {len(numbers)} 个数：{line}")
    return poses


def read_depth(path: Path, depth_scale: float) -> np.ndarray:
    """读取深度图并转换为米。"""
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"无法读取深度图：{path}")
        depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / depth_scale


def read_rgb(path: Path) -> np.ndarray:
    """读取 RGB 图像，并归一化到 0 到 1。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取 RGB 图像：{path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_label_map(path: Path) -> np.ndarray:
    """读取 cv2.imwrite 保存的单通道语义 label_map。"""
    label_map = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label_map is None:
        raise ValueError(f"无法读取语义 label_map：{path}")
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
    """把单帧 RGB-D + label_map 反投影成世界坐标点云。"""
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
    """为语义 label 生成稳定的伪随机颜色。"""
    if label == 0:
        return np.array([0.35, 0.35, 0.35], dtype=np.float64)
    value = int(label) * 2654435761 % 2**32
    r = ((value >> 0) & 255) / 255.0
    g = ((value >> 8) & 255) / 255.0
    b = ((value >> 16) & 255) / 255.0
    return np.array([r, g, b], dtype=np.float64)


def colorize_label_grid(label_grid: np.ndarray) -> np.ndarray:
    """把 BEV label grid 转成 RGB 彩色图。"""
    if label_grid.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    image = np.zeros((*label_grid.shape, 3), dtype=np.uint8)
    for label in np.unique(label_grid):
        image[label_grid == label] = (label_to_color(int(label)) * 255).astype(np.uint8)
    return image


def write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """保存 Open3D PLY 点云。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), point_cloud):
        raise RuntimeError(f"写入点云失败：{path}")


def write_label_json(path: Path, labels: np.ndarray) -> None:
    """保存 3D 点云对应的语义标签。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": labels.tolist()}, ensure_ascii=False), encoding="utf-8")


def write_bev_outputs(bev_map: SemanticBevMap, image_path: Path, label_npy_path: Path | None) -> None:
    """保存 BEV 彩色图、可选 label grid 和元信息。"""
    label_grid, origin_cell = bev_map.to_label_grid()
    if label_grid.size == 0:
        raise RuntimeError("BEV 地图为空，请检查高度过滤、mask 和深度参数")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image = colorize_label_grid(label_grid)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), bgr_image):
        raise RuntimeError(f"写入 BEV 图像失败：{image_path}")

    metadata = bev_map.metadata(origin_cell)
    metadata_path = image_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if label_npy_path is not None:
        label_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(label_npy_path, label_grid)
        label_npy_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_write_bev_snapshot(bev_map: SemanticBevMap | None, args: argparse.Namespace, fused_count: int) -> None:
    """按固定间隔保存 BEV 增量建图快照。"""
    if bev_map is None or len(bev_map) == 0 or args.bev_snapshot_dir is None or args.bev_snapshot_every <= 0:
        return
    if fused_count % args.bev_snapshot_every != 0:
        return
    snapshot_path = args.bev_snapshot_dir / f"bev_{fused_count:06d}.png"
    write_bev_outputs(bev_map, snapshot_path, None)


def validate_frame_counts(rgb_files: list[Path], depth_files: list[Path], mask_files: list[Path], poses: list[np.ndarray]) -> int:
    """检查 RGB、depth、mask、pose 是否存在可对齐帧。"""
    frame_count = min(len(rgb_files), len(depth_files), len(mask_files), len(poses))
    if frame_count == 0:
        raise ValueError("没有找到可对齐的 RGB/depth/mask/pose 帧")
    counts = {"rgb": len(rgb_files), "depth": len(depth_files), "mask": len(mask_files), "pose": len(poses)}
    if len(set(counts.values())) != 1:
        print(f"警告：各模态帧数不一致，将只使用前 {frame_count} 帧：{counts}")
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
    bev_map = None
    if not args.disable_bev:
        bev_map = SemanticBevMap(
            resolution=args.bev_resolution,
            up_axis=args.bev_up_axis,
            min_height=args.bev_min_height,
            max_height=args.bev_max_height,
        )

    fused_count = 0
    frame_indices = range(0, frame_count, args.frame_stride)
    for index in tqdm(frame_indices, desc="增量融合 RGB-D 语义帧"):
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
            if bev_map is not None:
                bev_map.integrate(points, semantic)
        fused_count += 1
        maybe_write_bev_snapshot(bev_map, args, fused_count)

    points, semantic_colors, labels = semantic_map.to_arrays()
    if len(points) == 0:
        raise RuntimeError("3D 融合结果为空，请检查 depth-scale、Twc、mask 和深度阈值")
    write_point_cloud(args.output, points, semantic_colors)
    if args.label_json is not None:
        write_label_json(args.label_json, labels)

    if bev_map is not None:
        write_bev_outputs(bev_map, args.bev_output, args.bev_label_npy)
        print(f"BEV 语义地图包含 {len(bev_map)} 个有效栅格，已写入 {args.bev_output}")
    print(f"3D 语义地图包含 {len(points)} 个 voxel，已写入 {args.output}")


if __name__ == "__main__":
    main()
