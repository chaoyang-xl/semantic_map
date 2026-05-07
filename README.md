# 3D Semantic Scene Reconstruction (RGB-D Semantic Fusion)

This repository contains a compact, classic RGB-D semantic scene reconstruction baseline for Replica-style data. It follows the traditional **SemanticFusion-style** pipeline:

1. back-project every RGB-D frame into 3D with camera intrinsics;
2. transform points from camera coordinates to the world frame with `Twc`;
3. fuse geometry in a voxel grid;
4. fuse per-pixel semantic labels by Bayesian/count voting;
5. export a colored semantic point cloud and optional voxel centers.

It is designed for your current `result/` directory layout, where you already have RGB images, depth images, camera intrinsics, camera poses `Twc`, and YOLOv8 segmentation masks saved with:

```python
cv2.imwrite(mask_path, label_map)
```

Here `label_map` is expected to be a single-channel image in which each pixel value is the semantic class id.

## Expected input layout

The script is intentionally configurable, but the default layout is:

```text
result/
├── rgb/                 # RGB frames: .png/.jpg
├── depth/               # depth frames: .png/.tiff/.exr/.npy
├── masks/               # single-channel semantic label maps
├── intrinsic.txt        # 3x3, 4x4, or fx fy cx cy
└── trajectory.txt       # one Twc pose per frame
```

Supported pose formats:

- one line with 16 numbers: row-major 4x4 `Twc`;
- TUM style: `timestamp tx ty tz qx qy qz qw`;
- stacked 4x4 matrices separated by blank lines.

Depth is converted to meters with `--depth-scale`. Replica PNG depth is commonly millimeters, so the default is `1000.0`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run on Replica-style results

```bash
python scripts/semantic_fusion.py \
  --result-dir result \
  --rgb-dir rgb \
  --depth-dir depth \
  --mask-dir masks \
  --intrinsic intrinsic.txt \
  --trajectory trajectory.txt \
  --depth-scale 1000 \
  --voxel-size 0.03 \
  --max-depth 8.0 \
  --output outputs/semantic_map.ply
```

The output PLY uses deterministic colors for semantic ids. You can open it with Open3D, MeshLab, or CloudCompare.

## Notes for YOLOv8 masks

YOLOv8 instance segmentation often gives instance masks first. For this baseline, convert them into a dense `label_map` before saving. A simple policy is to fill each instance region with its class id and keep background as `0`:

```python
label_map = np.zeros((height, width), dtype=np.uint16)
for mask, cls_id in zip(masks, class_ids):
    label_map[mask > 0] = int(cls_id) + 1  # reserve 0 for unknown/background
cv2.imwrite(mask_path, label_map)
```

Make sure RGB, depth, and mask frame filenames sort into the same order, or use `--frame-stride` / explicit renaming to align them.

## Algorithm details

For each valid depth pixel, the script computes:

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth(u, v)
p_world = Twc @ [x, y, z, 1]^T
```

Points are discretized into voxels. Each voxel stores:

- the running mean XYZ position;
- the running mean RGB color;
- a histogram of observed semantic labels.

The final semantic label for each voxel is the maximum-count label. This is a mature and reproducible classical baseline: simple, deterministic, easy to debug, and a good foundation before adding TSDF ray integration, CRF smoothing, or learned 3D refinement.
