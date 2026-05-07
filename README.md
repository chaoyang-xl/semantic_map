# 3D 语义场景重建与 BEV 语义地图增量融合

本仓库提供一个面向 Replica 风格数据的经典 RGB-D 语义重建基线。它不是端到端学习方法，而是一个成熟、稳定、容易调试的 **SemanticFusion 风格**复现：在逐帧读取 RGB、Depth、相机内参、相机位姿 `Twc` 和语义 mask 的同时，增量更新 3D 语义体素地图，并同步增量更新一张 BEV（Bird's-Eye View，俯视图）语义地图。

整体流程如下：

1. 用相机内参把每一帧 depth 反投影成相机坐标系下的 3D 点；
2. 用轨迹文件中的 `Twc` 把点从相机坐标系变换到世界坐标系；
3. 将世界坐标点增量融合到稀疏 3D voxel map；
4. 将同一批世界坐标点按水平面投影，增量融合到 2D BEV grid map；
5. 对每个 3D voxel / BEV cell 维护语义标签直方图，用投票得到最终语义类别；
6. 导出 3D 语义点云 `.ply`、BEV 语义图 `.png`，以及可选的标签数组文件。

## 输入数据格式

默认假设你的数据在 `result/` 下：

```text
result/
├── rgb/                 # RGB 图像：.png/.jpg 等
├── depth/               # 深度图：.png/.tiff/.exr/.npy 等
├── masks/               # YOLOv8 分割后保存的单通道语义 label_map
├── intrinsic.txt        # 相机内参：3x3、4x4，或 fx fy cx cy
└── trajectory.txt       # 每一帧一个 Twc
```

你提到的 mask 保存方式可以直接使用：

```python
cv2.imwrite(mask_path, label_map)
```

这里要求 `label_map` 是单通道图，每个像素值就是语义类别 id。建议保留 `0` 表示 unknown/background，其他类别从 `1` 开始。

## 支持的相机参数与轨迹格式

内参文件支持：

- `3x3` 相机内参矩阵；
- `4x4` 矩阵，会自动取左上角 `3x3`；
- 一行或多行中前四个数为 `fx fy cx cy`。

轨迹文件支持：

- 每行 16 个数：row-major 的 `4x4 Twc`；
- TUM 格式：`timestamp tx ty tz qx qy qz qw`；
- 多个用空行分隔的 `4x4` 矩阵。

> 注意：脚本默认把轨迹解释为 `Twc`，即 camera-to-world。如果你的文件是 `Tcw`，需要先取逆，或在数据预处理阶段转换。

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一键运行 3D + BEV 增量语义融合

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
  --pixel-stride 2 \
  --max-depth 8.0 \
  --output outputs/semantic_map.ply \
  --bev-output outputs/bev_semantic.png \
  --bev-label-npy outputs/bev_labels.npy
```

输出内容：

- `outputs/semantic_map.ply`：3D 语义点云，每个 voxel 用最终语义类别的颜色表示；
- `outputs/bev_semantic.png`：BEV 语义地图，可直接查看；
- `outputs/bev_labels.npy`：可选，保存 BEV 每个栅格的语义 label id，便于后续路径规划或评估；
- `outputs/bev_labels.json`：当保存 `.npy` 时同步保存坐标原点、分辨率、坐标轴等元信息。

## BEV 地图参数

BEV 地图是和 3D 语义地图一起在主循环里逐帧增量更新的。每一帧反投影得到的世界坐标点会同时进入：

- 3D sparse voxel map；
- 2D sparse BEV grid map。

常用参数：

```bash
--bev-resolution 0.05       # BEV 每个 grid cell 的边长，单位米
--bev-up-axis y             # 世界坐标中哪个轴是竖直方向：x/y/z，Replica/Habitat 常用 y
--bev-min-height -1.0       # 可选：只融合高度 >= 该值的点
--bev-max-height 2.0        # 可选：只融合高度 <= 该值的点
--bev-snapshot-dir outputs/bev_frames
--bev-snapshot-every 50     # 每融合 50 帧保存一次 BEV 中间结果，观察增量建图过程
```

如果你的世界坐标系是 `z` 轴朝上，请设置：

```bash
--bev-up-axis z
```

## YOLOv8 mask 转 label_map 的建议

YOLOv8 segmentation 通常先得到 instance masks。为了进行语义融合，需要把 instance masks 合成 dense label map。一个简单策略是：

```python
label_map = np.zeros((height, width), dtype=np.uint16)
for mask, cls_id in zip(masks, class_ids):
    # +1 是为了保留 0 给 unknown/background
    label_map[mask > 0] = int(cls_id) + 1
cv2.imwrite(mask_path, label_map)
```

请确保 RGB、depth、mask 的文件名排序后一一对应。例如 `000000.png`、`000001.png` 这种命名最稳妥。

## 算法细节

对每个有效深度像素 `(u, v)`：

```text
z = depth(v, u)
x = (u - cx) * z / fx
y = (v - cy) * z / fy
p_world = Twc @ [x, y, z, 1]^T
```

3D 融合：

```text
voxel_index = floor(p_world / voxel_size)
```

BEV 融合会先丢弃竖直轴，只保留两个水平轴。例如 `--bev-up-axis y` 时：

```text
bev_cell = floor([world_x, world_z] / bev_resolution)
```

每个 3D voxel 和 BEV cell 都维护一个语义直方图：

```text
hist[label] += 1
final_label = argmax(hist)
```

这种做法是经典的增量式语义融合 baseline：不依赖训练，行为确定，可复现，便于和 TSDF、CRF、语义先验、动态物体过滤等后续模块组合。
