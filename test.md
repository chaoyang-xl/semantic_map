#### 1、项目背景与目标

智慧康养移位机器人需要在室内护理、床椅对接、人员辅助转移等场景中理解周围环境与人体状态。视觉语义感知模块承担从相机图像中提取“人、家具、人体姿态”等语义信息的任务，并将关键语义结果转换到机器人导航地图坐标系中，为导航、避障、对接和上层任务决策提供结构化输入。系统环境以当前 Orange Pi 5 Plus +
Orbbec GEMINI 336L + ROS 2 + RKNN YOLO/YOLO-pose 环境为基础。



#### 2、总体架构设计

| **层级** | **主要功能**                                                 | **推荐 ROS 节点**                           |
| -------- | ------------------------------------------------------------ | ------------------------------------------- |
| 采集层   | 前置/后置相机  RGB-D 图像采集、时间戳同步、CameraInfo 发布   | orbbec_camera  或相机驱动节点               |
| 推理层   | NPU  加载 RKNN 模型，完成 YOLO 检测与 YOLO-pose 姿态估计     | front_vision_node、rear_pose_alignment_node |
| 语义层   | 目标类别过滤、姿态状态识别、目标跟踪、置信度平滑             | semantic_extractor                          |
| 投影层   | 2D 语义 + Depth + TF 转换为 map 坐标语义目标                 | semantic_projection_node                    |
| 地图层   | 将语义目标写入导航地图/代价地图，可按类别设置生命周期与代价值 | semantic_costmap_layer                      |
| 调试层   | 生成  debug 图像、JSON、Marker、日志和网页预览               | vision_debug_server  / recorder             |

#### 3、上游功能及数据

| **功能**     | **说明**                                                     | **输出示例**                                             |
| ------------ | ------------------------------------------------------------ | -------------------------------------------------------- |
| 家具识别     | 识别床、椅子、沙发、桌子、柜体、马桶等对护理场景有意义的家具；初版可基于 COCO 类别，后续补充护理场景数据微调。 | chair、couch、bed、dining  table、toilet、cabinet/custom |
| 人体检测     | 检测前方人员。                                               | person  bbox + map position                              |
| 人体姿态估计 | 输出人体关键点，辅助判断姿态语义。                           | keypoints[17]  + pose_state                              |



#### 4、订阅话题

| **话题**                                                  | **类型**                                            | **用途**       | **备注**                                                     |
| --------------------------------------------------------- | --------------------------------------------------- | -------------- | ------------------------------------------------------------ |
| /camera/color/image_raw  或 /front_camera/color/image_raw | sensor_msgs/msg/Image                               | 前置  RGB 输入 | 当前环境已有  /camera/color/image_raw，可后续重命名为 front 命名空间。 |
| /camera/depth/image_raw  或 /front_camera/depth/image_raw | sensor_msgs/msg/Image                               | 前置深度输入   | 用于  2D 结果反投影。                                        |
| /front_camera/color/camera_info                           | sensor_msgs/msg/CameraInfo                          | 前置相机内参   | 必须保证与图像分辨率一致。                                   |
| /rear_camera/color/image_raw                              | sensor_msgs/msg/Image                               | 后置  RGB 输入 | 用于对接阶段人体姿态估计。                                   |
| /rear_camera/depth/image_raw                              | sensor_msgs/msg/Image                               | 后置深度输入   | 用于人体相对位置估计。                                       |
| /tf、/tf_static                                           | tf2_msgs/msg/TFMessage                              | 坐标变换       | camera_link、base_link、map  必须连通。                      |
| /map、/odom                                               | nav_msgs/msg/OccupancyGrid  / nav_msgs/msg/Odometry | 地图和定位参考 | 用于语义地图融合和调试。                                     |