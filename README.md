## realsense-yolo-seg-trt

项目覆盖模型推理评测、深度相机接入、ROI/Mask 目标深度估计，以及目标在相机坐标系下的三维定位，适用于深度相机目标定位与工程验证等场景

### 功能
#### 1. 模型指标测试（Benchmark）
- 支持**det/obb/seg/pose**四类任务的离线精度与性能测试。
- 推理框架面向 **Ultralytics 系列模型**，包括 YOLOv5/YOLOv8/YOLO11/YOLO26等，并完成TensorRT工程化落地
- 通过 YAML 统一管理模型类型、引擎路径、阈值、数据集路径等参数，便于标准化评测与复现实验结果

#### 2. D435 深度 + ROI/Mask 目标坐标估计
- 支持 Intel RealSense D435 彩色图像与深度图输入。
- 融合模型输出的 ROI 与 Mask 信息，估计目标在相机坐标系下的三维坐标 **(X, Y, Z)**。

#### 3. 多策略深度估计
- 支持多种 **Z 深度采样策略** 与 **XY 像素选点策略**
- 策略均可通过 YAML 直接配置，方便实验对比与工程调优

**深度策略表**
| 配置项 | 可选值 | 说明 |
| --- | --- | --- |
| depth.z_strategy | bbox_center | 使用 bbox 中心点深度 |
| depth.z_strategy | mask_mean | 使用 mask 内深度均值 |
| depth.z_strategy | mask_median | 使用 mask 内深度中位数 |
| depth.z_strategy | masked_window_median | 在窗口内结合 mask 取中位数 |
| depth.z_strategy | eroded_mask_median | 先腐蚀 mask 再取中位数，抗边缘噪声 |
| depth.z_strategy | mask_percentile | 使用 mask 内深度分位值 |
| depth.xy_strategy | bbox_center | 使用 bbox 中心作为像素点 |
| depth.xy_strategy | mask_centroid | 使用 mask 质心 |
| depth.xy_strategy | mask_median | 使用 mask 中位像素 |
| depth.xy_strategy | mask_inner_point | 使用 mask 内部最稳定点 |
| depth.xy_strategy | eroded_mask_centroid | 使用腐蚀后 mask 质心 |
| depth.xy_strategy | nearest_depth_to_z | 选择与当前 z 最接近的 mask 像素 |
| depth.xy_fallback_strategy | 同 depth.xy_strategy 全部可选值 | 主 XY 策略失败时启用 |


### 环境配置
- 操作系统：Linux（推荐 Ubuntu 20.04 / 22.04）
- 编译工具：CMake >= 3.10、C++14
- 依赖项：CUDA、TensorRT、OpenCV（4.9+）、librealsense2
- 硬件建议：NVIDIA GPU + Intel RealSense D435

### 配置说明
项目主要通过 YAML 配置运行参数，包括但不限于：
- 相机参数：分辨率、帧率、对齐方式、重连策略
- 模型参数：任务类型、TensorRT 引擎路径、阈值、类别过滤
- 深度参数：Z 深度策略、XY 选点策略、点云相关配置
- 显示参数：窗口显示、调试开关、深度可视化范围
- Benchmark 参数：预热帧数、统计周期、性能摘要输出

### 快速开始
编译
```bash
./scripts/build_apps.sh
```

运行 Demo
```bash
./build/bin/yolo_seg_trt_app app d435_seg configs/seg_depth.yaml
```

运行 Benchmark 示例
```bash
./build/bin/yolo_seg_trt_app bench seg configs/bench_v11s_seg_dynamic.yaml
```

### 重要提示
- bench 命令中的 task 必须与 YAML 的 task_type 一致
- seg_depth 配置中 input.source 支持 camera 或视频路径
- batch_size 字段当前仅保留接口，运行时按 1 执行
- 若D435未识别，请先用realsense-viewer验证设备连通性
