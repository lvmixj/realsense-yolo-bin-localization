# RealSense YOLO Bin Localization

**基于RealSense与YOLO的垃圾桶三维定位（Windows）**

基于 Intel RealSense D435i 与 YOLO11 的垃圾桶检测与三维定位样机程序。系统从
RGB-D 数据中识别垃圾桶，融合深度信息计算目标相对相机的三维坐标，并通过多帧
稳定性判断输出可用的定位结果，用于侧装垃圾车的作业辅助与后续自动抓取研究。

~~~
RGB-D 采集 -> YOLO11 检测 -> 深度质量评估 -> 多帧稳定确认 -> 相机坐标 XYZ
~~~

## 主要特性

- **检测 + 三维定位**：YOLO11 检测垃圾桶，深度 ROI 融合给出 `X/Y/Z`（毫米）。
- **深度质量门控**：以 `depth_valid_ratio`、深度 MAD、框尺寸与宽高比等条件过滤
  不可信测量，避免"有框就出坐标"。
- **多帧稳定性确认**：1 秒窗口内累计确认次数，减少逐帧抖动与瞬时误检。
- **光流短时补帧**：已确认目标在 YOLO 短暂漏检时由 LK 光流维持显示（黄框），
  不影响写入结果。
- **昼夜双模型**：白天默认权重 + 夜间单类别权重，可按场景切换或统一入口运行。
- **离线回放**：直接读取 RealSense `.db3` 录包，无需安装 ROS 2。
- **结构化输出**：逐帧 JSONL 记录环境质量、检测框、三维坐标、跟踪与拒识原因，
  可直接用于论文统计与回归测试。
- **可选机械臂外参**：提供 `camera_to_base` 外参接口换算底座坐标（默认仅记录，
  不触发任何自动动作）。

## 目录结构

~~~
.
├── d435i_yolo11_3d_detection.py      # 主程序：采集/回放/检测/三维定位/可视化
├── detect_image.py                   # 单张图片检测（仅 RGB，不含深度）
├── test_grasp_pipeline.py            # 无相机、无模型推理的回归测试
├── verify_windows_setup.py           # 环境自检
├── extract_night_annotation_frames.py# 从 .db3 抽取夜间标注帧
├── label_bin_boxes.py                # 本地交互式标注器
├── build_night_bin_dataset.py        # 构建 YOLO 数据集划分
├── prepare_pseudo_seg_dataset.py     # 伪分割数据准备
├── setup_windows.bat / .ps1          # 创建虚拟环境并安装依赖
├── run_d435i_windows.bat             # 实时相机（默认权重）
├── run_night_model_windows.bat       # 夜间权重
├── run_day_night_model_windows.bat   # 昼夜统一权重
├── test_image_windows.bat            # 图片验证
├── camera_to_base.example.json       # 相机-底座外参格式示例
├── models/                           # 权重
├── sample/                           # 样例图片与检测结果
├── runs/night_bin_train_v1/          # 夜间微调训练曲线与指标
└── docs/                             # 方案文档
~~~

## 环境要求

- Windows 10/11 64 位
- Python 3.11（64 位）
- NVIDIA GPU（CUDA 版 PyTorch）+ USB 3.x 接口
- Intel RealSense D435i

## 获取源码

```sh
git clone https://github.com/lvmixj/realsense-yolo-bin-localization.git
cd realsense-yolo-bin-localization
```

## 安装

双击 `setup_windows.bat`，或在 PowerShell 中执行：

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
./setup_windows.ps1
~~~

脚本会在项目目录内创建 `.venv` 并安装 `requirements-windows.txt` 中的依赖。

## 快速开始

**图片验证**（仅 RGB 检测）

~~~bat
test_image_windows.bat sample/rubb_Color.png
~~~

**离线回放录包**

~~~bat
run_d435i_windows.bat --db3 "D:/path/recording.db3" --no_display --print_result
~~~

带结构化输出的小窗口回归：

~~~bat
run_d435i_windows.bat ^
  --db3 "D:/path/recording.db3" ^
  --no_display --no_warmup --max_frames 200 ^
  --jsonl "outputs/recording_200.jsonl" ^
  --out_video "outputs/recording_200.mp4"
~~~

**实时相机**

~~~bat
run_d435i_windows.bat --print_result
~~~

**回归测试**（无需相机与模型）

~~~bash
./.venv/Scripts/python.exe -m unittest -v test_grasp_pipeline.py
~~~

## 发布验证

发布前在Linux/Python3.10的隔离CPU环境执行了12项管线回归测试，三份公开权重均完成加载与样例图片推理。清理前后模型参数逐项相等，模型文件校验和见`SHA256SUMS.txt`。Windows安装入口随源码提供。

## 模型

| 文件 | 类别 | 适用场景 |
| --- | --- | --- |
| `models/bin_v3_yolo11s_960_best.pt` | 多类别（垃圾桶/散落垃圾/满溢/人员等） | 白天、通用场景，默认权重 |
| `models/night_bin_yolo11s_v1.pt` | 单类别 `garbage bin` | 夜间、低照度、暖色光源 |
| `models/day_night_bin_yolo11s_v1.pt` | 单类别 `garbage bin` | 统一启动入口；当前为夜间权重别名 |

当前`day_night_bin_yolo11s_v1.pt`与`night_bin_yolo11s_v1.pt`内容相同，统一入口使用夜间权重的别名。

夜间权重的训练/验证集按录包划分（同一录包的相邻帧不会被随机拆到不同集合），
避免因时间相关性造成的指标虚高。训练曲线与指标见 `runs/night_bin_train_v1/`。

## 常用参数

~~~text
--conf 0.45                     检测置信度阈值
--min_target_depth_valid_ratio  深度有效率下限（默认 0.35）
--max_target_depth_mad_mm       深度中位绝对偏差上限（默认 40）
--stability_window_frames       稳定性统计窗口
--min_stable_frames             最小稳定确认帧数
--display_track_max_missed_frames  光流补帧最大连续漏检帧数
--extrinsics                    相机到底座外参与工作空间
--show_runtime_osd              显示 FPS / 推理耗时 / 设备信息
~~~

完整参数见 `python d435i_yolo11_3d_detection.py --help`，或 `docs/` 内的使用说明。

## 输出字段

逐帧 JSONL 每条记录包含环境质量（亮度、暗像素比例、过曝比例、对比度、
全局深度有效率）与检测列表。每个目标至少包括：

~~~text
target_id, frame_id, class_name, detection_confidence
position_camera_xyz_mm, position_base_xyz_mm
object_width/height/depth_mm, depth_valid_ratio, depth_mad_mm, position_std_mm
grasp_score, graspable, status, reject_reason
~~~

常见拒识原因：`DEPTH_UNAVAILABLE`、`DEPTH_VALID_RATIO_TOO_LOW`、
`DEPTH_MAD_TOO_HIGH`、`POSITION_UNSTABLE`、`REQUIRED_JAW_WIDTH_EXCEEDED`、
`OUT_OF_ROBOT_WORKSPACE` 等。

## 数据说明

原始录包、夜间现场图像、标注样例与现场JSONL输出保存在开发者本地。公开仓库提供源码、模型、训练曲线和垃圾桶图片示例；自行采集后可用下列命令构建标注目录。

自行采集与标注的流程：

~~~bat
.venv\Scripts\python.exe extract_night_annotation_frames.py ^
  --db3 "D:/path/night_a.db3" --output "outputs/night_bin_labeling" ^
  --frames-per-recording 40

.venv\Scripts\python.exe label_bin_boxes.py "outputs/night_bin_labeling"
~~~

标注器操作：左键拖动添加框，`Backspace` 撤销，`C` 清空，`N`/`P` 切换帧，
`Q` 保存退出。无垃圾桶的帧应保存为空标签，作为负样本。

## 已知限制

- 首期只处理直立、可见、处于作业区域内的标准垃圾桶；倾倒或被遮挡的目标交由
  人工处理，程序不下发自动抓取指令。
- 夜间权重仅覆盖已标注的夜间场景，新环境建议重新采集并微调。
- 手眼外参为占位示例，实际部署需自行标定后填入 `camera_to_base.json`。
- Precision / Recall / mAP、深度 MAE/RMSE、抓取成功率等指标需按采集场次划分
  数据后测量，程序输出的字段可作为统计基础。

## 许可证

本项目的YOLO集成程序及随附模型以[AGPL-3.0](LICENSE)开源，使用[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)。原有自编代码的Apache-2.0声明保留于[LICENSE-APACHE-2.0](LICENSE-APACHE-2.0)，完整集成分发遵循AGPL-3.0。第三方依赖按各自许可证提供，详见[第三方许可说明](THIRD_PARTY.md)。
