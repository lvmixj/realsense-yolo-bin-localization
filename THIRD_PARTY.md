# 第三方许可与发布说明

- YOLO实现与模型：Ultralytics YOLO，AGPL-3.0，见https://github.com/ultralytics/ultralytics及https://docs.ultralytics.com/help/contributing/。模型内嵌许可声明保留。
- 原有自编代码的Apache-2.0许可文本保留在LICENSE-APACHE-2.0；本项目完整YOLO集成程序按AGPL-3.0分发。
- PyTorch、Torchvision、OpenCV、NumPy与RealSense Python绑定通过包管理器独立安装，各自许可随依赖发行包提供。
- 为保护隐私，发布副本将模型训练配置中的本机绝对路径改为相对路径；模型张量存储逐字节保持一致。SHA256SUMS.txt对应公开版本。
- 夜间现场帧、原始录包、标签清单与现场JSONL输出保留在开发者本地。
