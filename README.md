# 深度学习全栈开发项目实践

图像分类课程项目，包含两个 Flask 应用示例、ImageNet 数据整理与知识蒸馏脚本，以及部署架构图。

## 目录

| 路径 | 内容 |
| --- | --- |
| `fullstack-演示案例代码/` | 基础图像分类 Web 示例，使用 AlexNet |
| `self/` | 扩展版 Web 应用，支持单张、批量和 URL 图片识别，可选 AlexNet 或蒸馏 ConvNeXt-Nano |
| `distll/` | ImageNet 验证集整理、划分、标签检查和蒸馏训练脚本；目录名沿用原项目 |
| `output/architecture/` | 部署架构图的 LaTeX 源码及 PNG |

## 运行 Web 示例

建议使用 Python 3.12 和虚拟环境。两个应用的依赖与运行目录相互独立：

```bash
cd fullstack-演示案例代码
python -m pip install -r requirements.txt
python app.py
```

扩展版应用：

```bash
cd self
python -m pip install -r requirements.txt
python app.py
```

首次使用 AlexNet 时，`torchvision` 会下载预训练权重。扩展版的 ConvNeXt-Nano 需要自行训练模型，并把 `best_student_distilled.pth` 放到 `distll/distill_output/`。该权重未包含在仓库中；只有使用 ConvNeXt-Nano 时才需要它。`self/app.py` 使用的 `imghdr` 模块在 Python 3.13 起已移除，因此扩展版请使用 Python 3.12。

蒸馏流程见 [`distll/README.md`](distll/README.md)。ImageNet 数据集、训练输出和检查点未纳入仓库。

## 收录范围

仓库保留源代码、必要示例图片和架构图。课程指导书、个人实验报告、临时渲染文件、测试缓存、上传图片副本及模型权重未纳入；原始本地项目未被改动。
