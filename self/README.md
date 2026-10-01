## 快速开始

使用 Python 3.12，在本目录中运行：

```bash
python -m pip install -r requirements.txt
python app.py
```

使用 ConvNeXt-Nano 前，需要把训练得到的 `best_student_distilled.pth` 放到 `../distll/distill_output/`。首次使用 AlexNet 时会下载预训练权重。

## 简易深度学习图像识别Web应用

一个使用Gunicorn、Flask和PyTorch构建的简易图像分类Web应用。

## 功能特性

- 用户友好的Web界面
- 支持常见图片格式上传（JPG, PNG, JPEG, GIF）
- 使用AlexNet进行图像识别
- 返回前5个最可能的类别及其置信度
- 实时图片预览
- 可批量上传
- 可URL解析图片并识别

