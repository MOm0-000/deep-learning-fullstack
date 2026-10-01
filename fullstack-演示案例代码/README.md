# 基础图像分类 Web 示例

这是 Flask + PyTorch 的基础版图像分类应用，使用 AlexNet 预训练权重，对上传图片返回前五个预测类别。

在本目录中安装依赖并运行：

```bash
python -m pip install -r requirements.txt
python app.py
```

首次运行需要下载 AlexNet 权重。上传图片保存到 `static/uploads/`，该目录不会纳入 Git。
