# 本模块负责加载深度学习模型（AlexNet 和蒸馏后的 ConvNeXt-Nano），
# 并提供统一的图像预测接口。支持模型缓存、设备自动检测和 ImageNet 标签映射。

import json
import logging
import torch
import torch.nn as nn                           # 新增：用于自定义网络结构
from torchvision import transforms
from torchvision.models import AlexNet_Weights  # 仅保留权重类，不再导入 alexnet
import timm

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. 设备自动检测
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 2. ImageNet 标签加载
# ------------------------------------------------------------------
try:
    with open('imagenet_class_index.json', 'r', encoding='utf-8') as f:
        labels = {int(k): v[1] for k, v in json.load(f).items()}
    logger.info("成功加载ImageNet标签")
except Exception as e:
    logger.warning(f"加载标签失败，使用备用标签: {e}")
    labels = {i: f"class_{i}" for i in range(1000)}

# ------------------------------------------------------------------
# 3. 图像预处理流水线
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------
# 4. 全局模型缓存
# ------------------------------------------------------------------
_model_cache = {}

# ------------------------------------------------------------------
# 5. 自定义 AlexNet 结构（完全展开）
# ------------------------------------------------------------------
class AlexNet(nn.Module):
    """
    显式定义的 AlexNet 网络结构，与 torchvision 官方实现完全一致。
    包含特征提取层（features）、自适应池化层（avgpool）和分类头（classifier）。
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        # 卷积特征提取模块
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应平均池化，固定输出 6×6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------
# 6. 模型加载函数（各自独立）
# ------------------------------------------------------------------

def load_alexnet():
    """
    加载自定义 AlexNet 结构，并从 torchvision 官方预训练权重中读取参数。
    不再直接调用 torchvision.models.alexnet，而是手动实例化并加载权重。
    """
    # 1. 创建自定义网络
    model = AlexNet(num_classes=1000, dropout=0.5)
    
    # 2. 获取官方预训练权重（不依赖 alexnet 函数，仅使用权重类）
    weights = AlexNet_Weights.IMAGENET1K_V1
    state_dict = weights.get_state_dict(progress=True)  # 自动下载并返回 state_dict
    
    # 3. 加载权重到自定义网络（严格匹配）
    model.load_state_dict(state_dict)
    
    # 4. 迁移至设备并切换为推理模式
    model.to(device)
    model.eval()
    logger.info("成功加载自定义展开结构的 AlexNet，并使用官方预训练权重")
    return model

def load_convnext_nano(weights_path="best_student_distilled.pth"):
    """
    （保持不变）加载蒸馏训练得到的 ConvNeXt-Nano 模型。
    """
    try:
        model = timm.create_model('convnext_nano', pretrained=False, num_classes=1000)
        state_dict = torch.load(weights_path, map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)
        model.eval()
        logger.info(f"成功加载 ConvNeXt-Nano 模型，权重来自 {weights_path}")
        return model
    except Exception as e:
        logger.error(f"加载 ConvNeXt-Nano 失败: {e}")
        raise RuntimeError(f"加载 ConvNeXt-Nano 失败: {e}") from e

# ------------------------------------------------------------------
# 7. 统一模型获取接口（带缓存）
# ------------------------------------------------------------------
def get_model(model_name="alexnet", weights_path="best_student_distilled.pth"):
    """
    根据模型名称返回对应的模型对象，并使用 _model_cache 进行缓存。
    """
    key = model_name
    if key not in _model_cache:
        if model_name == "alexnet":
            _model_cache[key] = load_alexnet()
        elif model_name == "convnext":
            _model_cache[key] = load_convnext_nano(weights_path)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
    return _model_cache[key]

# ------------------------------------------------------------------
# 8. 图像预测接口（对外核心函数）
# ------------------------------------------------------------------
def predict_image(img, model_name="alexnet", weights_path="best_student_distilled.pth"):
    """
    对给定的 PIL Image 对象进行预测，返回 Top-5 的类别名称和置信度（百分比）。
    """
    model = get_model(model_name, weights_path)
    img_tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        softmax = torch.nn.functional.softmax(output, dim=1)[0] * 100
        _, indices = torch.sort(output, descending=True)

    return [(labels.get(idx.item(), f"未知{idx.item()}"), softmax[idx].item())
            for idx in indices[0][:5]]
