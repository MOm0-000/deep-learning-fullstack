# -*- coding: utf-8 -*-
"""
知识蒸馏脚本：ConvNeXt-Base (教师) → ConvNeXt-Nano (学生)
数据集：ImageNet-1K (完整版，1000类)
硬件：RTX 5070 (12GB VRAM)
技术点：
1. 阶段1 (generate): 教师离线生成软标签（FP16存储，节省硬盘）
2. 阶段2 (train): 学生加载软标签，进行蒸馏微调（温度T=3.0）
新增功能：断点重训（--resume）、早停（--patience）
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

# ======================== 配置参数 ========================
parser = argparse.ArgumentParser(description="ConvNeXt 知识蒸馏")
parser.add_argument('--mode', type=str, required=True, choices=['generate', 'train'],
                    help='generate: 教师生成软标签 | train: 蒸馏训练学生')
parser.add_argument('--data_dir', type=str, required=True,
                    help='ImageNet-1K 数据集根目录（包含 train/ 和 val/ 子目录）')
parser.add_argument('--output_dir', type=str, default='./distill_output',
                    help='存放软标签和检查点的目录')
parser.add_argument('--batch_size', type=int, default=256,
                    help='生成和训练阶段的批次大小 (RTX 5070 建议 256)')
parser.add_argument('--epochs', type=int, default=5,
                    help='蒸馏训练 Epoch 数 (建议 3~5)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='学生微调学习率')
parser.add_argument('--temperature', type=float, default=3.0,
                    help='蒸馏温度 (T)')
parser.add_argument('--alpha', type=float, default=0.7,
                    help='软标签损失权重 (1-alpha 为硬标签损失权重)')
# ========== 新增参数 ==========
parser.add_argument('--resume', action='store_true',
                    help='从最近的检查点恢复训练')
parser.add_argument('--patience', type=int, default=5,
                    help='早停容忍度（连续多少个 epoch 验证准确率未提升则停止）')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备: {device}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ PyTorch 缓存已清空")

# ======================== 1. 数据预处理 (统一标准) ========================
# 注意：教师和学生使用完全相同的预处理，以确保软标签与输入对齐
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================== 2. 阶段1：教师生成软标签 ========================
def generate_soft_labels():
    print("\n========== 阶段1: 教师生成软标签 ==========")
    print(f"教师模型: convnext_base.fb_in22k_ft_in1k")
    
    # 加载教师模型 (冻结)
    teacher = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    teacher.to(device)
    teacher.eval()
    
    # 加载训练集 (不进行随机打乱，确保顺序固定，方便后续索引对齐)
    train_dataset = ImageFolder(
        root=os.path.join(args.data_dir, 'train'),
        transform=transform_train
    )
    # 关键：shuffle=False 保证样本顺序与索引严格一致
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    print(f"📂 训练集样本数: {len(train_dataset)}")
    print(f"🔮 开始生成软标签 (FP16 存储)...")
    
    all_soft_labels = []
    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc="教师推理"):
            images = images.to(device)
            logits = teacher(images)
            all_soft_labels.append(logits.cpu().to(torch.float16))
    
    soft_labels_tensor = torch.cat(all_soft_labels, dim=0)
    save_path = os.path.join(args.output_dir, 'soft_labels.pt')
    torch.save(soft_labels_tensor, save_path)
    print(f"✅ 软标签已保存至: {save_path} (形状: {soft_labels_tensor.shape})")
    print("✅ 阶段1 完成！")

# ======================== 3. 自定义数据集：返回索引以便对齐软标签 ========================
class DistillImageFolder(ImageFolder):
    """重写 __getitem__，额外返回样本索引，用于精确对齐软标签"""
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

# ======================== 4. 阶段2：蒸馏训练学生（含断点重训和早停） ========================
def train_student():
    print("\n========== 阶段2: 蒸馏训练学生 ==========")
    print(f"学生模型: convnext_nano.d1h_in1k")
    
    # 加载软标签
    soft_label_path = os.path.join(args.output_dir, 'soft_labels.pt')
    if not os.path.exists(soft_label_path):
        raise FileNotFoundError(f"请先运行生成阶段: {soft_label_path} 不存在")
    
    print(f"📥 加载软标签: {soft_label_path}")
    soft_labels = torch.load(soft_label_path, map_location='cpu')  # [N, 1000] FP16
    print(f"软标签形状: {soft_labels.shape}")
    
    # ---------- 构建学生模型 ----------
    student = timm.create_model('convnext_nano.d1h_in1k', pretrained=False)
    # 指定本地权重路径（若存在则加载，否则尝试在线下载）
    ckpt_path = os.path.expanduser('~/.cache/torch/hub/checkpoints/convnext_nano.d1h_in1k.safetensors')
    if os.path.exists(ckpt_path):
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path)
        student.load_state_dict(state_dict, strict=True)
        print("✅ 从本地 safetensors 加载学生权重")
    else:
        print("⚠️ 本地权重不存在，尝试在线下载...")
        student = timm.create_model('convnext_nano.d1h_in1k', pretrained=True)
        print("✅ 在线下载并加载学生权重")
    
    student.to(device)
    student.train()
    
    # ---------- 数据加载 ----------
    train_dataset = DistillImageFolder(
        root=os.path.join(args.data_dir, 'train'),
        transform=transform_train
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_dataset = ImageFolder(
        root=os.path.join(args.data_dir, 'val'),
        transform=transform_val
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # ---------- 优化器与调度器 ----------
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ---------- 损失函数与超参数 ----------
    alpha = args.alpha
    temperature = args.temperature
    hard_loss_fn = nn.CrossEntropyLoss()
    
    print(f"⚙️ 蒸馏参数: T={temperature}, alpha={alpha}, lr={args.lr}, epochs={args.epochs}")
    print(f"📊 训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}")
    
    # ---------- 检查点路径 ----------
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
    best_model_path = os.path.join(args.output_dir, 'best_student_distilled.pth')
    
    # ---------- 初始化训练状态 ----------
    start_epoch = 1
    best_acc = 0.0
    patience_counter = 0
    
    # ---------- 断点重训逻辑 ----------
    if args.resume and os.path.exists(checkpoint_path):
        print(f"📥 从检查点恢复: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        student.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"✅ 恢复成功，从 Epoch {start_epoch} 继续，当前最佳准确率: {best_acc:.4f}")
    else:
        if args.resume:
            print("⚠️ 未找到检查点文件，从头开始训练")
        # 如果最佳模型权重存在，可加载作为初始权重（但不恢复优化器状态）
        if os.path.exists(best_model_path) and not args.resume:
            # 只加载最佳模型权重，优化器从零开始
            print(f"📥 加载之前的最佳模型权重作为初始权重: {best_model_path}")
            state_dict = torch.load(best_model_path, map_location='cpu')
            student.load_state_dict(state_dict)
            # 此时 best_acc 保留为 0，会重新追踪
            print("✅ 权重加载成功，优化器将重新初始化")
    
    # ---------- 训练循环 ----------
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        student.train()
        running_loss = 0.0
        
        # 训练一个 epoch
        pbar = tqdm(train_loader, desc="训练")
        for images, hard_labels, indices in pbar:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            teacher_logits = soft_labels[indices].to(device).to(torch.float32)
            
            student_logits = student(images)
            
            # 蒸馏损失
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
            
            # 硬标签损失
            hard_loss = hard_loss_fn(student_logits, hard_labels)
            
            # 总损失
            loss = alpha * distill_loss + (1 - alpha) * hard_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"📉 平均训练损失: {avg_loss:.4f}")
        
        # ---------- 验证 ----------
        student.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="验证"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = student(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"🎯 验证集 Top-1 准确率: {acc:.4f}")
        
        # ---------- 保存最佳模型与检查点 ----------
        if acc > best_acc:
            best_acc = acc
            # 保存最佳模型权重
            torch.save(student.state_dict(), best_model_path)
            print(f"✅ 最佳模型已保存: {best_model_path} (准确率: {best_acc:.4f})")
            # 重置早停计数器
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ 早停计数: {patience_counter}/{args.patience}")
        
        # 保存完整的检查点（包含优化器、调度器状态）
        checkpoint = {
            'epoch': epoch,
            'model_state': student.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_acc': best_acc,
            'patience_counter': patience_counter,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 检查点已保存: {checkpoint_path}")
        
        # ---------- 早停判断 ----------
        if patience_counter >= args.patience:
            print(f"🛑 早停触发：连续 {args.patience} 个 epoch 验证准确率未提升，停止训练。")
            break
    
    print(f"\n🏆 蒸馏训练完成！最佳准确率: {best_acc:.4f}")
    print(f"📁 所有输出文件位于: {args.output_dir}")

# ======================== 5. 主入口 ========================
if __name__ == '__main__':
    if args.mode == 'generate':
        generate_soft_labels()
    elif args.mode == 'train':
        train_student()
    else:
        raise ValueError("mode 必须为 'generate' 或 'train'")
