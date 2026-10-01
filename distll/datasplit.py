#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageNet 验证集拆分工具
功能：将每个类别下的图片按比例随机拆分为训练集和验证集，保留子文件夹结构。
用法：直接运行，或通过命令行参数覆盖配置。
"""

import os
import random
import shutil
from tqdm import tqdm

# ==================== 用户配置参数（请按需修改） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "ILSVRC2012_val_organized")  # 原始验证集目录（包含1000个子文件夹）
TARGET_DIR = os.path.join(BASE_DIR, "DATA")  # 目标根目录（将在其下创建 train/ 和 val/）
TRAIN_RATIO = 0.8                                      # 训练集比例（剩余为验证集）
RANDOM_SEED = 42                                       # 随机种子，确保可复现
MOVE_FILES = True                                      # True=移动文件，False=复制文件（复制会占用双倍空间）
OVERWRITE = False                                      # 如果目标文件已存在，是否覆盖（建议False，会跳过）
# ================================================================

def ensure_dir(path):
    """确保目录存在，若不存在则创建"""
    os.makedirs(path, exist_ok=True)

def split_dataset():
    """执行拆分主逻辑"""
    # 检查源目录是否存在
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"源目录不存在: {SOURCE_DIR}")
    
    # 创建目标目录
    train_root = os.path.join(TARGET_DIR, "train")
    val_root = os.path.join(TARGET_DIR, "val")
    ensure_dir(train_root)
    ensure_dir(val_root)
    
    # 获取所有类别子文件夹
    class_dirs = [d for d in os.listdir(SOURCE_DIR) 
                  if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    class_dirs.sort()  # 保持顺序一致
    print(f"发现 {len(class_dirs)} 个类别文件夹")
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    
    # 统计信息
    total_train = 0
    total_val = 0
    
    # 遍历每个类别
    for cls in tqdm(class_dirs, desc="处理类别"):
        src_cls_path = os.path.join(SOURCE_DIR, cls)
        # 获取该类别下所有图片文件（支持常见格式）
        images = [f for f in os.listdir(src_cls_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'))]
        if not images:
            print(f"⚠️ 类别 {cls} 中没有图片，跳过")
            continue
        
        # 打乱图片列表
        random.shuffle(images)
        
        # 计算分割点
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 创建目标子文件夹
        train_cls_path = os.path.join(train_root, cls)
        val_cls_path = os.path.join(val_root, cls)
        ensure_dir(train_cls_path)
        ensure_dir(val_cls_path)
        
        # 移动/复制训练集图片
        for img in train_images:
            src_path = os.path.join(src_cls_path, img)
            dst_path = os.path.join(train_cls_path, img)
            if not OVERWRITE and os.path.exists(dst_path):
                continue
            if MOVE_FILES:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        
        # 移动/复制验证集图片
        for img in val_images:
            src_path = os.path.join(src_cls_path, img)
            dst_path = os.path.join(val_cls_path, img)
            if not OVERWRITE and os.path.exists(dst_path):
                continue
            if MOVE_FILES:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        
        total_train += len(train_images)
        total_val += len(val_images)
    
    print("\n✅ 拆分完成！")
    print(f"  训练集图片总数: {total_train}")
    print(f"  验证集图片总数: {total_val}")
    print(f"  目标目录: {TARGET_DIR}")

if __name__ == "__main__":
    # 支持通过命令行参数覆盖配置（方便自动化）
    import argparse
    parser = argparse.ArgumentParser(description="拆分ImageNet验证集")
    parser.add_argument("--source", help="源目录（覆盖SOURCE_DIR）")
    parser.add_argument("--target", help="目标目录（覆盖TARGET_DIR）")
    parser.add_argument("--ratio", type=float, help="训练集比例（覆盖TRAIN_RATIO）")
    parser.add_argument("--seed", type=int, help="随机种子（覆盖RANDOM_SEED）")
    parser.add_argument("--copy", action="store_true", help="复制而非移动文件")
    args = parser.parse_args()
    
    # 用命令行参数覆盖配置变量
    if args.source:
        SOURCE_DIR = args.source
    if args.target:
        TARGET_DIR = args.target
    if args.ratio is not None:
        TRAIN_RATIO = args.ratio
    if args.seed is not None:
        RANDOM_SEED = args.seed
    if args.copy:
        MOVE_FILES = False
    
    # 检查比例合法性
    if not (0 < TRAIN_RATIO < 1):
        raise ValueError("TRAIN_RATIO 必须在 0 到 1 之间")
    
    print("===== 配置信息 =====")
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    print(f"训练集比例: {TRAIN_RATIO:.2f}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"操作模式: {'移动' if MOVE_FILES else '复制'}")
    print("===================")
    
    split_dataset()
