#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageNet 验证集整理脚本（基于官方 Devkit）
将散落的 JPEG 图片按类别移动到以 WordNet ID 命名的子文件夹中。
"""

import os
import shutil
import scipy.io as sio
from tqdm import tqdm

# ==================== 用户配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "ILSVRC2012_img_val")
OUTPUT_DIR = os.path.join(BASE_DIR, "ILSVRC2012_val_organized")
DEVKIT_DIR = os.path.join(BASE_DIR, "ILSVRC2012_devkit_t12", "data")
# =================================================

gt_file = os.path.join(DEVKIT_DIR, "ILSVRC2012_validation_ground_truth.txt")
mat_file = os.path.join(DEVKIT_DIR, "meta.mat")

# 读取数字标签（1~1000）
with open(gt_file, 'r') as f:
    labels = [int(line.strip()) for line in f.readlines()]

# 读取 meta.mat
meta = sio.loadmat(mat_file)

# 提取 WordNet ID（WNID）
# meta['synsets'] 是一个形状为 (1000, 1) 的结构数组，我们将其展平
synsets_struct = meta['synsets'].flatten()  # 现在长度为1000
wnids = [entry['WNID'][0] for entry in synsets_struct]  # 每个 entry 的 'WNID' 是字符串数组，取第一个元素

print(f"成功加载 {len(wnids)} 个类别")

# 获取图片列表
img_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.JPEG')])
if len(img_files) != 50000:
    print(f"⚠️ 图片数量 {len(img_files)}，预期 50000")
else:
    print("✅ 图片数量正确")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 移动图片
for idx, fname in enumerate(tqdm(img_files, desc="整理图片")):
    class_id = labels[idx] - 1  # 转为 0~999
    wnid = wnids[class_id]
    target_dir = os.path.join(OUTPUT_DIR, wnid)
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(
        os.path.join(SOURCE_DIR, fname),
        os.path.join(target_dir, fname)
    )

print("✅ 整理完成！")
