#!/usr/bin/env python3
"""
验证 ImageNet 整理后的类别文件夹
扫描指定目录，找出非空子目录，并打印其 WNID 和对应的英文名称。
"""

import scipy.io as sio
import os

# ==================== 用户配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "DATA", "train")
DEVKIT_DIR = os.path.join(BASE_DIR, "ILSVRC2012_devkit_t12", "data")
# ================================================

# 1. 检查必要文件
meta_path = os.path.join(DEVKIT_DIR, "meta.mat")
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"未找到 meta.mat: {meta_path}")
if not os.path.exists(OUTPUT_DIR):
    raise FileNotFoundError(f"未找到输出目录: {OUTPUT_DIR}")

# 2. 读取 meta.mat，建立 WNID -> words 映射
meta = sio.loadmat(meta_path)
synsets = meta['synsets'].flatten()
wnid_to_words = {}
for entry in synsets:
    wnid = entry['WNID'][0]      # 例如 'n01440764'
    words = entry['words'][0]    # 例如 'tench, Tinca tinca'
    wnid_to_words[wnid] = words

print(f"✅ 成功加载 {len(wnid_to_words)} 个类别映射")

# 3. 扫描 OUTPUT_DIR，找出非空子目录
all_dirs = [d for d in os.listdir(OUTPUT_DIR) 
            if os.path.isdir(os.path.join(OUTPUT_DIR, d))]

non_empty = []
for d in all_dirs:
    dir_path = os.path.join(OUTPUT_DIR, d)
    # 检查目录下是否有文件（至少有一张图片）
    files = [f for f in os.listdir(dir_path) 
             if os.path.isfile(os.path.join(dir_path, f))]
    if files:
        non_empty.append(d)

print(f"📁 总目录数: {len(all_dirs)}")
print(f"📂 非空目录数: {len(non_empty)}")
if len(non_empty) != 1000:
    print(f"⚠️ 非空目录数不是 1000，请检查整理是否完整")

# 4. 取前 5 个非空目录（按名称排序）并打印信息
non_empty_sorted = sorted(non_empty)
sample = non_empty_sorted[:5]

print("\n🔍 前 5 个非空类别（按文件夹名排序）：")
for i, wnid in enumerate(sample, 1):
    words = wnid_to_words.get(wnid, "未知（未在 meta 中找到）")
    dir_path = os.path.join(OUTPUT_DIR, wnid)
    img_count = len([f for f in os.listdir(dir_path) 
                     if os.path.isfile(os.path.join(dir_path, f))])
    print(f"{i:2d}: {wnid} -> {words}  (图片数: {img_count})")
