## 快速开始

```bash
cd /your/path/to/distll
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com
sudo apt update
sudo apt install aria2    #安装多线程下载工具，下载数据集更快
aria2c -x 16 -s 16   --max-tries=0   --retry-wait=1   --timeout=30   --connect-timeout=10   --split=16   --min-split-size=1M  "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"     #下载数据集，下载后请解压
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz     #下载映射关系，下载后请解压
python ./organize.py                   #将下载的原始数据集按照映射关系划分到子文件夹中，生成ILSVRC2012_val_organized
python ./datasplit.py                  #将 ILSVRC2012_val_organized切分成train与val，生成DATA
python distill.py --mode generate --data_dir /your/data --output_dir ./distill_output --batch_size 256 #第一步：教师生成软标签，生成soft_labels.pt
python distill.py --mode train \
                  --data_dir ./DATA \
                  --output_dir ./distill_output \
                  --batch_size 64 \
                  --epochs 5 \
                  --lr 2.5e-5 \
                  --temperature 3.0 \
                  --alpha 0.7 \
                  --resume \
                  --patience 3       #第二步：蒸馏训练学生，resume为启用断点重训，patience为早停机制
```

## 简易的蒸馏脚本

教师为 convnext_base.fb_in22k_ft_in1k（先在ImageNet-22K的1400万张图上预训练，再在1K上微调）；学生为ConvNeXt-Nano（只在1K上训练过）。教师有学生从未见过的1400万张图的视觉先验。选择ImageNet Large Scale Visual Recognition Challenge (ILSVRC)的验证集作为蒸馏的数据集，共5万张图片，切分为4万张训练集+1万张验证集。
