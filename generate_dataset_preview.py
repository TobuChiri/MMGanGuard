#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成数据集预览图
从E:/data中挑选代表性图像，展示AI生成图像和自然图像的对比
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import random


def load_and_resize_image(image_path, target_size=(200, 200)):
    """
    加载并调整图像大小

    Args:
        image_path (str): 图像文件路径
        target_size (tuple): 目标尺寸 (width, height)

    Returns:
        numpy.ndarray: 调整大小后的图像数组
    """
    try:
        # 使用PIL加载图像
        img = Image.open(image_path)
        # 调整大小
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        # 转换为numpy数组
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None


def generate_dataset_preview():
    """生成数据集预览图"""
    # 数据路径
    data_dir = "E:/data/train"
    ai_dir = os.path.join(data_dir, "ai")
    nature_dir = os.path.join(data_dir, "nature")

    # 获取图像文件列表
    ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    nature_images = [os.path.join(nature_dir, f) for f in os.listdir(nature_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"找到 {len(ai_images)} 张AI生成图像")
    print(f"找到 {len(nature_images)} 张自然图像")

    # 随机选择图像（确保多样性）
    num_samples = 8  # 每个类别选择8张图像

    # 从AI图像中随机选择，确保包含不同生成器
    selected_ai = []
    ai_generators = {}

    # 按生成器分组
    for img_path in ai_images:
        filename = os.path.basename(img_path)
        generator = filename.split('_')[1] if '_' in filename else 'unknown'
        if generator not in ai_generators:
            ai_generators[generator] = []
        ai_generators[generator].append(img_path)

    # 从每个生成器中选择图像
    for generator, images in ai_generators.items():
        if images:
            selected_ai.extend(random.sample(images, min(2, len(images))))

    # 如果还不够，随机补充
    if len(selected_ai) < num_samples:
        remaining = num_samples - len(selected_ai)
        additional = random.sample([img for img in ai_images if img not in selected_ai],
                                 min(remaining, len(ai_images) - len(selected_ai)))
        selected_ai.extend(additional)

    # 随机选择自然图像
    selected_nature = random.sample(nature_images, min(num_samples, len(nature_images)))

    print(f"选择了 {len(selected_ai)} 张AI生成图像和 {len(selected_nature)} 张自然图像")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

    # 创建图表
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    fig.suptitle('数据集预览 - AI生成图像 vs 自然图像', fontsize=20, fontweight='bold')

    # 显示AI生成图像
    for i, img_path in enumerate(selected_ai[:num_samples]):
        img_array = load_and_resize_image(img_path)
        if img_array is not None:
            axes[0, i].imshow(img_array)
            filename = os.path.basename(img_path)
            generator = filename.split('_')[1] if '_' in filename else 'unknown'
            axes[0, i].set_title(f'AI: {generator}', fontsize=10)
        axes[0, i].axis('off')

    # 显示自然图像
    for i, img_path in enumerate(selected_nature[:num_samples]):
        img_array = load_and_resize_image(img_path)
        if img_array is not None:
            axes[1, i].imshow(img_array)
            axes[1, i].set_title('自然图像', fontsize=10)
        axes[1, i].axis('off')

    # 添加类别标签
    axes[0, 0].text(-0.3, 0.5, 'AI生成图像', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, va='center')
    axes[1, 0].text(-0.3, 0.5, '自然图像', transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, va='center')

    plt.tight_layout()

    # 保存图表
    output_path = "dataset_preview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"数据集预览图已保存到: {output_path}")

    plt.show()


def generate_statistics():
    """生成数据集统计信息"""
    data_dir = "E:/data"

    print("\n数据集统计信息:")
    print("="*50)

    for split in ['train', 'val']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            ai_dir = os.path.join(split_dir, "ai")
            nature_dir = os.path.join(split_dir, "nature")

            ai_count = len([f for f in os.listdir(ai_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(ai_dir) else 0
            nature_count = len([f for f in os.listdir(nature_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(nature_dir) else 0

            print(f"{split.capitalize()} 集:")
            print(f"  AI生成图像: {ai_count} 张")
            print(f"  自然图像: {nature_count} 张")
            print(f"  总计: {ai_count + nature_count} 张")
            print(f"  类别比例: AI:{ai_count/(ai_count+nature_count):.1%} / 自然:{nature_count/(ai_count+nature_count):.1%}")
            print()


def main():
    """主函数"""
    print("正在生成数据集预览图...")

    # 生成统计信息
    generate_statistics()

    # 生成预览图
    generate_dataset_preview()

    print("\n数据集预览生成完成！")


if __name__ == "__main__":
    main()