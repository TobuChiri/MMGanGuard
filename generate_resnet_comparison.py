#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从JSON数据文件生成ResNet模型对比图表
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_json_data(json_path):
    """
    从JSON文件加载数据

    Args:
        json_path (str): JSON文件路径

    Returns:
        dict: 模型数据字典
    """
    if not os.path.exists(json_path):
        print(f"错误: 找不到JSON文件 {json_path}")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载JSON文件: {json_path}")
        return data
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None


def generate_comprehensive_comparison(data, output_dir):
    """
    生成综合对比图表（两行三列，六个子图）

    Args:
        data (dict): 模型数据
        output_dir (str): 输出目录
    """
    # 设置中文字体 - 使用系统可用的中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

    models = list(data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 获取本级文件夹名称作为标题
    current_dir_name = os.path.basename(output_dir)

    # 创建图表 - 2行3列
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'ResNet模型对比分析({current_dir_name})', fontsize=18, fontweight='bold')

    # 1. 验证准确率对比（柱状图）- 左上
    ax1 = axes[0, 0]
    best_val_accs = [data[model]['best_val_acc'] for model in models]
    bars1 = ax1.bar(models, best_val_accs, color=colors[:len(models)])
    ax1.set_title('验证准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 验证准确率曲线 - 中上
    ax2 = axes[0, 1]
    for i, model in enumerate(models):
        val_accs = data[model]['val_accs']
        epochs = range(1, len(val_accs) + 1)
        ax2.plot(epochs, val_accs, label=model, color=colors[i], linewidth=2)
    ax2.set_title('验证准确率曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 训练时间对比 - 右上
    ax3 = axes[0, 2]
    training_times = [data[model]['training_time'] for model in models]
    bars3 = ax3.bar(models, training_times, color=colors[:len(models)])
    ax3.set_title('训练时间对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('训练时间(秒)')
    ax3.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. 模型参数量对比 - 左下
    ax4 = axes[1, 0]
    total_params = [data[model]['total_params'] for model in models]
    bars4 = ax4.bar(models, total_params, color=colors[:len(models)])
    ax4.set_title('模型参数量对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('参数量')
    ax4.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(total_params)*0.02,
                f'{height:,}', ha='center', va='bottom', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. 训练损失曲线 - 中下
    ax5 = axes[1, 1]
    for i, model in enumerate(models):
        train_losses = data[model]['train_losses']
        epochs = range(1, len(train_losses) + 1)
        ax5.plot(epochs, train_losses, label=model, color=colors[i], linewidth=2)
    ax5.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax5.set_xlabel('训练轮次')
    ax5.set_ylabel('损失值')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 验证损失曲线 - 右下
    ax6 = axes[1, 2]
    for i, model in enumerate(models):
        val_losses = data[model]['val_losses']
        epochs = range(1, len(val_losses) + 1)
        ax6.plot(epochs, val_losses, label=model, color=colors[i], linewidth=2)
    ax6.set_title('验证损失曲线', fontsize=14, fontweight='bold')
    ax6.set_xlabel('训练轮次')
    ax6.set_ylabel('损失值')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'resnet_comprehensive_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"综合对比图表已保存到: {output_path}")

    plt.close()


def generate_performance_summary(data, output_dir):
    """
    生成性能汇总图表

    Args:
        data (dict): 模型数据
        output_dir (str): 输出目录
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

    models = list(data.keys())

    # 提取性能指标
    total_params = [data[model]['total_params'] for model in models]
    best_val_accs = [data[model]['best_val_acc'] for model in models]
    final_val_accs = [data[model]['final_val_acc'] for model in models]
    training_times = [data[model]['training_time'] for model in models]

    # 创建性能汇总图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ResNet模型性能汇总', fontsize=16, fontweight='bold')

    # 1. 参数量对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, total_params, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax1.set_title('模型参数量对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('参数量')
    ax1.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100000,
                f'{height:,}', ha='center', va='bottom', fontsize=10)

    # 2. 最佳验证准确率对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, best_val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax2.set_title('最佳验证准确率对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('准确率')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 3. 最终验证准确率对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, final_val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax3.set_title('最终验证准确率对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('准确率')
    ax3.set_ylim(0, 1.1)
    ax3.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 4. 训练时间对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, training_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax4.set_title('训练时间对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('训练时间(秒)')
    ax4.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'resnet_performance_summary_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"性能汇总图表已保存到: {output_path}")

    plt.close()


def print_comparison_table(data):
    """
    打印模型对比表格

    Args:
        data (dict): 模型数据
    """
    print("\n" + "="*80)
    print("ResNet模型性能对比表")
    print("="*80)
    print(f"{'模型':<12} {'参数量':<15} {'最佳验证准确率':<15} {'最终验证准确率':<15} {'训练时间(秒)':<15}")
    print("-"*80)

    for model in data.keys():
        model_data = data[model]
        print(f"{model:<12} {model_data['total_params']:<15,} {model_data['best_val_acc']:<15.4f} {model_data['final_val_acc']:<15.4f} {model_data['training_time']:<15.2f}")

    print("="*80)

    # 找出最佳模型
    best_model = max(data.keys(), key=lambda x: data[x]['best_val_acc'])
    print(f"\n最佳模型: {best_model} (最佳验证准确率: {data[best_model]['best_val_acc']:.4f})")


def main():
    """主函数"""
    # JSON文件路径
    json_path = "checkpoints/resnet_comparison/data160000/comparison_results.json"

    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 找不到JSON文件 {json_path}")
        print("请确保路径正确或先运行训练脚本生成数据")
        return

    # 加载数据
    data = load_json_data(json_path)
    if data is None:
        return

    # 输出目录
    output_dir = os.path.dirname(json_path)

    print("正在生成模型综合对比图表...")

    # 生成综合对比图表
    generate_comprehensive_comparison(data, output_dir)

    # 打印对比表格
    print_comparison_table(data)

    print(f"\n图表生成完成！")
    print(f"- 综合对比图表: {output_dir}/resnet_comprehensive_comparison_*.png")


if __name__ == "__main__":
    main()