#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练结果并生成可视化图表和对比表格
从TensorBoard日志中提取数据
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_tensorboard_data(log_dir):
    """从TensorBoard日志中提取训练数据"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    data = {}

    # 提取所有标量数据
    tags = event_acc.Tags()['scalars']

    for tag in tags:
        events = event_acc.Scalars(tag)
        values = [event.value for event in events]
        steps = [event.step for event in events]

        # 按标签类型分类数据
        if 'Accuracy' in tag:
            if 'Train' in tag:
                data['train_accuracy'] = values
                data['train_epochs'] = steps
            elif 'Validation' in tag:
                data['val_accuracy'] = values
                data['val_epochs'] = steps
        elif 'Loss' in tag:
            if 'Train' in tag:
                data['train_loss'] = values
            elif 'Validation' in tag:
                data['val_loss'] = values
        elif 'Individual' in tag:
            # 处理各个子模型的数据
            parts = tag.split('/')
            if len(parts) >= 3:
                model_name = parts[1]
                metric_type = parts[2]

                if model_name not in data:
                    data[model_name] = {}

                if 'accuracy' in metric_type:
                    data[model_name]['val_accuracy'] = values
                elif 'f1' in metric_type:
                    data[model_name]['val_f1'] = values

    return data


def plot_training_comparison(data, save_path='training_comparison.png'):
    """生成训练结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 准确率对比
    ax1 = axes[0, 0]
    if 'train_accuracy' in data and 'train_epochs' in data:
        ax1.plot(data['train_epochs'], data['train_accuracy'],
                label='Final Train', linewidth=2, color='blue')
    if 'val_accuracy' in data and 'val_epochs' in data:
        ax1.plot(data['val_epochs'], data['val_accuracy'],
                label='Final Val', linewidth=2, color='red')

    # 添加各个子模型的验证准确率
    colors = ['green', 'orange', 'purple', 'brown']
    model_names = ['resnet', 'densenet', 'co_occurrence', 'gram_net']

    for i, model_name in enumerate(model_names):
        if model_name in data and 'val_accuracy' in data[model_name]:
            ax1.plot(range(len(data[model_name]['val_accuracy'])),
                    data[model_name]['val_accuracy'],
                    label=f'{model_name.capitalize()} Val',
                    linewidth=1.5, linestyle='--', color=colors[i])

    ax1.set_title('模型准确率对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 损失对比
    ax2 = axes[0, 1]
    if 'train_loss' in data and 'train_epochs' in data:
        ax2.plot(data['train_epochs'], data['train_loss'],
                label='Final Train', linewidth=2, color='blue')
    if 'val_loss' in data and 'val_epochs' in data:
        ax2.plot(data['val_epochs'], data['val_loss'],
                label='Final Val', linewidth=2, color='red')

    ax2.set_title('模型损失对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. F1分数对比
    ax3 = axes[1, 0]
    for i, model_name in enumerate(model_names):
        if model_name in data and 'val_f1' in data[model_name]:
            ax3.plot(range(len(data[model_name]['val_f1'])),
                    data[model_name]['val_f1'],
                    label=f'{model_name.capitalize()}',
                    linewidth=2, color=colors[i])

    ax3.set_title('各模型F1分数对比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 最终性能对比柱状图
    ax4 = axes[1, 1]
    models = []
    final_accuracies = []

    # Final模型
    if 'val_accuracy' in data:
        models.append('Final')
        final_accuracies.append(data['val_accuracy'][-1] if data['val_accuracy'] else 0)

    # 各个子模型
    for model_name in model_names:
        if model_name in data and 'val_accuracy' in data[model_name]:
            models.append(model_name.capitalize())
            final_accuracies.append(data[model_name]['val_accuracy'][-1]
                                  if data[model_name]['val_accuracy'] else 0)

    bars = ax4.bar(models, final_accuracies,
                   color=['red'] + colors[:len(models)-1], alpha=0.7)

    # 在柱状图上添加数值
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    ax4.set_title('最终验证准确率对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim(0, 1.0)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"对比图已保存到: {save_path}")


def create_performance_table(data):
    """创建性能指标对比表格"""
    table_data = []

    # Final模型数据
    if 'val_accuracy' in data:
        final_acc = data['val_accuracy'][-1] if data['val_accuracy'] else 0
        final_loss = data['val_loss'][-1] if 'val_loss' in data and data['val_loss'] else 0

        table_data.append({
            '模型': 'Final',
            '验证准确率': f"{final_acc:.4f}",
            '验证损失': f"{final_loss:.4f}",
            '最佳epoch': len(data['val_accuracy']) if 'val_accuracy' in data else 0
        })

    # 各个子模型数据
    model_names = ['resnet', 'densenet', 'co_occurrence', 'gram_net']
    for model_name in model_names:
        if model_name in data:
            model_data = data[model_name]
            val_acc = model_data.get('val_accuracy', [0])[-1] if model_data.get('val_accuracy') else 0
            val_f1 = model_data.get('val_f1', [0])[-1] if model_data.get('val_f1') else 0

            table_data.append({
                '模型': model_name.capitalize(),
                '验证准确率': f"{val_acc:.4f}",
                '验证F1分数': f"{val_f1:.4f}",
                '最佳epoch': len(model_data.get('val_accuracy', []))
            })

    # 创建DataFrame
    df = pd.DataFrame(table_data)

    # 保存为Markdown表格
    markdown_table = df.to_markdown(index=False)

    # 保存为CSV
    csv_path = 'performance_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n=== 性能指标对比表格 ===")
    print(markdown_table)
    print(f"\nCSV表格已保存到: {csv_path}")

    return df


def main():
    """主函数"""
    # TensorBoard日志目录
    log_dir = "checkpoints/tensorboard_logs/final_20251106_234625"

    if not os.path.exists(log_dir):
        print(f"错误: TensorBoard日志目录不存在: {log_dir}")
        print("请确保已经运行过Final模型训练")
        return

    print("正在从TensorBoard日志中提取数据...")

    # 提取数据
    data = extract_tensorboard_data(log_dir)

    if not data:
        print("警告: 未找到有效的训练数据")
        return

    print("数据提取完成!")
    print(f"找到的数据标签: {list(data.keys())}")

    # 生成对比图
    print("\n正在生成训练结果对比图...")
    plot_training_comparison(data, 'training_comparison.png')

    # 生成性能表格
    print("\n正在生成性能指标对比表格...")
    create_performance_table(data)

    print("\n=== 分析完成 ===")


if __name__ == "__main__":
    main()