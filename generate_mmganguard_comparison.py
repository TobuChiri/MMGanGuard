#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从MMGuardian JSON数据文件生成子模型对比图表
分析四个子模型（ResNet、DenseNet、Co-occurrence、GramNet）和加权结果的性能对比
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_mmganguard_data(json_path):
    """
    从MMGuardian JSON文件加载数据

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


def extract_submodel_data(data):
    """
    从MMGuardian数据中提取子模型数据

    Args:
        data (dict): MMGuardian JSON数据

    Returns:
        dict: 子模型数据字典
    """
    submodel_data = {}

    # 主模型数据
    submodel_data['mmganguard'] = {
        'train_loss': data['training_history']['train_loss'],
        'train_accuracy': data['training_history']['train_accuracy'],
        'val_loss': data['training_history']['val_loss'],
        'val_accuracy': data['training_history']['val_accuracy'],
        'learning_rate': data['training_history']['learning_rate'],
        'epoch_time': data['training_history']['epoch_time'],
        'total_params': data['model_metrics']['total_parameters'],
        'model_size_mb': data['model_metrics']['model_size_mb'],
        'best_val_acc': data['best_epoch']['val_accuracy'],
        'best_val_loss': data['best_epoch']['val_loss'],
        'final_val_acc': data['training_summary']['final_val_accuracy'],
        'final_train_acc': data['training_summary']['final_train_accuracy'],
        'training_time': data['training_summary']['total_training_time']
    }

    # 子模型数据
    submodels = ['resnet', 'densenet', 'co_occurrence', 'gram_net']

    for submodel in submodels:
        submodel_data[submodel] = {
            'train_loss': data['submodel_history']['train_loss'][submodel],
            'train_accuracy': data['submodel_history']['train_accuracy'][submodel],
            'val_loss': data['submodel_history']['val_loss'][submodel],
            'val_accuracy': data['submodel_history']['val_accuracy'][submodel],
            'train_time': data['submodel_history']['train_time'][submodel],
            'val_time': data['submodel_history']['val_time'][submodel],
            'train_precision_recall_f1': data['submodel_history']['train_precision_recall_f1'][submodel],
            'val_precision_recall_f1': data['submodel_history']['val_precision_recall_f1'][submodel]
        }

        # 计算子模型的训练时间总和
        submodel_data[submodel]['total_training_time'] = sum(submodel_data[submodel]['train_time'])

        # 提取最佳验证准确率
        submodel_data[submodel]['best_val_acc'] = max(submodel_data[submodel]['val_accuracy']) if submodel_data[submodel]['val_accuracy'] else 0
        submodel_data[submodel]['final_val_acc'] = submodel_data[submodel]['val_accuracy'][-1] if submodel_data[submodel]['val_accuracy'] else 0
        submodel_data[submodel]['final_train_acc'] = submodel_data[submodel]['train_accuracy'][-1] if submodel_data[submodel]['train_accuracy'] else 0

        # 提取精确率、召回率、F1分数
        if submodel_data[submodel]['train_precision_recall_f1']:
            last_train_metrics = submodel_data[submodel]['train_precision_recall_f1'][-1]
            submodel_data[submodel]['train_macro_f1'] = last_train_metrics['macro_f1']
            submodel_data[submodel]['train_macro_precision'] = last_train_metrics['macro_precision']
            submodel_data[submodel]['train_macro_recall'] = last_train_metrics['macro_recall']

        if submodel_data[submodel]['val_precision_recall_f1']:
            last_val_metrics = submodel_data[submodel]['val_precision_recall_f1'][-1]
            submodel_data[submodel]['val_macro_f1'] = last_val_metrics['macro_f1']
            submodel_data[submodel]['val_macro_precision'] = last_val_metrics['macro_precision']
            submodel_data[submodel]['val_macro_recall'] = last_val_metrics['macro_recall']

    return submodel_data


def generate_comprehensive_comparison(data, output_dir, experiment_name):
    """
    生成综合对比图表（三行三列，九个子图）

    Args:
        data (dict): 子模型数据
        output_dir (str): 输出目录
        experiment_name (str): 实验名称
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

    models = list(data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 创建图表 - 3行3列
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle(f'MMGuardian模型对比分析({experiment_name})', fontsize=20, fontweight='bold')

    # 1. 验证准确率对比（柱状图）- 左上
    ax1 = axes[0, 0]
    best_val_accs = [data[model]['best_val_acc'] for model in models]
    bars1 = ax1.bar(models, best_val_accs, color=colors[:len(models)])
    ax1.set_title('最佳验证准确率对比', fontsize=14, fontweight='bold')
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
        val_accs = data[model]['val_accuracy']
        epochs = range(1, len(val_accs) + 1)
        ax2.plot(epochs, val_accs, label=model, color=colors[i], linewidth=2)
    ax2.set_title('验证准确率曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 训练时间对比 - 右上
    ax3 = axes[0, 2]
    training_times = [data[model]['total_training_time'] if 'total_training_time' in data[model] else data[model]['training_time'] for model in models]
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

    # 4. 训练损失曲线 - 中左
    ax4 = axes[1, 0]
    for i, model in enumerate(models):
        train_losses = data[model]['train_loss']
        epochs = range(1, len(train_losses) + 1)
        ax4.plot(epochs, train_losses, label=model, color=colors[i], linewidth=2)
    ax4.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('损失值')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 验证损失曲线 - 中中
    ax5 = axes[1, 1]
    for i, model in enumerate(models):
        val_losses = data[model]['val_loss']
        epochs = range(1, len(val_losses) + 1)
        ax5.plot(epochs, val_losses, label=model, color=colors[i], linewidth=2)
    ax5.set_title('验证损失曲线', fontsize=14, fontweight='bold')
    ax5.set_xlabel('训练轮次')
    ax5.set_ylabel('损失值')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 训练F1分数对比 - 中右
    ax6 = axes[1, 2]
    train_f1_scores = [data[model].get('train_macro_f1', 0) for model in models]
    bars6 = ax6.bar(models, train_f1_scores, color=colors[:len(models)])
    ax6.set_title('训练F1分数对比', fontsize=14, fontweight='bold')
    ax6.set_ylabel('F1分数')
    ax6.set_ylim(0, 1.1)
    ax6.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars6:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    ax6.grid(True, alpha=0.3)

    # 7. 验证F1分数对比 - 左下
    ax7 = axes[2, 0]
    val_f1_scores = [data[model].get('val_macro_f1', 0) for model in models]
    bars7 = ax7.bar(models, val_f1_scores, color=colors[:len(models)])
    ax7.set_title('验证F1分数对比', fontsize=14, fontweight='bold')
    ax7.set_ylabel('F1分数')
    ax7.set_ylim(0, 1.1)
    ax7.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars7:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    ax7.grid(True, alpha=0.3)

    # 8. 训练精确率对比 - 中下
    ax8 = axes[2, 1]
    train_precision = [data[model].get('train_macro_precision', 0) for model in models]
    bars8 = ax8.bar(models, train_precision, color=colors[:len(models)])
    ax8.set_title('训练精确率对比', fontsize=14, fontweight='bold')
    ax8.set_ylabel('精确率')
    ax8.set_ylim(0, 1.1)
    ax8.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars8:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    ax8.grid(True, alpha=0.3)

    # 9. 训练召回率对比 - 右下
    ax9 = axes[2, 2]
    train_recall = [data[model].get('train_macro_recall', 0) for model in models]
    bars9 = ax9.bar(models, train_recall, color=colors[:len(models)])
    ax9.set_title('训练召回率对比', fontsize=14, fontweight='bold')
    ax9.set_ylabel('召回率')
    ax9.set_ylim(0, 1.1)
    ax9.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars9:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'mmganguard_comprehensive_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"综合对比图表已保存到: {output_path}")

    plt.close()


def generate_performance_summary(data, output_dir, experiment_name):
    """
    生成性能汇总图表

    Args:
        data (dict): 子模型数据
        output_dir (str): 输出目录
        experiment_name (str): 实验名称
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

    models = list(data.keys())

    # 创建性能汇总图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'MMGuardian模型性能汇总({experiment_name})', fontsize=16, fontweight='bold')

    # 1. 最佳验证准确率对比
    ax1 = axes[0, 0]
    best_val_accs = [data[model]['best_val_acc'] for model in models]
    bars1 = ax1.bar(models, best_val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax1.set_title('最佳验证准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 2. 最终验证准确率对比
    ax2 = axes[0, 1]
    final_val_accs = [data[model]['final_val_acc'] for model in models]
    bars2 = ax2.bar(models, final_val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax2.set_title('最终验证准确率对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('准确率')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 3. 训练时间对比
    ax3 = axes[1, 0]
    training_times = [data[model]['total_training_time'] if 'total_training_time' in data[model] else data[model]['training_time'] for model in models]
    bars3 = ax3.bar(models, training_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax3.set_title('训练时间对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('训练时间(秒)')
    ax3.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # 4. 验证F1分数对比
    ax4 = axes[1, 1]
    val_f1_scores = [data[model].get('val_macro_f1', 0) for model in models]
    bars4 = ax4.bar(models, val_f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax4.set_title('验证F1分数对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('F1分数')
    ax4.set_ylim(0, 1.1)
    ax4.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'mmganguard_performance_summary_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"性能汇总图表已保存到: {output_path}")

    plt.close()


def print_comparison_table(data):
    """
    打印模型对比表格

    Args:
        data (dict): 子模型数据
    """
    print("\n" + "="*120)
    print("MMGuardian模型性能对比表")
    print("="*120)
    print(f"{'模型':<15} {'最佳验证准确率':<15} {'最终验证准确率':<15} {'训练时间(秒)':<15} {'验证F1分数':<15} {'训练F1分数':<15} {'验证精确率':<15} {'验证召回率':<15}")
    print("-"*120)

    for model in data.keys():
        model_data = data[model]
        print(f"{model:<15} {model_data['best_val_acc']:<15.4f} {model_data['final_val_acc']:<15.4f} "
              f"{model_data.get('total_training_time', model_data.get('training_time', 0)):<15.2f} "
              f"{model_data.get('val_macro_f1', 0):<15.4f} {model_data.get('train_macro_f1', 0):<15.4f} "
              f"{model_data.get('val_macro_precision', 0):<15.4f} {model_data.get('val_macro_recall', 0):<15.4f}")

    print("="*120)

    # 找出最佳模型
    best_model = max(data.keys(), key=lambda x: data[x]['best_val_acc'])
    print(f"\n最佳模型: {best_model} (最佳验证准确率: {data[best_model]['best_val_acc']:.4f})")

    # 分析加权组合效果
    if 'mmganguard' in data:
        mmganguard_acc = data['mmganguard']['best_val_acc']
        submodel_accs = [data[model]['best_val_acc'] for model in data.keys() if model != 'mmganguard']
        best_submodel_acc = max(submodel_accs)
        improvement = mmganguard_acc - best_submodel_acc
        print(f"加权组合相对于最佳子模型的提升: {improvement:.4f} ({improvement*100:.2f}%)")


def main():
    """主函数"""
    # JSON文件路径
    json_path = "checkpoints/final_20251111_014443.json"

    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 找不到JSON文件 {json_path}")
        print("请确保路径正确或先运行训练脚本生成数据")
        return

    # 加载数据
    data = load_mmganguard_data(json_path)
    if data is None:
        return

    # 提取子模型数据
    submodel_data = extract_submodel_data(data)

    # 输出目录
    output_dir = os.path.dirname(json_path)
    experiment_name = os.path.basename(json_path).replace('.json', '')

    print("正在生成MMGuardian模型综合对比图表...")

    # 生成综合对比图表
    generate_comprehensive_comparison(submodel_data, output_dir, experiment_name)

    # 生成性能汇总图表
    generate_performance_summary(submodel_data, output_dir, experiment_name)

    # 打印对比表格
    print_comparison_table(submodel_data)

    print(f"\n图表生成完成！")
    print(f"- 综合对比图表: {output_dir}/mmganguard_comprehensive_comparison_*.png")
    print(f"- 性能汇总图表: {output_dir}/mmganguard_performance_summary_*.png")


if __name__ == "__main__":
    main()