#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMGanGuard模型逐批次概率验证脚本
用于在别的数据集中验证已训练好的mmganguard模型
逐批次记录四个子模型的概率、加权概率和预测准确性
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# 导入模型
from models.co_occurrence_fast import create_co_occurrence_fast_model
from models.gram_net_paper import create_gram_net_paper_model as create_gram_net_model
import torchvision.models as models


class MMGanGuardBatchValidator:
    """MMGanGuard模型逐批次验证器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.weights = None

    def load_model(self, model_type, model_path):
        """
        加载指定类型的模型
        Args:
            model_type: 模型类型 ('resnet', 'densenet', 'co_occurrence', 'gram_net')
            model_path: 模型文件路径
        """
        print(f"加载 {model_type} 模型: {model_path}")

        # 创建模型
        if model_type == 'resnet':
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
        elif model_type == 'densenet':
            model = models.densenet201(pretrained=False)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 2)
        elif model_type == 'co_occurrence':
            model = create_co_occurrence_fast_model(num_classes=2, pretrained=False)
        elif model_type == 'gram_net':
            model = create_gram_net_model(num_classes=2, input_size=256)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        self.models[model_type] = model
        print(f"{model_type} 模型加载成功")

    def load_all_models(self, model_dir):
        """
        加载四个子模型
        Args:
            model_dir: 包含四个子模型文件的目录
        """
        print(f"加载四个子模型: {model_dir}")

        # 加载四个子模型
        model_files = {
            'resnet': os.path.join(model_dir, 'resnet.pth'),
            'densenet': os.path.join(model_dir, 'densenet.pth'),
            'co_occurrence': os.path.join(model_dir, 'co_occurrence.pth'),
            'gram_net': os.path.join(model_dir, 'gram_net.pth')
        }

        for model_type, model_path in model_files.items():
            if os.path.exists(model_path):
                self.load_model(model_type, model_path)
            else:
                print(f"警告: 找不到 {model_type} 模型文件: {model_path}")

        # 使用固定的软投票权重
        self.weights = torch.tensor([0.2, 0.6, 0.1, 0.1])  # resnet:0.2, densenet:0.6, co_occurrence:0.1, gram_net:0.1
        print(f"使用软投票权重: {self.weights}")

    def convert_to_probability(self, outputs, model_type):
        """将模型输出转换为真实图像的概率"""
        if model_type == 'co_occurrence':
            # 二元分类模型直接输出概率
            return outputs.squeeze()
        else:
            # 多分类模型：使用softmax获取真实图像的概率
            # 假设类别0是AI图像，类别1是真实图像
            probs = F.softmax(outputs, dim=1)
            return probs[:, 1]  # 返回真实图像的概率

    def validate_batch_probabilities(self, val_loader):
        """
        逐批次验证并记录概率
        Args:
            val_loader: 验证数据加载器
        Returns:
            list: 包含所有批次概率和预测结果的列表
        """
        if self.weights is None:
            raise ValueError("权重未设置，请先加载四个子模型")

        weights = self.weights.to(self.device)

        batch_results = []
        batch_counter = 0

        print("开始逐批次验证...")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="批次验证")):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)

                # 获取四个模型的输出概率
                model_probs = {}
                all_probs = []

                for model_name, model in self.models.items():
                    model = model.to(self.device)
                    model.eval()
                    output = model(inputs)

                    # 转换为概率
                    prob = self.convert_to_probability(output, model_name)
                    model_probs[model_name] = prob.cpu().numpy()
                    all_probs.append(prob)

                # 加权平均
                weighted_probs = torch.stack(all_probs, dim=1) * weights
                final_prob = weighted_probs.sum(dim=1)
                final_probs = final_prob.cpu().numpy()

                # 计算预测结果
                labels_np = labels.cpu().numpy()
                final_preds = (final_probs > 0.5).astype(int)
                is_correct = (final_preds == labels_np).astype(int)

                # 记录每个样本的结果
                for i in range(batch_size):
                    sample_result = {
                        'batch_id': batch_idx,
                        'sample_id': batch_counter + i,
                        'true_label': int(labels_np[i]),
                        'resnet_prob': float(model_probs['resnet'][i]),
                        'densenet_prob': float(model_probs['densenet'][i]),
                        'co_occurrence_prob': float(model_probs['co_occurrence'][i]),
                        'gram_net_prob': float(model_probs['gram_net'][i]),
                        'weighted_prob': float(final_probs[i]),
                        'predicted_label': int(final_preds[i]),
                        'is_correct': int(is_correct[i])
                    }
                    batch_results.append(sample_result)

                batch_counter += batch_size

        return batch_results

    def save_probability_results(self, batch_results, output_dir, dataset_name):
        """
        保存概率结果
        Args:
            batch_results: 批次结果列表
            output_dir: 输出目录
            dataset_name: 数据集名称
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON格式的详细结果
        json_path = os.path.join(output_dir, f"probability_results_{dataset_name}_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        print(f"详细概率结果已保存到: {json_path}")

        # 保存CSV格式的结果
        df = pd.DataFrame(batch_results)
        csv_path = os.path.join(output_dir, f"probability_summary_{dataset_name}_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"概率汇总结果已保存到: {csv_path}")

        # 计算并打印总体统计
        total_samples = len(batch_results)
        correct_predictions = sum(result['is_correct'] for result in batch_results)
        accuracy = correct_predictions / total_samples

        print(f"\n总体统计:")
        print(f"总样本数: {total_samples}")
        print(f"正确预测数: {correct_predictions}")
        print(f"准确率: {accuracy:.4f}")

        # 计算各模型的平均概率
        resnet_avg = np.mean([result['resnet_prob'] for result in batch_results])
        densenet_avg = np.mean([result['densenet_prob'] for result in batch_results])
        co_occurrence_avg = np.mean([result['co_occurrence_prob'] for result in batch_results])
        gram_net_avg = np.mean([result['gram_net_prob'] for result in batch_results])
        weighted_avg = np.mean([result['weighted_prob'] for result in batch_results])

        print(f"\n各模型平均概率:")
        print(f"ResNet: {resnet_avg:.4f}")
        print(f"DenseNet: {densenet_avg:.4f}")
        print(f"Co-occurrence: {co_occurrence_avg:.4f}")
        print(f"GramNet: {gram_net_avg:.4f}")
        print(f"加权平均: {weighted_avg:.4f}")

        return json_path, csv_path


def create_data_loader(data_dir, batch_size=32, image_size=256):
    """
    创建验证数据加载器
    Args:
        data_dir: 数据目录，应包含ai和nature子目录
        batch_size: 批次大小
        image_size: 图像尺寸
    Returns:
        DataLoader: 验证数据加载器
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 使用ImageFolder加载数据
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 检查类别
    if len(dataset.classes) != 2:
        print(f"警告: 数据集包含 {len(dataset.classes)} 个类别，期望2个类别 (ai, nature)")
        print(f"实际类别: {dataset.classes}")

    print(f"数据集: {data_dir}")
    print(f"样本数量: {len(dataset)}")
    print(f"类别: {dataset.classes}")

    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return loader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MMGanGuard模型逐批次概率验证脚本')

    # 模型参数
    parser.add_argument('--model_path', type=str, default='E:/code/mmganguard/checkpoints/best_final_model',
                       help='包含四个子模型的目录')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='E:/dataset/wukong/val',
                       help='验证数据集目录，应包含ai和nature子目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--image_size', type=int, default=256,
                       help='图像尺寸')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./probability_results',
                       help='结果输出目录')
    parser.add_argument('--dataset_name', type=str, default='unknown',
                       help='数据集名称（用于结果文件名）')

    args = parser.parse_args()

    print("MMGanGuard模型逐批次概率验证脚本")
    print("="*60)

    # 创建验证器
    validator = MMGanGuardBatchValidator()

    # 加载模型
    validator.load_all_models(args.model_path)

    # 创建数据加载器
    val_loader = create_data_loader(args.data_dir, args.batch_size, args.image_size)

    # 执行验证
    start_time = time.time()

    print("开始逐批次验证...")
    batch_results = validator.validate_batch_probabilities(val_loader)

    validation_time = time.time() - start_time

    # 保存结果
    json_path, csv_path = validator.save_probability_results(batch_results, args.output_dir, args.dataset_name)

    print(f"\n验证完成!")
    print(f"验证时间: {validation_time:.2f} 秒")
    print(f"详细概率结果: {json_path}")
    print(f"概率汇总结果: {csv_path}")


if __name__ == "__main__":
    main()