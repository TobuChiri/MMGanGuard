#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证集随机样本验证脚本
从验证集中随机选择4张AI图像和4张Nature图像
用四个训练好的模型验证并输出每个模型给出的概率
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
from PIL import Image

# 导入模型
from models.co_occurrence_fast import create_co_occurrence_fast_model
from models.gram_net_paper import create_gram_net_paper_model as create_gram_net_model
import torchvision.models as models

class RandomSampleValidator:
    """随机样本验证器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}

    def load_model(self, model_type, model_path):
        """加载指定类型的模型"""
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
        """加载四个子模型"""
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
                # 尝试在final_final_model目录中查找
                final_path = os.path.join("checkpoints/final_final_model", f"{model_type}.pth")
                if os.path.exists(final_path):
                    print(f"在 {final_path} 找到模型文件")
                    self.load_model(model_type, final_path)

    def predict_probabilities(self, image_tensor):
        """预测单个图像的概率"""
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            probabilities = {}

            # 计算每个模型的概率
            for model_type, model in self.models.items():
                output = model(image_tensor)

                # 根据模型类型处理输出
                if model_type == 'co_occurrence':
                    # Co-occurrence模型输出已经是sigmoid概率
                    ai_prob = output[0, 0].item()  # 单输出，直接取第一个元素
                else:
                    # ResNet, DenseNet, GramNet输出logits
                    prob = F.softmax(output, dim=1)
                    # 根据调试结果，类别0是nature，类别1是AI
                    # 但我们想要的是AI图像的概率，所以取类别1
                    ai_prob = prob[0, 1].item()

                probabilities[model_type] = ai_prob

            return probabilities

    def sample_images_from_dataset(self, data_dir, num_samples_per_class=4):
        """从数据集中采样图像"""
        # 定义数据变换
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载数据集
        dataset = datasets.ImageFolder(data_dir, transform=transform)

        # 根据文件路径判断真实标签
        # 如果路径包含 'ai' 则是AI图像，否则是nature图像
        def get_true_label(image_path):
            if 'ai' in image_path.lower():
                return 1  # AI图像
            else:
                return 0  # Nature图像

        # 按真实标签分组图像
        class_indices = {0: [], 1: []}  # 0: nature, 1: ai
        for idx, (image_path, _) in enumerate(dataset.samples):
            true_label = get_true_label(image_path)
            class_indices[true_label].append((idx, image_path))

        # 从每个类别中随机采样
        sampled_images = []
        for true_label, indices in class_indices.items():
            if len(indices) >= num_samples_per_class:
                sampled = random.sample(indices, num_samples_per_class)
                for idx, image_path in sampled:
                    sampled_images.append({
                        'index': idx,
                        'image_path': image_path,
                        'true_label': true_label,
                        'class_name': 'Nature' if true_label == 0 else 'AI Generated'
                    })

        print(f"采样结果: Nature图像 {len(class_indices[0])}张, AI图像 {len(class_indices[1])}张")
        return sampled_images, dataset

    def validate_samples(self, data_dir, num_samples_per_class=4):
        """验证随机样本"""
        print(f"从数据集 {data_dir} 中采样图像...")
        sampled_images, dataset = self.sample_images_from_dataset(data_dir, num_samples_per_class)

        print(f"采样了 {len(sampled_images)} 张图像")

        # 处理每个采样图像
        results = []
        for i, sample in enumerate(sampled_images):
            print(f"\n处理图像 {i+1}/{len(sampled_images)}:")
            print(f"  图像路径: {sample['image_path']}")
            print(f"  真实标签: {sample['class_name']}")

            # 加载并预处理图像
            image_tensor = dataset[sample['index']][0]

            # 预测概率
            probabilities = self.predict_probabilities(image_tensor)

            # 保存结果
            result = {
                'image_index': i + 1,
                'image_path': sample['image_path'],
                'true_label': sample['true_label'],
                'class_name': sample['class_name'],
                'probabilities': probabilities
            }
            results.append(result)

            # 打印当前图像的概率结果
            print("  各模型预测概率:")
            for model_type, prob in probabilities.items():
                prediction = "AI" if prob > 0.5 else "Nature"
                print(f"    {model_type}: {prob:.4f} ({prediction})")

        return results

    def print_summary(self, results):
        """打印验证结果摘要"""
        print("\n" + "="*80)
        print("验证结果摘要")
        print("="*80)

        # 按类别分组结果
        nature_results = [r for r in results if r['true_label'] == 0]
        ai_results = [r for r in results if r['true_label'] == 1]

        print(f"\nNature图像 ({len(nature_results)}张):")
        for result in nature_results:
            print(f"  {os.path.basename(result['image_path'])}:")
            for model_type, prob in result['probabilities'].items():
                prediction = "AI" if prob > 0.5 else "Nature"
                correct = "✓" if prediction == "Nature" else "✗"
                print(f"    {model_type}: {prob:.4f} ({prediction}) {correct}")

        print(f"\nAI图像 ({len(ai_results)}张):")
        for result in ai_results:
            print(f"  {os.path.basename(result['image_path'])}:")
            for model_type, prob in result['probabilities'].items():
                prediction = "AI" if prob > 0.5 else "Nature"
                correct = "✓" if prediction == "AI" else "✗"
                print(f"    {model_type}: {prob:.4f} ({prediction}) {correct}")

        # 计算准确率
        print("\n各模型准确率:")
        for model_type in ['resnet', 'densenet', 'co_occurrence', 'gram_net']:
            correct_count = 0
            total_count = len(results)

            for result in results:
                prob = result['probabilities'].get(model_type, 0)
                prediction = 1 if prob > 0.5 else 0
                if prediction == result['true_label']:
                    correct_count += 1

            accuracy = correct_count / total_count
            print(f"  {model_type}: {accuracy:.1%} ({correct_count}/{total_count})")

    def save_results(self, results, output_file):
        """保存结果到JSON文件"""
        # 转换为可序列化的格式
        serializable_results = []
        for result in results:
            serializable_result = {
                'image_index': result['image_index'],
                'image_path': result['image_path'],
                'true_label': result['true_label'],
                'class_name': result['class_name'],
                'probabilities': result['probabilities']
            }
            serializable_results.append(serializable_result)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n详细结果已保存到: {output_file}")

def main():
    # 配置参数
    data_dir = "E:/data/val"  # 验证集目录
    model_dir = "checkpoints/final_final_model"  # 模型目录
    output_file = "random_sample_validation_results.json"  # 输出文件

    # 创建验证器
    validator = RandomSampleValidator()

    # 加载模型
    print("加载模型...")
    validator.load_all_models(model_dir)

    # 验证随机样本
    print("\n开始验证随机样本...")
    results = validator.validate_samples(data_dir, num_samples_per_class=4)

    # 打印摘要
    validator.print_summary(results)

    # 保存结果
    validator.save_results(results, output_file)

    print("\n验证完成！")

if __name__ == "__main__":
    main()