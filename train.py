#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独训练各个模型并记录详细性能指标
支持: resnet, densenet, co_occurrence, gram_net
记录: 运行时间、精确率、召回率、F1-score、准确率、损失、学习率
数据保存到JSON文件和TensorBoard
"""

import os
import json
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm

# 导入模型
from models.co_occurrence_fast import create_co_occurrence_fast_model
from models.gram_net_paper import create_gram_net_paper_model as create_gram_net_model
import torchvision.models as models


class DetailedExperimentLogger:
    """详细实验数据记录器"""

    def __init__(self, save_dir, experiment_name):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        os.makedirs(save_dir, exist_ok=True)

        # 实验数据存储结构
        self.experiment_data = {
            'experiment_info': {
                'name': experiment_name,
                'start_time': None,
                'save_dir': save_dir,
                'end_time': None,
                'total_duration_seconds': 0
            },
            'hyperparameters': {},
            'training_history': {
                'epochs': [],
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
                'learning_rate': [],
                'epoch_time': []
            },
            'detailed_metrics': {
                'train_metrics': [],
                'val_metrics': []
            },
            'model_metrics': {},
            'dataset_info': {},
            'best_epoch': {
                'epoch': 0,
                'val_accuracy': 0,
                'val_loss': float('inf')
            },
            'training_summary': {}
        }

        # 开始时间
        self.experiment_data['experiment_info']['start_time'] = datetime.datetime.now().isoformat()

    def log_hyperparameters(self, hyperparams):
        """记录超参数"""
        self.experiment_data['hyperparameters'] = hyperparams

    def log_dataset_info(self, dataset_info):
        """记录数据集信息"""
        self.experiment_data['dataset_info'] = dataset_info

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time,
                  train_metrics=None, val_metrics=None):
        """记录每个epoch的数据并立即保存到JSON文件"""
        self.experiment_data['training_history']['epochs'].append(epoch)
        self.experiment_data['training_history']['train_loss'].append(train_loss)
        self.experiment_data['training_history']['train_accuracy'].append(train_acc)
        self.experiment_data['training_history']['val_loss'].append(val_loss)
        self.experiment_data['training_history']['val_accuracy'].append(val_acc)
        self.experiment_data['training_history']['learning_rate'].append(lr)
        self.experiment_data['training_history']['epoch_time'].append(epoch_time)

        if train_metrics:
            self.experiment_data['detailed_metrics']['train_metrics'].append(train_metrics)
        if val_metrics:
            self.experiment_data['detailed_metrics']['val_metrics'].append(val_metrics)

        # 更新最佳epoch
        if val_acc > self.experiment_data['best_epoch']['val_accuracy']:
            self.experiment_data['best_epoch']['epoch'] = epoch
            self.experiment_data['best_epoch']['val_accuracy'] = val_acc
            self.experiment_data['best_epoch']['val_loss'] = val_loss

        # 每轮训练后立即保存到JSON文件
        self.save_to_json()

    def save_to_json(self):
        """将当前实验数据保存到JSON文件"""
        json_path = self.get_json_path()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)
        print(f"实验数据已更新到: {json_path}")

    def log_model_metrics(self, model_metrics):
        """记录模型指标"""
        self.experiment_data['model_metrics'] = model_metrics

    def finalize(self):
        """完成实验记录"""
        end_time = datetime.datetime.now()
        start_time = datetime.datetime.fromisoformat(
            self.experiment_data['experiment_info']['start_time']
        )
        total_duration = (end_time - start_time).total_seconds()

        self.experiment_data['experiment_info']['end_time'] = end_time.isoformat()
        self.experiment_data['experiment_info']['total_duration_seconds'] = total_duration

        # 训练总结
        if self.experiment_data['training_history']['train_accuracy']:
            self.experiment_data['training_summary'] = {
                'total_training_time': total_duration,
                'final_train_accuracy': self.experiment_data['training_history']['train_accuracy'][-1],
                'final_val_accuracy': self.experiment_data['training_history']['val_accuracy'][-1],
                'final_train_loss': self.experiment_data['training_history']['train_loss'][-1],
                'final_val_loss': self.experiment_data['training_history']['val_loss'][-1],
                'best_val_accuracy': self.experiment_data['best_epoch']['val_accuracy'],
                'best_epoch': self.experiment_data['best_epoch']['epoch']
            }

        # 更新训练总结后再次保存JSON文件
        self.save_to_json()
        return self.get_json_path()

    def get_json_path(self):
        """获取JSON文件路径"""
        return os.path.join(self.save_dir, f"{self.experiment_name}.json")


def compute_detailed_metrics(outputs, labels, is_binary=False):
    """
    计算详细的性能指标

    Args:
        outputs: 模型输出
        labels: 真实标签
        is_binary: 是否为二元分类

    Returns:
        dict: 包含各项指标的字典
    """
    if is_binary:
        # 二元分类
        preds = (outputs > 0.5).long().squeeze()
        if preds.dim() > 1:
            preds = preds.squeeze(1)
    else:
        # 多分类
        _, preds = torch.max(outputs, 1)

    # 转换为numpy数组
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 计算整体指标
    accuracy = (preds_np == labels_np).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average='macro', zero_division=0
    )

    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels_np, preds_np, average=None, zero_division=0
    )

    # 计算混淆矩阵
    cm = confusion_matrix(labels_np, preds_np)

    # 分离AI图像和自然图像的指标
    ai_indices = labels_np == 0  # 假设0表示AI图像
    nature_indices = labels_np == 1  # 假设1表示自然图像

    ai_metrics = {
        'precision': precision_per_class[0],
        'recall': recall_per_class[0],
        'f1': f1_per_class[0]
    }

    nature_metrics = {
        'precision': precision_per_class[1],
        'recall': recall_per_class[1],
        'f1': f1_per_class[1]
    }

    return {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'ai_precision': ai_metrics['precision'],
        'ai_recall': ai_metrics['recall'],
        'ai_f1': ai_metrics['f1'],
        'nature_precision': nature_metrics['precision'],
        'nature_recall': nature_metrics['recall'],
        'nature_f1': nature_metrics['f1'],
        'confusion_matrix': cm.tolist()
    }


def create_model(model_type, num_classes=2, pretrained=True):
    """创建指定类型的模型"""
    if model_type == 'resnet':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model

    elif model_type == 'densenet':
        model = models.densenet201(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        return model

    elif model_type == 'co_occurrence':
        # 使用快速版共现矩阵模型
        return create_co_occurrence_fast_model(num_classes=num_classes, pretrained=pretrained)

    elif model_type == 'co_occurrence_resnet50':
        # 使用ResNet50处理三通道共现矩阵的模型
        return create_co_occurrence_resnet50_model(num_classes=num_classes, pretrained=pretrained)

    elif model_type == 'gram_net':
        return create_gram_net_model(num_classes=num_classes, input_size=256)

    elif model_type == 'final':
        # Final模型返回四个模型的集合
        return {
            'resnet': create_model('resnet', num_classes, pretrained),
            'densenet': create_model('densenet', num_classes, pretrained),
            'co_occurrence': create_model('co_occurrence', num_classes, pretrained),
            'gram_net': create_model('gram_net', num_classes, pretrained),
            'weights': torch.tensor([0.2, 0.6, 0.1, 0.1])  # 加权权重：resnet:0.2, densenet:0.6, co_occurrence:0.1, gram_net:0.1
        }

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_criterion_and_output_type(model_type):
    """根据模型类型返回对应的损失函数和输出类型"""
    if model_type == 'co_occurrence':
        return nn.BCELoss(), 'binary'
    elif model_type == 'final':
        # Final模型使用BCELoss，因为输出是概率
        return nn.BCELoss(), 'binary'
    else:
        return nn.CrossEntropyLoss(), 'multi_class'


def convert_to_probability(outputs, model_type, is_binary=False):
    """将模型输出转换为真实图像的概率"""
    if model_type == 'final':
        # Final模型已经输出概率
        return outputs

    if is_binary:
        # 二元分类模型直接输出概率
        return outputs
    else:
        # 多分类模型：使用softmax获取真实图像的概率
        # 假设类别0是AI图像，类别1是真实图像
        probs = F.softmax(outputs, dim=1)
        return probs[:, 1]  # 返回真实图像的概率


def log_probability_info(prob_info, epoch, batch_idx, sample_idx=0, log_file="probability_log.txt"):
    """将概率信息记录到文件中"""
    import datetime

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== Epoch {epoch}, Batch {batch_idx}, 样本 {sample_idx} ===\n")
        f.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 记录各个模型的概率
        f.write("各模型输出概率:\n")
        for model_name, prob in prob_info['individual_probs'].items():
            prob_value = prob[sample_idx].item() if prob.dim() > 0 else prob.item()
            f.write(f"  {model_name}: {prob_value:.4f}\n")

        # 记录权重
        f.write(f"\n权重: {prob_info['weights'].cpu().numpy()}\n")

        # 记录加权后的概率
        f.write("\n加权后概率:\n")
        weighted_probs = prob_info['weighted_probs'][sample_idx].cpu().numpy()
        for model_name, weighted_prob in zip(prob_info['individual_probs'].keys(), weighted_probs):
            f.write(f"  {model_name}: {weighted_prob:.4f}\n")

        # 记录最终概率
        final_prob = prob_info['final_prob'][sample_idx].item() if prob_info['final_prob'].dim() > 0 else prob_info['final_prob'].item()
        f.write(f"\n最终加权概率: {final_prob:.4f}\n")
        f.write(f"预测结果: {'真实图像' if final_prob > 0.5 else '生成图像'}\n")
        f.write("=" * 40 + "\n")


def log_validation_batch(model_type, outputs, labels, epoch, batch_idx, log_file="validation_detailed.log",
                        individual_model_outputs=None):
    """
    记录每个验证batch的详细输出概率和真实值
    Args:
        model_type: 模型类型
        outputs: 模型输出
        labels: 真实标签
        epoch: 当前epoch
        batch_idx: 当前batch索引
        log_file: 日志文件名
        individual_model_outputs: 各个子模型的输出字典 {model_name: output_tensor}
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"=== 验证批次详细日志 ===\n")
        f.write(f"时间: {timestamp}\n")
        f.write(f"Epoch: {epoch}, Batch: {batch_idx}\n")
        f.write(f"模型类型: {model_type}\n")

        # 处理主模型输出
        if outputs.dim() > 1:
            batch_size = outputs.size(0)
            for i in range(batch_size):
                if model_type in ['co_occurrence', 'final']:
                    # 二元分类模型
                    prob = outputs[i].item() if outputs[i].numel() == 1 else outputs[i].squeeze().item()
                    prediction = "真实图像" if prob > 0.5 else "生成图像"
                    true_label = "真实图像" if labels[i].item() == 1 else "生成图像"
                    correct = "正确" if (prob > 0.5) == (labels[i].item() == 1) else "错误"
                else:
                    # 多分类模型
                    probs = F.softmax(outputs[i], dim=0)
                    prob_real = probs[1].item()  # 假设类别1是真实图像
                    _, pred_class = torch.max(outputs[i], 0)
                    prediction = "真实图像" if pred_class.item() == 1 else "生成图像"
                    true_label = "真实图像" if labels[i].item() == 1 else "生成图像"
                    correct = "正确" if pred_class.item() == labels[i].item() else "错误"
                    prob = prob_real

                f.write(f"样本{i}: 概率={prob:.4f}, 预测={prediction}, 真实={true_label} ({correct})\n")
        else:
            # 单样本情况
            if model_type in ['co_occurrence', 'final']:
                prob = outputs.item() if outputs.numel() == 1 else outputs.squeeze().item()
                prediction = "真实图像" if prob > 0.5 else "生成图像"
                true_label = "真实图像" if labels.item() == 1 else "生成图像"
                correct = "正确" if (prob > 0.5) == (labels.item() == 1) else "错误"
            else:
                probs = F.softmax(outputs, dim=0)
                prob_real = probs[1].item()
                _, pred_class = torch.max(outputs, 0)
                prediction = "真实图像" if pred_class.item() == 1 else "生成图像"
                true_label = "真实图像" if labels.item() == 1 else "生成图像"
                correct = "正确" if pred_class.item() == labels.item() else "错误"
                prob = prob_real

            f.write(f"样本0: 概率={prob:.4f}, 预测={prediction}, 真实={true_label} ({correct})\n")

        # 记录各个子模型的输出（如果有）
        if individual_model_outputs:
            f.write(f"\n--- 各个子模型输出 ---\n")
            for model_name, model_output in individual_model_outputs.items():
                f.write(f"{model_name}:\n")

                if model_output.dim() > 1:
                    batch_size = model_output.size(0)
                    for i in range(min(3, batch_size)):  # 只记录前3个样本
                        if model_name in ['co_occurrence', 'final']:
                            prob = model_output[i].item() if model_output[i].numel() == 1 else model_output[i].squeeze().item()
                        else:
                            probs = F.softmax(model_output[i], dim=0)
                            prob = probs[1].item()  # 真实图像概率

                        f.write(f"  样本{i}: 概率={prob:.4f}\n")
                else:
                    if model_name in ['co_occurrence', 'final']:
                        prob = model_output.item() if model_output.numel() == 1 else model_output.squeeze().item()
                    else:
                        probs = F.softmax(model_output, dim=0)
                        prob = probs[1].item()
                    f.write(f"  样本0: 概率={prob:.4f}\n")

        # 计算批次统计
        if model_type in ['co_occurrence', 'final']:
            preds = (outputs > 0.5).long().squeeze()
        else:
            _, preds = torch.max(outputs, 1)

        batch_correct = (preds == labels).sum().item()
        batch_total = labels.size(0)
        batch_accuracy = batch_correct / batch_total

        f.write(f"\n批次统计: 正确数={batch_correct}/{batch_total}, 准确率={batch_accuracy:.4f}\n")

        # 计算各个子模型的准确率（如果有）
        if individual_model_outputs:
            f.write(f"\n--- 各个子模型准确率 ---\n")
            for model_name, model_output in individual_model_outputs.items():
                if model_name in ['co_occurrence', 'final']:
                    model_preds = (model_output > 0.5).long().squeeze()
                else:
                    _, model_preds = torch.max(model_output, 1)

                model_correct = (model_preds == labels).sum().item()
                model_accuracy = model_correct / batch_total
                f.write(f"{model_name}: {model_correct}/{batch_total}, 准确率={model_accuracy:.4f}\n")

        f.write("\n")


def forward_final_model(model_dict, x, device, record_probs=False):
    """Final模型的前向传播"""
    weights = model_dict['weights'].to(device)

    # 获取四个模型的输出概率
    probs = []
    model_names = []
    individual_probs = {}

    for model_name, model in model_dict.items():
        if model_name != 'weights':
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(x)
                # 转换为概率
                if model_name == 'co_occurrence':
                    prob = output.squeeze()  # 二元分类直接输出概率
                else:
                    prob = F.softmax(output, dim=1)[:, 1]  # 多分类取真实图像概率
                probs.append(prob)
                model_names.append(model_name)
                individual_probs[model_name] = prob

    # 加权平均
    weighted_probs = torch.stack(probs, dim=1) * weights
    final_prob = weighted_probs.sum(dim=1)

    # 记录概率信息（如果启用）
    if record_probs:
        prob_info = {
            'individual_probs': individual_probs,
            'weights': weights,
            'weighted_probs': weighted_probs,
            'final_prob': final_prob
        }
        return final_prob.unsqueeze(1), prob_info  # 保持形状一致性
    else:
        return final_prob.unsqueeze(1)  # 保持形状一致性


def train_epoch(model, train_loader, criterion, optimizer, device, is_binary=False, model_type=None):
    """训练一个epoch"""
    if model_type == 'final':
        return train_final_epoch(model, train_loader, criterion, optimizer, device)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []

    start_time = time.time()

    # 创建进度条
    pbar = tqdm(train_loader, desc=f"训练", leave=False)

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # 根据输出类型调整损失计算
        if is_binary:
            labels_binary = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels_binary)
        else:
            loss = criterion(outputs, labels)

        loss.backward()

        # 针对共现矩阵模型添加梯度裁剪
        if model_type == 'co_occurrence':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # 计算准确率
        if is_binary:
            preds = (outputs > 0.5).long().squeeze()
            if preds.dim() > 1:
                preds = preds.squeeze(1)
        else:
            _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 更新进度条显示
        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })

        # 收集输出和标签用于详细指标计算
        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())

    pbar.close()
    epoch_time = time.time() - start_time

    # 计算详细指标
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    detailed_metrics = compute_detailed_metrics(all_outputs, all_labels, is_binary)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, detailed_metrics, epoch_time


def train_final_epoch(model_dict, train_loader, criterion, optimizer, device):
    """训练Final模型的一个epoch"""
    # 分别训练四个模型
    models_metrics = {}
    total_time = 0

    # 创建主进度条显示四个模型的训练进度
    model_names = [name for name in model_dict.keys() if name != 'weights']
    pbar_main = tqdm(model_names, desc="训练Final模型", leave=False)

    for model_name in pbar_main:
        model = model_dict[model_name]
        model = model.to(device)

        # 更新主进度条描述
        pbar_main.set_description(f"训练 {model_name}")

        # 获取对应的损失函数和输出类型
        criterion_model, output_type = get_criterion_and_output_type(model_name)
        is_binary = (output_type == 'binary')

        # 为每个模型创建优化器
        if model_name == 'co_occurrence':
            # 共现矩阵模型使用更低的学率和权重衰减
            optimizer_model = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        else:
            optimizer_model = optim.Adam(model.parameters(), lr=1e-3)

        # 训练该模型
        loss, acc, metrics, time_taken = train_epoch(
            model, train_loader, criterion_model, optimizer_model, device, is_binary, model_name
        )

        models_metrics[model_name] = {
            'loss': loss,
            'accuracy': acc,
            'metrics': metrics,
            'time': time_taken
        }
        total_time += time_taken

        # 更新主进度条后处理信息
        pbar_main.set_postfix({
            'loss': f'{loss:.4f}',
            'acc': f'{acc:.4f}'
        })

    pbar_main.close()

    # 计算Final模型的指标
    final_loss = np.mean([m['loss'] for m in models_metrics.values()])
    final_acc = np.mean([m['accuracy'] for m in models_metrics.values()])

    # 合并详细指标
    final_metrics = {
        'accuracy': final_acc,
        'macro_precision': np.mean([m['metrics']['macro_precision'] for m in models_metrics.values()]),
        'macro_recall': np.mean([m['metrics']['macro_recall'] for m in models_metrics.values()]),
        'macro_f1': np.mean([m['metrics']['macro_f1'] for m in models_metrics.values()]),
        'ai_precision': np.mean([m['metrics']['ai_precision'] for m in models_metrics.values()]),
        'ai_recall': np.mean([m['metrics']['ai_recall'] for m in models_metrics.values()]),
        'ai_f1': np.mean([m['metrics']['ai_f1'] for m in models_metrics.values()]),
        'nature_precision': np.mean([m['metrics']['nature_precision'] for m in models_metrics.values()]),
        'nature_recall': np.mean([m['metrics']['nature_recall'] for m in models_metrics.values()]),
        'nature_f1': np.mean([m['metrics']['nature_f1'] for m in models_metrics.values()]),
        'confusion_matrix': None,  # 无法直接合并混淆矩阵
        'individual_models': models_metrics  # 保存各个子模型的训练指标
    }

    return final_loss, final_acc, final_metrics, total_time


def validate_epoch(model, val_loader, criterion, device, is_binary=False, model_type=None, epoch=0):
    """验证一个epoch"""
    if model_type == 'final':
        # 每个epoch都记录概率信息
        log_probabilities = True
        return validate_final_model(model, val_loader, criterion, device, epoch, log_probabilities)

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        # 创建验证进度条
        pbar = tqdm(val_loader, desc=f"验证", leave=False)

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 记录每个batch的详细验证信息到.log文件
            log_validation_batch(model_type, outputs, labels, epoch, batch_idx)

            # 记录概率信息（共现矩阵模型，每隔100个batch记录一次）
            if model_type == 'co_occurrence' and batch_idx % 100 == 0:
                prob_info = {
                    'individual_probs': {'co_occurrence': outputs.squeeze()},
                    'weights': torch.tensor([1.0]),  # 单独训练时权重为1
                    'weighted_probs': outputs.squeeze().unsqueeze(1),
                    'final_prob': outputs.squeeze()
                }
                log_probability_info(prob_info, epoch, batch_idx, sample_idx=0)

            # 根据输出类型调整损失计算
            if is_binary:
                labels_binary = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels_binary)
            else:
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # 计算准确率
            if is_binary:
                preds = (outputs > 0.5).long().squeeze()
                if preds.dim() > 1:
                    preds = preds.squeeze(1)
            else:
                _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 更新进度条显示
            current_loss = running_loss / total
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })

            # 收集输出和标签用于详细指标计算
            all_outputs.append(outputs)
            all_labels.append(labels)

        pbar.close()

    epoch_time = time.time() - start_time

    # 计算详细指标
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    detailed_metrics = compute_detailed_metrics(all_outputs, all_labels, is_binary)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, detailed_metrics, epoch_time


def validate_final_model(model_dict, val_loader, criterion, device, epoch=0, log_probabilities=False):
    """验证Final模型"""
    # 分别验证四个模型并计算加权结果
    models_metrics = {}
    all_final_outputs = []
    all_labels = []
    total_time = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 收集各个子模型的输出
            individual_outputs = {}
            for model_name, model in model_dict.items():
                if model_name != 'weights':
                    model = model.to(device)
                    model.eval()
                    output = model(inputs)
                    individual_outputs[model_name] = output

            # 计算Final模型的输出
            if log_probabilities and batch_idx % 100 == 0:  # 每隔100个batch记录一次
                final_outputs, prob_info = forward_final_model(model_dict, inputs, device, record_probs=True)
                # 记录第一个样本的概率信息
                log_probability_info(prob_info, epoch, batch_idx, sample_idx=0)
            else:
                final_outputs = forward_final_model(model_dict, inputs, device)

            # 记录Final模型和各个子模型的详细验证信息
            log_validation_batch('final', final_outputs, labels, epoch, batch_idx,
                               individual_model_outputs=individual_outputs)

            all_final_outputs.append(final_outputs)
            all_labels.append(labels)

    # 计算Final模型的指标
    all_final_outputs = torch.cat(all_final_outputs)
    all_labels = torch.cat(all_labels)

    # 计算损失
    labels_binary = all_labels.float().unsqueeze(1)
    final_loss = criterion(all_final_outputs, labels_binary).item()

    # 计算准确率
    final_preds = (all_final_outputs > 0.5).long().squeeze()
    final_acc = (final_preds == all_labels).float().mean().item()

    # 计算详细指标
    final_metrics = compute_detailed_metrics(all_final_outputs, all_labels, is_binary=True)

    # 分别验证每个模型
    for model_name, model in model_dict.items():
        if model_name != 'weights':
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                all_outputs = []
                all_labels_model = []

                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    all_outputs.append(outputs)
                    all_labels_model.append(labels)

                all_outputs = torch.cat(all_outputs)
                all_labels_model = torch.cat(all_labels_model)

                # 转换为概率
                prob_outputs = convert_to_probability(all_outputs, model_name,
                                                    model_name in ['co_occurrence', 'co_occurrence_optimized'])

                # 计算准确率
                preds = (prob_outputs > 0.5).long()
                # 确保预测和标签形状匹配，避免广播错误
                if preds.dim() > 1:
                    preds = preds.squeeze()
                acc = (preds == all_labels_model).float().mean().item()

                # 计算详细指标
                metrics = compute_detailed_metrics(prob_outputs.unsqueeze(1) if prob_outputs.dim() == 1 else prob_outputs,
                                                 all_labels_model, is_binary=True)

                models_metrics[model_name] = {
                    'accuracy': acc,
                    'metrics': metrics
                }

    # 添加各个模型的指标到final_metrics
    final_metrics['individual_models'] = models_metrics

    return final_loss, final_acc, final_metrics, 0  # 时间暂时设为0


def train_model(model_type, train_loader, val_loader, num_epochs, device, save_dir,
                learning_rate=1e-3, batch_size=32, experiment_name=None):
    """训练指定模型"""

    # 创建模型
    model = create_model(model_type)

    # Final模型特殊处理
    if model_type == 'final':
        # 将各个模型移到设备
        for model_name, sub_model in model.items():
            if model_name != 'weights':
                model[model_name] = sub_model.to(device)
        # Final模型使用虚拟优化器（实际训练时会在train_final_epoch中为每个子模型创建优化器）
        optimizer = optim.Adam([torch.tensor(0.0, requires_grad=True)], lr=learning_rate)  # 虚拟优化器
    else:
        model = model.to(device)
        # 针对共现矩阵模型使用更低的学率和权重衰减
        if model_type == 'co_occurrence':
            # 降低学习率，添加权重衰减
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        else:
            # 常规优化器
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 获取损失函数和输出类型
    criterion, output_type = get_criterion_and_output_type(model_type)
    is_binary = (output_type == 'binary')

    # 学习率调度器
    if model_type != 'final':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None  # Final模型不使用调度器

    # 实验记录器
    if experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_type}_{timestamp}"

    logger = DetailedExperimentLogger(save_dir, experiment_name)

    # TensorBoard记录器
    tb_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_logs', experiment_name))

    # 记录超参数
    hyperparams = {
        'model_type': model_type,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'device': device,
        'output_type': output_type
    }
    logger.log_hyperparameters(hyperparams)

    # 记录数据集信息
    dataset_info = {
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'batch_size': batch_size
    }
    logger.log_dataset_info(dataset_info)

    # 记录模型指标
    if model_type == 'final':
        # Final模型计算各个子模型的参数
        total_params = 0
        individual_params = {}
        for model_name, sub_model in model.items():
            if model_name != 'weights':
                params = sum(p.numel() for p in sub_model.parameters())
                total_params += params
                individual_params[model_name] = params

        model_metrics = {
            'total_parameters': total_params,
            'trainable_parameters': total_params,  # 假设所有参数都可训练
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'individual_models': individual_params
        }
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }

    logger.log_model_metrics(model_metrics)

    print(f"开始训练 {model_type} 模型...")
    if model_type == 'final':
        print(f"总参数数量: {total_params:,}")
        for model_name, params in individual_params.items():
            print(f"  {model_name}: {params:,} 参数")
    else:
        print(f"模型参数数量: {total_params:,}")

    print(f"训练数据: {len(train_loader.dataset)} 张图像")
    print(f"验证数据: {len(val_loader.dataset)} 张图像")

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_acc, train_metrics, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, is_binary, model_type
        )

        # 验证
        val_loss, val_acc, val_metrics, val_time = validate_epoch(
            model, val_loader, criterion, device, is_binary, model_type, epoch
        )

        # 学习率调度
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            current_lr = learning_rate  # Final模型使用固定学习率

        epoch_time = train_time + val_time

        # 记录到实验日志
        logger.log_epoch(
            epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time,
            train_metrics, val_metrics
        )

        # 记录到TensorBoard
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
        tb_writer.add_scalar('Learning_rate', current_lr, epoch)
        tb_writer.add_scalar('Time/epoch', epoch_time, epoch)

        # 记录详细指标
        tb_writer.add_scalar('Metrics/train_macro_f1', train_metrics['macro_f1'], epoch)
        tb_writer.add_scalar('Metrics/val_macro_f1', val_metrics['macro_f1'], epoch)
        tb_writer.add_scalar('Metrics/train_ai_f1', train_metrics['ai_f1'], epoch)
        tb_writer.add_scalar('Metrics/val_ai_f1', val_metrics['ai_f1'], epoch)
        tb_writer.add_scalar('Metrics/train_nature_f1', train_metrics['nature_f1'], epoch)
        tb_writer.add_scalar('Metrics/val_nature_f1', val_metrics['nature_f1'], epoch)

        # 对于Final模型，额外记录各个子模型的指标
        if model_type == 'final' and 'individual_models' in val_metrics:
            for model_name, model_metrics in val_metrics['individual_models'].items():
                tb_writer.add_scalar(f'Individual/{model_name}/val_accuracy', model_metrics['accuracy'], epoch)
                tb_writer.add_scalar(f'Individual/{model_name}/val_f1', model_metrics['metrics']['macro_f1'], epoch)

        # 记录各个子模型的训练指标（如果有）
        if model_type == 'final' and 'individual_models' in train_metrics:
            for model_name, model_metrics in train_metrics['individual_models'].items():
                tb_writer.add_scalar(f'Individual/{model_name}/train_accuracy', model_metrics['accuracy'], epoch)
                tb_writer.add_scalar(f'Individual/{model_name}/train_loss', model_metrics['loss'], epoch)
                tb_writer.add_scalar(f'Individual/{model_name}/train_f1', model_metrics['metrics']['macro_f1'], epoch)
                tb_writer.add_scalar(f'Individual/{model_name}/train_time', model_metrics['time'], epoch)

        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"训练时间: {train_time:.2f}s, 验证时间: {val_time:.2f}s")
        print(f"学习率: {current_lr:.6f}")
        print(f"AI图像F1: {val_metrics['ai_f1']:.4f}, 自然图像F1: {val_metrics['nature_f1']:.4f}")

        # 对于Final模型，打印各个子模型的准确率
        if model_type == 'final' and 'individual_models' in val_metrics:
            print("各个模型验证准确率:")
            for model_name, model_metrics in val_metrics['individual_models'].items():
                print(f"  {model_name}: {model_metrics['accuracy']:.4f}")

            # 打印各个子模型的训练准确率和时间（如果有）
            if 'individual_models' in train_metrics:
                print("各个模型训练准确率和时间:")
                for model_name, model_metrics in train_metrics['individual_models'].items():
                    print(f"  {model_name}: 准确率={model_metrics['accuracy']:.4f}, 时间={model_metrics['time']:.2f}s")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if model_type == 'final':
                # Final模型保存所有子模型
                model_path = os.path.join(save_dir, f"best_{model_type}_model")
                os.makedirs(model_path, exist_ok=True)
                for model_name, sub_model in model.items():
                    if model_name != 'weights':
                        sub_model_path = os.path.join(model_path, f"{model_name}.pth")
                        # 保存包含训练信息的checkpoint
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': sub_model.state_dict(),
                            'val_accuracy': val_acc,
                            'val_loss': val_loss,
                            'train_accuracy': train_acc,
                            'train_loss': train_loss,
                            'learning_rate': current_lr,
                            'best_val_accuracy': best_val_acc,
                            'val_metrics': val_metrics,
                            'train_metrics': train_metrics
                        }
                        torch.save(checkpoint, sub_model_path)
                # 保存权重
                weights_path = os.path.join(model_path, "weights.pth")
                torch.save(model['weights'], weights_path)
                print(f"保存最佳Final模型到: {model_path}")
            else:
                model_path = os.path.join(save_dir, f"best_{model_type}_model.pth")
                # 保存包含训练信息的checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc,
                    'train_loss': train_loss,
                    'learning_rate': current_lr,
                    'best_val_accuracy': best_val_acc,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics
                }
                torch.save(checkpoint, model_path)
                print(f"保存最佳模型到: {model_path}")

    # 保存最终模型
    if model_type == 'final':
        final_model_path = os.path.join(save_dir, f"final_{model_type}_model")
        os.makedirs(final_model_path, exist_ok=True)
        for model_name, sub_model in model.items():
            if model_name != 'weights':
                sub_model_path = os.path.join(final_model_path, f"{model_name}.pth")
                # 保存包含训练信息的checkpoint
                checkpoint = {
                    'epoch': num_epochs - 1,
                    'model_state_dict': sub_model.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc,
                    'train_loss': train_loss,
                    'learning_rate': current_lr,
                    'best_val_accuracy': best_val_acc,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics
                }
                torch.save(checkpoint, sub_model_path)
        # 保存权重
        weights_path = os.path.join(final_model_path, "weights.pth")
        torch.save(model['weights'], weights_path)
    else:
        final_model_path = os.path.join(save_dir, f"final_{model_type}_model.pth")
        # 保存包含训练信息的checkpoint
        checkpoint = {
            'epoch': num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'learning_rate': current_lr,
            'best_val_accuracy': best_val_acc,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics
        }
        torch.save(checkpoint, final_model_path)

    # 完成实验记录
    json_path = logger.finalize()
    tb_writer.close()

    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"实验数据: {json_path}")
    print(f"TensorBoard日志: {tb_writer.log_dir}")

    return best_val_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='单独训练各个模型')
    parser.add_argument('--model_type', type=str, default='final', choices=['resnet', 'densenet', 'co_occurrence', 'gram_net', 'final'], help='要训练的模型类型')
    parser.add_argument('--data_dir', type=str, default='E:/dataset/BigGAN', help='数据根目录')
    # parser.add_argument('--data_dir', type=str, default='E:/dataset/Human Faces Dataset', help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')

    args = parser.parse_args()

    print(f"使用设备: {args.device}")

    # 数据预处理
    train_transform = transforms.Compose([
    transforms.Resize((256, 256)),                    # 调整到256x256像素
    transforms.RandomHorizontalFlip(p=0.5),          # 随机水平翻转
    transforms.RandomRotation(10),                   # 随机旋转±10度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),                           # 转换为Tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"类别: {train_dataset.classes}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练模型
    best_acc = train_model(
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.save_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )

    print(f"\n{args.model_type} 模型训练完成!")
    print(f"最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    import torchvision
    main()