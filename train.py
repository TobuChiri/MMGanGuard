#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名：train.py
文件内容：单独训练各个模型并记录详细性能指标，包括resnet, densenet, co_occurrence, gram_net及final模型
日志记录: 运行时间、精确率、召回率、F1-score、准确率、损失、学习率，数据保存到JSON文件和TensorBoard
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

"""
类：DetailedExperimentLogger
功能：详细记录实验数据到JSON文件，包括超参数、训练历史、性能指标等
输入：save_dir (保存路径)、experiment_name (实验名称)
输出：无
"""
class DetailedExperimentLogger:

    def __init__(self, save_dir, experiment_name):
        """
        函数：
        功能：初始化实验记录器
        输入：
            save_dir (str): 保存实验数据的目录路径
            experiment_name (str): 实验名称，用于生成JSON文件名
        输出：无返回值。初始化以下实例属性：
                - self.save_dir: 数据保存目录
                - self.experiment_name: 实验名称  
                - self.experiment_data (dict): 存储所有实验数据，包含：
                * experiment_info: 实验基本信息（名称、开始/结束时间、总耗时）
                * hyperparameters: 超参数配置
                * training_history: 各epoch的训练记录
                * detailed_metrics: 详细性能指标
                * model_metrics: 模型参数统计
                * dataset_info: 数据集信息
                * best_epoch: 最佳epoch信息
                * training_summary: 训练总结
        """
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
        """
        函数：log_hyperparameters
        功能：记录模型训练的超参数
        输入：
            hyperparams (dict): 超参数字典，包含如下键值对：
                - model_type (str): 模型类型
                - batch_size (int): 批次大小
                - num_epochs (int): 训练轮数
                - learning_rate (float): 学习率
                - device (str): 计算设备
                - output_type (str): 输出类型
        输出：
            无返回值。将超参数存储到self.experiment_data['hyperparameters']中
        """
        self.experiment_data['hyperparameters'] = hyperparams

    def log_dataset_info(self, dataset_info):
        """
        函数：log_dataset_info
        功能：记录数据集基本信息
        输入：
            dataset_info (dict): 数据集信息字典，包含：
                - train_size (int): 训练集样本数
                - val_size (int): 验证集样本数
                - batch_size (int): 批次大小
        输出：
            无返回值。将数据集信息存储到self.experiment_data['dataset_info']中
        """
        self.experiment_data['dataset_info'] = dataset_info

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time,
                  train_metrics=None, val_metrics=None):
        """
        函数：log_epoch
        功能：记录单个epoch的训练和验证数据并立即保存到JSON文件
        输入：
            epoch (int): 当前epoch索引（从0开始）
            train_loss (float): 训练集损失值
            train_acc (float): 训练集准确率（0-1之间）
            val_loss (float): 验证集损失值
            val_acc (float): 验证集准确率（0-1之间）
            lr (float): 当前学习率
            epoch_time (float): 本epoch耗时（秒）
            train_metrics (dict, optional): 训练集详细指标（precision, recall, f1等）
            val_metrics (dict, optional): 验证集详细指标
        输出：
            无返回值。更新以下内容：
            - 将数据追加到training_history的各个列表中
            - 如果当前验证准确率更高，更新best_epoch信息
            - 自动调用save_to_json()保存数据到JSON文件
        """
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
        """
        函数：save_to_json
        功能：将当前实验数据保存到JSON文件
        输入：
            无输入参数。使用self.experiment_data和self.save_dir
        输出：
            无返回值。在文件系统中创建/更新JSON文件：
            - 文件路径：{save_dir}/{experiment_name}.json
            - 文件内容：将experiment_data序列化为格式化的JSON格式
        """
        json_path = self.get_json_path()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)
        print(f"实验数据已更新到: {json_path}")

    def log_model_metrics(self, model_metrics):
        """
        函数：log_model_metrics
        功能：记录模型参数统计和大小信息
        输入：
            model_metrics (dict): 模型指标字典，包含：
                - total_parameters (int): 总参数数量
                - trainable_parameters (int): 可训练参数数量
                - model_size_mb (float): 模型大小（MB）
                - individual_models (dict, optional): 各子模型参数统计（仅Final模型）
        输出：
            无返回值。将模型指标存储到self.experiment_data['model_metrics']中
        """
        self.experiment_data['model_metrics'] = model_metrics

    def finalize(self):
        """
        函数：finalize
        功能：完成实验记录并生成训练总结
        输入：
            无输入参数。使用self.experiment_data中的历史数据
        输出：
            json_path (str): 保存的JSON文件路径
            同时更新以下内容：
            - 计算并记录总训练耗时
            - 生成training_summary包含：
              * total_training_time: 总耗时（秒）
              * final_train_accuracy: 最终训练准确率
              * final_val_accuracy: 最终验证准确率
              * final_train_loss: 最终训练损失
              * final_val_loss: 最终验证损失
              * best_val_accuracy: 最佳验证准确率
              * best_epoch: 最佳epoch索引
            - 自动保存更新后的JSON文件
        """
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
        """
        函数：get_json_path
        功能：获取实验数据保存的JSON文件路径
        输入：
            无输入参数。使用self.save_dir和self.experiment_name
        输出：
            json_path (str): JSON文件的完整路径，格式为 {save_dir}/{experiment_name}.json
        """
        return os.path.join(self.save_dir, f"{self.experiment_name}.json")


def compute_detailed_metrics(outputs, labels, is_binary=False):
    """
    函数：compute_detailed_metrics
    功能：计算详细的性能指标，包括精确率、召回率、F1-score等
    输入：
        outputs (torch.Tensor): 模型输出，形状为(batch_size,) 或 (batch_size, num_classes)
        labels (torch.Tensor): 真实标签，形状为(batch_size,)
        is_binary (bool): 是否为二元分类，默认False
    输出：
        dict: 包含以下性能指标的字典：
            - accuracy (float): 整体准确率
            - macro_precision (float): 宏平均精确率
            - macro_recall (float): 宏平均召回率
            - macro_f1 (float): 宏平均F1-score
            - ai_precision (float): AI类别精确率
            - ai_recall (float): AI类别召回率
            - ai_f1 (float): AI类别F1-score
            - nature_precision (float): 自然类别精确率
            - nature_recall (float): 自然类别召回率
            - nature_f1 (float): 自然类别F1-score
            - confusion_matrix (list): 混淆矩阵的列表形式
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
    """
    函数：create_model
    功能：创建指定类型的模型
    输入：
        model_type (str): 模型类型，支持：'resnet', 'densenet', 'co_occurrence', 'gram_net', 'final'
        num_classes (int): 分类类数，默认为2（二元分类）
        pretrained (bool): 是否加载预训练权重，默认为True
    输出：
        model (torch.nn.Module 或 dict): 
            - 对于其他模型类型：返回PyTorch模型对象
            - 对于'final'类型：返回字典，包含：
              * 'resnet': ResNet50模型
              * 'densenet': DenseNet201模型
              * 'co_occurrence': 共现矩阵模型
              * 'gram_net': GramNet模型
              * 'weights': 权重张量[0.2, 0.6, 0.1, 0.1]
    异常：
        ValueError: 当model_type不在支持列表中时抛出
    """
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
    """
    函数：get_criterion_and_output_type
    功能：根据模型类型返回对应的损失函数和输出类型
    输入：
        model_type (str): 模型类型，支持：'resnet', 'densenet', 'co_occurrence', 'gram_net', 'final'
    输出：
        tuple: (criterion, output_type)
            - criterion (torch.nn.Module): 损失函数对象
              * co_occurrence或final模型：nn.BCELoss()（二元交叉熵）
              * 其他模型：nn.CrossEntropyLoss()（交叉熵）
            - output_type (str): 输出类型，'binary' 或 'multi_class'
    """
    if model_type == 'co_occurrence':
        return nn.BCELoss(), 'binary'
    elif model_type == 'final':
        # Final模型使用BCELoss，因为输出是概率
        return nn.BCELoss(), 'binary'
    else:
        return nn.CrossEntropyLoss(), 'multi_class'


def convert_to_probability(outputs, model_type, is_binary=False):
    """
    函数：convert_to_probability
    功能：将模型输出转换为真实图像的概率值
    输入：
        outputs (torch.Tensor): 模型输出，shape为(batch_size, num_classes)或(batch_size,)
        model_type (str): 模型类型
        is_binary (bool): 是否为二元分类模型
    输出：
        torch.Tensor: 真实图像的概率值
            - 对于二元分类（co_occurrence或final）：直接返回输出
            - 对于多分类（resnet、densenet、gram_net）：返回softmax后第1类的概率
    """
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
    """
    函数：log_probability_info
    功能：将概率信息详细记录到文本日志文件中
    输入：
        prob_info (dict): 概率信息字典，包含individual_probs、weights、weighted_probs、final_prob
        epoch (int): 当前epoch索引
        batch_idx (int): 当前batch索引
        sample_idx (int): 样本索引，默认为0
        log_file (str): 日志文件名，默认为"probability_log.txt"
    输出：
        无返回值。在日志文件中追加写入：
            - Epoch和Batch信息
            - 各个模型的输出概率
            - 权重值和加权后的概率
            - 最终加权概率和预测结果
    """
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
    函数：log_validation_batch
    功能：记录每个验证batch的详细输出概率、预测和真实值到日志文件
    输入：
        model_type (str): 模型类型
        outputs (torch.Tensor): 主模型输出
        labels (torch.Tensor): 真实标签
        epoch (int): 当前epoch索引
        batch_idx (int): 当前batch索引
        log_file (str): 日志文件名，默认为"validation_detailed.log"
        individual_model_outputs (dict, optional): 各子模型的输出字典 {model_name: output_tensor}
    输出：
        无返回值。在日志文件中追加写入：
            - 验证批次的时间戳和基本信息
            - 每个样本的输出概率、预测结果和真实标签
            - 各子模型的输出（如有）
            - 批次准确率统计和各子模型的准确率（如有）
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
    """
    函数：forward_final_model
    功能：Final模型的前向传播，计算各子模型的加权平均输出
    输入：
        model_dict (dict): 包含四个子模型的字典 {resnet, densenet, co_occurrence, gram_net, weights}
        x (torch.Tensor): 输入张量，shape为(batch_size, 3, 256, 256)
        device (str): 计算设备('cuda'或'cpu')
        record_probs (bool): 是否记录各子模型的概率，默认False
    输出：
        如果record_probs为False：
            final_prob (torch.Tensor): 最终加权概率，shape为(batch_size, 1)
        如果record_probs为True：
            tuple: (final_prob, prob_info)其中prob_info包含各模型概率和权重信息
    """
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
    """
    函数：train_epoch
    功能：训练模型一个epoch，更新模型参数并计算性能指标
    输入：
        model (torch.nn.Module 或 dict): 模型对象（Final模型时为字典）
        train_loader (DataLoader): 训练数据加载器
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        device (str): 计算设备
        is_binary (bool): 是否为二元分类，默认False
        model_type (str): 模型类型，用于特殊处理
    输出：
        tuple: (epoch_loss, epoch_acc, detailed_metrics, epoch_time)
            - epoch_loss (float): 整个epoch的平均损失
            - epoch_acc (float): 整个epoch的准确率
            - detailed_metrics (dict): 详细性能指标
            - epoch_time (float): epoch耗时（秒）
    特殊处理：
        - 当model_type为'final'时，调用train_final_epoch
        - 当model_type为'co_occurrence'时，使用梯度裁剪
    """
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
    """
    函数：train_final_epoch
    功能：训练Final模型的一个epoch（分别训练四个子模型）
    输入：
        model_dict (dict): 包含四个子模型的字典
        train_loader (DataLoader): 训练数据加载器
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 虚拟优化器（实际为每个子模型创建独立优化器）
        device (str): 计算设备
    输出：
        tuple: (final_loss, final_acc, final_metrics, total_time)
            - final_loss (float): 四个模型损失的平均值
            - final_acc (float): 四个模型准确率的平均值
            - final_metrics (dict): 合并的详细指标及individual_models子模型指标
            - total_time (float): 总耗时（秒）
    特殊处理：
        - co_occurrence模型使用较低学习率和权重衰减
        - 返回四个模型指标的平均值
    """
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
    """
    函数：validate_epoch
    功能：验证模型在验证集上的性能，计算各项指标
    输入：
        model (torch.nn.Module 或 dict): 模型对象（Final模型时为字典）
        val_loader (DataLoader): 验证数据加载器
        criterion (torch.nn.Module): 损失函数
        device (str): 计算设备
        is_binary (bool): 是否为二元分类，默认False
        model_type (str): 模型类型
        epoch (int): 当前epoch索引，用于日志记录
    输出：
        tuple: (epoch_loss, epoch_acc, detailed_metrics, epoch_time)
            - epoch_loss (float): 整个epoch的平均损失
            - epoch_acc (float): 整个epoch的准确率
            - detailed_metrics (dict): 详细性能指标
            - epoch_time (float): epoch耗时（秒）
    特殊处理：
        - 当model_type为'final'时，调用validate_final_model
        - 记录验证batch详细信息到日志文件
        - co_occurrence模型每100个batch记录一次概率信息
    """
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
    """
    函数：validate_final_model
    功能：验证Final模型（计算各子模型加权结果）
    输入：
        model_dict (dict): 包含四个子模型的字典 {resnet, densenet, co_occurrence, gram_net, weights}
        val_loader (DataLoader): 验证数据加载器
        criterion (torch.nn.Module): 损失函数
        device (str): 计算设备
        epoch (int): 当前epoch索引，默认为0
        log_probabilities (bool): 是否记录概率信息，默认False
    输出：
        tuple: (final_loss, final_acc, final_metrics, time_taken)
            - final_loss (float): Final模型的损失值
            - final_acc (float): Final模型的准确率
            - final_metrics (dict): 详细指标，包含：
              * 整体和各类别的precision, recall, f1
              * individual_models: 各子模型的验证指标
            - time_taken (float): 耗时（秒，暂时设为0）
    功能细节：
        - 分别验证每个子模型
        - 计算Final模型的加权输出
        - 每100个batch记录一次概率信息（如果启用）
        - 记录验证详细信息到日志文件
    """
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
    """
    函数：train_model
    功能：训练指定模型的主函数，包括初始化、训练循环、模型保存等
    输入：
        model_type (str): 模型类型 ('resnet', 'densenet', 'co_occurrence', 'gram_net', 'final')
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        num_epochs (int): 训练轮数
        device (str): 计算设备('cuda'或'cpu')
        save_dir (str): 模型保存目录
        learning_rate (float): 初始学习率，默认1e-3
        batch_size (int): 批次大小，默认32
        experiment_name (str, optional): 实验名称，默认自动生成
    输出：
        best_val_acc (float): 最佳验证准确率（0-1之间）
    主要功能：
        - 创建或加载模型
        - 初始化损失函数、优化器和学习率调度器
        - 创建DetailedExperimentLogger记录实验数据
        - 循环训练num_epochs个epoch，每个epoch进行：
          * 训练阶段
          * 验证阶段
          * 记录指标到TensorBoard和JSON
          * 保存最佳和最终模型
    输出文件：
        - 最佳模型：{save_dir}/best_{model_type}_model.pth (或目录)
        - 最终模型：{save_dir}/final_{model_type}_model.pth (或目录)
        - 实验数据：{save_dir}/tensorboard_logs/{experiment_name}.json
        - TensorBoard日志：{save_dir}/tensorboard_logs/{experiment_name}/
    """

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
    """
    函数：main
    功能：程序入口函数，解析命令行参数、加载数据和执行模型训练
    输入：
        无直接输入，通过命令行参数传入（见argparse定义）：
        - --model_type: 模型类型，默认'final'
        - --data_dir: 数据根目录，默认'E:/dataset/BigGAN'
        - --batch_size: 批次大小，默认32
        - --num_epochs: 训练轮数，默认5
        - --learning_rate: 学习率，默认1e-3
        - --save_dir: 模型保存目录，默认'./checkpoints'
        - --num_workers: 数据加载工作进程数，默认4
        - --device: 计算设备，默认'cuda'（如可用）或'cpu'
    输出：
        无返回值。执行以下操作：
        - 加载并预处理数据（训练集和验证集）
        - 构建数据加载器
        - 调用train_model()训练模型
        - 打印训练结果和最佳准确率
    数据预处理：
        训练集：调整大小(256x256)、随机水平翻转、随机旋转(±10度)、颜色抖动、标准化
        验证集：调整大小(256x256)、标准化
    输出示例：
        使用设备: cuda
        训练集大小: 10000
        验证集大小: 2000
        开始训练 final 模型...
        ...
        最佳验证准确率: 0.9234
    """
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