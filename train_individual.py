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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

# 导入模型
from models.co_occurrence_fast import create_co_occurrence_fast_model as create_co_occurrence_model
from models.gram_net_paper import create_gram_net_paper_model as create_gram_net_model


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
        """记录每个epoch的数据"""
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

        # 保存到JSON文件
        json_path = os.path.join(self.save_dir, f"{self.experiment_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)

        print(f"实验数据已保存到: {json_path}")
        return json_path


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
        model = torchvision.models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model

    elif model_type == 'densenet':
        model = torchvision.models.densenet201(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        return model

    elif model_type == 'co_occurrence':
        return create_co_occurrence_model(num_classes=num_classes, pretrained=pretrained)

    elif model_type == 'gram_net':
        return create_gram_net_model(num_classes=num_classes, input_size=256)

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_criterion_and_output_type(model_type):
    """根据模型类型返回对应的损失函数和输出类型"""
    if model_type == 'co_occurrence':
        return nn.BCELoss(), 'binary'
    else:
        return nn.CrossEntropyLoss(), 'multi_class'


def train_epoch(model, train_loader, criterion, optimizer, device, is_binary=False):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []

    start_time = time.time()

    for inputs, labels in train_loader:
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

        # 收集输出和标签用于详细指标计算
        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())

    epoch_time = time.time() - start_time

    # 计算详细指标
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    detailed_metrics = compute_detailed_metrics(all_outputs, all_labels, is_binary)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, detailed_metrics, epoch_time


def validate_epoch(model, val_loader, criterion, device, is_binary=False):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

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

            # 收集输出和标签用于详细指标计算
            all_outputs.append(outputs)
            all_labels.append(labels)

    epoch_time = time.time() - start_time

    # 计算详细指标
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    detailed_metrics = compute_detailed_metrics(all_outputs, all_labels, is_binary)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, detailed_metrics, epoch_time


def train_model(model_type, train_loader, val_loader, num_epochs, device, save_dir,
                learning_rate=1e-3, batch_size=32, experiment_name=None):
    """训练指定模型"""

    # 创建模型
    model = create_model(model_type)
    model = model.to(device)

    # 获取损失函数和输出类型
    criterion, output_type = get_criterion_and_output_type(model_type)
    is_binary = (output_type == 'binary')

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_metrics = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
    }
    logger.log_model_metrics(model_metrics)

    print(f"开始训练 {model_type} 模型...")
    print(f"模型参数数量: {total_params:,}")
    print(f"训练数据: {len(train_loader.dataset)} 张图像")
    print(f"验证数据: {len(val_loader.dataset)} 张图像")

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_acc, train_metrics, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, is_binary
        )

        # 验证
        val_loss, val_acc, val_metrics, val_time = validate_epoch(
            model, val_loader, criterion, device, is_binary
        )

        # 学习率调度
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

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

        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"训练时间: {train_time:.2f}s, 验证时间: {val_time:.2f}s")
        print(f"学习率: {current_lr:.6f}")
        print(f"AI图像F1: {val_metrics['ai_f1']:.4f}, 自然图像F1: {val_metrics['nature_f1']:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, f"best_{model_type}_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"保存最佳模型到: {model_path}")

    # 保存最终模型
    final_model_path = os.path.join(save_dir, f"final_{model_type}_model.pth")
    torch.save(model.state_dict(), final_model_path)

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
    parser.add_argument('--model_type', type=str, default='gram_net', choices=['resnet', 'densenet', 'co_occurrence', 'gram_net'], help='要训练的模型类型')
    parser.add_argument('--data_dir', type=str, default='E:/data', help='数据根目录')
    # parser.add_argument('--data_dir', type=str, default='E:/code/data_test', help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')

    args = parser.parse_args()

    print(f"使用设备: {args.device}")

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((100, 100)),
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