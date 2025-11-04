import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import argparse
import json
import datetime
from torch.utils.tensorboard import SummaryWriter

# 导入新模型
from models.co_occurrence_fast import create_co_occurrence_fast_model as create_co_occurrence_model
from models.gram_net_paper import create_gram_net_paper_model as create_gram_net_model


class ExperimentLogger:
    """实验数据记录器，用于保存详细的训练数据到JSON文件"""

    def __init__(self, save_dir, experiment_name=None):
        """
        初始化实验记录器
        Args:
            save_dir: 保存目录
            experiment_name: 实验名称，如果为None则自动生成
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 生成实验名称
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name

        # 实验数据存储结构
        self.experiment_data = {
            'experiment_info': {
                'name': self.experiment_name,
                'start_time': datetime.datetime.now().isoformat(),
                'save_dir': save_dir
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
            'model_metrics': {
                'total_parameters': 0,
                'trainable_parameters': 0,
                'model_size_mb': 0
            },
            'dataset_info': {},
            'best_epoch': {
                'epoch': 0,
                'val_accuracy': 0.0,
                'val_loss': 0.0
            },
            'training_summary': {
                'total_training_time': 0.0,
                'final_train_accuracy': 0.0,
                'final_val_accuracy': 0.0,
                'final_train_loss': 0.0,
                'final_val_loss': 0.0
            }
        }

        # 子模型数据（针对MMGuardian模型）
        self.submodel_data = {
            'train_loss': {},
            'train_accuracy': {},
            'val_loss': {},
            'val_accuracy': {},
            'train_precision_recall_f1': {},
            'val_precision_recall_f1': {},
            'train_time': {},
            'val_time': {}
        }

        self.start_time = time.time()
        self.epoch_start_time = None

    def set_hyperparameters(self, hyperparams):
        """设置超参数"""
        self.experiment_data['hyperparameters'] = hyperparams

    def set_dataset_info(self, dataset_info):
        """设置数据集信息"""
        self.experiment_data['dataset_info'] = dataset_info

    def set_model_metrics(self, model):
        """设置模型指标"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 计算模型大小（近似）
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2

        self.experiment_data['model_metrics'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(size_mb, 2)
        }

    def start_epoch(self):
        """开始新的epoch"""
        self.epoch_start_time = time.time()

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, learning_rate, submodel_metrics=None):
        """记录epoch数据"""
        epoch_time = time.time() - self.epoch_start_time

        # 记录主模型数据
        self.experiment_data['training_history']['epochs'].append(epoch)
        self.experiment_data['training_history']['train_loss'].append(float(train_loss))
        self.experiment_data['training_history']['train_accuracy'].append(float(train_acc))
        self.experiment_data['training_history']['val_loss'].append(float(val_loss))
        self.experiment_data['training_history']['val_accuracy'].append(float(val_acc))
        self.experiment_data['training_history']['learning_rate'].append(float(learning_rate))
        self.experiment_data['training_history']['epoch_time'].append(float(epoch_time))

        # 记录子模型数据（针对MMGuardian）
        if submodel_metrics:
            for model_name, metrics in submodel_metrics.items():
                # 初始化子模型数据结构
                if model_name not in self.submodel_data['train_loss']:
                    self.submodel_data['train_loss'][model_name] = []
                    self.submodel_data['train_accuracy'][model_name] = []
                    self.submodel_data['val_loss'][model_name] = []
                    self.submodel_data['val_accuracy'][model_name] = []
                    self.submodel_data['train_precision_recall_f1'][model_name] = []
                    self.submodel_data['val_precision_recall_f1'][model_name] = []
                    self.submodel_data['train_time'][model_name] = []
                    self.submodel_data['val_time'][model_name] = []

                # 记录基础指标
                self.submodel_data['train_loss'][model_name].append(float(metrics.get('train_loss', 0)))
                self.submodel_data['train_accuracy'][model_name].append(float(metrics.get('train_acc', 0)))
                self.submodel_data['val_loss'][model_name].append(float(metrics.get('val_loss', 0)))
                self.submodel_data['val_accuracy'][model_name].append(float(metrics.get('val_acc', 0)))

                # 记录精确率、召回率、F1分数
                if 'train_precision_recall_f1' in metrics:
                    self.submodel_data['train_precision_recall_f1'][model_name].append(metrics['train_precision_recall_f1'])
                if 'val_precision_recall_f1' in metrics:
                    self.submodel_data['val_precision_recall_f1'][model_name].append(metrics['val_precision_recall_f1'])

                # 记录训练时间
                if 'train_time' in metrics:
                    self.submodel_data['train_time'][model_name].append(float(metrics['train_time']))
                if 'val_time' in metrics:
                    self.submodel_data['val_time'][model_name].append(float(metrics['val_time']))

        # 更新最佳epoch
        if val_acc > self.experiment_data['best_epoch']['val_accuracy']:
            self.experiment_data['best_epoch'] = {
                'epoch': epoch,
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss)
            }

    def finalize(self, final_train_loss, final_train_acc, final_val_loss, final_val_acc):
        """完成训练，记录最终总结"""
        total_time = time.time() - self.start_time

        self.experiment_data['experiment_info']['end_time'] = datetime.datetime.now().isoformat()
        self.experiment_data['experiment_info']['total_duration_seconds'] = total_time

        self.experiment_data['training_summary'] = {
            'total_training_time': round(total_time, 2),
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss)
        }

        # 添加子模型数据
        if self.submodel_data['train_loss']:
            self.experiment_data['submodel_history'] = self.submodel_data

        # 保存到文件
        self.save()

    def save(self):
        """保存实验数据到JSON文件"""
        filename = os.path.join(self.save_dir, f"{self.experiment_name}.json")

        # 确保所有数据都是JSON可序列化的
        def make_json_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj

        serializable_data = make_json_serializable(self.experiment_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        print(f"实验数据已保存到: {filename}")

    def get_file_path(self):
        """获取实验数据文件路径"""
        return os.path.join(self.save_dir, f"{self.experiment_name}.json")


def compute_precision_recall_f1(outputs, labels, is_binary=False):
    """
    计算精确率、召回率和F1分数
    Args:
        outputs: 模型输出
        labels: 真实标签
        is_binary: 是否是二元分类
    Returns:
        dict: 包含精确率、召回率、F1分数的字典
    """
    if is_binary:
        # 二元分类：AI (0) vs Nature (1)
        # 假设AI是正类(positive class)
        preds = (outputs > 0.5).long()
        if preds.dim() > 1:
            preds = preds.squeeze(1)
    else:
        # 多分类
        _, preds = torch.max(outputs, 1)

    # 转换为numpy数组用于计算
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 计算混淆矩阵
    cm = compute_confusion_matrix(labels_np, preds_np, class_labels=[0, 1])

    # 计算各类别的指标
    metrics = {}
    for class_idx, class_name in [(0, 'ai'), (1, 'nature')]:
        # TP: 预测为正类且实际为正类
        tp = cm[class_idx, class_idx]
        # FP: 预测为正类但实际为负类
        fp = cm[:, class_idx].sum() - tp
        # FN: 预测为负类但实际为正类
        fn = cm[class_idx, :].sum() - tp

        # 计算精确率、召回率、F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'{class_name}_precision'] = float(precision)
        metrics[f'{class_name}_recall'] = float(recall)
        metrics[f'{class_name}_f1'] = float(f1)

    # 计算宏平均
    ai_precision = metrics['ai_precision']
    nature_precision = metrics['nature_precision']
    ai_recall = metrics['ai_recall']
    nature_recall = metrics['nature_recall']
    ai_f1 = metrics['ai_f1']
    nature_f1 = metrics['nature_f1']

    metrics['macro_precision'] = float((ai_precision + nature_precision) / 2)
    metrics['macro_recall'] = float((ai_recall + nature_recall) / 2)
    metrics['macro_f1'] = float((ai_f1 + nature_f1) / 2)

    return metrics


def compute_confusion_matrix(true_labels, pred_labels, class_labels=None):
    """
    计算混淆矩阵
    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
        class_labels: 类别标签列表
    Returns:
        numpy.ndarray: 混淆矩阵
    """
    if class_labels is None:
        class_labels = np.unique(np.concatenate([true_labels, pred_labels]))

    n_classes = len(class_labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(true_labels, pred_labels):
        true_idx = np.where(class_labels == true_label)[0][0]
        pred_idx = np.where(class_labels == pred_label)[0][0]
        cm[true_idx, pred_idx] += 1

    return cm


def average_precision_recall_f1(prf1_list):
    """
    计算精确率、召回率、F1分数的平均值
    Args:
        prf1_list: 包含多个批次的精确率、召回率、F1分数的列表
    Returns:
        dict: 平均后的精确率、召回率、F1分数
    """
    if not prf1_list:
        return {}

    # 初始化累加器
    total_metrics = {}
    count = len(prf1_list)

    # 累加所有批次的指标
    for prf1 in prf1_list:
        for key, value in prf1.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value

    # 计算平均值
    avg_metrics = {}
    for key, total in total_metrics.items():
        avg_metrics[key] = total / count

    return avg_metrics


def custom_collate_fn(batch):
    """自定义collate函数，过滤掉None样本"""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


class AINatureDataset(Dataset):
    """自定义数据集类，用于加载AI和Nature图片"""

    def __init__(self, data_dir, transform=None):
        """
        初始化数据集
        Args:
            data_dir: 数据目录，包含ai和nature子目录
            transform: 数据预处理变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = ['ai', 'nature']
        self.class_to_idx = {'ai': 0, 'nature': 1}

        # 加载AI图片
        ai_dir = os.path.join(data_dir, 'ai')
        if os.path.exists(ai_dir):
            for img_name in os.listdir(ai_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(ai_dir, img_name)
                    self.samples.append((img_path, 0))  # 0表示AI类

        # 加载Nature图片
        nature_dir = os.path.join(data_dir, 'nature')
        if os.path.exists(nature_dir):
            for img_name in os.listdir(nature_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(nature_dir, img_name)
                    self.samples.append((img_path, 1))  # 1表示Nature类

        print(f"数据集加载完成: {len(self.samples)} 张图片")
        print(f"AI图片: {len([s for s in self.samples if s[1] == 0])} 张")
        print(f"Nature图片: {len([s for s in self.samples if s[1] == 1])} 张")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图片，添加错误处理
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 返回None表示跳过此样本
            return None, None

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label


def create_model(model_type='resnet', num_classes=2, pretrained=True):
    """
    创建模型
    Args:
        model_type: 模型类型，可选 'resnet', 'densenet', 'mmganguard', 'co_occurrence', 'gram_net'
        num_classes: 类别数
        pretrained: 是否使用预训练权重
    """

    if model_type == 'resnet':
        """创建ResNet模型并加载预训练权重"""
        model = models.resnet50(pretrained=pretrained)
        # 修改最后的全连接层
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model

    elif model_type == 'densenet':
        """创建DenseNet模型并加载预训练权重"""
        model = models.densenet201(pretrained=pretrained)
        # 修改最后的全连接层
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        return model

    elif model_type == 'co_occurrence':
        """创建共现矩阵模型"""
        return create_co_occurrence_model(num_classes=num_classes, pretrained=pretrained)

    elif model_type == 'gram_net':
        """创建Gram-Net模型"""
        return create_gram_net_model(num_classes=num_classes, input_size=256)
    
    elif model_type == 'mmganguard':
        """创建MMGuardian模型，使用ResNet*2 + DenseNet*6 + (CoOccurrence + GramNet)*0.1的加权组合"""
        class MMGuardian(nn.Module):
            def __init__(self, num_classes=2, pretrained=True):
                super(MMGuardian, self).__init__()

                # 创建ResNet模型
                self.resnet = models.resnet50(pretrained=pretrained)
                num_features_resnet = self.resnet.fc.in_features
                self.resnet.fc = nn.Linear(num_features_resnet, num_classes)

                # 创建DenseNet模型
                self.densenet = models.densenet201(pretrained=pretrained)
                num_features_densenet = self.densenet.classifier.in_features
                self.densenet.classifier = nn.Linear(num_features_densenet, num_classes)

                # 创建共现矩阵模型
                self.co_occurrence = create_co_occurrence_model(num_classes=num_classes, pretrained=pretrained)

                # 创建Gram-Net模型
                self.gram_net = create_gram_net_model(num_classes=num_classes, input_size=256)

                # 权重系数
                self.resnet_weight = 2.0
                self.densenet_weight = 6.0
                self.co_occurrence_weight = 0.1
                self.gram_net_weight = 0.1
                self.total_weight = (self.resnet_weight + self.densenet_weight +
                                   self.co_occurrence_weight + self.gram_net_weight)

            def forward(self, x, return_individual=False):
                # 分别获取四个模型的输出
                resnet_output = self.resnet(x)
                densenet_output = self.densenet(x)
                co_occurrence_output = self.co_occurrence(x)
                gram_net_output = self.gram_net(x)

                # 计算加权组合: ResNet*2 + DenseNet*6 + (CoOccurrence + GramNet)*0.1
                # weighted_output = (resnet_output * self.resnet_weight +
                #                  densenet_output * self.densenet_weight +
                #                  co_occurrence_output * self.co_occurrence_weight +
                #                  gram_net_output * self.gram_net_weight) / self.total_weight
                
                weighted_output = (resnet_output * self.resnet_weight +
                                 densenet_output * self.densenet_weight +
                                 co_occurrence_output * self.co_occurrence_weight +
                                 gram_net_output * self.gram_net_weight) / 4

                if return_individual:
                    return {
                        'final': weighted_output,
                        'resnet': resnet_output,
                        'densenet': densenet_output,
                        'co_occurrence': co_occurrence_output,
                        'gram_net': gram_net_output
                    }
                else:
                    return weighted_output

        return MMGuardian(num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def compute_model_metrics(outputs_dict, labels, criterion, model_names, is_binary=False, compute_detailed=False):
    """计算各个模型的损失和准确率"""
    metrics = {}
    for name in model_names:
        if name in outputs_dict:
            outputs = outputs_dict[name]

            # 根据是否是二元分类调整损失计算
            if is_binary or name == 'co_occurrence':
                # 对于二元分类，需要将labels转换为float并调整形状
                labels_binary = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels_binary)
                # 模型输出已经是概率值，直接进行二元分类预测
                preds = (outputs > 0.5).long()
                # 确保preds形状与labels匹配
                if preds.dim() > 1:
                    preds = preds.squeeze(1)
                else:
                    preds = preds.squeeze()
                # 准确率计算
                acc = torch.sum(preds == labels.data).float() / len(labels)
            else:
                # 确保标签是long类型，适用于CrossEntropyLoss
                labels_long = labels.long()
                loss = criterion(outputs, labels_long)
                _, preds = torch.max(outputs, 1)
                # 准确率计算
                acc = torch.sum(preds == labels_long.data).float() / len(labels)

            metrics[f'{name}_loss'] = loss.item()
            metrics[f'{name}_acc'] = acc.item()

            # 如果需要计算详细指标
            if compute_detailed:
                precision_recall_f1 = compute_precision_recall_f1(outputs, labels, is_binary or name == 'co_occurrence')
                metrics[f'{name}_precision_recall_f1'] = precision_recall_f1

    return metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir, is_binary=False, experiment_name=None, hyperparams=None, dataset_info=None):
    """训练模型"""

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard_logs'))

    # 创建实验数据记录器
    experiment_logger = ExperimentLogger(save_dir, experiment_name)

    # 设置模型指标
    experiment_logger.set_model_metrics(model)

    # 设置超参数和数据集信息
    if hyperparams:
        experiment_logger.set_hyperparameters(hyperparams)
    if dataset_info:
        experiment_logger.set_dataset_info(dataset_info)

    # 检查是否是MMGuardian模型
    is_mmganguard = hasattr(model, 'resnet') and hasattr(model, 'densenet')
    model_names = ['final', 'resnet', 'densenet', 'co_occurrence', 'gram_net'] if is_mmganguard else ['final']

    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # 开始epoch计时
        experiment_logger.start_epoch()

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}')
        train_individual_losses = {name: 0.0 for name in model_names}
        train_individual_corrects = {name: 0 for name in model_names}

        # 对于MMGuardian模型，收集详细的子模型数据
        if is_mmganguard:
            train_submodel_precision_recall_f1 = {name: [] for name in model_names if name != 'final'}
            train_submodel_time = {name: 0.0 for name in model_names if name != 'final'}

        for inputs, labels in train_pbar:
            # 跳过空批次
            if inputs is None or labels is None:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            if is_mmganguard:
                # 记录前向传播时间
                forward_start_time = time.time()
                outputs_dict = model(inputs, return_individual=True)
                outputs = outputs_dict['final']
                forward_time = time.time() - forward_start_time

                # 计算各个模型的指标
                batch_metrics = compute_model_metrics(outputs_dict, labels, criterion, model_names, is_binary, compute_detailed=True)

                # 累加各个模型的损失和正确预测数
                for name in model_names:
                    if f'{name}_loss' in batch_metrics:
                        train_individual_losses[name] += batch_metrics[f'{name}_loss'] * inputs.size(0)
                    if f'{name}_acc' in batch_metrics:
                        train_individual_corrects[name] += int(batch_metrics[f'{name}_acc'] * len(labels))

                    # 收集详细的精确率、召回率、F1分数
                    if name != 'final' and f'{name}_precision_recall_f1' in batch_metrics:
                        train_submodel_precision_recall_f1[name].append(batch_metrics[f'{name}_precision_recall_f1'])
                        train_submodel_time[name] += forward_time / len(model_names)  # 平均分配时间

                # 使用最终输出的损失进行反向传播 - MMGuardian最终输出使用多分类
                loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                outputs_dict = {'final': outputs}
                batch_metrics = compute_model_metrics(outputs_dict, labels, criterion, model_names, is_binary)

                # 使用最终输出的损失进行反向传播 - MMGuardian最终输出使用多分类
                loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            # 根据是否是二元分类调整预测计算
            if is_binary:
                # 模型输出已经是概率值，直接进行二元分类预测
                preds = (outputs > 0.5).long()
                # 确保preds形状与labels匹配
                if preds.dim() > 1:
                    preds = preds.squeeze(1)
                else:
                    preds = preds.squeeze()
            else:
                _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{torch.sum(preds == labels.data).item() / len(labels):.4f}'
            })

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 记录训练指标到TensorBoard - 将同一指标的不同模型绘制在同一图像中
        if is_mmganguard:
            # 使用add_scalars方法在同一图像中显示所有模型
            train_loss_dict = {}
            train_acc_dict = {}

            # 添加最终结果
            train_loss_dict['Final'] = epoch_loss
            train_acc_dict['Final'] = epoch_acc.item()

            # 添加各个子模型
            for name in model_names:
                if name != 'final':
                    individual_loss = train_individual_losses[name] / len(train_loader.dataset)
                    individual_acc = train_individual_corrects[name] / len(train_loader.dataset)
                    train_loss_dict[name.capitalize()] = individual_loss
                    train_acc_dict[name.capitalize()] = individual_acc

            # 记录到同一图像
            writer.add_scalars('Loss/Train', train_loss_dict, epoch)
            writer.add_scalars('Accuracy/Train', train_acc_dict, epoch)
        else:
            # 单模型训练
            writer.add_scalar('Loss/Train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/Train', epoch_acc, epoch)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        val_pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}')
        val_individual_losses = {name: 0.0 for name in model_names}
        val_individual_corrects = {name: 0 for name in model_names}

        # 对于MMGuardian模型，收集详细的子模型数据
        if is_mmganguard:
            val_submodel_precision_recall_f1 = {name: [] for name in model_names if name != 'final'}
            val_submodel_time = {name: 0.0 for name in model_names if name != 'final'}

        with torch.no_grad():
            for inputs, labels in val_pbar:
                # 跳过空批次
                if inputs is None or labels is None:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)

                if is_mmganguard:
                    # 记录前向传播时间
                    forward_start_time = time.time()
                    outputs_dict = model(inputs, return_individual=True)
                    outputs = outputs_dict['final']
                    forward_time = time.time() - forward_start_time

                    # 计算各个模型的指标
                    batch_metrics = compute_model_metrics(outputs_dict, labels, criterion, model_names, is_binary, compute_detailed=True)

                    # 累加各个模型的损失和正确预测数
                    for name in model_names:
                        if f'{name}_loss' in batch_metrics:
                            val_individual_losses[name] += batch_metrics[f'{name}_loss'] * inputs.size(0)
                        if f'{name}_acc' in batch_metrics:
                            val_individual_corrects[name] += int(batch_metrics[f'{name}_acc'] * len(labels))

                        # 收集详细的精确率、召回率、F1分数
                        if name != 'final' and f'{name}_precision_recall_f1' in batch_metrics:
                            val_submodel_precision_recall_f1[name].append(batch_metrics[f'{name}_precision_recall_f1'])
                            val_submodel_time[name] += forward_time / len(model_names)  # 平均分配时间

                    # 使用最终输出的损失进行统计 - MMGuardian最终输出使用多分类
                    loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    outputs_dict = {'final': outputs}
                    batch_metrics = compute_model_metrics(outputs_dict, labels, criterion, model_names, is_binary)

                    # 使用最终输出的损失进行统计 - MMGuardian最终输出使用多分类
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                # 根据是否是二元分类调整预测计算
                if is_binary and not is_mmganguard:
                    # 模型输出已经是概率值，直接进行二元分类预测
                    preds = (outputs > 0.5).long()
                    # 确保preds形状与labels匹配
                    if preds.dim() > 1:
                        preds = preds.squeeze(1)
                    else:
                        preds = preds.squeeze()
                else:
                    _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{torch.sum(preds == labels.data).item() / len(labels):.4f}'
                })

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        # 记录验证指标到TensorBoard - 将同一指标的不同模型绘制在同一图像中
        if is_mmganguard:
            # 使用add_scalars方法在同一图像中显示所有模型
            val_loss_dict = {}
            val_acc_dict = {}

            # 添加最终结果
            val_loss_dict['Final'] = val_epoch_loss
            val_acc_dict['Final'] = val_epoch_acc.item()

            # 添加各个子模型
            for name in model_names:
                if name != 'final':
                    individual_loss = val_individual_losses[name] / len(val_loader.dataset)
                    individual_acc = val_individual_corrects[name] / len(val_loader.dataset)
                    val_loss_dict[name.capitalize()] = individual_loss
                    val_acc_dict[name.capitalize()] = individual_acc

            # 记录到同一图像
            writer.add_scalars('Loss/Validation', val_loss_dict, epoch)
            writer.add_scalars('Accuracy/Validation', val_acc_dict, epoch)
        else:
            # 单模型训练
            writer.add_scalar('Loss/Validation', val_epoch_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_epoch_acc, epoch)

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # 准备子模型指标（针对MMGuardian）
        submodel_metrics = None
        if is_mmganguard:
            submodel_metrics = {}
            for name in model_names:
                if name != 'final':
                    # 计算平均精确率、召回率、F1分数
                    train_prf1_avg = {}
                    val_prf1_avg = {}

                    if name in train_submodel_precision_recall_f1 and train_submodel_precision_recall_f1[name]:
                        train_prf1_avg = average_precision_recall_f1(train_submodel_precision_recall_f1[name])

                    if name in val_submodel_precision_recall_f1 and val_submodel_precision_recall_f1[name]:
                        val_prf1_avg = average_precision_recall_f1(val_submodel_precision_recall_f1[name])

                    submodel_metrics[name] = {
                        'train_loss': train_individual_losses[name] / len(train_loader.dataset),
                        'train_acc': train_individual_corrects[name] / len(train_loader.dataset),
                        'val_loss': val_individual_losses[name] / len(val_loader.dataset),
                        'val_acc': val_individual_corrects[name] / len(val_loader.dataset),
                        'train_precision_recall_f1': train_prf1_avg,
                        'val_precision_recall_f1': val_prf1_avg,
                        'train_time': train_submodel_time[name] if name in train_submodel_time else 0.0,
                        'val_time': val_submodel_time[name] if name in val_submodel_time else 0.0
                    }

        # 记录epoch数据到实验日志
        experiment_logger.log_epoch(
            epoch=epoch,
            train_loss=epoch_loss,
            train_acc=epoch_acc.item(),
            val_loss=val_epoch_loss,
            val_acc=val_epoch_acc.item(),
            learning_rate=current_lr,
            submodel_metrics=submodel_metrics
        )

        # # 记录模型权重和梯度（每10个epoch记录一次）
        # if epoch % 10 == 0:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             writer.add_histogram(f'Weights/{name}', param, epoch)
        #             if param.grad is not None:
        #                 writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # 学习率调度
        if scheduler:
            scheduler.step()

        # 保存最佳模型
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_acc': best_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'保存最佳模型，验证准确率: {best_acc:.4f}')

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_acc': best_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }, os.path.join(save_dir, 'latest_model.pth'))
  
    print(f'训练完成，最佳验证准确率: {best_acc:.4f}')

    # 完成实验记录
    final_train_loss = train_losses[-1] if train_losses else 0.0
    final_train_acc = train_accs[-1] if train_accs else 0.0
    final_val_loss = val_losses[-1] if val_losses else 0.0
    final_val_acc = val_accs[-1] if val_accs else 0.0

    experiment_logger.finalize(final_train_loss, final_train_acc, final_val_loss, final_val_acc)

    # 关闭TensorBoard写入器
    writer.close()

    return train_losses, val_losses, train_accs, val_accs


def main():
    parser = argparse.ArgumentParser(description='AI vs Nature 二分类训练')
    # parser.add_argument('--data_dir', type=str, default='E:/data', help='数据根目录')
    parser.add_argument('--data_dir', type=str, default='E:/code/data_test', help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载工作进程数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'densenet', 'co_occurrence','gram_net','mmganguard'], help='模型类型')

    args = parser.parse_args()

    print(f"使用设备: {args.device}")

    # 设置实验名称（基于模型类型和时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.model_type}_{timestamp}"

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

    # 创建训练集和验证集
    train_dataset = AINatureDataset(os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_dataset = AINatureDataset(os.path.join(args.data_dir, 'val'), transform=val_transform)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    # 创建模型
    model = create_model(model_type=args.model_type, num_classes=2, pretrained=True)
    model = model.to(args.device)

    print(f"使用模型类型: {args.model_type}")

    # 设置实验记录器的超参数和数据集信息
    hyperparams = {
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'num_workers': args.num_workers,
        'is_binary': args.model_type == 'co_occurrence'
    }

    dataset_info = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'train_ai_count': len([s for s in train_dataset.samples if s[1] == 0]),
        'train_nature_count': len([s for s in train_dataset.samples if s[1] == 1]),
        'val_ai_count': len([s for s in val_dataset.samples if s[1] == 0]),
        'val_nature_count': len([s for s in val_dataset.samples if s[1] == 1])
    }

    # 损失函数和优化器 - 根据模型类型选择适当的损失函数
    # 对于共现矩阵模型使用BCELoss（模型已包含sigmoid），其他模型使用CrossEntropyLoss
    is_binary = args.model_type == 'co_occurrence'
    if is_binary:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"开始训练...")

    # 训练模型
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        args.num_epochs, args.device, args.save_dir, is_binary, experiment_name,
        hyperparams, dataset_info
    )

    print("训练完成!")


if __name__ == '__main__':
    main()