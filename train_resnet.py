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
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
import glob


def check_training_completed(model_name, save_dir, expected_epochs):
    """
    检查模型是否已经完成训练
    Args:
        model_name: 模型名称
        save_dir: 保存目录
        expected_epochs: 期望的训练轮数
    Returns:
        bool: 如果训练已完成返回True，否则返回False
    """
    model_save_dir = os.path.join(save_dir, model_name)

    # 检查模型文件是否存在
    best_model_path = os.path.join(model_save_dir, 'best_model.pth')
    latest_model_path = os.path.join(model_save_dir, 'latest_model.pth')

    if not os.path.exists(best_model_path) or not os.path.exists(latest_model_path):
        return False

    try:
        # 加载检查点检查训练轮数
        checkpoint = torch.load(latest_model_path, map_location='cpu')
        completed_epochs = checkpoint['epoch'] + 1  # epoch是从0开始的

        # 如果已经完成了期望的轮数，则认为训练已完成
        if completed_epochs >= expected_epochs:
            print(f"[OK] {model_name} 已经完成了 {completed_epochs} 轮训练，跳过训练")
            return True
        else:
            print(f"[WARN] {model_name} 只完成了 {completed_epochs}/{expected_epochs} 轮训练，需要继续训练")
            return False

    except Exception as e:
        print(f"[ERROR] 检查 {model_name} 训练状态时出错: {e}")
        return False


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


def create_resnet_model(model_type='resnet50', num_classes=2, pretrained=True):
    """
    创建ResNet模型
    Args:
        model_type: ResNet类型，可选 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        num_classes: 类别数
        pretrained: 是否使用预训练权重
    """

    # 支持的ResNet模型
    resnet_models = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
    }

    if model_type not in resnet_models:
        raise ValueError(f"不支持的ResNet模型: {model_type}。支持的模型: {list(resnet_models.keys())}")

    # 创建模型
    model = resnet_models[model_type](pretrained=pretrained)

    # 修改最后的全连接层
    if model_type in ['resnet18', 'resnet34']:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    return model


def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_dir, is_binary=False, shared_writer=None):
    """训练模型"""

    # 创建保存目录
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # 创建TensorBoard写入器
    if shared_writer is None:
        writer = SummaryWriter(log_dir=os.path.join(model_save_dir, 'tensorboard_logs'))
    else:
        writer = shared_writer

    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 记录训练开始时间
    import time
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'\n{model_name} - Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}')

        for inputs, labels in train_pbar:
            # 跳过空批次
            if inputs is None or labels is None:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算损失
            if is_binary:
                labels_binary = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels_binary)
            else:
                loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)

            # 计算准确率
            if is_binary:
                preds = (torch.sigmoid(outputs) > 0.5).long()
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

        # 记录训练指标到TensorBoard
        if shared_writer is None:
            # 独立训练时使用常规记录
            writer.add_scalar('Loss/Train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
        else:
            # 对比训练时使用多模型记录
            writer.add_scalars('Loss/Train', {model_name: epoch_loss}, epoch)
            writer.add_scalars('Accuracy/Train', {model_name: epoch_acc.item()}, epoch)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        val_pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}')

        with torch.no_grad():
            for inputs, labels in val_pbar:
                # 跳过空批次
                if inputs is None or labels is None:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                # 计算损失
                if is_binary:
                    labels_binary = labels.float().unsqueeze(1)
                    loss = criterion(outputs, labels_binary)
                else:
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)

                # 计算准确率
                if is_binary:
                    preds = (torch.sigmoid(outputs) > 0.5).long()
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

        # 记录验证指标到TensorBoard
        if shared_writer is None:
            # 独立训练时使用常规记录
            writer.add_scalar('Loss/Validation', val_epoch_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_epoch_acc, epoch)
        else:
            # 对比训练时使用多模型记录
            writer.add_scalars('Loss/Validation', {model_name: val_epoch_loss}, epoch)
            writer.add_scalars('Accuracy/Validation', {model_name: val_epoch_acc.item()}, epoch)

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        if shared_writer is None:
            writer.add_scalar('Learning_rate', current_lr, epoch)
        else:
            writer.add_scalars('Learning_rate', {model_name: current_lr}, epoch)

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
            }, os.path.join(model_save_dir, 'best_model.pth'))
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
        }, os.path.join(model_save_dir, 'latest_model.pth'))

    # 计算训练时间
    end_time = time.time()
    training_time = end_time - start_time

    print(f'{model_name} 训练完成，最佳验证准确率: {best_acc:.4f}, 训练时间: {training_time:.2f}秒')

    # 重新保存模型，包含训练时间信息
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'training_time': training_time
    }, os.path.join(model_save_dir, 'latest_model.pth'))

    # 关闭TensorBoard写入器（仅当是独立写入器时）
    if shared_writer is None:
        writer.close()

    return {
        'model_name': model_name,
        'best_val_acc': best_acc.item(),
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'training_time': training_time
    }


def compare_resnet_models():
    """比较不同ResNet模型的性能"""

    parser = argparse.ArgumentParser(description='ResNet模型性能对比')
    parser.add_argument('--data_dir', type=str, default='E:/data', help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/resnet_comparison', help='模型保存目录')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='要比较的ResNet模型列表')
    parser.add_argument('--force_retrain', action='store_true',
                       help='强制重新训练所有模型，即使已有完整训练记录')

    args = parser.parse_args()

    print(f"使用设备: {args.device}")
    print(f"比较的模型: {args.models}")

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
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

    # 存储所有模型的结果
    all_results = {}

    # 创建共享的TensorBoard写入器用于对比训练
    comparison_log_dir = os.path.join(args.save_dir, 'tensorboard_comparison')
    os.makedirs(comparison_log_dir, exist_ok=True)
    shared_writer = SummaryWriter(log_dir=comparison_log_dir)

    # 训练每个模型
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"检查 {model_type} 训练状态")
        print(f"{'='*60}")

        # 检查模型是否已经完成训练（除非强制重新训练）
        if not args.force_retrain and check_training_completed(model_type, args.save_dir, args.num_epochs):
            # 如果训练已完成，加载现有结果
            try:
                # 使用latest_model.pth获取完整的训练历史
                checkpoint = torch.load(os.path.join(args.save_dir, model_type, 'latest_model.pth'), map_location='cpu')

                # 创建结果字典
                result = {
                    'model_name': model_type,
                    'best_val_acc': checkpoint['best_acc'].item() if torch.is_tensor(checkpoint['best_acc']) else checkpoint['best_acc'],
                    'final_train_acc': checkpoint['train_accs'][-1] if checkpoint['train_accs'] else 0,
                    'final_val_acc': checkpoint['val_accs'][-1] if checkpoint['val_accs'] else 0,
                    'train_losses': checkpoint['train_losses'],
                    'val_losses': checkpoint['val_losses'],
                    'train_accs': checkpoint['train_accs'],
                    'val_accs': checkpoint['val_accs'],
                    'training_time': checkpoint.get('training_time', 0),  # 从模型文件获取训练时间，如果不存在则设为0
                    'total_params': sum(p.numel() for p in create_resnet_model(model_type=model_type, num_classes=2, pretrained=True).parameters())
                }

                all_results[model_type] = result
                print(f"[INFO] {model_type} 使用latest_model.pth的完整训练历史")
                continue  # 跳过训练
            except Exception as e:
                print(f"[ERROR] 加载 {model_type} 现有结果失败: {e}")
                print(f"重新训练 {model_type}")

        print(f"开始训练 {model_type}")

        # 创建模型
        model = create_resnet_model(model_type=model_type, num_classes=2, pretrained=True)
        model = model.to(args.device)

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{model_type} 参数数量: {total_params:,}")

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 训练模型
        start_time = time.time()

        result = train_model(
            model=model,
            model_name=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            device=args.device,
            save_dir=args.save_dir,
            is_binary=False,
            shared_writer=shared_writer
        )

        end_time = time.time()
        training_time = end_time - start_time

        # 添加训练时间信息
        result['training_time'] = training_time
        result['total_params'] = total_params

        all_results[model_type] = result

    # 关闭共享的TensorBoard写入器
    shared_writer.close()

    # 输出比较结果
    print(f"\n{'='*80}")
    print("ResNet模型性能对比结果")
    print(f"{'='*80}")

    print(f"{'模型':<12} {'状态':<10} {'参数数量':<15} {'最佳验证准确率':<15} {'最终验证准确率':<15} {'训练时间(秒)':<15}")
    print(f"{'-'*80}")

    for model_type in args.models:
        result = all_results[model_type]
        status = "跳过" if result.get('training_time', 0) == 0 else "训练"
        print(f"{model_type:<12} {status:<10} {result['total_params']:<15,} {result['best_val_acc']:<15.4f} {result['final_val_acc']:<15.4f} {result['training_time']:<15.2f}")

    # 保存结果到JSON文件
    results_file = os.path.join(args.save_dir, 'comparison_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到: {results_file}")

    # 找出最佳模型
    best_model = max(all_results.items(), key=lambda x: x[1]['best_val_acc'])
    print(f"\n最佳模型: {best_model[0]} (验证准确率: {best_model[1]['best_val_acc']:.4f})")

    return all_results


if __name__ == '__main__':
    compare_resnet_models()