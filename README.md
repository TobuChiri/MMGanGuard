## 项目概述

MMGanGuard 项目主要用于：
- 训练和评估多种深度学习模型（ResNet、DenseNet、Co-Occurrence、GramNet）
- 详细记录训练过程中的性能指标
- 生成模型性能对比报告和可视化结果
- 验证模型输出概率和随机样本预测

## 支持的模型

- **ResNet**: 经典的残差网络架构
- **DenseNet**: 密集连接的卷积网络
- **Co-Occurrence Matrix**: 基于共现矩阵的模型
- **GramNet**: 基于语法特征的网络模型

## 环境要求

### Python 版本
- Python 3.7+

### 依赖包

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.0.0（深度学习框架）
- torchvision >= 0.15.0（图像处理工具）
- numpy >= 1.24.0（数值计算）
- pandas >= 2.0.0（数据处理）
- scikit-learn >= 1.3.0（机器学习指标）
- matplotlib >= 3.7.0（可视化）
- tensorboard >= 2.13.0（训练监控）
- Pillow >= 10.0.0（图像处理）
- tqdm >= 4.66.0（进度条）

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/TobuChiri/MMGanGuard.git
cd MMGanGuard

# 安装依赖
pip install -r requirements.txt
```

### 2. 单个模型训练

```bash
# 训练多种不同卷积层数的ResNet模型
python train_resnet.py

# 训练多种不同卷积层数的DenseNet模型
python train_densenet.py

# 训练单一模型或所有模型
python train.py --model resnet
python train.py --model densenet
python train.py --model co_occurrence
python train.py --model gram_net
python train.py --model final
```

### 3. 性能评估

```bash
# 验证批量概率
python validate_batch_probabilities.py

# 验证随机样本
python validate_random_samples.py

# 分析训练结果
python analyze_training_results.py
```

### 4. 生成对比报告

```bash
# 生成综合对比
python generate_comprehensive_comparison.py

# 生成DenseNet对比
python generate_densenet_comparison.py

# 生成ResNet对比
python generate_resnet_comparison.py

# 生成MMGanGuard对比
python generate_mmganguard_comparison.py

# 生成训练进度对比
python generate_training_progress_comparison.py
```

## 项目结构

```
MMGanGuard/
├── train.py                          # 主训练脚本
├── train_resnet.py                   # ResNet模型训练
├── train_densenet.py                 # DenseNet模型训练
├── models/                           # 模型定义
│   ├── co_occurrence_fast.py         # Co-Occurrence模型
│   └── gram_net_paper.py             # GramNet模型
├── checkpoints/                      # 模型检查点和权重
│   ├── best_final_model/
│   ├── densenet_comparison/
│   └── tensorboard_logs/
├── validate_batch_probabilities.py   # 批量概率验证
├── validate_random_samples.py        # 随机样本验证
├── analyze_training_results.py       # 训练结果分析
├── generate_*.py                     # 各种对比报告生成脚本
├── weight_analysis_table.py          # 权重分析工具
└── print_pth.py                      # 模型权重打印工具
```

```
dataset/                 
├── train/                            # 训练集
│   ├── ai/                           # ai生成的图片
│   └── nature/                       # 真实图像
└── train/                            # 验证集集
    ├── ai/                           # ai生成的图片
    └── nature/                       # 真实图像
```

## 关键功能

### 训练特性

- **详细性能记录**: 记录运行时间、精确率、召回率、F1-score、准确率、损失和学习率
- **TensorBoard集成**: 实时监控训练过程
- **JSON数据保存**: 完整的实验数据记录
- **多模型支持**: 同时或单独训练不同的模型

### 评估特性

- **概率验证**: 验证模型输出的概率分布
- **随机样本测试**: 对随机样本进行预测验证
- **混淆矩阵生成**: 详细的分类性能分析
- **性能对比**: 多模型性能对比分析

## 数据集

项目使用 CIFAR-10 数据集进行训练和评估。数据会自动下载到项目目录。

## 输出文件

- **模型权重**: 保存在 `checkpoints/` 目录
- **训练日志**: 保存在 `tensorboard_logs/` 目录
- **性能指标**: 保存为 JSON 和 CSV 文件
- **可视化结果**: HTML 和图像文件保存在 `visualization_results/` 目录

## 贡献

欢迎提交问题和拉取请求！

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有任何问题或建议，请联系项目维护者。

---

**最后更新**: 2025年11月19日
