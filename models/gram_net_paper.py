import torch
import torch.nn as nn
import torch.nn.functional as F


class GramBlock(nn.Module):
    """
    按照论文配图描述的Gram块架构
    包含: 池化层 + 3×3 conv 32/2 + 3×3 conv 16/2 + Gram矩阵 + 3×3 conv 32
    """
    def __init__(self, in_channels):
        super(GramBlock, self).__init__()

        # 池化层
        self.pool = nn.AdaptiveAvgPool2d((32, 32))  # 统一到固定大小

        # 3×3卷积 32通道 / 步长2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 3×3卷积 16通道 / 步长2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Gram矩阵计算后的3×3卷积 32通道
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 全局池化层用于输出
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def compute_gram_matrix(self, x):
        """
        计算Gram矩阵
        公式: G_ij^l = Σ_k F_ik^l F_jk^l
        """
        batch_size, channels, height, width = x.size()

        # 将特征图转换为向量化空间维度
        features = x.view(batch_size, channels, height * width)  # [B, C, H*W]

        # 计算Gram矩阵: G = F * F^T
        gram = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]

        # 归一化
        gram = gram / (channels * height * width)

        return gram

    def forward(self, x):
        # 池化到固定大小
        x = self.pool(x)

        # 第一个卷积: 3×3 conv 32/2
        x = self.conv1(x)  # [B, 32, 16, 16]

        # 第二个卷积: 3×3 conv 16/2
        x = self.conv2(x)  # [B, 16, 8, 8]

        # 计算Gram矩阵
        gram = self.compute_gram_matrix(x)  # [B, 16, 16]

        # 使用Gram矩阵特征 - 展平Gram矩阵作为风格特征
        gram_features = gram.view(gram.size(0), -1)  # [B, 256]

        # 第三个卷积: 3×3 conv 32
        x = self.conv3(x)  # [B, 32, 8, 8]

        # 全局池化获取空间特征
        spatial_features = self.global_pool(x)  # [B, 32, 1, 1]
        spatial_features = spatial_features.view(spatial_features.size(0), -1)  # [B, 32]

        # 合并Gram特征和空间特征
        combined_features = torch.cat([gram_features, spatial_features], dim=1)  # [B, 288]

        return combined_features


class ResidualBlock(nn.Module):
    """
    残差块，包含两个3×3卷积层和跳跃连接
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GramNetPaper(nn.Module):
    """
    按照论文描述重新实现的Gram-Net模型
    结构: 输入 → GramBlock → 7×7 conv 64/2 → GramBlock → 池化层 → GramBlock →
          4×3×3 conv 64 → GramBlock → 4×3×3 conv 128/2 → GramBlock →
          4×3×3 conv 256/2 → GramBlock → 4×3×3 conv 512/2 →
          池化 → fc256 → 与GramBlock结果相加 → FC2 → softmax
    """
    def __init__(self, num_classes=2, input_size=100):
        super(GramNetPaper, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Gram块 - 按照描述的位置放置
        self.gram_blocks = nn.ModuleList([
            GramBlock(3),      # 输入图像处
            GramBlock(64),     # 在7×7卷积后
            GramBlock(64),     # 在池化层后
            GramBlock(64),     # 在第一个4×3×3 conv后
            GramBlock(128),    # 在第二个4×3×3 conv后
            GramBlock(256),    # 在第三个4×3×3 conv后
        ])

        # 输入层: 7×7卷积 64通道 / 步长2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 256x256 -> 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 128x128 -> 64x64

        # 第一个阶段: 2个残差块 64通道
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),  # 第一个残差块
            ResidualBlock(64, 64),  # 第二个残差块
        )

        # 第二个阶段: 2个残差块 128通道 / 步长2
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # 64x64 -> 32x32
            ResidualBlock(128, 128),
        )

        # 第三个阶段: 2个残差块 256通道 / 步长2
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),  # 32x32 -> 16x16
            ResidualBlock(256, 256),
        )

        # 第四个阶段: 2个残差块 512通道 / 步长2
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),  # 16x16 -> 8x8
            ResidualBlock(512, 512),
        )

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc256 = nn.Linear(512, 256)  # fc256层
        self.fc2 = nn.Linear(256, num_classes)  # FC2层

        # 初始化权重
        self._initialize_weights()


    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        gram_features = []

        # Gram块0: 在输入图像处
        gram0 = self.gram_blocks[0](x)
        gram_features.append(gram0)

        # 7×7卷积 64通道 / 步长2
        x = self.conv1(x)

        # Gram块1: 在7×7卷积后
        gram1 = self.gram_blocks[1](x)
        gram_features.append(gram1)

        # 池化层
        x = self.maxpool(x)

        # Gram块2: 在池化层后
        gram2 = self.gram_blocks[2](x)
        gram_features.append(gram2)

        # 第一个阶段: 2个残差块 64通道
        x = self.stage1(x)

        # Gram块3: 在第一个残差块后
        gram3 = self.gram_blocks[3](x)
        gram_features.append(gram3)

        # 第二个阶段: 2个残差块 128通道 / 步长2
        x = self.stage2(x)

        # Gram块4: 在第二个残差块后
        gram4 = self.gram_blocks[4](x)
        gram_features.append(gram4)

        # 第三个阶段: 2个残差块 256通道 / 步长2
        x = self.stage3(x)

        # Gram块5: 在第三个残差块后
        gram5 = self.gram_blocks[5](x)
        gram_features.append(gram5)

        # 第四个阶段: 2个残差块 512通道 / 步长2
        x = self.stage4(x)

        # 全局平均池化
        spatial_features = self.global_avg_pool(x)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)  # [B, 512]

        # fc256层
        fc256_features = self.fc256(spatial_features)  # [B, 256]

        # 将所有Gram特征相加
        all_gram_features = torch.stack(gram_features, dim=0).sum(dim=0)  # [B, 288]

        # 由于Gram特征维度(288)与fc256特征维度(256)不同，需要调整
        # 使用一个线性层将Gram特征映射到256维度
        if not hasattr(self, 'gram_adapter'):
            self.gram_adapter = nn.Linear(288, 256).to(x.device)
        adapted_gram_features = self.gram_adapter(all_gram_features)  # [B, 256]

        # 与GramBlock结果相加
        combined_features = fc256_features + adapted_gram_features  # [B, 256]

        # FC2层 + softmax
        output = self.fc2(combined_features)

        return output


def create_gram_net_paper_model(num_classes=2, input_size=100, pretrained=True):
    """
    创建按照论文配图描述的Gram-Net模型
    Args:
        num_classes: 分类数量
        input_size: 输入图像大小
        pretrained: 为了接口兼容性保留的参数（GramNetPaper不使用预训练权重）
    Returns:
        model: 按照论文配图实现的GramNet模型
    """
    return GramNetPaper(num_classes=num_classes, input_size=input_size)


if __name__ == "__main__":
    # 测试模型
    model = GramNetPaper(num_classes=2, input_size=100)

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")

    # 测试前向传播
    x = torch.randn(4, 3, 100, 100)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")