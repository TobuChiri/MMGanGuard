import torch
import torch.nn as nn
import torch.nn.functional as F


class CoOccurrenceMatrixFast(nn.Module):
    """
    按照论文方法实现的共现矩阵模型
    使用256x256共现矩阵和论文中的CNN架构
    """

    def __init__(self, num_classes=2, num_bins=256, distance=1):
        super(CoOccurrenceMatrixFast, self).__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins  # 论文使用256
        self.distance = distance

        # 按照论文描述的CNN架构
        self.cnn_classifier = nn.Sequential(
            # 输入: [B, 3, 256, 256]
            # 32个3x3卷积 + ReLU
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 32个5x5卷积 + 最大池化
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 32, 128, 128]

            # 64个3x3卷积 + ReLU
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 64个5x5卷积 + 最大池化
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, 64, 64]

            # 128个3x3卷积 + ReLU
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 128个5x5卷积 + 最大池化
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 128, 32, 32]

            # 展平
            nn.Flatten(),

            # 256全连接层
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(inplace=True),

            # 256全连接层
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            # 输出层 - 按照论文使用sigmoid，输出单个值
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def compute_co_occurrence_fast(self, images):
        """
        快速计算共现矩阵 - 使用向量化和优化策略
        Args:
            images: 输入图像 [B, 3, H, W] - 已经反标准化到[0,1]范围
        Returns:
            co_matrices: 共现矩阵 [B, 3, num_bins, num_bins]
        """
        batch_size, num_channels, H, W = images.shape

        # 初始化共现矩阵数组
        co_matrices = torch.zeros((batch_size, num_channels, self.num_bins, self.num_bins),
                                 device=images.device, dtype=torch.float32)

        # 将像素值量化为num_bins个级别
        quantized_images = (images * (self.num_bins - 1)).long()  # [B, 3, H, W]

        # 只计算水平和垂直方向，大幅减少计算量
        for b in range(batch_size):
            for c in range(num_channels):
                channel_quantized = quantized_images[b, c]  # [H, W]
                co_matrix = torch.zeros((self.num_bins, self.num_bins),
                                       device=images.device, dtype=torch.float32)

                # 水平方向 - 向量化计算
                if W > self.distance:
                    # 获取所有水平相邻像素对
                    pixels_h1 = channel_quantized[:, :-self.distance]  # [H, W-d]
                    pixels_h2 = channel_quantized[:, self.distance:]   # [H, W-d]

                    # 使用bincount进行快速统计
                    indices_h = pixels_h1.flatten() * self.num_bins + pixels_h2.flatten()
                    counts_h = torch.bincount(indices_h, minlength=self.num_bins * self.num_bins)
                    co_matrix += counts_h.reshape(self.num_bins, self.num_bins)

                    # 对称方向
                    indices_h_sym = pixels_h2.flatten() * self.num_bins + pixels_h1.flatten()
                    counts_h_sym = torch.bincount(indices_h_sym, minlength=self.num_bins * self.num_bins)
                    co_matrix += counts_h_sym.reshape(self.num_bins, self.num_bins)

                # 垂直方向 - 向量化计算
                if H > self.distance:
                    # 获取所有垂直相邻像素对
                    pixels_v1 = channel_quantized[:-self.distance, :]  # [H-d, W]
                    pixels_v2 = channel_quantized[self.distance:, :]   # [H-d, W]

                    # 使用bincount进行快速统计
                    indices_v = pixels_v1.flatten() * self.num_bins + pixels_v2.flatten()
                    counts_v = torch.bincount(indices_v, minlength=self.num_bins * self.num_bins)
                    co_matrix += counts_v.reshape(self.num_bins, self.num_bins)

                    # 对称方向
                    indices_v_sym = pixels_v2.flatten() * self.num_bins + pixels_v1.flatten()
                    counts_v_sym = torch.bincount(indices_v_sym, minlength=self.num_bins * self.num_bins)
                    co_matrix += counts_v_sym.reshape(self.num_bins, self.num_bins)

                # 使用原始计数，不进行归一化或变换
                # 论文中直接使用原始共现矩阵计数
                co_matrix = co_matrix.float()

                co_matrices[b, c] = co_matrix

        return co_matrices

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, H, W] - 已经经过ImageNet标准化
        Returns:
            output: 分类输出 [B, num_classes]
        """
        batch_size = x.shape[0]

        # 反标准化到[0,1]范围
        x_denorm = x * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device) + \
                   torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)

        # 确保在[0, 1]范围内
        x_denorm = torch.clamp(x_denorm, min=0.0, max=1.0)

        # 使用快速方法计算共现矩阵
        co_matrices = self.compute_co_occurrence_fast(x_denorm)  # [B, 3, num_bins, num_bins]

        # 通过CNN分类器
        output = self.cnn_classifier(co_matrices)

        return output


def create_co_occurrence_fast_model(num_classes=2, num_bins=256, distance=1, pretrained=True):
    """
    创建快速版本的共现矩阵模型
    Args:
        num_classes: 分类数量
        num_bins: 量化级别数
        distance: 像素距离
        pretrained: 为了接口一致性保留的参数
    Returns:
        model: 快速版本的CoOccurrenceMatrixFast模型
    """
    return CoOccurrenceMatrixFast(
        num_classes=num_classes,
        num_bins=num_bins,
        distance=distance
    )