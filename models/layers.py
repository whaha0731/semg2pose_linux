# -*- coding: utf-8 -*-
"""
基础层组件

包含轻量化神经网络的核心组件：
1. 深度可分离卷积 (Depthwise Separable Convolution)
2. 轻量级 ResNet 块 (Lightweight ResNet Block)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积 (Depthwise Separable Convolution)

    核心思想：
    将标准卷积分解为两步：
    1. Depthwise 卷积：每个输入通道独立进行卷积（groups=in_channels）
    2. Pointwise 卷积：使用 1x1 卷积混合通道信息

    参数节省：
    - 标准卷积: in_ch × out_ch × kernel_size
    - 深度可分离: in_ch × kernel_size + in_ch × out_ch
    - 对于 Conv1d(32, 64, k=3):
      * 标准: 32 × 64 × 3 = 6,144
      * 深度可分离: 32 × 3 + 32 × 64 = 2,144
      * 节省: 65% 参数！

    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步幅（默认 1）
        padding: 填充（默认 0）

    示例：
        >>> conv = DepthwiseSeparableConv1d(16, 32, kernel_size=3, padding=1)
        >>> x = torch.randn(2, 16, 100)  # [Batch, Channels, Time]
        >>> y = conv(x)  # [2, 32, 100]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()

        # ====================================================================
        # 步骤 1: Depthwise 卷积 - 每个通道独立卷积
        # ====================================================================
        # groups=in_channels 表示：
        # - 将输入分成 in_channels 组
        # - 每组只有 1 个通道
        # - 每组独立进行卷积，互不干扰
        #
        # 例如：16 个输入通道 → 16 个独立的 1D 卷积核
        # 每个卷积核只处理对应的那 1 个通道
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,  # 输出通道数 = 输入通道数
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 关键参数！每个通道独立
            bias=False  # 使用 BatchNorm，不需要 bias
        )

        # ====================================================================
        # 步骤 2: Pointwise 卷积 - 1x1 卷积混合通道信息
        # ====================================================================
        # kernel_size=1 表示：
        # - 不考虑时间维度的邻近关系
        # - 只在通道维度上进行线性组合
        # - 将 in_channels 个特征混合为 out_channels 个特征
        #
        # 例如：将 16 个独立特征混合成 32 个新特征
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 1x1 卷积
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            x: 输入张量 [Batch, in_channels, Time]

        返回：
            输出张量 [Batch, out_channels, Time]

        计算流程：
            输入 [B, C_in, T]
              ↓ Depthwise（每个通道独立卷积）
            中间 [B, C_in, T']  (T' 取决于 stride)
              ↓ Pointwise（1x1 卷积混合通道）
            输出 [B, C_out, T']
        """
        # 步骤 1: Depthwise 卷积
        # 每个通道独立提取时间特征
        x = self.depthwise(x)  # [B, C_in, T] → [B, C_in, T']

        # 步骤 2: Pointwise 卷积
        # 混合通道信息，改变通道数
        x = self.pointwise(x)  # [B, C_in, T'] → [B, C_out, T']

        return x


class LightweightResBlock1D(nn.Module):
    """轻量级 1D ResNet 块 - 倒残差结构 (Inverted Residual Block)

    设计理念：
    受 MobileNetV2 启发，使用"先扩展后压缩"的倒残差结构：
    1. 1x1 卷积扩展通道（C → C×expansion_factor）
    2. 3x3 Depthwise 卷积提取特征
    3. 1x1 卷积压缩通道（C×expansion_factor → C）
    4. 残差连接（Residual Connection）

    为什么叫"倒"残差？
    - 标准 ResNet: 压缩 → 卷积 → 扩展（瓶颈结构）
    - 倒残差: 扩展 → 卷积 → 压缩（中间维度高）

    优势：
    - 在低维空间进行残差连接，节省内存
    - 在高维空间提取特征，保持表达能力
    - 使用深度可分离卷积，大幅减少参数

    结构示意：
        输入 [B, C, T]
            │
            ├──────────────────┐ (Residual 捷径)
            │                  │
            ▼                  │
          扩展 (C → C×2)       │
            ↓                  │
        Depthwise 卷积         │
            ↓                  │
          压缩 (C×2 → C)       │
            ↓                  │
          Dropout              │
            │                  │
            └────( + )─────────┘
                  ↓
            输出 [B, C, T]

    参数：
        channels: 输入输出通道数（保持不变）
        expansion_factor: 扩展因子（默认 2，即中间层通道数翻倍）
        kernel_size: Depthwise 卷积核大小（默认 3）

    示例：
        >>> block = LightweightResBlock1D(channels=32, expansion_factor=2)
        >>> x = torch.randn(2, 32, 100)
        >>> y = block(x)  # [2, 32, 100] - 维度不变
    """

    def __init__(
        self,
        channels: int,
        expansion_factor: int = 2,
        kernel_size: int = 3
    ):
        super().__init__()

        # 计算中间层的扩展通道数
        expanded_channels = channels * expansion_factor

        # ====================================================================
        # 阶段 1: 扩展层 (Expansion)
        # ====================================================================
        # 作用：增加通道数，在高维空间提取更丰富的特征
        # 使用 1x1 卷积快速改变通道数，不改变时间维度
        self.conv1 = nn.Sequential(
            # 1x1 标准卷积：C → C×expansion_factor
            nn.Conv1d(channels, expanded_channels, kernel_size=1, bias=False),

            # BatchNorm1d: 批量归一化
            # 对每个通道独立进行归一化
            nn.BatchNorm1d(expanded_channels),

            # ReLU6: 输出限制在 [0, 6] 范围内
            # 优势：更适合低精度量化（INT8），在移动端部署时有用
            nn.ReLU6(inplace=True),
        )

        # ====================================================================
        # 阶段 2: Depthwise 卷积 (Depthwise Convolution)
        # ====================================================================
        # 作用：在每个通道上独立提取时间特征
        # 使用 groups=expanded_channels，每个通道独立卷积
        self.dwconv = nn.Sequential(
            # 3x3 Depthwise 卷积
            # padding=kernel_size//2 确保输出时间维度不变
            nn.Conv1d(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # padding=1（当 k=3 时）
                groups=expanded_channels,  # 关键！每个通道独立
                bias=False
            ),

            nn.BatchNorm1d(expanded_channels),
            nn.ReLU6(inplace=True),
        )

        # ====================================================================
        # 阶段 3: 投影层 (Projection)
        # ====================================================================
        # 作用：压缩回原始通道数，准备进行残差连接
        # 注意：这里没有激活函数！（线性投影）
        self.conv2 = nn.Sequential(
            # 1x1 标准卷积：C×expansion_factor → C
            nn.Conv1d(expanded_channels, channels, kernel_size=1, bias=False),

            # 最后只做归一化，不使用激活函数
            # 原因：残差连接前保持线性，让梯度更好地流动
            nn.BatchNorm1d(channels),
        )

        # ====================================================================
        # 正则化: Dropout
        # ====================================================================
        # 作用：训练时随机丢弃部分神经元，防止过拟合
        # drop_prob=0.1: 10% 的神经元被置零
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            x: 输入张量 [Batch, channels, Time]

        返回：
            输出张量 [Batch, channels, Time]（维度不变）

        详细计算流程：
            输入 [B, C, T]
              ↓
            保存输入作为残差 (identity = x)
              ↓
            扩展: [B, C, T] → [B, C×2, T]
              ↓ (1x1 Conv + GroupNorm + ReLU6)
            Depthwise: [B, C×2, T] → [B, C×2, T]
              ↓ (3x3 DWConv + GroupNorm + ReLU6)
            压缩: [B, C×2, T] → [B, C, T]
              ↓ (1x1 Conv + GroupNorm)
            Dropout: [B, C, T] → [B, C, T]
              ↓
            残差相加: out + identity
              ↓
            输出 [B, C, T]
        """
        # 保存输入用于残差连接
        identity = x  # [B, C, T]

        # 倒残差路径
        out = self.conv1(x)    # 扩展: [B, C, T] → [B, C×2, T]
        out = self.dwconv(out)  # Depthwise: [B, C×2, T] → [B, C×2, T]
        out = self.conv2(out)   # 压缩: [B, C×2, T] → [B, C, T]
        out = self.dropout(out) # Dropout: [B, C, T] → [B, C, T]

        # 残差连接：输出 = 转换后的特征 + 原始输入
        # 这是 ResNet 的核心：允许梯度直接流向更早的层
        out = out + identity  # [B, C, T]

        return out


# ============================================================================
# 测试代码（运行此文件时自动执行）
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("测试深度可分离卷积 (DepthwiseSeparableConv1d)")
    print("=" * 80)

    # 创建深度可分离卷积层
    dsconv = DepthwiseSeparableConv1d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1
    )

    # 测试输入
    x = torch.randn(2, 16, 100)  # [Batch=2, Channels=16, Time=100]
    print(f"输入维度: {x.shape}")

    # 前向传播
    y = dsconv(x)
    print(f"输出维度: {y.shape}")
    print(f"期望维度: [2, 32, 100]")

    # 参数量统计
    params = sum(p.numel() for p in dsconv.parameters())
    print(f"参数量: {params:,}")

    # 对比标准卷积的参数量
    standard_params = 16 * 32 * 3
    print(f"标准卷积参数量: {standard_params:,}")
    print(f"节省参数: {(standard_params - params) / standard_params * 100:.1f}%")

    print("\n" + "=" * 80)
    print("测试轻量级 ResNet 块 (LightweightResBlock1D)")
    print("=" * 80)

    # 创建 ResNet 块
    resblock = LightweightResBlock1D(channels=32, expansion_factor=2, kernel_size=3)

    # 测试输入
    x = torch.randn(2, 32, 100)
    print(f"输入维度: {x.shape}")

    # 前向传播
    y = resblock(x)
    print(f"输出维度: {y.shape}")
    print(f"期望维度: [2, 32, 100] (维度不变)")

    # 参数量统计
    params = sum(p.numel() for p in resblock.parameters())
    print(f"参数量: {params:,}")

    # 验证残差连接
    print("\n验证残差连接：")
    print(f"输入范数: {x.norm().item():.4f}")
    print(f"输出范数: {y.norm().item():.4f}")
    print("残差连接使得输出保留了输入的信息")

    print("\n✅ 所有测试通过！")
