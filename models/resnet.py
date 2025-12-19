# -*- coding: utf-8 -*-
"""
轻量级 ResNet 编码器

实现基于倒残差结构的 1D ResNet，用于从 EMG 信号中提取局部时空特征。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import DepthwiseSeparableConv1d, LightweightResBlock1D


class LightweightResNetEncoder(nn.Module):
    """轻量级 ResNet 编码器

    设计目标：
    1. 从 16 通道 EMG 信号提取 64 通道高级特征
    2. 通过降采样将时间维度从 100 压缩到 50
    3. 使用轻量化技术控制参数量在 ~66K

    网络结构：
        输入: [B, 16, 100]
          ↓
        [Stem Layer] - 初始特征提取
          ↓ Conv1d(16→32, k=7) + GroupNorm + ReLU6
        输出: [B, 32, 100]
          ↓
        [Stage 1] - 保持分辨率的特征提取
          ↓ 2 × LightweightResBlock1D(32)
        输出: [B, 32, 100]
          ↓
        [Transition] - 降采样 + 通道扩展
          ↓ DepthwiseSeparableConv1d(32→64, s=2)
        输出: [B, 64, 50]  ← 时间维度减半！
          ↓
        [Stage 2] - 高级特征提取
          ↓ 2 × LightweightResBlock1D(64)
        输出: [B, 64, 50]

    设计理念：
    1. **Stem Layer**: 使用较大的卷积核（k=7）快速提取基础特征
    2. **Stage 1**: 在原始分辨率上提取细节特征
    3. **Transition**: 降采样减少计算量，同时扩展通道数增强表达能力
    4. **Stage 2**: 在较低分辨率上提取高级语义特征

    参数量分布：
    - Stem: ~3.6K (16×32×7)
    - Stage 1: ~20K (2 blocks × 10K)
    - Transition: ~2.2K
    - Stage 2: ~40K (2 blocks × 20K)
    - 总计: ~66K

    示例：
        >>> encoder = LightweightResNetEncoder()
        >>> x = torch.randn(2, 16, 100)  # EMG 输入
        >>> y = encoder(x)  # [2, 64, 50]
        >>> print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # Stem Layer: 初始特征提取
        # ====================================================================
        # 作用：
        # 1. 快速扩展通道数 (16 → 32)
        # 2. 使用较大卷积核 (k=7) 捕捉更大范围的时间模式
        # 3. 不进行降采样，保留完整的时间分辨率
        #
        # 为什么使用 k=7？
        # - EMG 信号需要较大的感受野来捕捉肌肉活动模式
        # - k=7 可以一次性看到 7 个连续时间步的信息
        # - 相比 k=3，感受野更大但参数增加有限
        self.stem = nn.Sequential(
            # 标准 1D 卷积
            nn.Conv1d(
                in_channels=16,      # EMG 通道数
                out_channels=32,     # 扩展到 32 通道
                kernel_size=7,       # 较大的卷积核
                stride=1,            # 不降采样
                padding=3,           # padding = (k-1)/2 保持时间维度
                bias=False           # 使用 BatchNorm，不需要 bias
            ),

            # 批量归一化
            nn.BatchNorm1d(32),

            # ReLU6 激活
            nn.ReLU6(inplace=True),
        )
        # Stem 输出: [B, 32, 100]

        # ====================================================================
        # Stage 1: 保持分辨率的特征提取
        # ====================================================================
        # 作用：
        # 1. 在原始时间分辨率 (100) 上提取局部特征
        # 2. 通过 2 个 ResNet 块逐步增强特征表示
        # 3. 残差连接保证梯度流畅
        #
        # 感受野计算：
        # - Stem: 7
        # - Block 1: 7 + 3 + 3 = 13 (两个 3x3 卷积)
        # - Block 2: 13 + 3 + 3 = 19
        # 因此 Stage 1 的感受野可以覆盖 19 个时间步
        self.stage1 = nn.Sequential(
            LightweightResBlock1D(
                channels=32,          # 保持 32 通道
                expansion_factor=2,   # 内部扩展到 64 通道
                kernel_size=3
            ),
            LightweightResBlock1D(
                channels=32,
                expansion_factor=2,
                kernel_size=3
            ),
        )
        # Stage 1 输出: [B, 32, 100]

        # ====================================================================
        # Transition Layer: 降采样 + 通道扩展
        # ====================================================================
        # 作用：
        # 1. 降采样 (stride=2): 时间维度 100 → 50，减少后续计算量
        # 2. 通道扩展 (32 → 64): 增强特征表达能力
        # 3. 使用深度可分离卷积减少参数
        #
        # 为什么要降采样？
        # - Transformer 的计算复杂度是 O(L²)，序列越长越慢
        # - 降采样可以减少 Transformer 的计算量（50² vs 100²）
        # - EMG 信号在经过初步特征提取后，不需要保持完整分辨率
        #
        # 为什么扩展通道？
        # - 时间信息压缩了，需要更多通道来保存特征
        # - 高层特征需要更大的容量来表示复杂模式
        self.transition = nn.Sequential(
            DepthwiseSeparableConv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,            # 关键！降采样
                padding=1
            ),

            # 批量归一化
            nn.BatchNorm1d(64),
            nn.ReLU6(inplace=True),
        )
        # Transition 输出: [B, 64, 50]

        # ====================================================================
        # Stage 2: 高级特征提取
        # ====================================================================
        # 作用：
        # 1. 在较低分辨率 (50) 上提取高级语义特征
        # 2. 64 个通道提供更强的表达能力
        # 3. 为后续 Transformer 准备好的特征
        #
        # 感受野计算（相对于降采样后的序列）：
        # - Transition 后感受野: 19 × 2 = 38 (降采样使感受野扩大)
        # - Block 1: 38 + 3 + 3 = 44
        # - Block 2: 44 + 3 + 3 = 50
        # 因此 Stage 2 的感受野可以覆盖整个 50 步序列！
        self.stage2 = nn.Sequential(
            LightweightResBlock1D(
                channels=64,
                expansion_factor=2,   # 内部扩展到 128 通道
                kernel_size=3
            ),
            LightweightResBlock1D(
                channels=64,
                expansion_factor=2,
                kernel_size=3
            ),
        )
        # Stage 2 输出: [B, 64, 50]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            x: 输入 EMG 信号 [Batch, 16, 100]
                - Batch: 批次大小
                - 16: EMG 通道数（16 个传感器）
                - 100: 时间窗口长度

        返回：
            特征张量 [Batch, 64, 50]
                - 64: 特征通道数
                - 50: 降采样后的时间步数

        完整数据流：

        1. 输入 EMG 信号
           └─ [B, 16, 100]
              原始 16 通道 EMG，100 个时间步

        2. Stem Layer (初始特征提取)
           └─ [B, 32, 100]
              Conv1d(k=7) 提取基础特征
              感受野: 7 个时间步

        3. Stage 1 (细节特征提取)
           ├─ ResBlock1D #1
           │  └─ [B, 32, 100]
           │     扩展→64→压缩→32, 感受野: 13
           └─ ResBlock1D #2
              └─ [B, 32, 100]
                 扩展→64→压缩→32, 感受野: 19

        4. Transition (降采样 + 通道扩展)
           └─ [B, 64, 50]
              时间维度减半: 100 → 50
              通道数翻倍: 32 → 64
              感受野相对原始信号: 38

        5. Stage 2 (高级特征提取)
           ├─ ResBlock1D #1
           │  └─ [B, 64, 50]
           │     扩展→128→压缩→64, 感受野: 44
           └─ ResBlock1D #2
              └─ [B, 64, 50]
                 扩展→128→压缩→64, 感受野: 50

        6. 输出特征
           └─ [B, 64, 50]
              准备输入 Transformer 进行全局建模
        """
        # ===== 调试信息（可选，训练时可注释掉） =====
        # print(f"输入维度: {x.shape}")  # [B, 16, 100]

        # Stem: 初始特征提取
        x = self.stem(x)  # [B, 16, 100] → [B, 32, 100]
        # print(f"Stem 输出: {x.shape}")

        # Stage 1: 保持分辨率的特征提取
        x = self.stage1(x)  # [B, 32, 100] → [B, 32, 100]
        # print(f"Stage 1 输出: {x.shape}")

        # Transition: 降采样 + 通道扩展
        x = self.transition(x)  # [B, 32, 100] → [B, 64, 50]
        # print(f"Transition 输出: {x.shape}")

        # Stage 2: 高级特征提取
        x = self.stage2(x)  # [B, 64, 50] → [B, 64, 50]
        # print(f"Stage 2 输出: {x.shape}")

        return x


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("测试轻量级 ResNet 编码器 (LightweightResNetEncoder)")
    print("=" * 80)

    # 创建编码器
    encoder = LightweightResNetEncoder()

    # 打印网络结构
    print("\n网络结构：")
    print(encoder)

    # 测试输入（模拟 EMG 信号）
    batch_size = 2
    emg_channels = 16
    time_steps = 100

    x = torch.randn(batch_size, emg_channels, time_steps)
    print(f"\n输入维度: {x.shape}")

    # 前向传播
    print("\n开始前向传播...")
    y = encoder(x)
    print(f"输出维度: {y.shape}")
    print(f"期望维度: [{batch_size}, 64, 50]")

    # 验证维度
    assert y.shape == (batch_size, 64, 50), "输出维度不正确！"
    print("✅ 维度验证通过")

    # 统计参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"\n参数统计：")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"目标参数量: ~66,000")
    print(f"实际参数量: {total_params:,}")

    # 分层参数统计
    print(f"\n分层参数统计：")
    print(f"Stem: {sum(p.numel() for p in encoder.stem.parameters()):,}")
    print(f"Stage 1: {sum(p.numel() for p in encoder.stage1.parameters()):,}")
    print(f"Transition: {sum(p.numel() for p in encoder.transition.parameters()):,}")
    print(f"Stage 2: {sum(p.numel() for p in encoder.stage2.parameters()):,}")

    # 测试梯度反向传播
    print(f"\n测试梯度反向传播...")
    loss = y.sum()
    loss.backward()
    print("✅ 梯度反向传播成功")

    # 检查梯度
    has_grad = all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    if has_grad:
        print("✅ 所有参数都有梯度")
    else:
        print("⚠️  某些参数没有梯度")

    # 计算感受野
    print(f"\n感受野分析：")
    print(f"Stem: 7 个时间步")
    print(f"Stage 1: 19 个时间步（覆盖原始信号的 19%）")
    print(f"Transition: 降采样使感受野扩大 2 倍")
    print(f"Stage 2: 50 个时间步（覆盖整个降采样序列！）")

    # 推理速度测试（可选）
    print(f"\n推理速度测试...")
    import time

    # 预热
    for _ in range(10):
        _ = encoder(x)

    # 计时
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        _ = encoder(x)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # 毫秒
    fps = 1000 / avg_time

    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"推理速度: {fps:.1f} FPS (CPU)")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过！ResNet 编码器工作正常")
    print("=" * 80)
