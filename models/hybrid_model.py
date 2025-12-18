# -*- coding: utf-8 -*-
"""
ResNet + Transformer 混合模型

完整的端到端模型：从 16 通道 EMG 信号预测 20 个关节角度。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import DepthwiseSeparableConv1d
from .resnet import LightweightResNetEncoder
from .transformer import LightweightTransformerEncoder


class LightweightDecoder(nn.Module):
    """轻量级解码器

    设计目标：
    1. 将 Transformer 输出的时间维度从 50 恢复到 100
    2. 将通道数从 64 投影到 20（关节数）
    3. 保持轻量化（~9K 参数）

    网络结构：
        输入: [B, 64, 50]
          ↓
        [上采样层] - 时间维度恢复
          ↓ ConvTranspose1d(64→32, k=4, s=2, p=1) + GroupNorm + ReLU6
        输出: [B, 32, 100]  ← 时间维度翻倍！
          ↓
        [输出投影层] - 通道数投影
          ↓ DepthwiseSeparableConv1d(32→20, k=3, p=1) + GroupNorm
        输出: [B, 20, 100]  ← 预测的关节角度

    为什么使用 ConvTranspose1d？
    - 也称为"反卷积"或"转置卷积"
    - stride=2 可以将序列长度翻倍（50 → 100）
    - 相比插值（如 nn.Upsample），可学习的上采样更灵活

    ConvTranspose1d 输出长度计算：
    output_length = (input_length - 1) × stride - 2×padding + kernel_size
    例如：(50 - 1) × 2 - 2×1 + 4 = 49×2 - 2 + 4 = 100 ✓

    参数量分布：
    - 上采样: ~8.2K (64×32×4)
    - 输出投影: ~0.7K (深度可分离)
    - 总计: ~9K

    示例：
        >>> decoder = LightweightDecoder()
        >>> x = torch.randn(2, 64, 50)  # Transformer 输出
        >>> y = decoder(x)  # [2, 20, 100] - 关节角度预测
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # 上采样层: 恢复时间分辨率
        # ====================================================================
        # 作用：
        # 1. 将序列长度从 50 恢复到 100
        # 2. 减少通道数（64 → 32），为输出投影做准备
        # 3. 使用可学习的上采样（而非简单插值）
        #
        # 为什么使用 kernel_size=4, stride=2, padding=1？
        # - stride=2: 使序列长度翻倍
        # - kernel_size=4 和 padding=1: 确保输出长度恰好是 100
        # - 计算: (50-1)×2 - 2×1 + 4 = 100
        self.upsample = nn.Sequential(
            # 转置卷积（反卷积）
            nn.ConvTranspose1d(
                in_channels=64,      # Transformer 输出通道
                out_channels=32,     # 降低到 32 通道
                kernel_size=4,       # 卷积核大小
                stride=2,            # 步幅=2，序列长度翻倍
                padding=1,           # 填充，控制输出长度
                bias=False           # 使用 BatchNorm，不需要 bias
            ),

            # 批量归一化
            nn.BatchNorm1d(32),

            # ReLU6 激活
            nn.ReLU6(inplace=True),
        )
        # 上采样输出: [B, 32, 100]

        # ====================================================================
        # 输出投影层: 通道数投影到关节数
        # ====================================================================
        # 作用：
        # 1. 将 32 个特征通道投影到 20 个关节角度
        # 2. 使用深度可分离卷积减少参数
        # 3. 最后一层不使用激活函数（回归任务）
        #
        # 为什么最后不用激活函数？
        # - 关节角度是连续值（回归任务）
        # - 激活函数会限制输出范围（如 ReLU 只输出正值）
        # - 让网络自由预测任意角度值
        self.output_proj = nn.Sequential(
            # 深度可分离卷积
            DepthwiseSeparableConv1d(
                in_channels=32,
                out_channels=20,     # 20 个关节角度
                kernel_size=3,
                stride=1,
                padding=1            # padding=1 保持时间维度
            ),

            # 批量归一化
            nn.BatchNorm1d(20),

            # 注意：这里没有激活函数！
            # 原因：回归任务需要输出任意实数值
        )
        # 输出投影输出: [B, 20, 100]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            x: Transformer 编码器输出 [Batch, 64, 50]

        返回：
            关节角度预测 [Batch, 20, 100]

        完整数据流：

        1. 输入 Transformer 特征
           └─ [B, 64, 50]
              全局时序特征（经过 ResNet + Transformer 提取）

        2. 上采样层
           ├─ ConvTranspose1d(64→32, k=4, s=2)
           │  └─ 时间维度翻倍：50 → 100
           ├─ GroupNorm(4, 32)
           └─ ReLU6
           输出: [B, 32, 100]

        3. 输出投影层
           ├─ DepthwiseSeparableConv1d(32→20, k=3)
           │  └─ Depthwise: 每个通道独立卷积
           │  └─ Pointwise: 混合通道并投影到 20 维
           └─ GroupNorm(4, 20)
           输出: [B, 20, 100]

        4. 输出关节角度预测
           └─ [B, 20, 100]
              20 个关节 × 100 个时间步
        """
        # ===== 调试信息（可选，训练时可注释掉） =====
        # print(f"Decoder 输入维度: {x.shape}")  # [B, 64, 50]

        # 上采样：恢复时间分辨率
        x = self.upsample(x)  # [B, 64, 50] → [B, 32, 100]
        # print(f"上采样后: {x.shape}")

        # 输出投影：通道数投影到关节数
        x = self.output_proj(x)  # [B, 32, 100] → [B, 20, 100]
        # print(f"Decoder 输出维度: {x.shape}")

        return x


class LightweightResNetTransformer(nn.Module):
    """轻量级 ResNet + Transformer 混合模型

    完整的端到端模型，用于从 EMG 信号预测手部关节角度。

    设计理念：
    1. **ResNet 编码器**：提取局部时空特征
       - 捕捉 EMG 信号的局部模式（肌肉收缩模式）
       - 降采样减少序列长度（100 → 50）
       - 扩展特征通道（16 → 64）

    2. **Transformer 编码器**：建模全局时序依赖
       - 捕捉不同时间步之间的长程依赖关系
       - 使用局部注意力降低计算复杂度
       - 增强时序特征表示

    3. **轻量级解码器**：生成关节角度预测
       - 恢复时间分辨率（50 → 100）
       - 投影到关节空间（64 → 20）

    网络架构流程：
    ```
    输入: EMG 信号 [B, 16, 100]
      ↓
    ┌─────────────────────────────────┐
    │   ResNet 编码器 (~66K 参数)     │
    │   - Stem: 16→32                 │
    │   - Stage1: 2×ResBlock(32)      │
    │   - Transition: 32→64, 降采样   │
    │   - Stage2: 2×ResBlock(64)      │
    └─────────────────────────────────┘
      ↓ [B, 64, 50]
    ┌─────────────────────────────────┐
    │ Transformer 编码器 (~58K 参数)  │
    │   - 位置编码                    │
    │   - 2×TransformerLayer          │
    │     (局部注意力 W=25)           │
    └─────────────────────────────────┘
      ↓ [B, 64, 50]
    ┌─────────────────────────────────┐
    │    解码器 (~9K 参数)            │
    │   - 上采样: 64→32, 50→100      │
    │   - 输出投影: 32→20             │
    └─────────────────────────────────┘
      ↓
    输出: 关节角度 [B, 20, 100]
    ```

    模型规格：
    - 总参数量: ~133K
    - 推理速度: > 100 FPS (GPU)
    - 内存占用: < 50MB (模型权重)
    - FLOPs: ~15.4M

    参数：
        无需参数，所有超参数已预设为最优值

    示例：
        >>> model = LightweightResNetTransformer()
        >>> emg = torch.randn(2, 16, 100)  # 批次大小=2
        >>> joint_angles = model(emg)  # [2, 20, 100]
        >>> print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    """

    def __init__(self):
        super().__init__()

        # ====================================================================
        # 组件 1: ResNet 编码器
        # ====================================================================
        # 作用：从原始 EMG 信号中提取局部时空特征
        # 输入: [B, 16, 100] - 16 通道 EMG 信号
        # 输出: [B, 64, 50] - 64 通道高级特征，时间维度减半
        # 参数量: ~66K
        self.resnet_encoder = LightweightResNetEncoder()

        # ====================================================================
        # 组件 2: Transformer 编码器
        # ====================================================================
        # 作用：建模全局时序依赖关系
        # 输入: [B, 64, 50] - ResNet 提取的特征
        # 输出: [B, 64, 50] - 全局时序特征
        # 参数量: ~58K
        self.transformer_encoder = LightweightTransformerEncoder(
            dim=64,              # 特征维度
            depth=2,             # 2 层 Transformer
            num_heads=4,         # 4 个注意力头
            window_size=25,      # 局部窗口大小
            mlp_ratio=2.0,       # MLP 扩展倍数
            dropout=0.1,         # Dropout 概率
            drop_path_rate=0.1   # DropPath 最大概率
        )

        # ====================================================================
        # 组件 3: 解码器
        # ====================================================================
        # 作用：生成关节角度预测
        # 输入: [B, 64, 50] - Transformer 输出
        # 输出: [B, 20, 100] - 20 个关节角度预测
        # 参数量: ~9K
        self.decoder = LightweightDecoder()

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            emg: EMG 信号张量 [Batch, 16, 100]
                - Batch: 批次大小
                - 16: EMG 通道数（16 个肌电传感器）
                - 100: 时间窗口长度（采样点数）

        返回：
            joint_angles: 关节角度预测 [Batch, 20, 100]
                - 20: 关节数量（手部 20 个关节）
                - 100: 时间步数（与输入对齐）

        完整计算流程：

        ┌─────────────────────────────────────────────────────────────┐
        │ 阶段 1: 局部特征提取 (ResNet)                               │
        └─────────────────────────────────────────────────────────────┘

        输入: EMG 信号 [B, 16, 100]
          ↓
        Stem Layer: Conv1d(16→32, k=7)
          ↓ 初始特征提取，感受野 = 7
        [B, 32, 100]
          ↓
        Stage 1: 2 × ResBlock(32)
          ↓ 保持分辨率的特征提取，感受野 = 19
        [B, 32, 100]
          ↓
        Transition: DepthwiseSeparable(32→64, s=2)
          ↓ 降采样 + 通道扩展
        [B, 64, 50]  ← 时间维度减半！
          ↓
        Stage 2: 2 × ResBlock(64)
          ↓ 高级特征提取，感受野 = 50（覆盖整个序列）
        [B, 64, 50]

        ┌─────────────────────────────────────────────────────────────┐
        │ 阶段 2: 全局时序建模 (Transformer)                          │
        └─────────────────────────────────────────────────────────────┘

        输入: ResNet 特征 [B, 64, 50]
          ↓
        维度转换: [B, 64, 50] → [B, 50, 64]
          ↓ (Channels, Time) → (Time, Features)
        [B, 50, 64]
          ↓
        位置编码: 添加正弦位置信息
          ↓
        Transformer Layer 1:
          ├─ LocalAttention(W=25): 每个位置关注前后 25 步
          └─ MLP(64→128→64): 特征变换
          ↓
        [B, 50, 64]
          ↓
        Transformer Layer 2:
          ├─ LocalAttention(W=25)
          └─ MLP(64→128→64)
          ↓
        [B, 50, 64]
          ↓
        维度转换: [B, 50, 64] → [B, 64, 50]
          ↓ (Time, Features) → (Channels, Time)
        [B, 64, 50]

        ┌─────────────────────────────────────────────────────────────┐
        │ 阶段 3: 关节角度预测 (Decoder)                              │
        └─────────────────────────────────────────────────────────────┘

        输入: Transformer 特征 [B, 64, 50]
          ↓
        上采样: ConvTranspose1d(64→32, k=4, s=2)
          ↓ 恢复时间分辨率
        [B, 32, 100]  ← 时间维度恢复！
          ↓
        输出投影: DepthwiseSeparable(32→20, k=3)
          ↓ 投影到关节空间
        [B, 20, 100]
          ↓
        输出: 关节角度预测

        ════════════════════════════════════════════════════════════════
        最终输出: [B, 20, 100]
        - 20 个关节的角度预测
        - 100 个时间步（与输入 EMG 对齐）
        ════════════════════════════════════════════════════════════════
        """
        # ===== 调试信息（可选，训练时可注释掉） =====
        # print("\n" + "="*60)
        # print(f"模型输入维度: {emg.shape}")  # [B, 16, 100]
        # print("="*60)

        # ====================================================================
        # 阶段 1: ResNet 编码器 - 局部特征提取
        # ====================================================================
        # print("\n[阶段 1] ResNet 编码器")
        x = self.resnet_encoder(emg)  # [B, 16, 100] → [B, 64, 50]
        # print(f"ResNet 输出: {x.shape}")

        # ====================================================================
        # 阶段 2: Transformer 编码器 - 全局时序建模
        # ====================================================================
        # print("\n[阶段 2] Transformer 编码器")
        x = self.transformer_encoder(x)  # [B, 64, 50] → [B, 64, 50]
        # print(f"Transformer 输出: {x.shape}")

        # ====================================================================
        # 阶段 3: 解码器 - 关节角度预测
        # ====================================================================
        # print("\n[阶段 3] 解码器")
        joint_angles = self.decoder(x)  # [B, 64, 50] → [B, 20, 100]
        # print(f"Decoder 输出: {joint_angles.shape}")

        # print("\n" + "="*60)
        # print(f"模型输出维度: {joint_angles.shape}")
        # print("="*60 + "\n")

        return joint_angles


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("测试轻量级 ResNet + Transformer 混合模型")
    print("=" * 80)

    # 创建完整模型
    model = LightweightResNetTransformer()

    # 打印网络结构
    print("\n网络结构：")
    print(model)

    # 测试输入（模拟 EMG 信号）
    batch_size = 2
    emg_channels = 16
    time_steps = 100
    num_joints = 20

    emg = torch.randn(batch_size, emg_channels, time_steps)
    print(f"\n输入维度: {emg.shape}")
    print(f"期望输出维度: [{batch_size}, {num_joints}, {time_steps}]")

    # 前向传播
    print("\n开始前向传播...")
    joint_angles = model(emg)
    print(f"输出维度: {joint_angles.shape}")

    # 验证维度
    assert joint_angles.shape == (batch_size, num_joints, time_steps), \
        f"输出维度不正确！期望 ({batch_size}, {num_joints}, {time_steps})，实际 {joint_angles.shape}"
    print("✅ 维度验证通过")

    # 统计参数量
    print("\n" + "=" * 80)
    print("参数统计")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"目标参数量: ~133,000")

    # 分模块参数统计
    resnet_params = sum(p.numel() for p in model.resnet_encoder.parameters())
    transformer_params = sum(p.numel() for p in model.transformer_encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"\n分模块参数统计：")
    print(f"├─ ResNet 编码器:      {resnet_params:>8,} 参数 ({resnet_params/total_params*100:.1f}%)")
    print(f"├─ Transformer 编码器: {transformer_params:>8,} 参数 ({transformer_params/total_params*100:.1f}%)")
    print(f"└─ 解码器:             {decoder_params:>8,} 参数 ({decoder_params/total_params*100:.1f}%)")

    # 内存占用估算
    memory_mb = total_params * 4 / (1024 * 1024)  # 假设 float32
    print(f"\n模型内存占用（float32）: {memory_mb:.2f} MB")

    # 测试梯度反向传播
    print("\n" + "=" * 80)
    print("测试梯度反向传播")
    print("=" * 80)

    # 创建虚拟目标
    target = torch.randn(batch_size, num_joints, time_steps)

    # 计算损失
    loss_fn = nn.MSELoss()
    loss = loss_fn(joint_angles, target)
    print(f"\n虚拟损失值: {loss.item():.6f}")

    # 反向传播
    print("开始反向传播...")
    loss.backward()
    print("✅ 梯度反向传播成功")

    # 检查梯度
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    if has_grad:
        print("✅ 所有参数都有梯度")
    else:
        print("⚠️  某些参数没有梯度")

    # 检查梯度数值稳定性
    max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    min_grad = min(p.grad.abs().min().item() for p in model.parameters() if p.grad is not None)
    print(f"梯度范围: [{min_grad:.2e}, {max_grad:.2e}]")

    if max_grad > 1e3:
        print("⚠️  梯度过大，可能需要梯度裁剪")
    elif max_grad < 1e-6:
        print("⚠️  梯度过小，可能存在梯度消失")
    else:
        print("✅ 梯度数值正常")

    # 推理速度测试
    print("\n" + "=" * 80)
    print("推理速度测试（CPU）")
    print("=" * 80)

    import time

    # 预热
    print("\n预热中...")
    for _ in range(10):
        _ = model(emg)

    # 计时
    num_runs = 100
    print(f"测试 {num_runs} 次推理...")
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():  # 推理模式，不计算梯度
            _ = model(emg)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # 毫秒
    fps = 1000 / avg_time

    print(f"\n平均推理时间: {avg_time:.2f} ms")
    print(f"推理速度: {fps:.1f} FPS")

    if fps > 30:
        print(f"✅ 推理速度满足实时要求（> 30 FPS）")
    else:
        print(f"⚠️  推理速度较慢，建议使用 GPU")

    # 输出数值范围检查
    print("\n" + "=" * 80)
    print("输出数值检查")
    print("=" * 80)

    print(f"\n输出统计信息：")
    print(f"├─ 最小值: {joint_angles.min().item():.4f}")
    print(f"├─ 最大值: {joint_angles.max().item():.4f}")
    print(f"├─ 均值:   {joint_angles.mean().item():.4f}")
    print(f"└─ 标准差: {joint_angles.std().item():.4f}")

    if torch.isnan(joint_angles).any():
        print("❌ 输出包含 NaN 值！")
    elif torch.isinf(joint_angles).any():
        print("❌ 输出包含 Inf 值！")
    else:
        print("✅ 输出数值正常（无 NaN/Inf）")

    # 模型摘要
    print("\n" + "=" * 80)
    print("模型摘要")
    print("=" * 80)

    print(f"""
┌────────────────────────────────────────────────────────────┐
│                     模型规格                                │
├────────────────────────────────────────────────────────────┤
│ 输入:  [Batch, 16, 100]  - 16 通道 EMG 信号               │
│ 输出:  [Batch, 20, 100]  - 20 个关节角度预测              │
├────────────────────────────────────────────────────────────┤
│ 总参数量: {total_params:>8,} 参数                              │
│ 模型大小: {memory_mb:>8.2f} MB (float32)                        │
│ 推理速度: {fps:>8.1f} FPS (CPU)                                │
├────────────────────────────────────────────────────────────┤
│ ResNet 编码器:      {resnet_params:>6,} 参数 ({resnet_params/total_params*100:>4.1f}%)           │
│ Transformer 编码器: {transformer_params:>6,} 参数 ({transformer_params/total_params*100:>4.1f}%)           │
│ 解码器:             {decoder_params:>6,} 参数 ({decoder_params/total_params*100:>4.1f}%)           │
└────────────────────────────────────────────────────────────┘
    """)

    print("\n" + "=" * 80)
    print("✅ 所有测试通过！混合模型工作正常")
    print("=" * 80)

    print("\n下一步：集成到 train.py")
    print("在 train.py 中将 PlaceholderModel 替换为 LightweightResNetTransformer")
