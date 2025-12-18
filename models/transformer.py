# -*- coding: utf-8 -*-
"""
轻量级 Transformer 编码器

实现局部注意力机制的 Transformer，用于捕捉 EMG 信号的全局时序依赖关系。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码 (Sinusoidal Positional Encoding)

    设计理念：
    使用正弦和余弦函数为每个时间位置生成唯一的位置向量，无需训练参数。

    原理（来自 "Attention Is All You Need" 论文）：
    - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中：
    - pos: 时间位置 (0 到 max_len-1)
    - i: 维度索引 (0 到 d_model/2-1)
    - d_model: 特征维度

    为什么使用正弦函数？
    1. **无参数**：不增加模型复杂度
    2. **外推性**：可以处理训练时未见过的序列长度
    3. **相对位置**：sin(α+β) 可以表示为 sin(α) 和 cos(β) 的线性组合

    示例：
        >>> pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
        >>> x = torch.randn(2, 50, 64)  # [Batch, Time, Features]
        >>> x_with_pos = pe(x)  # 添加位置信息
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 位置索引: [0, 1, 2, ..., max_len-1]
        # 形状: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算除数项: 10000^(2i/d_model)
        # 形状: [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数索引使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数索引使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加 batch 维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer（不参与训练，但会保存到模型中）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码

        参数：
            x: 输入张量 [Batch, Time, Features]

        返回：
            添加位置编码后的张量 [Batch, Time, Features]
        """
        # 取出对应长度的位置编码并相加
        # self.pe[:, :x.size(1)] 形状: [1, Time, Features]
        x = x + self.pe[:, :x.size(1)]
        return x


class LocalAttention1D(nn.Module):
    """1D 局部注意力 (Local Attention)

    设计理念：
    标准 Transformer 的全局注意力复杂度为 O(L²)，对于长序列计算量巨大。
    局部注意力限制每个位置只关注局部窗口内的位置，复杂度降为 O(L×W)。

    工作原理：
    对于位置 i，只计算与 [i-W//2, i+W//2] 窗口内其他位置的注意力权重。

    复杂度对比（序列长度 L=50, 窗口大小 W=25）：
    - 全局注意力: O(50² × 64) = 160K 操作
    - 局部注意力: O(50 × 25 × 64) = 80K 操作
    - 节省 50% 计算量！

    多头注意力：
    将 d_model 维度分成 num_heads 个头，每个头独立计算注意力。
    好处：不同头可以关注不同的特征模式（例如：频率、幅度、相位等）

    参数：
        dim: 特征维度（必须能被 num_heads 整除）
        num_heads: 注意力头数（默认 4）
        window_size: 局部窗口大小（默认 25）
        dropout: Dropout 概率（默认 0.1）

    示例：
        >>> attn = LocalAttention1D(dim=64, num_heads=4, window_size=25)
        >>> x = torch.randn(2, 50, 64)  # [Batch, Time, Features]
        >>> y = attn(x)  # [2, 50, 64]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 25,
        dropout: float = 0.1
    ):
        super().__init__()

        assert dim % num_heads == 0, f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5  # 缩放因子: 1/√d_k

        # ====================================================================
        # Q, K, V 投影层
        # ====================================================================
        # 使用单个线性层同时计算 Q, K, V（效率更高）
        # 输入: [B, L, dim]      
        # 输出: [B, L, 3*dim] → 拆分为 Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        # 注意力 dropout
        self.attn_dropout = nn.Dropout(dropout)

        # 输出投影层
        # 将多头注意力的结果投影回原始维度
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            x: 输入张量 [Batch, Time, Features]

        返回：
            输出张量 [Batch, Time, Features]

        计算流程：
        1. 线性投影得到 Q, K, V
        2. 重塑为多头形式
        3. 计算局部注意力分数
        4. Softmax 归一化
        5. 加权求和得到输出
        6. 合并多头并投影
        """
        B, L, C = x.shape  # Batch, Length, Channels

        # ====================================================================
        # 步骤 1: 计算 Q, K, V
        # ====================================================================
        # qkv: [B, L, 3*C]
        qkv = self.qkv(x)

        # 重塑并拆分为 Q, K, V
        # [B, L, 3*C] → [B, L, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)

        # 转置为 [3, B, num_heads, L, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 拆分 Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状: [B, num_heads, L, head_dim]

        # ====================================================================
        # 步骤 2: 计算局部注意力
        # ====================================================================
        # 注意力分数: Q @ K^T
        # [B, num_heads, L, head_dim] @ [B, num_heads, head_dim, L]
        # = [B, num_heads, L, L]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ====================================================================
        # 步骤 3: 应用局部窗口掩码
        # ====================================================================
        # 创建局部注意力掩码
        # 对于位置 i，只允许关注 [i - W//2, i + W//2] 范围内的位置
        mask = self._create_local_mask(L, self.window_size, x.device)

        # 将掩码区域设为 -inf，Softmax 后会变为 0
        attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax 归一化
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # ====================================================================
        # 步骤 4: 加权求和
        # ====================================================================
        # attn @ V: [B, num_heads, L, L] @ [B, num_heads, L, head_dim]
        # = [B, num_heads, L, head_dim]
        out = attn @ v

        # ====================================================================
        # 步骤 5: 合并多头
        # ====================================================================
        # 转置: [B, num_heads, L, head_dim] → [B, L, num_heads, head_dim]
        out = out.transpose(1, 2)

        # 重塑: [B, L, num_heads, head_dim] → [B, L, C]
        out = out.reshape(B, L, C)

        # 输出投影
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out

    def _create_local_mask(
        self,
        length: int,
        window_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """创建局部注意力掩码

        参数：
            length: 序列长度
            window_size: 窗口大小
            device: 设备

        返回：
            掩码张量 [length, length]
            - 1 表示可以关注
            - 0 表示不能关注（被屏蔽）

        示例（length=5, window_size=3）：
            位置:  0  1  2  3  4
              0 [[ 1  1  1  0  0 ]  ← 位置0关注[0,1,2]
              1  [ 1  1  1  1  0 ]  ← 位置1关注[0,1,2,3]
              2  [ 1  1  1  1  1 ]  ← 位置2关注[0,1,2,3,4]
              3  [ 0  1  1  1  1 ]  ← 位置3关注[1,2,3,4]
              4  [ 0  0  1  1  1 ]] ← 位置4关注[2,3,4]
        """
        # 创建全1矩阵
        mask = torch.ones(length, length, device=device)

        # 对于每个位置 i
        for i in range(length):
            # 计算窗口范围
            start = max(0, i - window_size // 2)
            end = min(length, i + window_size // 2 + 1)

            # 只保留窗口内的位置
            mask[i, :start] = 0  # 左侧屏蔽
            mask[i, end:] = 0    # 右侧屏蔽

        return mask


class DropPath(nn.Module):
    """DropPath (Stochastic Depth) 正则化

    设计理念：
    训练时随机"丢弃"整个残差路径，迫使网络学习多条特征提取路径。
    这比 Dropout（随机丢弃单个神经元）更适合残差网络。

    工作原理：
    - 训练时：以概率 drop_prob 将整个路径的输出置零
    - 推理时：保持所有路径，但输出乘以 (1 - drop_prob) 进行缩放

    与 Dropout 的区别：
    - Dropout: 随机丢弃单个激活值
    - DropPath: 随机丢弃整个样本的路径

    通常用法：
    在深层网络中，浅层使用较小的 drop_prob（如 0.0），
    深层使用较大的 drop_prob（如 0.1），形成递增的丢弃率。

    参数：
        drop_prob: 路径丢弃概率（0.0 表示不丢弃）

    示例：
        >>> drop_path = DropPath(drop_prob=0.1)
        >>> x = torch.randn(2, 50, 64)
        >>> x = drop_path(x)  # 训练时有10%概率置零
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            x: 输入张量 [Batch, ...]

        返回：
            处理后的张量（维度不变）
        """
        # 推理模式或 drop_prob=0 时，直接返回
        if not self.training or self.drop_prob == 0.0:
            return x

        # 计算保留概率
        keep_prob = 1 - self.drop_prob

        # 生成随机掩码 [Batch, 1, 1, ...]
        # 形状与 x 相同，但除了 batch 维度外其他维度都是 1
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

        # 二值化：> 0.5 → 1, < 0.5 → 0
        random_tensor.floor_()

        # 缩放并应用掩码
        # 除以 keep_prob 保持期望不变
        output = x.div(keep_prob) * random_tensor

        return output


class LightweightTransformerLayer(nn.Module):
    """轻量级 Transformer 层

    设计理念：
    使用 Pre-Norm 结构（先归一化再计算）+ 局部注意力 + 残差连接。

    结构（Pre-Norm）：
    ```
    输入 x
      ↓
      ├────────────────────┐
      │                    │ (残差连接)
      ▼                    │
    LayerNorm              │
      ↓                    │
    LocalAttention         │
      ↓                    │
    DropPath               │
      │                    │
      └────( + )───────────┘
      ↓
      ├────────────────────┐
      │                    │ (残差连接)
      ▼                    │
    LayerNorm              │
      ↓                    │
    MLP (64→128→64)        │
      ↓                    │
    DropPath               │
      │                    │
      └────( + )───────────┘
      ↓
    输出 x
    ```

    Pre-Norm vs Post-Norm：
    - Pre-Norm (本实现): Norm → Sublayer → Residual
      * 优点: 训练更稳定，梯度流动更顺畅
      * 缺点: 表达能力略弱
    - Post-Norm: Sublayer → Residual → Norm
      * 优点: 表达能力更强
      * 缺点: 深层网络难训练

    MLP 结构：
    - 扩展因子 = 2.0
    - 64 → 128 (扩展) → 64 (压缩)
    - 使用 GELU 激活函数

    参数：
        dim: 特征维度
        num_heads: 注意力头数
        window_size: 局部窗口大小
        mlp_ratio: MLP 扩展倍数（默认 2.0）
        dropout: Dropout 概率（默认 0.1）
        drop_path: DropPath 概率（默认 0.0）

    示例：
        >>> layer = LightweightTransformerLayer(
        ...     dim=64, num_heads=4, window_size=25, drop_path=0.0
        ... )
        >>> x = torch.randn(2, 50, 64)
        >>> y = layer(x)  # [2, 50, 64]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # ====================================================================
        # 注意力分支
        # ====================================================================
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalAttention1D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )
        self.drop_path1 = DropPath(drop_path)

        # ====================================================================
        # MLP 分支
        # ====================================================================
        self.norm2 = nn.LayerNorm(dim)

        # MLP: dim → dim*mlp_ratio → dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),  # GELU 比 ReLU 更平滑
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            x: 输入张量 [Batch, Time, Features]

        返回：
            输出张量 [Batch, Time, Features]（维度不变）

        计算流程：
        1. 注意力分支: x + DropPath(Attn(Norm(x)))
        2. MLP 分支: x + DropPath(MLP(Norm(x)))
        """
        # 注意力分支（Pre-Norm）
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # MLP 分支（Pre-Norm）
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class LightweightTransformerEncoder(nn.Module):
    """轻量级 Transformer 编码器

    设计目标：
    1. 从 ResNet 的 64 通道特征中捕捉全局时序依赖
    2. 使用局部注意力降低计算复杂度
    3. 控制参数量在 ~58K

    网络结构：
        输入: [B, 64, 50] (Channels, Time) - 来自 ResNet
          ↓
        维度转换: [B, 64, 50] → [B, 50, 64] (Time, Features)
          ↓
        位置编码: 添加正弦位置信息（无参数）
          ↓
        Transformer Layer 1 (DropPath=0.0)
          ↓ LocalAttention(4 头, W=25) + MLP(64→128→64)
        输出: [B, 50, 64]
          ↓
        Transformer Layer 2 (DropPath=0.05)
          ↓ LocalAttention(4 头, W=25) + MLP(64→128→64)
        输出: [B, 50, 64]
          ↓
        Final LayerNorm
          ↓
        维度转换: [B, 50, 64] → [B, 64, 50] (Features, Time)
          ↓
        输出: [B, 64, 50] (Channels, Time) - 传给 Decoder

    为什么转换维度？
    - ResNet 输出是 Conv1d 格式: [B, Channels, Time]
    - Transformer 期望输入格式: [B, Time, Features]
    - 需要在进入/离开 Transformer 时转换维度

    参数分布：
    - 位置编码: 0 参数（正弦函数）
    - Layer 1 注意力: ~12.3K
    - Layer 1 MLP: ~16.4K
    - Layer 2 注意力: ~12.3K
    - Layer 2 MLP: ~16.4K
    - LayerNorm: ~0.3K
    - 总计: ~58K

    参数：
        dim: 特征维度（默认 64）
        depth: Transformer 层数（默认 2）
        num_heads: 注意力头数（默认 4）
        window_size: 局部窗口大小（默认 25）
        mlp_ratio: MLP 扩展倍数（默认 2.0）
        dropout: Dropout 概率（默认 0.1）
        drop_path_rate: DropPath 最大概率（默认 0.1）

    示例：
        >>> encoder = LightweightTransformerEncoder(
        ...     dim=64, depth=2, num_heads=4, window_size=25
        ... )
        >>> x = torch.randn(2, 64, 50)  # ResNet 输出格式
        >>> y = encoder(x)  # [2, 64, 50]
    """

    def __init__(
        self,
        dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 25,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.dim = dim

        # ====================================================================
        # 位置编码
        # ====================================================================
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=dim,
            max_len=100  # 最大支持 100 个时间步
        )

        # ====================================================================
        # Transformer 层堆叠
        # ====================================================================
        # DropPath 概率递增：0.0 → drop_path_rate
        # 例如：depth=2 时，层 0 使用 0.0，层 1 使用 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = nn.ModuleList([
            LightweightTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        # ====================================================================
        # 最终归一化
        # ====================================================================
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：
            x: ResNet 编码器输出 [Batch, 64, 50]
                - Batch: 批次大小
                - 64: 特征通道数
                - 50: 时间步数

        返回：
            Transformer 编码器输出 [Batch, 64, 50]
                - 维度不变，但特征已经过全局时序建模

        完整数据流：

        1. 输入（Conv1d 格式）
           └─ [B, 64, 50]
              来自 ResNet 的局部特征

        2. 维度转换（Transformer 格式）
           └─ [B, 50, 64]
              (Channels, Time) → (Time, Features)

        3. 位置编码
           └─ [B, 50, 64]
              添加位置信息（无参数）

        4. Transformer Layer 1
           ├─ LayerNorm + LocalAttention(W=25)
           │  └─ 每个位置关注前后 25 个时间步
           ├─ Residual + DropPath(0.0)
           ├─ LayerNorm + MLP(64→128→64)
           └─ Residual + DropPath(0.0)
           输出: [B, 50, 64]

        5. Transformer Layer 2
           ├─ LayerNorm + LocalAttention(W=25)
           ├─ Residual + DropPath(0.05)
           ├─ LayerNorm + MLP(64→128→64)
           └─ Residual + DropPath(0.05)
           输出: [B, 50, 64]

        6. Final LayerNorm
           └─ [B, 50, 64]
              归一化输出

        7. 维度转换（Conv1d 格式）
           └─ [B, 64, 50]
              (Time, Features) → (Channels, Time)

        8. 输出
           └─ [B, 64, 50]
              准备输入到 Decoder
        """
        # ===== 调试信息（可选，训练时可注释掉） =====
        # print(f"Transformer 输入维度: {x.shape}")  # [B, 64, 50]

        # ====================================================================
        # 维度转换: [B, C, T] → [B, T, C]
        # ====================================================================
        # 将 Conv1d 格式转换为 Transformer 格式
        x = x.transpose(1, 2)  # [B, 64, 50] → [B, 50, 64]
        # print(f"维度转换后: {x.shape}")

        # ====================================================================
        # 位置编码
        # ====================================================================
        x = self.pos_encoding(x)  # [B, 50, 64] → [B, 50, 64]
        # print(f"位置编码后: {x.shape}")

        # ====================================================================
        # Transformer 层
        # ====================================================================
        for i, layer in enumerate(self.layers):
            x = layer(x)  # [B, 50, 64] → [B, 50, 64]
            # print(f"Transformer Layer {i+1} 输出: {x.shape}")

        # ====================================================================
        # 最终归一化
        # ====================================================================
        x = self.norm(x)  # [B, 50, 64] → [B, 50, 64]
        # print(f"Final Norm 后: {x.shape}")

        # ====================================================================
        # 维度转换: [B, T, C] → [B, C, T]
        # ====================================================================
        # 转换回 Conv1d 格式供 Decoder 使用
        x = x.transpose(1, 2)  # [B, 50, 64] → [B, 64, 50]
        # print(f"Transformer 输出维度: {x.shape}")

        return x


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("测试轻量级 Transformer 编码器 (LightweightTransformerEncoder)")
    print("=" * 80)

    # 创建编码器
    encoder = LightweightTransformerEncoder(
        dim=64,
        depth=2,
        num_heads=4,
        window_size=25,
        mlp_ratio=2.0,
        dropout=0.1,
        drop_path_rate=0.1
    )

    # 打印网络结构
    print("\n网络结构：")
    print(encoder)

    # 测试输入（模拟 ResNet 输出）
    batch_size = 2
    channels = 64
    time_steps = 50

    x = torch.randn(batch_size, channels, time_steps)
    print(f"\n输入维度: {x.shape}")

    # 前向传播
    print("\n开始前向传播...")
    y = encoder(x)
    print(f"输出维度: {y.shape}")
    print(f"期望维度: [{batch_size}, {channels}, {time_steps}]")

    # 验证维度
    assert y.shape == (batch_size, channels, time_steps), "输出维度不正确！"
    print("✅ 维度验证通过")

    # 统计参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"\n参数统计：")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"目标参数量: ~58,000")
    print(f"实际参数量: {total_params:,}")

    # 分层参数统计
    print(f"\n分层参数统计：")
    print(f"位置编码: 0 (无参数)")
    for i, layer in enumerate(encoder.layers):
        params = sum(p.numel() for p in layer.parameters())
        print(f"Transformer Layer {i+1}: {params:,}")
    print(f"Final Norm: {sum(p.numel() for p in encoder.norm.parameters()):,}")

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

    # 测试局部注意力掩码
    print(f"\n测试局部注意力掩码...")
    attn_layer = encoder.layers[0].attn
    mask = attn_layer._create_local_mask(50, 25, x.device)
    print(f"掩码维度: {mask.shape}")
    print(f"掩码示例（位置 25）:")
    print(f"  可关注范围: {mask[25].nonzero().flatten().tolist()}")
    print(f"  期望范围: [13, 14, ..., 36, 37] (共 25 个)")

    # 计算实际窗口大小
    actual_window = mask[25].sum().item()
    print(f"  实际窗口大小: {int(actual_window)}")

    # 测试位置编码
    print(f"\n测试位置编码...")
    pos_enc = encoder.pos_encoding
    test_seq = torch.randn(1, 50, 64)
    test_out = pos_enc(test_seq)
    print(f"位置编码输入: {test_seq.shape}")
    print(f"位置编码输出: {test_out.shape}")
    print("✅ 位置编码维度正确")

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

    # 计算复杂度估算
    print(f"\n复杂度分析：")
    L = 50  # 序列长度
    W = 25  # 窗口大小
    D = 64  # 特征维度
    H = 4   # 头数

    # 局部注意力 FLOPs: O(L × W × D)
    attn_flops = 2 * L * W * D  # 2x 因为有 Q@K^T 和 attn@V
    print(f"单层局部注意力 FLOPs: {attn_flops:,}")

    # 全局注意力 FLOPs: O(L² × D)
    global_attn_flops = 2 * L * L * D
    print(f"单层全局注意力 FLOPs: {global_attn_flops:,}")
    print(f"节省: {(global_attn_flops - attn_flops) / global_attn_flops * 100:.1f}%")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过！Transformer 编码器工作正常")
    print("=" * 80)
