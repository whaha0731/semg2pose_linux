# train.py
"""
基于 MultiFileWindowedEMGDataset 的深度学习训练流程
包含数据加载、训练循环、验证循环等完整框架
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from semgdataset import MultiFileWindowedEMGDataset


# ============================================================================
# 配置参数
# ============================================================================

class Config:
    """训练配置"""
    # 数据路径
    data_dir: Path = Path("emg2pose_dataset_mini")

    # Dataset 参数
    window_length: int = 100
    stride: int = 50
    skip_invalid: bool = True
    target_sample_rate: int = 200     # 降采样至 200 Hz（从原始2000Hz降采样10x）
    use_envelope: bool = True          # 使用Hilbert包络特征替代原始EMG信号

    # DataLoader 参数
    batch_size: int = 16  # 从 32 降低到 16（小数据集 + 轻量模型）
    num_workers: int = 4
    train_val_split: float = 0.85  # 从 0.8 提高到 0.85（给训练集更多数据）

    # 训练参数
    num_epochs: int = 100  # 从 50 提高到 100（轻量模型需要更多 epoch）
    learning_rate: float = 3e-4  # 从 1e-3 降低到 3e-4（更稳定的学习率）
    weight_decay: float = 1e-4  # 从 1e-5 提高到 1e-4（更强的 L2 正则化）

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 随机种子
    seed: int = 42

    # 模型保存
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 10  # 每 N 个 epoch 保存一次


# ============================================================================
# 占位符模型（这里应该替换为实际的网络结构）
# ============================================================================

class PlaceholderModel(nn.Module):
    """
    占位符模型 - 请替换为实际的网络结构

    输入：EMG 信号 [B, 16, 100]
    输出：关节角度预测 [B, J, 100]，J 为关节数
    """

    def __init__(self, emg_channels: int = 16, num_joints: int = 20, window_length: int = 100):
        super().__init__()

        # 这里是一个简单的占位符网络
        # 实际应用中应该使用 TCN、Transformer、LSTM 等更复杂的架构
        self.encoder = nn.Sequential(
            nn.Conv1d(emg_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_joints, kernel_size=3, padding=1),
        )

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emg: [B, 16, 100] - EMG 信号

        Returns:
            joint_angles: [B, 20, 100] - 预测的关节角度
        """
        x = self.encoder(emg)  # [B, 128, 100]
        x = self.decoder(x)     # [B, 20, 100]
        return x


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """
    创建训练集和验证集的 DataLoader

    Returns:
        train_loader, val_loader
    """
    # 获取所有 HDF5 文件
    hdf5_files = sorted(config.data_dir.glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files")

    # 创建完整 dataset
    full_dataset = MultiFileWindowedEMGDataset(
        hdf5_paths=hdf5_files,
        window_length=config.window_length,
        stride=config.stride,
        skip_invalid=config.skip_invalid,
        target_sample_rate=config.target_sample_rate,
        use_envelope=config.use_envelope,
    )

    print(f"Total windows in dataset: {len(full_dataset)}")

    # 划分训练集和验证集
    train_size = int(config.train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
    )

    return train_loader, val_loader


# ============================================================================
# 训练和验证函数
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    """
    训练一个 epoch

    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # 将数据移到设备上
        emg = batch["emg"].to(device)              # [B, 16, 100]
        target = batch["joint_angles"].to(device)  # [B, 20, 100]

        # 前向传播
        optimizer.zero_grad()
        output = model(emg)  # [B, 20, 100]

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item()
        num_batches += 1

        # 打印进度
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    验证模型

    Returns:
        平均验证损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # 将数据移到设备上
            emg = batch["emg"].to(device)
            target = batch["joint_angles"].to(device)

            # 前向传播
            output = model(emg)

            # 计算损失
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: Path,
):
    """保存模型检查点"""
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    """主训练函数"""
    # 初始化配置
    config = Config()

    # 设置随机种子
    set_seed(config.seed)

    # 打印配置
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Data directory: {config.data_dir}")
    print(f"Window length: {config.window_length}")
    print(f"Stride: {config.stride}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"Num epochs: {config.num_epochs}")
    print("=" * 80)

    # 创建数据加载器
    print("\nCreating data loaders...")
    train_loader, val_loader = get_data_loaders(config)

    # 创建模型（轻量级 ResNet + Transformer 混合模型）
    print("\nInitializing model...")
    from models.hybrid_model import LightweightResNetTransformer

    model = LightweightResNetTransformer().to(config.device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 分模块参数统计
    resnet_params = sum(p.numel() for p in model.resnet_encoder.parameters())
    transformer_params = sum(p.numel() for p in model.transformer_encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  ├─ ResNet encoder:      {resnet_params:>6,} ({resnet_params/total_params*100:.1f}%)")
    print(f"  ├─ Transformer encoder: {transformer_params:>6,} ({transformer_params/total_params*100:.1f}%)")
    print(f"  └─ Decoder:             {decoder_params:>6,} ({decoder_params/total_params*100:.1f}%)")

    # 定义损失函数（MSE Loss 适用于回归任务）
    criterion = nn.MSELoss()

    # 定义优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # 训练循环
    print("\nStarting training...")
    print("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 80)

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )

        # 验证
        val_loss = validate(model, val_loader, criterion, config.device)

        # 更新学习率
        scheduler.step(val_loss)

        # 打印结果
        if(epoch % 5 == 0) or (epoch == 1):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config.checkpoint_dir
            )
            print(f"New best model saved! (Val Loss: {val_loss:.6f})")

        # 定期保存检查点
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config.checkpoint_dir
            )

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
