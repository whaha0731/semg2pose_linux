# dataset.py
"""
多文件 HDF5 Dataset 实现
支持从多个独立的 HDF5 文件中按索引获取时间窗口数据
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import  ClassVar

import h5py
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset


@dataclass
class MultiSessionHDF5Data:
    """单个 HDF5 会话文件的只读接口
    
    支持以下数据结构：
    - HDF5 文件 → Group → Dataset（compound dtype）
    - Dataset 字段：time, joint_angles, emg
    """
    
    HDF5_GROUP: ClassVar[str] = "emg2pose"
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG: ClassVar[str] = "emg"
    JOINT_ANGLES: ClassVar[str] = "joint_angles"
    TIMESTAMPS: ClassVar[str] = "time"
    
    hdf5_path: Path
    
    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        emg2pose_group = self._file[self.HDF5_GROUP]
        self.timeseries = emg2pose_group[self.TIMESERIES]
        
        # 验证数据结构
        assert self.timeseries.dtype.fields is not None
        assert self.EMG in self.timeseries.dtype.fields
        assert self.JOINT_ANGLES in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._file.close()
    
    def __len__(self) -> int:
        return len(self.timeseries)
    
    '''核心方法：按切片读取数据'''
    def __getitem__(self, key: slice) -> np.ndarray:
        return self.timeseries[key]
    
    @property
    def timestamps(self) -> np.ndarray:
        """获取时间戳数组（注意会触发磁盘 IO）"""
        if not hasattr(self, "_timestamps"):
            self._timestamps = self.timeseries[self.TIMESTAMPS]
        return self._timestamps


@dataclass
class MultiFileWindowedEMGDataset(Dataset):
    """从多个 HDF5 文件中按索引获取窗口数据的 PyTorch Dataset

    Args:
        hdf5_paths (list[Path]): 所有 HDF5 文件的路径列表
        window_length (int): 时间窗口长度（样本数），默认 100
        stride (int): 窗口滑动步幅，默认等于 window_length（无重叠）
        skip_invalid (bool): 是否跳过包含 NaN 的窗口
        target_sample_rate (int | None): 目标采样率（Hz），None 表示不降采样（保持 2000Hz）
            - 原始采样率: 2000 Hz
            - ⚠️ 警告: 数据经过 20-850Hz 带通滤波，降采样至 <1700Hz 会丢失高频信息
        use_envelope (bool): 是否使用 Hilbert 包络特征替代原始 EMG 信号，默认 True
    Returns:
        dict 包含：
            - "emg": Tensor[1, 16, 100]，EMG 信号（1×通道数×时间长度）
            - "joint_angles": Tensor[20, 100]，关节角度标签（关节数×时间长度）
            - "session_idx": int，该窗口来自哪个 HDF5 文件
            - "window_start": int，窗口在该文件中的起始索引
    """

    hdf5_paths: list[Path]
    window_length: int = 100
    stride: int = 100
    skip_invalid: bool = False
    target_sample_rate: int = 100   # None = 不降采样，保持 2000Hz
    use_envelope: bool = True       # 是否使用Hilbert包络特征替代原始EMG信号
    
    def __post_init__(self) -> None:
        # 生成各session hdf5路径列表 hdf5_paths
        self.hdf5_paths = [Path(p) for p in self.hdf5_paths]

        # 参数验证
        assert self.window_length > 0, "window_length 必须 > 0"
        assert self.stride > 0, "stride 必须 > 0"
        
        '''降采样参数计算'''
        self.original_sample_rate = 2000  # Hz，HDF5 文件的原始采样率
        if self.target_sample_rate is None:
            self.target_sample_rate = self.original_sample_rate  # 不降采样

        assert self.target_sample_rate > 0, "target_sample_rate 必须 > 0"
        assert self.target_sample_rate <= self.original_sample_rate, \
            f"目标采样率 {self.target_sample_rate} 不能超过原始采样率 {self.original_sample_rate}"
        
        self.downsample_factor = self.original_sample_rate // self.target_sample_rate

        '''计算实际时间跨度'''
        actual_time_ms = (self.window_length / self.target_sample_rate) * 1000
        print(f"[INFO] 数据集配置:")
        print(f"   - 原始采样率: {self.original_sample_rate} Hz")
        print(f"   - 目标采样率: {self.target_sample_rate} Hz")
        print(f"   - 降采样因子: {self.downsample_factor}x")
        print(f"   - 窗口长度: {self.window_length} 样本")
        print(f"   - 实际时间跨度: {actual_time_ms:.1f} ms ({actual_time_ms/1000:.3f} s)")
        print(f"   - 使用Hilbert包络: {'是 (在降采样后计算)' if self.use_envelope else '否 (原始EMG信号)'}")

        # 懒加载会话
        self._sessions: dict[int, MultiSessionHDF5Data] = {}

        # 预计算每个文件的窗口信息
        self._compute_window_mapping()
    
    def _get_session(self, session_idx: int) -> MultiSessionHDF5Data:
        """懒加载会话"""
        if session_idx not in self._sessions:
            self._sessions[session_idx] = MultiSessionHDF5Data(
                self.hdf5_paths[session_idx]
            )
        return self._sessions[session_idx]
    
    def _compute_window_mapping(self) -> None:
        """预计算所有hdf5文件中 所有窗口的位置映射

        建立从全局 dataset idx → (session_idx, local_offset) 的映射
        注意：这里的 offset 是基于**降采样后**的长度计算的
        """
        self.window_mapping = []  # list[(session_idx, offset, session_length)]
        self.session_lengths = []

        for session_idx, hdf5_path in enumerate(self.hdf5_paths):
            # 获取该会话的原始长度
            with MultiSessionHDF5Data(hdf5_path) as session:
                original_length = len(session)

            '''计算单个session降采样后的长度,在该长度下计算窗口数量'''
            downsampled_length = original_length // self.downsample_factor
            self.session_lengths.append(downsampled_length)

            # 计算该会话内能产生多少个窗口
            if downsampled_length < self.window_length:
                # 文件太短，跳过
                continue

            # 计算窗口数量
            num_windows = (downsampled_length - self.window_length) // self.stride + 1

            # 为每个窗口记录映射关系
            for window_idx in range(num_windows):
                offset = window_idx * self.stride
                self.window_mapping.append((session_idx, offset, downsampled_length))
    
    def __len__(self) -> int:
        """返回 dataset 中的总窗口数"""
        return len(self.window_mapping)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """核心方法：根据全局索引获取一个窗口的数据

        流程：
        1. 从映射表查询该索引对应的 (session_idx, offset)
        2. 加载对应的 HDF5 文件
        3. 计算原始 HDF5 中需要读取的范围（考虑降采样因子）
        4. 读取窗口数据
        5. 提取 EMG 和关节角度
        6. 应用降采样
        7. 转换为 tensor 并返回
        """

        # 步骤 1：查询窗口位置（基于降采样后的坐标）
        session_idx, offset, downsampled_length = self.window_mapping[idx]

        # 步骤 2：加载会话
        session = self._get_session(session_idx)

        # 步骤 3：计算原始 HDF5 中的读取范围
        # offset 是降采样后的位置，需要转换为原始位置
        original_start = offset * self.downsample_factor
        original_end = min(
            (offset + self.window_length) * self.downsample_factor,
            len(session)
        )

        # 步骤 4：从 HDF5 读取原始数据（结构化数组）
        window_data = session[original_start:original_end]

        emg_raw = window_data[MultiSessionHDF5Data.EMG]  # shape: (T_original, 16)
        joint_angles_raw = window_data[MultiSessionHDF5Data.JOINT_ANGLES]  # shape: (T_original, J)

        # 步骤 5：检查有效性（可选）
        if self.skip_invalid:
            if np.isnan(emg_raw).any() or np.isnan(joint_angles_raw).any():
                # 如果包含 NaN，返回零张量或跳过（这里简单返回零）
                emg_tensor = torch.zeros(16, self.window_length)
                joint_tensor = torch.zeros(joint_angles_raw.shape[1], self.window_length)
            else:
                # 转换为 tensor（包含降采样）
                emg_tensor = self._prepare_emg_tensor(emg_raw)
                joint_tensor = self._prepare_joint_tensor(joint_angles_raw)
        else:
            # 不检查，直接转换（包含降采样）
            emg_tensor = self._prepare_emg_tensor(emg_raw)
            joint_tensor = self._prepare_joint_tensor(joint_angles_raw)

        # 步骤 6：确保输出形状正确
        actual_length = emg_tensor.shape[-1]

        # 如果实际长度不足 window_length（session末尾window），进行 padding
        if actual_length < self.window_length:
            emg_tensor = torch.nn.functional.pad(
                emg_tensor,
                (0, self.window_length - actual_length),
                mode='constant',
                value=0
            )
            joint_tensor = torch.nn.functional.pad(
                joint_tensor,
                (0, self.window_length - actual_length),
                mode='constant',
                value=0
            )

        elif actual_length > self.window_length:
            emg_tensor = emg_tensor[:, :self.window_length]
            joint_tensor = joint_tensor[:, :self.window_length]

        return {
            "emg": emg_tensor,  # [16, 100]
            "joint_angles": joint_tensor,  # [20, 100]
        }
    
    def _prepare_emg_tensor(self, emg_raw: np.ndarray) -> torch.Tensor:
        """EMG 数据预处理（支持Hilbert包络提取）

        Input: shape (T_original, 16) - 时间 × 通道
        Output: shape (16, T_downsampled) - 通道 × 时间

        处理步骤：
        1. 转置为 (T, 16) → (16, T)
        2. 应用抗混叠降采样（如果需要）
        3. [可选] 使用 Hilbert 变换提取包络
        4. 转换为 PyTorch tensor
        """
        # 转置为 (16, T_original)
        emg_t = emg_raw.T  # shape: (16, T_original)

        # 应用降采样（如果需要）
        if self.downsample_factor > 1:
            # 使用 scipy.signal.decimate 进行抗混叠降采样
            # 对每个通道独立降采样
            emg_downsampled = []
            for channel in emg_t:
                # decimate 会自动应用 Chebyshev Type I 低通滤波器
                downsampled = signal.decimate(
                    channel,
                    self.downsample_factor,
                    ftype='iir',  # IIR 滤波器，速度快
                    zero_phase=True  # 零相位滤波，避免相位失真
                )
                emg_downsampled.append(downsampled)

            emg_t = np.stack(emg_downsampled, axis=0)  # shape: (16, T_downsampled)

        # 包络提取（在降采样之后）
        if self.use_envelope:
            emg_envelope = []
            for channel in emg_t:
                # 步骤1: 镜像padding，减少边界效应
                # 在信号两端进行镜像扩展，避免边界突变
                pad_width = min(20, len(channel) // 4)  # padding长度，不超过信号长度的1/4
                padded = np.pad(channel, pad_width=pad_width, mode='reflect')

                # 步骤2: Hilbert 变换 → 解析信号
                analytic_signal = signal.hilbert(padded)

                # 步骤3: 取幅度 → 瞬时包络
                envelope_padded = np.abs(analytic_signal)

                # 步骤4: 移除padding部分，恢复原始长度
                envelope = envelope_padded[pad_width:-pad_width]

                emg_envelope.append(envelope)
            emg_t = np.stack(emg_envelope, axis=0)  # 替换为包络信号

        emg_tensor = torch.as_tensor(emg_t, dtype=torch.float32)

        return emg_tensor
    
    def _prepare_joint_tensor(self, joint_raw: np.ndarray) -> torch.Tensor:
        """关节角度预处理（支持Hilbert包络提取）

        Input: shape (T_original, J) - 时间 × 关节数
        Output: shape (J, T_downsampled) - 关节数 × 时间

        处理步骤：
        1. 转置为 (T, J) → (J, T)
        2. 应用抗混叠降采样（如果需要）
        3. [可选] 使用 Hilbert 变换提取包络
        4. 转换为 PyTorch tensor
        """
        # 转置为 (J, T_original)
        joint_t = joint_raw.T  # shape: (J, T_original)

        # 应用降采样（如果需要）
        if self.downsample_factor > 1:
            # 使用 scipy.signal.decimate 进行抗混叠降采样
            # 对每个关节独立降采样
            joint_downsampled = []
            for joint in joint_t:
                # decimate 会自动应用 Chebyshev Type I 低通滤波器
                downsampled = signal.decimate(
                    joint,
                    self.downsample_factor,
                    ftype='iir',  # IIR 滤波器，速度快
                    zero_phase=True  # 零相位滤波，避免相位失真
                )
                joint_downsampled.append(downsampled)

            joint_t = np.stack(joint_downsampled, axis=0)  # shape: (J, T_downsampled)

        # 包络提取（在降采样之后）
        if self.use_envelope:
            joint_envelope = []
            for joint in joint_t:
                # Hilbert 变换 → 解析信号
                analytic_signal = signal.hilbert(joint)
                # 取幅度 → 瞬时包络
                envelope = np.abs(analytic_signal)
                joint_envelope.append(envelope)
            joint_t = np.stack(joint_envelope, axis=0)  # 替换为包络信号

        joint_tensor = torch.as_tensor(joint_t, dtype=torch.float32)

        return joint_tensor
    
    def __del__(self):
        """析构时关闭所有打开的 HDF5 文件"""
        for session in self._sessions.values():
            try:
                session._file.close()
            except:
                pass



# # 使用示例
# if __name__ == "__main__":
#     from pathlib import Path

#     # 准备数据文件列表
#     data_dir = Path("emg2pose_dataset_mini")
#     hdf5_files = sorted(data_dir.glob("*.hdf5"))[:5]  # 只用前 5 个文件做测试

#     print(f"Found {len(hdf5_files)} HDF5 files")
#     print(f"Files: {[f.name for f in hdf5_files]}")

#     print("\n" + "="*80)
#     print("测试 1: 不降采样 (2000 Hz)")
#     print("="*80)
#     dataset = MultiFileWindowedEMGDataset(
#         hdf5_paths=hdf5_files,
#         window_length=100,
#         stride=50,
#         target_sample_rate=None  # 不降采样
#     )
#     print(f"\nDataset length: {len(dataset)}")
#     print(f"Session lengths: {dataset.session_lengths}")

#     # 获取第一个样本
#     sample = dataset[0]
#     print(f"\nSample 0:")
#     print(f"  EMG shape: {sample['emg'].shape}")
#     print(f"  Joint angles shape: {sample['joint_angles'].shape}")

#     print("\n" + "="*80)
#     print("测试 2: 降采样至 1000 Hz (2x 降采样)")
#     print("="*80)
#     dataset_1000 = MultiFileWindowedEMGDataset(
#         hdf5_paths=hdf5_files,
#         window_length=100,
#         stride=50,
#         target_sample_rate=1000  # 降采样至 1000 Hz
#     )
#     print(f"\nDataset length: {len(dataset_1000)}")
#     print(f"Session lengths: {dataset_1000.session_lengths}")

#     # 获取第一个样本
#     sample_1000 = dataset_1000[0]
#     print(f"\nSample 0:")
#     print(f"  EMG shape: {sample_1000['emg'].shape}")
#     print(f"  Joint angles shape: {sample_1000['joint_angles'].shape}")

#     print("\n" + "="*80)
#     print("测试 3: 降采样至 500 Hz (4x 降采样)")
#     print("="*80)
#     dataset_500 = MultiFileWindowedEMGDataset(
#         hdf5_paths=hdf5_files,
#         window_length=100,
#         stride=50,
#         target_sample_rate=500  # 降采样至 500 Hz
#     )
#     print(f"\nDataset length: {len(dataset_500)}")
#     print(f"Session lengths: {dataset_500.session_lengths}")

#     # 获取第一个样本
#     sample_500 = dataset_500[0]
#     print(f"\nSample 0:")
#     print(f"  EMG shape: {sample_500['emg'].shape}")
#     print(f"  Joint angles shape: {sample_500['joint_angles'].shape}")