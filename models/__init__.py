# -*- coding: utf-8 -*-
"""
轻量级 ResNet + Transformer 混合模型

模块组织:
- layers.py: 基础层组件（深度可分离卷积、轻量级 ResNet 块）
- resnet.py: ResNet 编码器
- transformer.py: Transformer 编码器
- hybrid_model.py: 完整混合模型
"""

from .layers import DepthwiseSeparableConv1d, LightweightResBlock1D
from .resnet import LightweightResNetEncoder
from .transformer import (
    SinusoidalPositionalEncoding,
    LocalAttention1D,
    DropPath,
    LightweightTransformerLayer,
    LightweightTransformerEncoder,
)
from .hybrid_model import LightweightDecoder, LightweightResNetTransformer

__all__ = [
    # 基础层
    "DepthwiseSeparableConv1d",
    "LightweightResBlock1D",
    # ResNet
    "LightweightResNetEncoder",
    # Transformer
    "SinusoidalPositionalEncoding",
    "LocalAttention1D",
    "DropPath",
    "LightweightTransformerLayer",
    "LightweightTransformerEncoder",
    # 混合模型
    "LightweightDecoder",
    "LightweightResNetTransformer",
]
