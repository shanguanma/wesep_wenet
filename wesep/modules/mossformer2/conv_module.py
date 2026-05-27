"""Conformer-style convolution module for MossFormer2.

Ported from ClearerVoice-Studio:
https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction/models/av_mossformer2
"""

import torch.nn as nn
from torch import Tensor


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution (groups == in_channels)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise + depthwise + residual."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 17,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(
                in_channels, in_channels, kernel_size,
                stride=1, padding=(kernel_size - 1) // 2
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.sequential(inputs).transpose(1, 2)
