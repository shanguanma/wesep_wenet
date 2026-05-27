"""Normalization layers for MossFormer2.

Ported from ClearerVoice-Studio:
https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction/models/av_mossformer2
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Wrapper around torch.nn.LayerNorm that accepts input_size or input_shape."""

    def __init__(
        self,
        input_size=None,
        input_shape=None,
        eps=1e-05,
        elementwise_affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        )

    def forward(self, x):
        return self.norm(x)


class CLayerNorm(nn.LayerNorm):
    """Channel-wise layer normalization for (B, C, T) tensors."""

    def __init__(self, *args, **kwargs):
        super(CLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, sample):
        if sample.dim() != 3:
            raise RuntimeError(
                '{} only accept 3-D tensor as input'.format(self.__name__))
        sample = torch.transpose(sample, 1, 2)
        sample = super().forward(sample)
        sample = torch.transpose(sample, 1, 2)
        return sample


class ScaleNorm(nn.Module):
    """Scale normalization with learnable gain."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g
