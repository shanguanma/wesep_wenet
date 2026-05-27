"""Dual-path processing model for MossFormer2 speech separation/extraction.

Ported from ClearerVoice-Studio:
https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction/models/av_mossformer2
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from wesep.modules.mossformer2.transformer import TransformerEncoder_FLASH_DualA_FSMN


class ScaledSinuEmbedding(nn.Module):
    """Learnable-scale sinusoidal positional embedding."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale


class Linear(nn.Module):
    """Linear transformation with optional 4D combine_dims support."""

    def __init__(self, n_neurons, input_shape=None, input_size=None,
                 bias=True, combine_dims=False):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return self.w(x)


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization for 3D or 4D tensors."""

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Cumulative Layer Normalization (channel-wise)."""

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            x = super().forward(x)
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class SBFLASHBlock_DualA(nn.Module):
    """SpeechBrain-style FLASH Block with Dual Attention for intra-chunk processing."""

    def __init__(
        self, num_layers, d_model, nhead, d_ffn=2048,
        input_shape=None, kdim=None, vdim=None, dropout=0.1,
        activation="relu", use_positional_encoding=False,
        norm_before=False, attention_type="regularMHA",
    ):
        super(SBFLASHBlock_DualA, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder_FLASH_DualA_FSMN(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, N) - batch, time, features
        Returns:
            output: (B, L, N)
        """
        return self.mdl(x)


class Dual_Computation_Block(nn.Module):
    """Single-path computation block (intra-chunk processing only)."""

    def __init__(
        self, intra_mdl, out_channels, norm="ln",
        skip_around_intra=True, linear_layer_after_inter_intra=True,
    ):
        super(Dual_Computation_Block, self).__init__()
        self.intra_mdl = intra_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

        if linear_layer_after_inter_intra:
            self.intra_linear = Linear(out_channels, input_size=out_channels)

    def forward(self, x):
        """
        Args:
            x: (B, N, S) - batch, features, sequence
        Returns:
            out: (B, N, S)
        """
        B, N, S = x.shape
        intra = x.permute(0, 2, 1).contiguous()
        intra = self.intra_mdl(intra)

        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        if self.skip_around_intra:
            intra = intra + x

        return intra


class Dual_Path_Model(nn.Module):
    """MossFormer2 mask estimation network with positional encoding and gated output."""

    def __init__(
        self, in_channels, out_channels, intra_model,
        num_layers=1, norm="ln", K=200, num_spks=2,
        skip_around_intra=True, linear_layer_after_inter_intra=True,
        use_global_pos_enc=True, max_length=20000,
    ):
        super(Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, L) - batch, channels, sequence length
        Returns:
            out: (spks, B, N, L)
        """
        if self.use_global_pos_enc:
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1)
            x = base + emb

        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
            x = self.prelu(x)

        x = self.conv1d_out(x)
        B, _, S = x.shape

        x = x.view(B * self.num_spks, -1, S)
        x = self.output(x) * self.output_gate(x)
        x = self.conv1_decoder(x)

        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        x = x.transpose(0, 1)

        return x
