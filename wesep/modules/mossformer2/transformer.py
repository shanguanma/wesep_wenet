"""FLASH Transformer with Dual Attention and FSMN for MossFormer2.

Ported from ClearerVoice-Studio:
https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction/models/av_mossformer2

Authors: Shengkui Zhao, Jia Qi Yip (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from typing import Optional

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from wesep.modules.mossformer2.fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from wesep.modules.mossformer2.normalization import LayerNorm, CLayerNorm, ScaleNorm
from wesep.modules.mossformer2.conv_module import ConvModule


def exists(val):
    return val is not None


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


def default(val, d):
    return val if exists(val) else d


class FFConvM(nn.Module):
    """Feed-forward with convolution module."""

    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mdl(x)


class Gated_FSMN_dilated(nn.Module):
    """Gated FSMN with dilated convolutions."""

    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        super().__init__()
        self.to_u = FFConvM(
            dim_in=in_channels, dim_out=hidden_size,
            norm_klass=nn.LayerNorm, dropout=0.1,
        )
        self.to_v = FFConvM(
            dim_in=in_channels, dim_out=hidden_size,
            norm_klass=nn.LayerNorm, dropout=0.1,
        )
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x


class Gated_FSMN_Block_Dilated(nn.Module):
    """Gated FSMN Block with dilated convolutions and normalization."""

    def __init__(self, dim, inner_channels=256, group_size=256, norm_type='scalenorm'):
        super(Gated_FSMN_Block_Dilated, self).__init__()
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = Gated_FSMN_dilated(
            inner_channels, inner_channels, lorder=20, hidden_size=inner_channels
        )
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2, 1) + input


class OffsetScale(nn.Module):
    """Learnable offset and scale for multi-head projections."""

    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class FLASH_ShareA_FFConvM(nn.Module):
    """FLASH attention with shared attention, feed-forward, and convolution module."""

    def __init__(
        self, *, dim, group_size=256, query_key_dim=128,
        expansion_factor=1., causal=False, dropout=0.1,
        rotary_pos_emb=None, norm_klass=nn.LayerNorm, shift_tokens=True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = FFConvM(
            dim_in=dim, dim_out=hidden_dim,
            norm_klass=norm_klass, dropout=dropout,
        )
        self.to_qk = FFConvM(
            dim_in=dim, dim_out=query_key_dim,
            norm_klass=norm_klass, dropout=dropout,
        )
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = FFConvM(
            dim_in=dim * 2, dim_out=dim,
            norm_klass=norm_klass, dropout=dropout,
        )
        self.gateActivate = nn.Sigmoid()

    def forward(self, x, *, mask=None):
        normed_x = x
        residual = x

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)

        out = (att_u * v) * self.gateActivate(att_v * u)
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys,
                (quad_q, lin_q, quad_k, lin_k)
            )

        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.),
                (quad_q, quad_k, lin_q, lin_k, v, u)
            )
            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v, u)
        )

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)

        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g
        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        return map(
            lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n],
            (quad_out_v + lin_out_v, quad_out_u + lin_out_u)
        )


class FLASHTransformer_DualA_FSMN(nn.Module):
    """FLASH Transformer with Dual Attention and FSMN layers."""

    def __init__(
        self, *, dim, depth, group_size=256, query_key_dim=128,
        expansion_factor=4., causal=False, attn_dropout=0.1,
        norm_type='scalenorm', shift_tokens=True
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm')

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))

        self.fsmn = nn.ModuleList(
            [Gated_FSMN_Block_Dilated(dim) for _ in range(depth)]
        )
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim, group_size=group_size, query_key_dim=query_key_dim,
                expansion_factor=expansion_factor, causal=causal,
                dropout=attn_dropout, rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass, shift_tokens=shift_tokens
            )
            for _ in range(depth)
        ])

    def forward(self, x, *, mask=None):
        for ii, flash in enumerate(self.layers):
            x = flash(x, mask=mask)
            x = self.fsmn[ii](x)
        return x


class TransformerEncoder_FLASH_DualA_FSMN(nn.Module):
    """MossFormer2 Transformer Encoder combining FLASH attention with FSMN."""

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
    ):
        super().__init__()
        self.flashT = FLASHTransformer_DualA_FSMN(dim=d_model, depth=num_layers)
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        output = self.flashT(src)
        output = self.norm(output)
        return output
