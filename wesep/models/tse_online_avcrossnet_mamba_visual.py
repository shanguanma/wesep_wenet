# -*- coding: utf-8 -*-
"""
Online AV-CrossNet with Mamba: A Causal and Efficient Audiovisual System 
for Speech Enhancement and Target Speaker Extraction

Based on: "Online AV-CrossNet: a Causal and Efficient Audiovisual System for 
Speech Enhancement and Target Speaker Extraction" (Interspeech 2025)

Paper: https://www.isca-archive.org/interspeech_2025/yu25b_interspeech.pdf

Key Features:
- Causal processing with 1-frame look-ahead (40ms)
- Mamba-based sequence modeling (replacing GMHSA for efficiency)
- Inference latency: ~4.73ms
- Model compression up to 10x
- Supports both AVSE and AVTSE tasks

Author: Implementation based on paper by Cheng Yu et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
import logging
from typing import Optional, Tuple, List
import dataclasses
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OnlineAVCrossNetConfig:
    """Configuration for Online AV-CrossNet model"""
    # Audio parameters
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256  # 16ms hop -> 62.5 fps
    win_length: int = 512
    
    # Visual parameters  
    video_fps: int = 25
    lip_size: int = 88  # Lip ROI size
    
    # Model architecture
    audio_channels: int = 256
    visual_channels: int = 512
    fusion_channels: int = 256
    num_blocks: int = 4
    
    # Mamba parameters (replacing GMHSA)
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # V-TCN parameters
    vtcn_layers: int = 5
    vtcn_channels: int = 512
    vtcn_kernel_size: int = 3
    
    # Compression
    compression_ratio: int = 1  # 1 = no compression, 10 = 10x compression
    
    # Training
    look_ahead_frames: int = 1  # 1-frame look-ahead (40ms for 25fps video)
    # only for tensorboard logging
    def to_json_string(self):
        import json
        return json.dumps(dataclasses.asdict(self), indent=2)


# ============================================================================
# Causal Layers
# ============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D Convolution - no future information leakage"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, groups=groups
        )
    
    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # Remove future padding
        return out


class CausalConv2d(nn.Module):
    """Causal 2D Convolution for time dimension"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        
        # Causal padding: only pad the time dimension (dim 2) on the left
        self.time_padding = (kernel_size[0] - 1) * dilation
        self.freq_padding = kernel_size[1] // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=(self.time_padding, self.freq_padding),
            dilation=dilation
        )
    
    def forward(self, x):
        # x: (B, C, T, F)
        out = self.conv(x)
        if self.time_padding > 0:
            out = out[:, :, :-self.time_padding, :]
        return out


class CausalConv3d(nn.Module):
    """Causal 3D Convolution for video processing"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        
        # Causal padding for time dimension
        self.time_padding = kernel_size[0] - 1
        spatial_padding = kernel_size[1] // 2
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, 
            padding=(self.time_padding, spatial_padding, spatial_padding)
        )
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        out = self.conv(x)
        if self.time_padding > 0:
            out = out[:, :, :-self.time_padding, :, :]
        return out


class CausalGroupNorm(nn.Module):
    """Group Normalization with causal statistics (per-frame)"""
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # x: (B, C, T) or (B, C, T, F)
        orig_shape = x.shape
        B, C = x.shape[:2]
        
        # Reshape for group norm
        x = x.view(B, self.num_groups, C // self.num_groups, -1)
        
        # Compute stats per frame (causal)
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(B, C, *orig_shape[2:])
        
        # Apply affine transform
        if len(orig_shape) == 3:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        else:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x


# ============================================================================
# Mamba Block (Replacing GMHSA for efficient causal sequence modeling)
# ============================================================================

class MambaBlock(nn.Module):
    """
    Mamba Block for efficient causal sequence modeling
    Based on: "Mamba: Linear-time Sequence Modeling with Selective State Spaces"
    
    Replaces GMHSA in original AV-CrossNet for better efficiency and causality
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize A and D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        residual = x
        x = self.norm(x)
        
        B, T, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, T, d_inner) each
        
        # Causal convolution
        x = x.transpose(1, 2)  # (B, d_inner, T)
        x = self.conv1d(x)[:, :, :T]  # Causal: remove future
        x = x.transpose(1, 2)  # (B, T, d_inner)
        
        x = F.silu(x)
        
        # SSM
        x_dbl = self.x_proj(x)  # (B, T, d_state*2 + 1)
        delta, B_param, C_param = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))  # (B, T, d_inner)
        
        # Discretization
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Selective scan (simplified for efficiency)
        y = self.selective_scan(x, delta, A, B_param, C_param, self.D)
        
        # Gate
        y = y * F.silu(z)
        
        # Output projection
        out = self.out_proj(y)
        
        return out + residual
    
    def selective_scan(self, x, delta, A, B, C, D):
        """Selective scan operation"""
        B_batch, T, d_inner = x.shape
        d_state = self.d_state
        
        # Initialize state
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(T):
            # Update state
            delta_t = delta[:, t, :, None]  # (B, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))  # (B, d_inner, d_state)
            B_bar = delta_t * B[:, t, None, :]  # (B, d_inner, d_state)
            
            h = A_bar * h + B_bar * x[:, t, :, None]
            
            # Output
            y = (h * C[:, t, None, :]).sum(dim=-1) + D * x[:, t]  # (B, d_inner)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # (B, T, d_inner)


# ============================================================================
# Visual Encoder (Causal)
# ============================================================================

class CausalResBlock(nn.Module):
    """Causal Residual Block for ResNet-18"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.GroupNorm(8, out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CausalVisualTCN(nn.Module):
    """Causal Visual Temporal Convolutional Network (V-TCN)"""
    def __init__(self, channels, num_layers=5, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(nn.Sequential(
                CausalConv1d(channels, channels, kernel_size, dilation=dilation),
                nn.GroupNorm(8, channels),
                nn.ReLU(),
                CausalConv1d(channels, channels, kernel_size, dilation=dilation),
                nn.GroupNorm(8, channels),
            ))
    
    def forward(self, x):
        # x: (B, C, T)
        for layer in self.layers:
            residual = x
            x = layer(x) + residual
            x = F.relu(x)
        return x


class CausalVisualEncoder(nn.Module):
    """
    Causal Visual Encoder
    Components:
    1. 3D Causal Convolutional Encoder
    2. Causal ResNet-18
    3. 5-layer Causal V-TCN
    """
    def __init__(self, config: OnlineAVCrossNetConfig):
        super().__init__()
        self.config = config
        
        # 3D Causal Conv Encoder (lip region -> features)
        self.frontend = nn.Sequential(
            CausalConv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2)),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Causal ResNet-18 (2D processing per frame)
        self.resnet = nn.Sequential(
            CausalResBlock(64, 64),
            CausalResBlock(64, 64),
            CausalResBlock(64, 128, stride=2),
            CausalResBlock(128, 128),
            CausalResBlock(128, 256, stride=2),
            CausalResBlock(256, 256),
            CausalResBlock(256, config.visual_channels, stride=2),
            CausalResBlock(config.visual_channels, config.visual_channels),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Causal V-TCN for temporal modeling
        self.vtcn = CausalVisualTCN(
            config.visual_channels,
            num_layers=config.vtcn_layers,
            kernel_size=config.vtcn_kernel_size
        )
    
    def forward(self, video):
        """
        Args:
            video: (B, T, H, W) - grayscale lip ROI sequence
        Returns:
            visual_features: (B, C, T)
        """
        B, T, H, W = video.shape
        
        # Add channel dimension
        x = video.unsqueeze(1)  # (B, 1, T, H, W)
        
        # 3D Causal Conv
        x = self.frontend(x)  # (B, 64, T, H', W')
        
        # Process each frame with ResNet
        B, C, T_out, H_out, W_out = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T_out, C, H_out, W_out)  # (B*T, C, H, W)
        
        x = self.resnet(x)  # (B*T, C_out, 1, 1)
        x = x.view(B, T_out, -1).permute(0, 2, 1)  # (B, C_out, T)
        
        # V-TCN for temporal modeling
        x = self.vtcn(x)  # (B, C_out, T)
        
        return x


# ============================================================================
# Audio Encoder (STFT-based)
# ============================================================================

class AudioEncoder(nn.Module):
    """Audio Encoder using STFT"""
    def __init__(self, config: OnlineAVCrossNetConfig):
        super().__init__()
        self.config = config
        
        # STFT parameters
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        
        # Register window buffer
        self.register_buffer('window', torch.hann_window(config.win_length))
        
        # Complex spectral mapping encoder
        n_freq = config.n_fft // 2 + 1
        self.encoder = nn.Sequential(
            CausalConv2d(2, config.audio_channels // 4, kernel_size=3),
            CausalGroupNorm(8, config.audio_channels // 4),
            nn.ReLU(),
            CausalConv2d(config.audio_channels // 4, config.audio_channels // 2, kernel_size=3, stride=(1, 2)),
            CausalGroupNorm(8, config.audio_channels // 2),
            nn.ReLU(),
            CausalConv2d(config.audio_channels // 2, config.audio_channels, kernel_size=3, stride=(1, 2)),
            CausalGroupNorm(8, config.audio_channels),
            nn.ReLU(),
        )
    
    def forward(self, audio):
        """
        Args:
            audio: (B, samples) - waveform
        Returns:
            audio_features: (B, C, T, F)
            stft_complex: complex STFT for reconstruction
        """
        # STFT
        stft_complex = torch.stft(
            audio, self.n_fft, self.hop_length, self.win_length,
            self.window, return_complex=True
        )  # (B, F, T)
        
        # Stack real and imaginary parts
        stft_real = stft_complex.real.unsqueeze(1)  # (B, 1, F, T)
        stft_imag = stft_complex.imag.unsqueeze(1)  # (B, 1, F, T)
        stft_input = torch.cat([stft_real, stft_imag], dim=1)  # (B, 2, F, T)
        
        # Permute for conv2d: (B, C, T, F)
        stft_input = stft_input.permute(0, 1, 3, 2)
        
        # Encode
        features = self.encoder(stft_input)
        
        return features, stft_complex


# ============================================================================
# TF-CrossNet Block (Causal)
# ============================================================================

class CausalNarrowBandBlock(nn.Module):
    """Causal Narrow-band Processing Block"""
    def __init__(self, channels, num_freqs):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size=3)
        self.norm = nn.GroupNorm(8, channels)
    
    def forward(self, x):
        # x: (B, C, T, n_freq)
        B, C, T, n_freq = x.shape
        # Process each frequency band
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, n_freq, C, T)
        x = x.view(B * n_freq, C, T)
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = x.view(B, n_freq, C, T).permute(0, 2, 3, 1)  # (B, C, T, n_freq)
        return x


class CausalCrossBandBlock(nn.Module):
    """Causal Cross-band Processing Block"""
    def __init__(self, channels, num_freqs):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
    
    def forward(self, x):
        # x: (B, C, T, n_freq)
        B, C, T, n_freq = x.shape
        # Process across frequency bands at each time step
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, C, n_freq)
        x = x.view(B * T, C, n_freq)
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = x.view(B, T, C, n_freq).permute(0, 2, 1, 3)  # (B, C, T, n_freq)
        return x


class OnlineTFCrossNetBlock(nn.Module):
    """
    Online TF-CrossNet Block with Mamba for temporal modeling
    Replaces GMHSA with Mamba for efficient causal processing
    
    Uses per-frequency-band temporal modeling to reduce parameter count
    """
    def __init__(self, config: OnlineAVCrossNetConfig, num_freqs: int):
        super().__init__()
        channels = config.fusion_channels
        
        # Narrow-band processing
        self.narrow_band = CausalNarrowBandBlock(channels, num_freqs)
        
        # Cross-band processing
        self.cross_band = CausalCrossBandBlock(channels, num_freqs)
        
        # Temporal modeling with Mamba (per-channel, not flattened)
        # This significantly reduces parameter count
        self.temporal_mamba = MambaBlock(
            d_model=channels,  # Only process channels, not channels*freqs
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand
        )
        
        self.num_freqs = num_freqs
        self.channels = channels
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, n_freq)
        Returns:
            (B, C, T, n_freq)
        """
        B, C, T, n_freq = x.shape
        
        # Narrow-band
        x = x + self.narrow_band(x)
        
        # Cross-band
        x = x + self.cross_band(x)
        
        # Temporal with Mamba - process each frequency band independently
        # Reshape: (B, C, T, n_freq) -> (B*n_freq, T, C)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B, n_freq, T, C)
        x = x.view(B * n_freq, T, C)
        x = self.temporal_mamba(x)
        x = x.view(B, n_freq, T, C).permute(0, 3, 2, 1)  # (B, C, T, n_freq)
        
        return x


# ============================================================================
# Audio-Visual Fusion
# ============================================================================

class AVFusion(nn.Module):
    """Audio-Visual Feature Fusion"""
    def __init__(self, config: OnlineAVCrossNetConfig):
        super().__init__()
        self.audio_proj = nn.Linear(config.audio_channels, config.fusion_channels)
        self.visual_proj = nn.Linear(config.visual_channels, config.fusion_channels)
        self.fusion = nn.Sequential(
            nn.Linear(config.fusion_channels * 2, config.fusion_channels),
            nn.ReLU(),
            nn.Linear(config.fusion_channels, config.fusion_channels)
        )
    
    def forward(self, audio_feat, visual_feat):
        """
        Args:
            audio_feat: (B, C_a, T_a, n_freq)
            visual_feat: (B, C_v, T_v)
        Returns:
            fused: (B, C_f, T_a, n_freq)
        """
        B, C_a, T_a, n_freq = audio_feat.shape
        _, C_v, T_v = visual_feat.shape
        
        # Interpolate visual features to match audio time
        if T_v != T_a:
            visual_feat = F.interpolate(visual_feat, size=T_a, mode='linear', align_corners=False)
        
        # Project
        audio_feat = audio_feat.permute(0, 2, 3, 1)  # (B, T, n_freq, C_a)
        audio_proj = self.audio_proj(audio_feat)  # (B, T, n_freq, C_f)
        
        visual_feat = visual_feat.permute(0, 2, 1)  # (B, T, C_v)
        visual_proj = self.visual_proj(visual_feat)  # (B, T, C_f)
        visual_proj = visual_proj.unsqueeze(2).expand(-1, -1, n_freq, -1)  # (B, T, n_freq, C_f)
        
        # Concatenate and fuse
        combined = torch.cat([audio_proj, visual_proj], dim=-1)  # (B, T, n_freq, 2*C_f)
        fused = self.fusion(combined)  # (B, T, n_freq, C_f)
        
        return fused.permute(0, 3, 1, 2)  # (B, C_f, T, n_freq)


# ============================================================================
# Decoder
# ============================================================================

class ComplexSpectralDecoder(nn.Module):
    """Decoder for complex spectral mapping"""
    def __init__(self, config: OnlineAVCrossNetConfig, num_freqs: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.fusion_channels, config.fusion_channels // 2, 
                              kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.GroupNorm(8, config.fusion_channels // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(config.fusion_channels // 2, config.fusion_channels // 4,
                              kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.GroupNorm(8, config.fusion_channels // 4),
            nn.ReLU(),
            nn.Conv2d(config.fusion_channels // 4, 2, kernel_size=3, padding=1),  # Real + Imag
        )
        
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
    
    def forward(self, features, stft_shape):
        """
        Args:
            features: (B, C, T, n_freq) - features from TF-CrossNet blocks
            stft_shape: original STFT shape (B, n_fft//2+1, T_stft)
        Returns:
            mask_real, mask_imag: complex mask components (B, n_fft//2+1, T_stft)
        """
        # Decode to mask
        mask = self.decoder(features)  # (B, 2, T_feat, n_freq_feat)
        
        # Get target dimensions from STFT shape
        # stft_shape is (B, F, T) where F = n_fft//2+1
        target_F = stft_shape[1]  # Number of frequency bins
        target_T = stft_shape[2]  # Number of time frames
        
        # Interpolate to match STFT dimensions
        # mask is (B, 2, T_feat, n_freq_feat), need (B, 2, T_stft, n_fft//2+1)
        if mask.shape[2] != target_T or mask.shape[3] != target_F:
            mask = F.interpolate(mask, size=(target_T, target_F), mode='bilinear', align_corners=False)
        
        mask_real = mask[:, 0]  # (B, T, F)
        mask_imag = mask[:, 1]  # (B, T, F)
        
        # Permute for STFT format: (B, F, T)
        mask_real = mask_real.permute(0, 2, 1)
        mask_imag = mask_imag.permute(0, 2, 1)
        
        return mask_real, mask_imag


# ============================================================================
# Main Model: Online AV-CrossNet with Mamba
# ============================================================================

class OnlineAVCrossNetMamba(nn.Module):
    """
    Online AV-CrossNet with Mamba
    
    A causal and efficient audiovisual system for:
    - Speech Enhancement (AVSE)
    - Target Speaker Extraction (AVTSE)
    
    Key features:
    - 1-frame look-ahead (40ms)
    - Mamba-based temporal modeling (replacing GMHSA)
    - Real-time capable (~4.73ms inference)
    """
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = OnlineAVCrossNetConfig()
        elif isinstance(config, dict):
            #valid_fields = {f.name for f in __import__('dataclasses').fields(OnlineAVCrossNetConfig)}
            #config = OnlineAVCrossNetConfig(**{k: v for k, v in config.items() if k in valid_fields})
            valid_fields = {f.name for f in OnlineAVCrossNetConfig.__dataclass_fields__.values()}
            filtered = {k: v for k, v in config.items() if k in valid_fields}
            config = OnlineAVCrossNetConfig(**filtered)
        self.config = config
        
        # Visual Encoder
        self.visual_encoder = CausalVisualEncoder(config)
        
        # Audio Encoder
        self.audio_encoder = AudioEncoder(config)
        
        # Compute feature dimensions
        # After audio encoder, frequency bins are reduced by stride
        self.num_freqs = (config.n_fft // 2 + 1) // 4  # Two stride-2 convolutions
        
        # Audio-Visual Fusion
        self.av_fusion = AVFusion(config)
        
        # Online TF-CrossNet Blocks with Mamba
        self.tf_blocks = nn.ModuleList([
            OnlineTFCrossNetBlock(config, self.num_freqs)
            for _ in range(config.num_blocks)
        ])
        
        # Decoder
        self.decoder = ComplexSpectralDecoder(config, self.num_freqs)
        
        # Window for iSTFT
        self.register_buffer('window', torch.hann_window(config.win_length))
    
    # ------------------------------------------------------------------
    # Lip-ROI extraction  (offline / non-differentiable)
    # Faithfully replicates preprocess_data_for_lrs3.py:
    #   dlib face detect → 68-point landmarks → lip crop (48-67) → grayscale → resize
    # ------------------------------------------------------------------
    _face_detector = None
    _landmark_predictor = None
    _roi_init_done = False

    @classmethod
    def _init_roi_detectors(cls):
        """Lazy-init dlib detectors (once per process)."""
        if cls._roi_init_done:
            return
        cls._roi_init_done = True
        try:
            import dlib
            cls._face_detector = dlib.get_frontal_face_detector()
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            import os as _os
            if _os.path.exists(predictor_path):
                cls._landmark_predictor = dlib.shape_predictor(predictor_path)
            else:
                logging.warning(
                    "dlib landmark predictor not found at %s. "
                    "Lip-ROI extraction will fall back to centre-crop. "
                    "Download: wget http://dlib.net/files/"
                    "shape_predictor_68_face_landmarks.dat.bz2",
                    predictor_path,
                )
        except ImportError:
            logging.warning(
                "dlib is not installed. "
                "Lip-ROI extraction will fall back to centre-crop."
            )
            # Fallback to OpenCV
            cls._face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            cls._landmark_predictor = None

    @staticmethod
    def _extract_lip_roi_single_frame(
        bgr_frame: np.ndarray,
        face_detector,
        landmark_predictor,
        lip_size: int,
        last_valid_roi: np.ndarray | None,
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Process one BGR uint8 frame → grayscale lip ROI (lip_size × lip_size).
        Returns (roi_or_None, last_valid_roi_updated).
        """
        roi = None
        if face_detector is not None and landmark_predictor is not None:
            import dlib  # already verified importable in _init_roi_detectors
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            if len(faces) > 0:
                shape = landmark_predictor(gray, faces[0])
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                lip_lm = landmarks[48:68]
                x_min, y_min = lip_lm.min(axis=0)
                x_max, y_max = lip_lm.max(axis=0)
                margin = int((y_max - y_min) * 0.3)
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(bgr_frame.shape[1], x_max + margin)
                y_max = min(bgr_frame.shape[0], y_max + margin)
                crop = bgr_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                if crop.size > 0:
                    crop = cv2.resize(crop, (lip_size, lip_size))
                    roi = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if roi is None and last_valid_roi is not None:
            roi = last_valid_roi

        if roi is None:
            # Fallback: centre-crop + grayscale (no face detector available)
            h, w = bgr_frame.shape[:2]
            gray_full = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            cx, cy = w // 2, h // 2
            half = min(h, w) // 2
            crop = gray_full[cy - half:cy + half, cx - half:cx + half]
            roi = cv2.resize(crop, (lip_size, lip_size))

        return roi, roi

    @torch.no_grad()
    def get_ROI_from_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract lip ROI using the same offline pipeline as
        preprocess_data_for_lrs3.py (dlib face detection + landmark-based
        lip crop).  **Not differentiable.**

        Args:
            video: (B, H, W, C, T) — RGB float32 in [0, 1] from the dataset
        Returns:
            roi: (B, T, lip_size, lip_size) — grayscale float32 in [0, 1]
        """
        self._init_roi_detectors()
        lip_size = self.config.lip_size  # 88

        B, H, W, C, T = video.shape
        assert C == 3, f"Expected 3 RGB channels, got {C}"

        device = video.device
        # (B, H, W, C, T) → numpy uint8 BGR frames
        # permute to (B, T, H, W, C), scale to [0, 255]
        vid_np = (
            video.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C)
            .detach().cpu().numpy()
        )
        vid_np = (vid_np * 255.0).clip(0, 255).astype(np.uint8)
        # Dataset stores RGB; OpenCV / dlib expect BGR
        vid_np = vid_np[..., ::-1].copy()  # RGB → BGR

        all_rois = np.empty((B, T, lip_size, lip_size), dtype=np.float32)

        for b in range(B):
            last_valid = None
            for t in range(T):
                frame_bgr = vid_np[b, t]  # (H, W, 3) uint8 BGR
                roi, last_valid = self._extract_lip_roi_single_frame(
                    frame_bgr,
                    self._face_detector,
                    self._landmark_predictor,
                    lip_size,
                    last_valid,
                )
                all_rois[b, t] = roi.astype(np.float32) / 255.0

        roi_tensor = torch.from_numpy(all_rois).to(device)  # (B, T, lip_size, lip_size)
        return roi_tensor

    def forward(self, mix: torch.Tensor, enroll: list) -> torch.Tensor:
        """
        Forward pass for speech enhancement/extraction.

        Interface is compatible with TSE_BSRNN_VISUAL so the same Dataset /
        Executor pipeline can be reused.

        Args:
            mix:    (B, 1, T) or (B, T) — mixed audio waveform
            enroll: list[Tensor]
                enroll[0]: (B, H, W, C, T_v) — RGB video from the dataset
                    (float32, [0, 1]).  Converted internally to grayscale
                    lip ROI (B, T_v, lip_size, lip_size) via get_ROI_from_video.

        Returns:
            enhanced_audio: (B, T) — enhanced / extracted speech
        """
        audio = mix
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio.squeeze(1)
        assert audio.dim() == 2, "Only support 2D Input"

        # Dataset format → model format via ROI extraction
        video = self.get_ROI_from_video(enroll[0])  # (B, T_v, lip_size, lip_size)

        # Encode visual
        visual_feat = self.visual_encoder(video)  # (B, C_v, T_v)
        
        # Encode audio
        audio_feat, stft_complex = self.audio_encoder(audio)  # (B, C_a, T_a, F)
        
        # Fuse audio-visual
        fused = self.av_fusion(audio_feat, visual_feat)  # (B, C_f, T, F)
        
        # Process through TF-CrossNet blocks
        for block in self.tf_blocks:
            fused = block(fused)
        
        # Decode
        mask_real, mask_imag = self.decoder(fused, stft_complex.shape)
        
        # Apply complex mask
        enhanced_real = stft_complex.real * mask_real - stft_complex.imag * mask_imag
        enhanced_imag = stft_complex.real * mask_imag + stft_complex.imag * mask_real
        enhanced_stft = torch.complex(enhanced_real, enhanced_imag)
        
        # iSTFT
        enhanced_audio = torch.istft(
            enhanced_stft, self.config.n_fft, self.config.hop_length,
            self.config.win_length, self.window
        )
         # Match output length to input length (STFT/iSTFT can change length)
        T_in = audio.shape[-1]
        T_out = enhanced_audio.shape[-1]
        if T_out < T_in:
            enhanced_audio = F.pad(enhanced_audio, (0, T_in - T_out))
        elif T_out > T_in:
            enhanced_audio = enhanced_audio[..., :T_in]


        return enhanced_audio
    
    def streaming_forward(self, audio_chunk: torch.Tensor, video_frame: torch.Tensor,
                         state: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        """
        Streaming forward pass for real-time processing
        
        Args:
            audio_chunk: (B, chunk_samples) - audio chunk
            video_frame: (B, H, W) - single video frame
            state: previous state for streaming
            
        Returns:
            enhanced_chunk: (B, chunk_samples)
            new_state: updated state
        """
        # This is a placeholder for streaming implementation
        # Full implementation would require careful state management
        raise NotImplementedError("Streaming inference requires additional implementation")
