# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: wesep v2 network component.

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.common.deep_update import deep_update
from wesep.modules.visual.visual_frontend import VisualFrontend

# Old checkpoints: visual encoders lived under ``visual_ft.{muse,blaze}_visual``.
# Current ``VisualFrontend`` registers them as ``visual_ft.features.{muse,blaze}_visual``.
_VISUAL_CKPT_PREFIX_REMAPS = (
    ("module.visual_ft.muse_visual.", "module.visual_ft.features.muse_visual."),
    ("module.visual_ft.blaze_visual.", "module.visual_ft.features.blaze_visual."),
    ("visual_ft.muse_visual.", "visual_ft.features.muse_visual."),
    ("visual_ft.blaze_visual.", "visual_ft.features.blaze_visual."),
)


def _remap_visual_frontend_checkpoint_keys(state_dict):
    out = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _VISUAL_CKPT_PREFIX_REMAPS:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break
        out[new_key] = value
    return out


class TSE_BSRNN_VISUAL(nn.Module):
    # Transformers Trainer reads this when resuming (see Trainer._issue_warnings_after_load).
    _keys_to_ignore_on_save = None

    def __init__(self, config):
        super().__init__()

        # ===== Merge configs =====
        sep_configs = dict(
            sr=16000,
            win=512,
            stride=128,
            feature_dim=128,
            num_repeat=6,
            causal=False,
            nspk=1,  # For Separation (multiple output)
            spec_dim=2  # For TSE feature, used in self.subband_norm
        )
        sep_configs = {**sep_configs, **config["separator"]}
        visual_configs = {
            "features": {
                "muse_visual": {
                    "enabled": True,
                    "vf_pretrained": "./pretrain_networks/visual_frontend.pt",
                    "vtcn_channels": 512,
                    "vtcn_layers": 5,
                    "upsample": True,
                    "mix_dim": sep_configs["feature_dim"],
                    "fusion": "concat",
                },
                "blaze_visual": {
                    "enabled": False,
                    "causal": True,
                    "image_size": 128,
                    "embed_dim": 224,
                    "upsample": True,
                    "mix_dim": sep_configs["feature_dim"],
                    "fusion": "concat",
                },
            }
        }
        self.visual_configs = deep_update(visual_configs, config['visual'])
        # ===== Separator Loading =====
        self.sep_model = BSRNN(**sep_configs)

        # ===== Visual Loading =====
        self.visual_ft = VisualFrontend(self.visual_configs)

    def load_state_dict(self, state_dict, strict=True):
        remapped = _remap_visual_frontend_checkpoint_keys(dict(state_dict))
        return super().load_state_dict(remapped, strict=strict)

    def forward(self, mix, enroll):
        """
        Args:
            mix:  Tensor [B, 1, T]
            enroll: list[Tensor]
                each Tensor: [B, H, W, 3, T_image], if 1s video,its T_image is 25,T is 16000
        """

        if mix.dim() == 3 and mix.size(1) == 1:
            mix = mix.squeeze(1)
        assert mix.dim() == 2, "Only support 2D Input"

        wav_mix = mix
        visual_enroll = enroll[0]
        ###########################################################
        # S1. Convert into frequency-domain
        spec = self.sep_model.stft(wav_mix)[-1]  # (B, F, T) complex
        # S2. Concat real and imag
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # (B, 2, F, T)
        # Split to subbands
        subband_spec = self.sep_model.band_split(
            spec_RI)  # list of (B, spec_dim, BW, T)
        subband_mix_spec = self.sep_model.band_split(
            spec)  # list of (B, BW, T) complex
        # S3. Normalization and bottleneck
        subband_feature = self.sep_model.subband_norm(
            subband_spec)  # (B, nband, feat, T)
        ###########################################################
        # V1. Visual cue (Muse or BlazeNet64 via ``VisualFrontend.active_visual_encoder``)
        visual_enc = self.visual_ft.active_visual_encoder()
        if visual_enc is not None:
            visual_feat = visual_enc.compute(visual_enroll, mix=subband_feature)
            visual_feat = visual_feat.unsqueeze(1)  # (B, 1, F_v, T)
            subband_feature = visual_enc.post(subband_feature, visual_feat)
        ###########################################################
        # S4. Separation
        sep_output = self.sep_model.separator(
            subband_feature)  # (B, nband, feat, T)
        # S5. Complex Mask
        est_spec_RI = self.sep_model.band_masker(
            sep_output, subband_mix_spec)  # (B, 2, S, F, T)
        est_complex = torch.complex(est_spec_RI[:, 0],
                                    est_spec_RI[:, 1])  # (B, S, F, T)
        # S6. Back into waveform
        output = self.sep_model.istft(est_complex)  # (B, S, T)
        s = torch.squeeze(output, dim=1)  # (B, T)
        return s


def check_causal(model):
    fs = 16000
    fs_v = 25
    input = torch.randn(1, fs * 8).clamp_(-1, 1)
    visual = torch.randn(1, 244, 244, 3, fs_v * 8).clamp_(-1, 1)
    model = model.eval()
    with torch.no_grad():
        out1 = model(input, enroll=[visual])
        for t in range(1, 4, 1):
            inputs2 = input.clone()
            i_m = int(t * fs)
            inputs2[..., i_m:] = 1 + torch.rand_like(inputs2[..., i_m:])
            visual2 = visual.clone()
            i_v = int(t * fs_v)
            visual2[..., i_v:] = 1 + torch.rand_like(visual2[..., i_v:])
            out2 = model(inputs2, enroll=[visual2])
            print((((out1[0] - out2[0]).abs() > 1e-8).float().argmax()) / fs)
            print((((inputs2 - input).abs() > 1e-8).float().argmax()) / fs)


if __name__ == "__main__":
    config = dict()
    config["separator"] = dict(
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        causal=True,
        nspk=1,
    )
    config["visual"] = {
        "features": {
            "muse_visual": {
                "enabled": True,
                "vf_pretrained":
                None,  # "./pretrain_networks/visual_frontend.pt",
                "vtcn_channels": 512,
                "vtcn_layers": 5,
                "upsample": True,
                "mix_dim": 128,
                "fusion": "concat",
            }
        }
    }

    model = TSE_BSRNN_VISUAL(config)
    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print("# of parameters: " + str(s / 1024.0 / 1024.0))

    mix = torch.randn(4, 32000)
    visual_feat = torch.randn(4, 224, 224, 3, 75)  # (B, H, W, 3, T_v)

    model = model.eval()
    with torch.no_grad():
        out = model(mix, enroll=[visual_feat])
        print("output shape: ", out.shape)

    check_causal(model)
