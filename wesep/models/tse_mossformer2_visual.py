"""Audio-Visual MossFormer2 for Target Speaker Extraction.

Ported from ClearerVoice-Studio av_mossformer2:
https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction/models/av_mossformer2

Adapted to wesep interface: forward(mix, enroll) compatible with online mix datasets.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.mossformer2 import Dual_Path_Model, SBFLASHBlock_DualA
from wesep.modules.visual.visual_frontend import VisualFrontend
from wesep.modules.common.deep_update import deep_update

EPS = 1e-8

# Old checkpoints: visual encoders lived under ``visual_ft.{muse,blaze}_visual``.
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


def overlap_and_add(signal, frame_step):
    """Reconstruct a signal from framed representation via overlap-add.

    Args:
        signal: [..., frames, frame_length] tensor
        frame_step: hop size between frames

    Returns:
        [..., output_size] tensor where output_size = (frames-1)*frame_step + frame_length
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long().to(signal.device)
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class Encoder(nn.Module):
    """1D convolutional encoder for waveform-to-latent representation."""

    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: (B, T) waveform
        Returns:
            mixture_w: (B, N, K) encoded representation
        """
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.conv1d_U(mixture))
        return mixture_w


class Decoder(nn.Module):
    """Linear decoder with overlap-add reconstruction."""

    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: (B, N, K) encoded mixture
            est_mask: (B, N, K) estimated mask (squeezed from separator)
        Returns:
            est_source: (B, T) reconstructed waveform
        """
        est_source = mixture_w * est_mask
        est_source = torch.transpose(est_source, 2, 1)
        est_source = self.basis_signals(est_source)
        est_source = overlap_and_add(est_source, self.L // 2)
        return est_source


class Separator(nn.Module):
    """MossFormer2 separator with audio-visual fusion."""

    def __init__(self, encoder_out_nchannels, visual_emb_size,
                 intra_numlayers=4, intra_nhead=8, intra_dffn=1024,
                 intra_dropout=0.1, intra_use_positional=False,
                 intra_norm_before=True,
                 masknet_numlayers=1, masknet_norm="ln",
                 masknet_chunksize=200, masknet_numspks=1,
                 masknet_extraskipconnection=True,
                 masknet_useextralinearlayer=True):
        super(Separator, self).__init__()

        self.layer_norm = nn.GroupNorm(1, encoder_out_nchannels, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(
            encoder_out_nchannels, encoder_out_nchannels, 1, bias=False
        )

        intra_model = SBFLASHBlock_DualA(
            num_layers=intra_numlayers,
            d_model=encoder_out_nchannels,
            nhead=intra_nhead,
            d_ffn=intra_dffn,
            dropout=intra_dropout,
            use_positional_encoding=intra_use_positional,
            norm_before=intra_norm_before,
        )

        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            num_layers=masknet_numlayers,
            norm=masknet_norm,
            K=masknet_chunksize,
            num_spks=masknet_numspks,
            skip_around_intra=masknet_extraskipconnection,
            linear_layer_after_inter_intra=masknet_useextralinearlayer,
        )

        # Audio-visual fusion: concatenate visual and audio features, project back
        self.av_conv = nn.Conv1d(
            encoder_out_nchannels + visual_emb_size,
            encoder_out_nchannels, 1, bias=True
        )

    def forward(self, x, visual):
        """
        Args:
            x: (B, N, D) encoded audio
            visual: (B, V, T_v) visual embedding (will be interpolated)
        Returns:
            est_mask: (B, N, D) estimated mask
        """
        M, N, D = x.size()

        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        # Interpolate visual features to match audio time resolution
        visual = F.interpolate(visual, size=(D,), mode='linear')
        x = torch.cat((x, visual), 1)
        x = self.av_conv(x)

        x = self.masknet(x)
        x = x.squeeze(0)  # Remove spks dim (num_spks=1)

        return x


class TSE_MOSSFORMER2_VISUAL(nn.Module):
    """Audio-Visual MossFormer2 for Target Speaker Extraction.

    Compatible with wesep online mix dataset interface:
        forward(mix, enroll) where:
            mix: (B, 1, T) or (B, T) mixed audio waveform
            enroll: list[Tensor] where enroll[0] is (B, H, W, 3, T_image) video
    """

    _keys_to_ignore_on_save = None

    def __init__(self, config):
        super().__init__()

        # ===== Merge configs with defaults =====
        sep_configs = dict(
            encoder_out_nchannels=256,
            encoder_kernel_size=16,
            intra_numlayers=4,
            intra_nhead=8,
            intra_dffn=1024,
            intra_dropout=0.1,
            intra_use_positional=False,
            intra_norm_before=True,
            masknet_numlayers=1,
            masknet_norm="ln",
            masknet_chunksize=200,
            masknet_numspks=1,
            masknet_extraskipconnection=True,
            masknet_useextralinearlayer=True,
        )
        sep_configs.update(config.get("separator", {}))

        # ===== Visual config (unified: muse_visual / blaze_visual / resnet18_visual) =====
        visual_configs = {
            "features": {
                "muse_visual": {
                    "enabled": False,
                    "vf_pretrained": "./pretrain_networks/visual_frontend.pt",
                    "vtcn_channels": 512,
                    "vtcn_layers": 5,
                    "upsample": True,
                    "mix_dim": sep_configs["encoder_out_nchannels"],
                    "fusion": "concat",
                },
                "blaze_visual": {
                    "enabled": False,
                    "causal": True,
                    "image_size": 128,
                    "embed_dim": 224,
                    "upsample": True,
                    "mix_dim": sep_configs["encoder_out_nchannels"],
                    "fusion": "concat",
                },
                "resnet18_visual": {
                    "enabled": True,
                    "emb_size": 256,
                    "causal": False,
                    "pretrained_path": None,
                    "freeze_frontend": True,
                    "vtcn_layers": 5,
                    "mix_dim": sep_configs["encoder_out_nchannels"],
                    "fusion": "concat",
                },
            }
        }
        self.visual_configs = deep_update(visual_configs, config.get('visual', {}))

        visual_emb_size = self._get_visual_emb_size()

        N = sep_configs["encoder_out_nchannels"]
        L = sep_configs["encoder_kernel_size"]

        # ===== Build sub-modules =====
        self.encoder = Encoder(L, N)
        self.separator = Separator(
            encoder_out_nchannels=N,
            visual_emb_size=visual_emb_size,
            intra_numlayers=sep_configs["intra_numlayers"],
            intra_nhead=sep_configs["intra_nhead"],
            intra_dffn=sep_configs["intra_dffn"],
            intra_dropout=sep_configs["intra_dropout"],
            intra_use_positional=sep_configs["intra_use_positional"],
            intra_norm_before=sep_configs["intra_norm_before"],
            masknet_numlayers=sep_configs["masknet_numlayers"],
            masknet_norm=sep_configs["masknet_norm"],
            masknet_chunksize=sep_configs["masknet_chunksize"],
            masknet_numspks=sep_configs["masknet_numspks"],
            masknet_extraskipconnection=sep_configs["masknet_extraskipconnection"],
            masknet_useextralinearlayer=sep_configs["masknet_useextralinearlayer"],
        )
        self.decoder = Decoder(N, L)

        # ===== Visual frontend (unified) =====
        self.visual_ft = VisualFrontend(self.visual_configs)

        # Xavier initialization (skip frozen pretrained params)
        for name, p in self.named_parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_normal_(p)

    def _get_visual_emb_size(self):
        """Determine visual embedding dimension from the enabled visual encoder config."""
        features = self.visual_configs.get("features", {})
        if features.get("resnet18_visual", {}).get("enabled", False):
            return features["resnet18_visual"].get("emb_size", 256)
        elif features.get("muse_visual", {}).get("enabled", False):
            return features["muse_visual"].get("vtcn_channels", 512)
        elif features.get("blaze_visual", {}).get("enabled", False):
            return features["blaze_visual"].get("embed_dim", 224)
        return 256

    def load_state_dict(self, state_dict, strict=True):
        remapped = _remap_visual_frontend_checkpoint_keys(dict(state_dict))
        return super().load_state_dict(remapped, strict=strict)

    def forward(self, mix, enroll):
        """
        Args:
            mix: (B, 1, T) or (B, T) mixed audio waveform
            enroll: list[Tensor]
                enroll[0]: (B, H, W, 3, T_image) video tensor from online mix dataset
                    All visual backends (muse/blaze/resnet18) handle this format.

        Returns:
            s: (B, T) extracted target speaker waveform
        """
        if mix.dim() == 3 and mix.size(1) == 1:
            mix = mix.squeeze(1)
        assert mix.dim() == 2, "Only support 2D input (B, T)"

        visual_enroll = enroll[0]

        # Encode mixture audio
        mixture_w = self.encoder(mix)  # (B, N, K)

        # Extract visual features via the unified VisualFrontend
        visual_enc = self.visual_ft.active_visual_encoder()
        if visual_enc is not None:
            visual_feat = visual_enc.compute(visual_enroll)
            if visual_feat.dim() == 2:
                visual_feat = visual_feat.unsqueeze(-1)
            if visual_feat.dim() == 3 and visual_feat.size(1) != self._get_visual_emb_size():
                visual_feat = visual_feat.transpose(1, 2)
        else:
            B = mix.size(0)
            visual_feat = torch.zeros(
                B, self._get_visual_emb_size(), 1,
                device=mix.device, dtype=mix.dtype
            )

        # Separate with AV fusion
        est_mask = self.separator(mixture_w, visual_feat)  # (B, N, K)

        # Decode to waveform
        est_source = self.decoder(mixture_w, est_mask)

        # Fix length mismatch from conv1d encoder
        T_origin = mix.size(-1)
        T_conv = est_source.size(-1)
        if T_conv < T_origin:
            est_source = F.pad(est_source, (0, T_origin - T_conv))
        elif T_conv > T_origin:
            est_source = est_source[..., :T_origin]

        return est_source


if __name__ == "__main__":
    import numpy as np

    config = dict()
    config["separator"] = dict(
        encoder_out_nchannels=256,
        encoder_kernel_size=16,
        intra_numlayers=4,
        intra_nhead=8,
        intra_dffn=1024,
        intra_dropout=0.1,
        intra_use_positional=False,
        intra_norm_before=True,
        masknet_numlayers=1,
        masknet_norm="ln",
        masknet_chunksize=200,
        masknet_numspks=1,
        masknet_extraskipconnection=True,
        masknet_useextralinearlayer=True,
    )
    config["visual"] = {
        "features": {
            "muse_visual": {
                "enabled": True,
                "vf_pretrained": None,
                "vtcn_channels": 512,
                "vtcn_layers": 5,
                "upsample": True,
                "mix_dim": 256,
                "fusion": "concat",
            }
        }
    }

    model = TSE_MOSSFORMER2_VISUAL(config)
    s = sum(np.prod(p.size()) for p in model.parameters())
    print("# of parameters: {:.2f}M".format(s / 1024.0 / 1024.0))

    mix = torch.randn(2, 48000)
    visual_feat = torch.randn(2, 224, 224, 3, 75)  # (B, H, W, 3, T_v)

    model = model.eval()
    with torch.no_grad():
        out = model(mix, enroll=[visual_feat])
        print("output shape:", out.shape)
