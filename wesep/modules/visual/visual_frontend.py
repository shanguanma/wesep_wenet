# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: wesep v2 network component.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.deep_update import deep_update
from wesep.modules.fusion.speech import SpeakerFuseLayer
from wesep.modules.visual.blazenet64 import visualNet
from wesep.modules.visual.muse_visual_frontend import (
    Muse_LipROIProcessor,
    Muse_VisualFrontend,
    Muse_load_model,
    Muse_VisualConv1D,
)
from wesep.modules.visual.resnet18_visual import ResNet18VisualEncoder


class BaseVisualFeature(nn.Module):

    def compute(self, video, mix=None):
        raise NotImplementedError

    def post(self, mix_repr, feat_repr):
        return mix_repr


class MuseVisualFeature(BaseVisualFeature):
    """Muse visual encoder with support for gradual unfreezing.

    Unfreeze levels (0-5) peel layers from the output end inward:

    ======  ===========================================================
    Level   Layers unfrozen (cumulative)
    ======  ===========================================================
    0       None — fully frozen (default, same as original behaviour)
    1       ``resnet.layer4`` + ``resnet.avgpool``
    2       + ``resnet.layer3``
    3       + ``resnet.layer2``
    4       + ``resnet.layer1``
    5       + ``frontend3D`` (everything trainable)
    ======  ===========================================================
    """

    _UNFREEZE_GROUPS: list[list[str]] = [
        ["visual_frontend.resnet.layer4", "visual_frontend.resnet.avgpool"],
        ["visual_frontend.resnet.layer3"],
        ["visual_frontend.resnet.layer2"],
        ["visual_frontend.resnet.layer1"],
        ["visual_frontend.frontend3D"],
    ]
    MAX_UNFREEZE_LEVEL: int = len(_UNFREEZE_GROUPS)  # 5

    def __init__(self, config):
        super().__init__()

        self.roi = Muse_LipROIProcessor()
        self.visual_frontend = Muse_VisualFrontend()

        if config["vf_pretrained"] is not None:
            self.visual_frontend = Muse_load_model(
                self.visual_frontend,
                config["vf_pretrained"],
            )
            for key, param in self.visual_frontend.named_parameters():
                param.requires_grad = False

        ve_blocks = []
        for x in range(config.get("vtcn_layers", 5)):
            ve_blocks += [Muse_VisualConv1D(channels=config["vtcn_channels"])]
        self.vtcn = nn.Sequential(*ve_blocks)

        self.upsample_to_audio = config.get("upsample", False)

        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=config["vtcn_channels"],
            feat_dim=config["mix_dim"],
            fuse_type=config["fusion"],
        )

        self._unfreeze_level: int = 0

    # ------------------------------------------------------------------
    # Gradual unfreezing API
    # ------------------------------------------------------------------

    @property
    def unfreeze_level(self) -> int:
        return self._unfreeze_level

    def set_unfreeze_level(self, level: int) -> list[nn.Parameter]:
        """Unfreeze visual-frontend layers up to *level*.

        Returns the list of **newly** unfrozen ``nn.Parameter`` objects so
        the caller can add them to the optimizer with a custom LR.
        """
        level = max(0, min(level, self.MAX_UNFREEZE_LEVEL))
        if level <= self._unfreeze_level:
            return []

        newly_unfrozen: list[nn.Parameter] = []
        for lvl in range(self._unfreeze_level, level):
            prefixes = self._UNFREEZE_GROUPS[lvl]
            for name, param in self.named_parameters():
                if any(name.startswith(pfx) for pfx in prefixes):
                    if not param.requires_grad:
                        param.requires_grad = True
                        newly_unfrozen.append(param)

        self._unfreeze_level = level
        self._sync_frozen_bn_eval()
        return newly_unfrozen

    def _sync_frozen_bn_eval(self) -> None:
        """Keep BatchNorm layers whose params are frozen in eval mode so
        they do not corrupt running statistics during training."""
        for module in self.visual_frontend.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                if all(not p.requires_grad for p in module.parameters()):
                    module.eval()

    def train(self, mode: bool = True):
        """Override so that frozen BN layers stay in eval mode."""
        super().train(mode)
        if mode:
            self._sync_frozen_bn_eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def compute(self, video, mix=None):
        """
        video: (B, H, W, 3, T_v)
        return:
            muse_visual: (B, 512, T_v)
        """

        feat = self.roi(video)  # (B, T, 112, 112)
        feat = feat.transpose(0, 1)
        feat = feat.unsqueeze(2)  # (T, B, 1, 112, 112)

        if self._unfreeze_level == 0:
            with torch.no_grad():
                feat = self.visual_frontend(feat)  # (B, 512, T_v)
        else:
            feat = self.visual_frontend(feat)

        feat = self.vtcn(feat)  # (B, 512, T_v)

        if self.upsample_to_audio and mix is not None:
            T_audio = mix.shape[-1]
            feat = F.interpolate(feat, size=T_audio,
                                 mode="linear")  # (B, 512, T)

        return feat

    def post(self, mix_repr, feat_repr):
        return self.fusionLayer(mix_repr, feat_repr)


class BlazeVisualFeature(BaseVisualFeature):
    """
    BlazeNet64 ``visualNet`` lip encoder (ClearerVoice-style), same video layout
    as ``MuseVisualFeature``: ``(B, H, W, 3, T_v)`` float in [0, 1].

    Expects square frames; if ``H != image_size``, frames are bilinearly resized
    to ``(image_size, image_size)`` before ``prepared_input`` crop logic.

    Upstream ``visualNet`` outputs ``(B, 224, T_v)`` for the back_model head.
    """

    BLAZE_EMBED_DIM = 224

    def __init__(self, config):
        super().__init__()
        self.image_size = int(config.get("image_size", 128))
        self.causal = bool(config.get("causal", True))
        self.upsample_to_audio = config.get("upsample", False)
        self.blaze = visualNet(causal=self.causal, image_size=self.image_size)
        embed_dim = int(config.get("embed_dim", self.BLAZE_EMBED_DIM))
        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=embed_dim,
            feat_dim=config["mix_dim"],
            fuse_type=config["fusion"],
        )

    def compute(self, video, mix=None):
        if video.shape[1] != self.image_size or video.shape[2] != self.image_size:
            b, _, _, c, t = video.shape
            h, w = video.shape[1], video.shape[2]
            v = video.permute(0, 4, 3, 1, 2).reshape(b * t, c, h, w).float()
            v = F.interpolate(
                v,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            video = (
                v.reshape(b, t, c, self.image_size, self.image_size)
                .permute(0, 3, 4, 2, 1)
                .contiguous()
            )
        feat = self.blaze(video)
        if self.upsample_to_audio and mix is not None:
            t_audio = mix.shape[-1]
            feat = F.interpolate(feat, size=t_audio, mode="linear")
        return feat

    def post(self, mix_repr, feat_repr):
        return self.fusionLayer(mix_repr, feat_repr)


class ResNet18VisualFeature(BaseVisualFeature):
    """ResNet18 lip-reading visual encoder (ClearerVoice-Studio style).

    Input: (B, H, W, 3, T_v) RGB video — auto-converted to grayscale 112x112 lip ROI.
    Output: (B, emb_size, T_v) visual embedding.

    Config keys:
        enabled: bool
        emb_size: int (default 256)
        causal: bool (default False)
        pretrained_path: str or null
        freeze_frontend: bool (default True)
        vtcn_layers: int (default 5)
        mix_dim: int (feature dim of audio for fusion)
        fusion: str ("concat" or "add")
    """

    def __init__(self, config):
        super().__init__()
        self.emb_size = int(config.get("emb_size", 256))
        causal = bool(config.get("causal", False))
        pretrained_path = config.get("pretrained_path", None)
        freeze_frontend = bool(config.get("freeze_frontend", True))
        vtcn_layers = int(config.get("vtcn_layers", 5))

        self.encoder = ResNet18VisualEncoder(
            emb_size=self.emb_size,
            causal=causal,
            pretrained_path=pretrained_path,
            freeze_frontend=freeze_frontend,
            vtcn_layers=vtcn_layers,
        )

        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=self.emb_size,
            feat_dim=config.get("mix_dim", 128),
            fuse_type=config.get("fusion", "concat"),
        )

    def compute(self, video, mix=None):
        """
        Args:
            video: (B, H, W, 3, T_v) RGB video OR (B, T_v, 112, 112) grayscale lip ROI

        Returns:
            (B, emb_size, T_v) visual embedding
        """
        if video.dim() == 5 and video.size(3) == 3:
            video = self._rgb_to_grayscale_lip(video)
        # video: (B, T_v, 112, 112)
        return self.encoder(video)

    def post(self, mix_repr, feat_repr):
        return self.fusionLayer(mix_repr, feat_repr)

    @staticmethod
    def _rgb_to_grayscale_lip(video):
        """Convert (B, H, W, 3, T_v) RGB to (B, T_v, 112, 112) grayscale."""
        B, H, W, C, T_v = video.shape
        # (B, H, W, 3, T_v) -> (B, T_v, 3, H, W)
        video = video.permute(0, 4, 3, 1, 2).contiguous()
        gray = 0.2989 * video[:, :, 0] + 0.5870 * video[:, :, 1] + 0.1140 * video[:, :, 2]
        if H != 112 or W != 112:
            gray = gray.view(B * T_v, 1, H, W)
            gray = F.interpolate(gray, size=(112, 112), mode='bilinear', align_corners=False)
            gray = gray.view(B, T_v, 112, 112)
        return gray


class VisualFrontend(nn.Module):

    def __init__(self, config):
        super().__init__()

        DEFAULT_CONFIG = {
            "features": {
                "muse_visual": {
                    "enabled": True,
                    "vf_pretrained": "./pretrain_networks/visual_frontend.pt",
                    "vtcn_channels": 512,
                    "vtcn_layers": 5,
                    "upsample": False,
                    "mix_dim": 128,
                    "fusion": "concat",
                },
                "blaze_visual": {
                    "enabled": False,
                    "causal": True,
                    "image_size": 128,
                    "embed_dim": 224,
                    "upsample": False,
                    "mix_dim": 128,
                    "fusion": "concat",
                },
                "resnet18_visual": {
                    "enabled": False,
                    "emb_size": 256,
                    "causal": False,
                    "pretrained_path": None,
                    "freeze_frontend": True,
                    "vtcn_layers": 5,
                    "mix_dim": 128,
                    "fusion": "concat",
                },
            }
        }

        self.config = deep_update(DEFAULT_CONFIG, config)
        feats = self.config["features"]

        muse_on = bool(feats.get("muse_visual", {}).get("enabled"))
        blaze_on = bool(feats.get("blaze_visual", {}).get("enabled"))
        resnet18_on = bool(feats.get("resnet18_visual", {}).get("enabled"))
        enabled_count = sum([muse_on, blaze_on, resnet18_on])
        if enabled_count > 1:
            raise ValueError(
                "Enable only one of visual.features: muse_visual, blaze_visual, "
                "or resnet18_visual (set the others to enabled: false)."
            )

        self.features = nn.ModuleDict()
        if muse_on:
            self.features["muse_visual"] = MuseVisualFeature(feats["muse_visual"])
        if blaze_on:
            self.features["blaze_visual"] = BlazeVisualFeature(feats["blaze_visual"])
        if resnet18_on:
            self.features["resnet18_visual"] = ResNet18VisualFeature(feats["resnet18_visual"])

    @property
    def muse_visual(self) -> Optional[MuseVisualFeature]:
        """Alias for ``self.features['muse_visual']`` (not a second registered child)."""
        if "muse_visual" in self.features:
            return self.features["muse_visual"]
        return None

    @property
    def blaze_visual(self) -> Optional[BlazeVisualFeature]:
        """Alias for ``self.features['blaze_visual']`` (not a second registered child)."""
        if "blaze_visual" in self.features:
            return self.features["blaze_visual"]
        return None

    def active_visual_encoder(self) -> Optional[BaseVisualFeature]:
        if "resnet18_visual" in self.features:
            return self.features["resnet18_visual"]
        if "blaze_visual" in self.features:
            return self.features["blaze_visual"]
        if "muse_visual" in self.features:
            return self.features["muse_visual"]
        return None

    def compute_all(self, enroll, mix=None):
        out = {}
        for name, module in self.features.items():
            out[name] = module.compute(enroll, mix)
        return out
