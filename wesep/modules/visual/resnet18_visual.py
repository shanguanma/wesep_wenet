"""ResNet18-based visual frontend for lip-reading target speaker extraction.

Ported from ClearerVoice-Studio:
https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction/models/visual_frontend/resnet18.py

Original: Copyright 2020 Smeet Shah, MIT License.

Input: grayscale lip ROI video (B, T, 112, 112)
Output: visual embedding (B, emb_size, T)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    """A ResNet layer with two residual blocks."""

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 1),
                                    stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    """An 18-layer ResNet architecture for visual feature extraction."""

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class ResNet18VisualFrontend(nn.Module):
    """3D conv frontend + ResNet18 backbone for lip video.

    Input: (B, 1, T, 112, 112) grayscale lip video
    Output: (B, 512, T)
    """

    def __init__(self, causal=False):
        super(ResNet18VisualFrontend, self).__init__()
        self.causal = causal
        if self.causal:
            padding = (4, 3, 3)
        else:
            padding = (2, 3, 3)

        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                      padding=padding, bias=False),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet = ResNet()

    def forward(self, batch):
        """
        Args:
            batch: (B, 1, T, 112, 112) grayscale video
        Returns:
            (B, 512, T)
        """
        batchsize = batch.shape[0]

        batch = self.frontend3D[0](batch)
        if self.causal:
            batch = batch[:, :, :-4, :, :]
        batch = self.frontend3D[1](batch)
        batch = self.frontend3D[2](batch)
        batch = self.frontend3D[3](batch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(
            batch.shape[0] * batch.shape[1],
            batch.shape[2], batch.shape[3], batch.shape[4]
        )
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1, 2)
        return outputBatch


class VisualConv1D(nn.Module):
    """1D temporal convolution block for visual feature refinement."""

    def __init__(self, V=256, H=512, kernel_size=3, dilation=1, causal=False):
        super(VisualConv1D, self).__init__()
        self.causal = causal

        self.relu_0 = nn.ReLU()
        self.norm_0 = nn.BatchNorm1d(V)
        self.conv1x1 = nn.Conv1d(V, H, 1, bias=False)
        self.relu = nn.ReLU()
        self.norm_1 = nn.BatchNorm1d(H)
        self.dconv_pad = ((dilation * (kernel_size - 1)) // 2 if not self.causal
                          else (dilation * (kernel_size - 1)))
        self.dsconv = nn.Conv1d(H, H, kernel_size, stride=1,
                                padding=self.dconv_pad, dilation=1, groups=H)
        self.prelu = nn.PReLU()
        self.norm_2 = nn.BatchNorm1d(H)
        self.pw_conv = nn.Conv1d(H, V, 1, bias=False)

    def forward(self, x):
        out = self.relu_0(x)
        out = self.norm_0(out)
        out = self.conv1x1(out)
        out = self.relu(out)
        out = self.norm_1(out)
        out = self.dsconv(out)
        if self.causal:
            out = out[:, :, :-self.dconv_pad]
        out = self.prelu(out)
        out = self.norm_2(out)
        out = self.pw_conv(out)
        return out + x


class ResNet18VisualEncoder(nn.Module):
    """Complete ResNet18 visual encoder for TSE.

    Combines:
    1. ResNet18VisualFrontend (3D conv + ResNet18): (B, 1, T, 112, 112) -> (B, 512, T)
    2. 1x1 conv downproject: (B, 512, T) -> (B, emb_size, T)
    3. Visual TCN adaptor (5 layers): temporal refinement

    Input: grayscale lip ROI (B, T, 112, 112) or (B, 1, T, 112, 112)
    Output: (B, emb_size, T) visual embedding
    """

    def __init__(self, emb_size=256, causal=False, pretrained_path=None,
                 freeze_frontend=True, vtcn_layers=5):
        super(ResNet18VisualEncoder, self).__init__()
        self.causal = causal
        self.emb_size = emb_size

        self.v_frontend = ResNet18VisualFrontend(causal=causal)
        self.v_ds = nn.Conv1d(512, emb_size, 1, bias=False)

        if pretrained_path is not None and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path, freeze_frontend)
        elif not causal and freeze_frontend:
            self._try_download_pretrained(freeze_frontend)

        stacks = []
        for _ in range(vtcn_layers):
            stacks.append(VisualConv1D(V=emb_size, H=512, causal=causal))
        self.visual_conv = nn.Sequential(*stacks)

    def _load_pretrained(self, path, freeze=True):
        """Load pretrained weights for the visual frontend."""
        pretrained_model = torch.load(path, map_location='cpu')
        self.v_frontend.load_state_dict(pretrained_model, strict=False)
        if freeze:
            for param in self.v_frontend.parameters():
                param.requires_grad = False

    def _try_download_pretrained(self, freeze=True):
        """Try to download pretrained resnet18 lip-reading model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            lip_resnet18_path = hf_hub_download(
                repo_id="alibabasglab/lip_reading_resnet18",
                filename="resnet18.pth"
            )
            self._load_pretrained(lip_resnet18_path, freeze)
        except Exception:
            pass

    def forward(self, visual):
        """
        Args:
            visual: (B, T, 112, 112) or (B, 1, T, 112, 112) grayscale lip ROI

        Returns:
            (B, emb_size, T) visual embedding
        """
        if visual.dim() == 4:
            visual = visual.unsqueeze(1)  # (B, 1, T, 112, 112)
        visual = self.v_frontend(visual)  # (B, 512, T)
        visual = self.v_ds(visual)  # (B, emb_size, T)

        if not self.causal:
            visual = self.visual_conv(visual)  # (B, emb_size, T)
        return visual
