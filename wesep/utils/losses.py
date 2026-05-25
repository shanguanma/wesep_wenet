import auraloss
from typing import Optional
import torch
import torch.nn as nn
import torchmetrics.audio as audio_metrics
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
# ============================================================================
# Loss Functions
# ============================================================================

class SISDRLoss(nn.Module):
    """Scale-Invariant Signal-to-Distortion Ratio Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est: (B, samples) - estimated signal
            ref: (B, samples) - reference signal
        Returns:
            loss: scalar
        """
        # Zero-mean
        est = est - est.mean(dim=-1, keepdim=True)
        ref = ref - ref.mean(dim=-1, keepdim=True)

        # SI-SDR
        dot = (est * ref).sum(dim=-1, keepdim=True)
        s_ref = (ref ** 2).sum(dim=-1, keepdim=True)
        proj = dot * ref / (s_ref + 1e-8)

        noise = est - proj

        si_sdr = 10 * torch.log10(
            (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + 1e-8) + 1e-8
        )

        return -si_sdr.mean()


class HybridContinuityLoss(nn.Module):
    """
    Hybrid Continuity Loss for reducing over-suppression
    Based on: "A hybrid continuity loss to reduce over-suppression for 
    time-domain target speaker extraction" (Interspeech 2022)
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.si_sdr = SISDRLoss()

    def forward(self, est: torch.Tensor, ref: torch.Tensor,
                mix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est: estimated signal
            ref: reference signal
            mix: mixture signal
        Returns:
            loss: hybrid loss
        """
        # SI-SDR loss
        loss_sdr = self.si_sdr(est, ref)

        # Continuity loss (penalize difference from mixture when reference is silent)
        ref_energy = (ref ** 2).sum(dim=-1, keepdim=True)
        weight = torch.sigmoid(-ref_energy / (ref_energy.mean() + 1e-8))
        loss_cont = (weight * (est - mix) ** 2).mean()

        return loss_sdr + self.alpha * loss_cont


class OnlineAVCrossNetLoss(nn.Module):
    """Combined loss for Online AV-CrossNet training"""
    def __init__(self, use_continuity: bool = True, alpha: float = 0.5):
        super().__init__()
        if use_continuity:
            self.loss_fn = HybridContinuityLoss(alpha)
        else:
            self.loss_fn = SISDRLoss()
        self.use_continuity = use_continuity

    def forward(self, est: torch.Tensor, ref: torch.Tensor,
                mix: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_continuity and mix is not None:
            return self.loss_fn(est, ref, mix)
        else:
            return self.loss_fn.si_sdr(est, ref) if self.use_continuity else self.loss_fn(est, ref)


class SISDR_MRSTFTLoss(nn.Module):
    """
    Hybrid SI-SDR + Multi-resolution STFT loss for stage 98+.

    L = sisdr_weight * auraloss.time.SISDRLoss(est, ref).mean()
        + mrstft_weight * auraloss.freq.MultiResolutionSTFTLoss(est, ref).mean()

    Rationale: pure SI-SDR plateaus quickly (~34.36 in stage 78 by epoch 4.5)
    because gradient information at low SDR regimes is dominated by
    waveform-level scaling. Adding multi-resolution STFT magnitude loss
    reintroduces frequency-aware gradients and typically delivers an
    additional 0.5-1.5 dB SI-SDR on offline target speaker extraction.

    NOTE: This is an additive, opt-in loss; existing stages 78/88 continue
    to use ``SISDR`` and are unaffected.
    """

    def __init__(self, sisdr_weight: float = 1.0, mrstft_weight: float = 0.5):
        super().__init__()
        # Imported lazily to avoid surprising imports in test paths that don't use this loss.
        import auraloss  # noqa: WPS433
        self.sisdr_weight = float(sisdr_weight)
        self.mrstft_weight = float(mrstft_weight)
        self._sisdr = auraloss.time.SISDRLoss()
        self._mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # auraloss MultiResolutionSTFTLoss requires (B, C, T); SI-SDR is
        # shape-agnostic. TSE_BSRNN_VISUAL may return (B, T); normalize both
        # sides to 3-D before computing MRSTFT.
        if est.dim() == 2:
            est = est.unsqueeze(1)
        if ref.dim() == 2:
            ref = ref.unsqueeze(1)
        if est.shape[1] != ref.shape[1]:
            if est.shape[1] == 1:
                est = est.expand(-1, ref.shape[1], -1)
            elif ref.shape[1] == 1:
                ref = ref.expand(-1, est.shape[1], -1)
        sdr_term = self._sisdr(est, ref).mean()
        mag_term = self._mrstft(est, ref).mean()
        return self.sisdr_weight * sdr_term + self.mrstft_weight * mag_term



"""Get a loss function with its name from the configuration file."""
valid_losses = {}

torch_losses = {
    "L1": nn.L1Loss(),
    "L2": nn.MSELoss(),
    "CE": nn.CrossEntropyLoss(),
}

torchmetrics_losses = {
    # Not tested
    "PIT":
    audio_metrics.PermutationInvariantTraining(
        scale_invariant_signal_noise_ratio),
}

auraloss_losses = {
    "STFT": auraloss.freq.STFTLoss(),
    "MultiResolutionSTFT": auraloss.freq.MultiResolutionSTFTLoss(),
    "SISDR": auraloss.time.SISDRLoss(),
    "SISNR": auraloss.time.SISDRLoss(),
    "SNR": auraloss.time.SNRLoss(),
}
consumed_losses = {
    "OnlineAVCrossNetLoss": OnlineAVCrossNetLoss(),
    # Additive entry for stage 98+ (does not affect existing keys above).
    "SISDR_MRSTFT": SISDR_MRSTFTLoss(),
}

valid_losses.update(torch_losses)
valid_losses.update(auraloss_losses)
valid_losses.update(torchmetrics_losses) 
valid_losses.update(consumed_losses)


def parse_loss(loss):
    loss_functions = []
    if not isinstance(loss, list):
        loss = [loss]
    for i in range(len(loss)):
        loss_name = loss[i]
        loss_functions.append(valid_losses.get(loss_name))
    return loss_functions


