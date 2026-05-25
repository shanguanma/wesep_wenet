# Copyright (c) 2026 Duo Ma (maduo@cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-corpus online training: **train and validation** use
:class:`wesep.dataset.online_multi_dataset.OnlineMixIterableDataset` with a
**merged** MP4 inventory (VoxCeleb2 / LRS3 / Chinese_lips).

- **VoxCeleb2 & LRS3:** no official dev split — speakers are partitioned by
  ``--train_speaker_fraction`` (same mechanism as ``split_speaker_ids_train_val``).
- **Chinese_lips:** uses the dataset's own ``train`` / ``val`` directory layout;
  no random speaker split.

Train inventory = merge(all training-side sources); val inventory = merge(all
dev-side sources). Optional ``--num_speaker_pool`` subsamples within each side.

Validation mixture count per epoch defaults to ``train * 5000/20000`` unless set
via ``--sample_num_per_epoch_val``.

**YAML:** ``--model_config`` model-only (``model`` + ``model_args``).

Optional overrides: ``--visual_frontend muse|blaze`` (mutually exclusive
``muse_visual.enabled`` / ``blaze_visual.enabled``), ``--blaze_visual_causal true|false``,
``--separator_causal true|false`` (``model_args.tse_model.separator.causal``).

Requires ``--sample_num_per_epoch`` > 0 (default 20000). Enable sources with
``--use_voxceleb2``, ``--use_lrs3``, ``--use_chinese_lips`` and set roots.
"""

import argparse
import auraloss
import faulthandler
import logging
import math
import os
import random
import re
from functools import partial
from packaging import version
from typing import Any, Optional
from collections import defaultdict
faulthandler.enable()
from contextlib import nullcontext
from pprint import pformat
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from wesep.dataset.collate import (
    BASE_COLLECT_KEYS,
    build_collect_keys,
    build_collect_keys_online,
    tse_collate_fn,
)
from wesep.dataset.online_multi_dataset import (
    DEFAULT_CHINESE_LIPS_ROOT,
    DEFAULT_LRS3_ROOT,
    OnlineMixIterableDataset,
    build_dataset_args_from_tse_online_data_args,
    build_train_val_merged_audio_visual_inventories,
    default_online_val_samples_per_epoch,
    ensure_online_pipeline_defaults,
    resolve_speaker_pool,
    subset_inventory,
)
from wesep.models import get_model
from wesep.utils.checkpoint import (
    load_checkpoint,
    load_pretrained_model,
    save_checkpoint,
)
from wesep.utils.funcs import clip_gradients
from wesep.utils.losses import parse_loss
from wesep.utils.utils import set_seed, setup_logger
from wesep.utils.file_utils import load_yaml
from wesep.utils.executor import Executor
import transformers
from transformers import (
    get_scheduler,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    HfArgumentParser,
)

#MAX_NUM_log_files = 100
#logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
#logger = logging.get_logger(__name__)
from dataclasses import dataclass, field
import os
import logging
from datetime import datetime
from transformers import TrainerCallback

# 配置根日志记录器，让所有 print 都通过 logging 模块输出，以便统一控制格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class CustomLoggingCallback(TrainerCallback):
    """自定义回调，在日志中添加时间戳和文件名"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 获取当前执行的文件名
            current_file = os.path.basename(__file__)  # 获取当前脚本文件名
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 格式化输出，包含文件名、时间戳和日志内容
            log_message = f"[{current_file}] [{timestamp}] {logs}"
            logger.info(log_message)  # 使用配置好的 logger 输出


def load_yaml_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def normalize_dataloader_args(da: dict) -> None:
    """Drop keys invalid when num_workers == 0."""
    if int(da.get("num_workers", 0)) <= 0:
        da.pop("prefetch_factor", None)
        da.pop("persistent_workers", None)
        da.pop("multiprocessing_context", None)


def _parse_lrs3_subsets(s: str) -> tuple[str, ...]:
    """Comma-separated LRS3 subset folder names, e.g. ``trainval`` or ``trainval,test``."""
    t = tuple(x.strip() for x in str(s).split(",") if x.strip())
    return t if t else ("trainval",)


def _value_shape_for_log(v):
    """Shape summary for collated batch values (tensors, lists of tensors, or other)."""
    if torch.is_tensor(v):
        return tuple(v.shape)
    if isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
        return [tuple(t.shape) for t in v]
    if isinstance(v, (list, tuple)):
        return f"<len {len(v)}>"
    return type(v).__name__


def _format_collated_batch_shapes(batch: dict) -> str:
    lines = [f"  {k}: {_value_shape_for_log(batch[k])}" for k in sorted(batch.keys())]
    return "First train batch (collated) shapes:\n" + "\n".join(lines)


# ------------------------------------------------------------------
# ClearerVoice-Studio TSE-online LR (warmup + optional plateau halving)
# ------------------------------------------------------------------
# Ref: modelscope/ClearerVoice-Studio train/target_speaker_extraction_online/solver.py
# Warmup: lr = init_learning_rate / 0.001 * (64 ** (-0.5)) * step_num * (warmup_steps ** (-1.5))
# for step_num in [1, warmup_steps), then hold at the last warmup value (upstream stops updating).


def clearervoice_tse_online_lr_multiplier_at_internal_step(
    internal_step: int, warmup_steps: int
) -> float:
    """
    Multiplier relative to ``--learning_rate`` (ClearerVoice ``init_learning_rate``).

    ``internal_step`` matches upstream ``step_num`` **before** ``step_num += 1`` in
    ``_adjust_lr_warmup`` (1-based while ``step_num < warmup_steps``).
    """
    if warmup_steps <= 0:
        return 1.0
    scale = (64 ** (-0.5)) / 0.001
    if internal_step < warmup_steps:
        return float(scale * internal_step / (warmup_steps**1.5))
    return float(scale * (warmup_steps - 1) / (warmup_steps**1.5))


def _clearervoice_lambda_lr_mult(warmup_steps: int):
    """``LambdaLR`` factor: ``last_epoch`` is HF/PyTorch scheduler state (0 after init step)."""

    def lr_lambda(last_epoch: int) -> float:
        # After LRScheduler init, last_epoch is 0 and multiplier must match internal step 1.
        s = last_epoch + 1
        return clearervoice_tse_online_lr_multiplier_at_internal_step(s, warmup_steps)

    return lr_lambda


class ClearerVoicePlateauHalvingCallback(TrainerCallback):
    """
    When ``eval_loss`` fails to improve for ``patience_evals`` consecutive evaluations,
    multiply all optimizer param-group LRs by 0.5 (ClearerVoice ``Solver.train`` halving).

    Upstream also reloads ``last_best_checkpoint.pt`` before halving; that is not done here
    to avoid brittle HF/DDP state — enable ``--load_best_model_at_end`` or resume manually
    from a saved best checkpoint if you need weight rollback.
    """

    def __init__(self, patience_evals: int, min_delta: float):
        self.patience_evals = int(patience_evals)
        self.min_delta = float(min_delta)
        self._bad = 0
        self._best = float("inf")
        self._trainer = None

    def attach_trainer(self, trainer: "TSETrainer") -> None:
        self._trainer = trainer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._trainer is None or not metrics or "eval_loss" not in metrics:
            return control
        loss = float(metrics["eval_loss"])
        if loss < self._best - self.min_delta:
            self._best = loss
            self._bad = 0
            return control
        self._bad += 1
        if self._bad < self.patience_evals:
            return control
        self._bad = 0
        opt = self._trainer.optimizer
        if opt is None:
            return control
        new_lrs: list[float] = []
        for g in opt.param_groups:
            g["lr"] = float(g["lr"]) * 0.5
            new_lrs.append(float(g["lr"]))
        if self._trainer.is_world_process_zero():
            logger.info(
                "ClearerVoice-style plateau: halved LR after %s evals without "
                "eval_loss improvement (min_delta=%s). New LRs (first group): %s",
                self.patience_evals,
                self.min_delta,
                new_lrs[:4],
            )
        return control


class EMACallback(TrainerCallback):
    """
    Exponential Moving Average of model weights, saved alongside primary
    checkpoints into ``<output_dir>/ema_model.pt`` (overwritten at each
    ``save_steps`` event) and ``<output_dir>/ema_final.pt`` at train end.

    Notes
    -----
    * The shadow lives on CPU to keep VRAM unchanged.
    * Only floating-point parameters are EMA-updated; integer buffers are
      copied as-is.
    * Resume from EMA snapshot is **not** automatic; for stage 98 we run
      from scratch so this is fine. To inspect the EMA model later, load
      ``ema_model.pt`` (or ``ema_final.pt``) into a fresh model instance.
    * Single-GPU / DDP both supported via ``accelerator.unwrap_model``.
    """

    def __init__(self, decay: float, save_filename: str = "ema_model.pt"):
        self.decay = float(decay)
        self.save_filename = save_filename
        self._shadow: dict[str, torch.Tensor] | None = None
        self._trainer = None

    def attach_trainer(self, trainer: "TSETrainer") -> None:
        self._trainer = trainer

    def _unwrap_model(self):
        m = self._trainer.model
        try:
            return self._trainer.accelerator.unwrap_model(m)
        except Exception:
            inner = getattr(m, "module", m)
            return getattr(inner, "model", inner)

    def on_train_begin(self, args, state, control, **kwargs):
        if self._trainer is None:
            return control
        m_inner = self._unwrap_model()
        self._shadow = {
            k: v.detach().clone().cpu()
            for k, v in m_inner.state_dict().items()
        }
        if self._trainer.is_world_process_zero():
            n_float = sum(1 for v in self._shadow.values() if v.dtype.is_floating_point)
            logger.info(
                "EMA: initialized shadow weights (decay=%.4f, %d float tensors).",
                self.decay,
                n_float,
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._shadow is None or self._trainer is None:
            return control
        m_inner = self._unwrap_model()
        d = self.decay
        with torch.no_grad():
            for k, v in m_inner.state_dict().items():
                s = self._shadow.get(k)
                if s is None:
                    continue
                if v.dtype.is_floating_point:
                    s.mul_(d).add_(v.detach().to(device=s.device, dtype=s.dtype),
                                   alpha=1.0 - d)
                else:
                    s.copy_(v.detach().to(device=s.device))
        return control

    def _save(self, args, filename: str) -> None:
        if self._shadow is None:
            return
        if self._trainer is not None and not self._trainer.is_world_process_zero():
            return
        try:
            out = pathlib.Path(args.output_dir) / filename
            torch.save(self._shadow, out)
            logger.info("EMA: saved shadow weights to %s", out)
        except Exception as e:
            logger.warning("EMA save to %s failed: %s", filename, e)

    def on_save(self, args, state, control, **kwargs):
        self._save(args, self.save_filename)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._save(args, "ema_final.pt")
        return control


class VisualUnfreezeCallback(TrainerCallback):
    """Gradually unfreeze visual-frontend layers based on eval metrics or step count.

    Works with :class:`MuseVisualFeature.set_unfreeze_level`.  When the
    trigger condition is met the callback:

    1. Calls ``set_unfreeze_level(next_level)`` on the active visual encoder.
    2. Adds the newly unfrozen parameters to the optimizer as a separate
       param-group with ``lr = base_lr * lr_scale``.

    Supported modes:

    * **metric** — on each ``on_evaluate``, compare *metric* against
      *thresholds* (one per level).  Threshold *i* gates level *i+1*.
    * **step** — on each ``on_step_begin``, compare ``global_step`` against
      *steps* (one per level).
    """

    def __init__(
        self,
        mode: str,
        metric: str,
        thresholds: list[float],
        steps: list[int],
        lr_scale: float,
    ):
        self.mode = mode
        self.metric = metric
        self.thresholds = thresholds
        self.steps = steps
        self.lr_scale = float(lr_scale)
        self._trainer: "TSETrainer | None" = None
        self._current_level: int = 0

    def attach_trainer(self, trainer: "TSETrainer") -> None:
        self._trainer = trainer

    # ------------------------------------------------------------------

    def _get_visual_encoder(self):
        from wesep.modules.visual.visual_frontend import MuseVisualFeature

        if self._trainer is None:
            return None
        model = self._trainer.model
        inner = getattr(model, "module", model)
        vis_enc = inner.visual_ft.active_visual_encoder()
        if isinstance(vis_enc, MuseVisualFeature):
            return vis_enc
        return None

    def _try_unfreeze(self, target_level: int) -> None:
        if target_level <= self._current_level:
            return
        vis_enc = self._get_visual_encoder()
        if vis_enc is None:
            return

        newly_unfrozen = vis_enc.set_unfreeze_level(target_level)
        if not newly_unfrozen:
            self._current_level = target_level
            return

        opt = self._trainer.optimizer
        if opt is not None:
            base_lr = float(opt.param_groups[0]["lr"])
            scaled_lr = base_lr * self.lr_scale
            opt.add_param_group({
                "params": newly_unfrozen,
                "lr": scaled_lr,
                "weight_decay": float(
                    opt.param_groups[0].get("weight_decay", 0.0)
                ),
            })

        n_params = sum(p.numel() for p in newly_unfrozen)
        old_level = self._current_level
        self._current_level = target_level
        if self._trainer.is_world_process_zero():
            logger.info(
                "VisualUnfreeze: level %d → %d  |  %d params unfrozen  |  "
                "lr_scale=%.4f  (visual_frontend now has %d trainable params)",
                old_level,
                target_level,
                n_params,
                self.lr_scale,
                sum(
                    p.numel()
                    for p in vis_enc.visual_frontend.parameters()
                    if p.requires_grad
                ),
            )

    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        """Re-arm unfreeze state on resume (step mode)."""
        if self.mode == "step" and state.global_step > 0:
            target = self._current_level
            for i, s in enumerate(self.steps):
                if i + 1 > self._current_level and state.global_step >= s:
                    target = i + 1
            if target > self._current_level:
                self._try_unfreeze(target)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.mode != "metric" or not metrics:
            return control
        val = metrics.get(self.metric)
        if val is None:
            return control
        val = float(val)

        target = self._current_level
        for i, threshold in enumerate(self.thresholds):
            if i + 1 > self._current_level and val >= threshold:
                target = i + 1
        self._try_unfreeze(target)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if self.mode != "step":
            return control
        target = self._current_level
        for i, s in enumerate(self.steps):
            if i + 1 > self._current_level and state.global_step >= s:
                target = i + 1
        self._try_unfreeze(target)
        return control


def _per_mix_num_speakers_from_collate_flat(nums: list[int]) -> list[int] | None:
    """
    Recover per-mixture speaker counts from collated ``num_speaker`` metadata.

    ``tse_collate_fn`` repeats each mixture's ``num_speaker`` value ``ns`` times
    (one row per target speaker), e.g. [2,2, 3,3,3] → [2, 3].
    """
    if not nums:
        return None
    out: list[int] = []
    i = 0
    n = len(nums)
    while i < n:
        k = int(nums[i])
        if k <= 0 or i + k > n:
            return None
        for j in range(1, k):
            if int(nums[i + j]) != k:
                return None
        out.append(k)
        i += k
    return out


def _cue_ablation_scope_active(scope: str, *, training: bool) -> bool:
    """Whether ``visual_cue_ablation`` applies in this phase (train vs eval)."""
    s = (scope or "both").strip().lower()
    if s == "both":
        return True
    if s == "train":
        return training
    if s == "eval":
        return not training
    logger.warning("[visual_cue_ablation_scope] unknown value %r → treating as both", scope)
    return True


def _apply_visual_cue_ablation(
    cues: list[torch.Tensor],
    inputs: dict[str, Any],
    *,
    mode: str,
    scope: str,
    training: bool,
    trainer: Optional["TSETrainer"] = None,
) -> list[torch.Tensor]:
    """Mute or shuffle cues for ablation tests (deterministic structure via collate)."""
    m = (mode or "none").strip().lower()
    if m == "none":
        return cues
    if not _cue_ablation_scope_active(str(scope), training=training):
        return cues

    if m == "mute":
        return [torch.zeros_like(c) for c in cues]

    if m == "shuffle_within_mixture":
        ns_flat = inputs.get("num_speaker")
        if not isinstance(ns_flat, (list, tuple)) or len(ns_flat) == 0:
            logger.warning("[visual_cue_ablation shuffle] missing inputs['num_speaker'] → no-op")
            return cues
        try:
            flat = [int(x) for x in ns_flat]
            per_mix_ns = _per_mix_num_speakers_from_collate_flat(flat)
        except (TypeError, ValueError):
            per_mix_ns = None
        b_eff = int(cues[0].shape[0])
        if per_mix_ns is None or sum(per_mix_ns) != b_eff:
            if trainer is None or not getattr(
                trainer, "_warned_shuffle_parse_failure", False
            ):
                logger.warning(
                    "[visual_cue_ablation shuffle] cannot parse mixture layout "
                    "(B_eff=%s num_speaker=%s) → no-op",
                    b_eff,
                    list(ns_flat)[:16] if len(ns_flat) > 16 else list(ns_flat),
                )
                if trainer is not None:
                    setattr(trainer, "_warned_shuffle_parse_failure", True)
            return cues
        out = [c.clone() for c in cues]
        idx = 0
        for ns in per_mix_ns:
            if ns >= 2:
                perm = torch.randperm(ns, device=out[0].device)
                for li in range(len(out)):
                    sl = out[li][idx:idx + ns]
                    out[li][idx:idx + ns] = sl[perm]
            idx += ns
        return out

    logger.warning("[visual_cue_ablation] unknown mode %r → no-op", mode)
    return cues


def _ensure_sisdr_loss_module_device(
    sisdr_mod: torch.nn.Module, ref_tensor: torch.Tensor
) -> None:
    """Move cached auraloss ``SISDRLoss`` to *ref_tensor*'s device at most once per device."""
    dev = ref_tensor.device
    if getattr(sisdr_mod, "_wesep_sisdr_device", None) != dev:
        sisdr_mod.to(dev)
        sisdr_mod._wesep_sisdr_device = dev


def _mean_sisdr_loss_auraloss_normalized(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Batch SI-SDR **loss** (auraloss: positive when SI-SDR is negative).

    Uses cached ``auraloss.time.SISDRLoss`` with default reduction (scalar mean over
    batch×channel SI-SDR values). Matches the pre-refactor training path — **does not
    cast to FP32**, so AMP / bf16 behavior stays aligned with whatever dtype *est*
    carries (critical for throughput vs forcing ``est.float()`` on every step).
    """
    sisdr_mod = _get_cached_sisdr_module()
    _ensure_sisdr_loss_module_device(sisdr_mod, est)
    est_t, ref_t = _normalize_audio_pair_3d(est, ref)
    return sisdr_mod(est_t, ref_t)


class TSETrainer(Trainer):
    """
    Hugging Face Trainer adapted for Target Speaker Extraction (TSE) models.

    Handles the custom batch format from ``tse_collate_fn``, the TSE model
    forward signature ``model(mix, cues)``, and configurable loss functions
    (SISDR, SISNR, SNR, OnlineAVCrossNetLoss, or weighted multi-criterion).

    The Trainer (via accelerate) manages DDP wrapping, optimizer, scheduler,
    mixed precision, gradient clipping, logging, and checkpointing.
    """

    def __init__(
        self,
        model=None,
        args=None,
        criterion=None,
        loss_type="SISDR",
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.criterion = criterion or []
        self.loss_type = loss_type
        self._custom_train_dl = train_dataloader
        self._custom_eval_dl = eval_dataloader
        self._executor = Executor()
        self._metrics = defaultdict(list)
        self._logged_first_train_batch_shapes = False
        # Optional per-sample SI-SDR loss buffer for evaluate() stats. Stays
        # None unless --eval_extra_stats is enabled, so existing eval paths
        # are bit-for-bit unchanged.
        self._eval_sisdr_buffer: list[torch.Tensor] | None = None
        self._eval_sisdr_target_ot_buf: list[torch.Tensor] | None = None
        self._eval_sisdr_target_mt_buf: list[torch.Tensor] | None = None
        self._cv_eval_active = False
        self._cv_eval_acc: dict[str, float] | None = None
        self._warned_shuffle_parse_failure = False

    def _accumulate_clearervoice_val_metrics(
        self,
        mix: torch.Tensor,
        target: torch.Tensor,
        est: torch.Tensor,
    ) -> None:
        """Accumulate ClearerVoice ``solver.evaluate`` metrics (per-utterance means).

        Aligns with
        https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction/solver.py#L223
        """
        acc = self._cv_eval_acc
        if acc is None:
            return
        sr = int(getattr(self.args, "eval_audio_sr", 16000))
        tgt_bt = _to_mono_bt(target)
        est_bt = _to_mono_bt(est)
        mix_bt = _to_mono_bt(mix)
        b = int(tgt_bt.shape[0])
        l = min(int(tgt_bt.shape[1]), int(est_bt.shape[1]), int(mix_bt.shape[1]))
        if b <= 0 or l <= 0:
            return
        tgt_bt = tgt_bt[:, :l]
        est_bt = est_bt[:, :l]
        mix_bt = mix_bt[:, :l]

        sisnr_est = _clearervoice_cal_sisnr_torch(tgt_bt, est_bt)
        sisnr_mix = _clearervoice_cal_sisnr_torch(tgt_bt, mix_bt)
        sisnri = sisnr_est - sisnr_mix
        acc["sisnri"] += float(sisnri.sum().detach().cpu())
        acc["n"] += float(b)

        for bi in range(b):
            gt = tgt_bt[bi].detach().cpu().numpy()
            e = est_bt[bi].detach().cpu().numpy()
            m = mix_bt[bi].detach().cpu().numpy()
            try:
                acc["sdri"] += _clearervoice_sdri_numpy(gt, e, m)
            except Exception:  # noqa: BLE001
                pass
            peak = float(np.max(np.abs(e)) + 1e-12)
            e_n = e / peak
            try:
                from pesq import PesqError, pesq

                pr_e = pesq(
                    sr,
                    gt.astype(np.float64),
                    e_n.astype(np.float64),
                    "wb",
                    on_error=PesqError.RETURN_VALUES,
                )
                pr_m = pesq(
                    sr,
                    gt.astype(np.float64),
                    m.astype(np.float64),
                    "wb",
                    on_error=PesqError.RETURN_VALUES,
                )
                if (
                    pr_e != PesqError.NO_UTTERANCES_DETECTED
                    and pr_m != PesqError.NO_UTTERANCES_DETECTED
                ):
                    acc["pesqi"] += float(pr_e - pr_m)
                    acc["n_pesq"] += 1.0
            except Exception:  # noqa: BLE001
                pass
            try:
                from pystoi.stoi import stoi

                se = float(stoi(gt, e_n, fs_sig=sr, extended=False))
                sm = float(stoi(gt, m, fs_sig=sr, extended=False))
                acc["stoii"] += se - sm
                acc["n_stoi"] += 1.0
            except Exception:  # noqa: BLE001
                pass

    def _current_mrstft_weight(self) -> float:
        """Return the (possibly ramped) MRSTFT weight for the current step.

        ``mrstft_warmup_steps == 0`` (default for stages 98/99) → constant
        ``mrstft_weight``, identical to the previous hardcoded 0.5 path.
        """
        target = float(getattr(self.args, "mrstft_weight", 0.5))
        warmup = int(getattr(self.args, "mrstft_warmup_steps", 0))
        if warmup <= 0:
            return target
        step = int(getattr(self.state, "global_step", 0))
        return target * min(1.0, max(0.0, step / float(warmup)))

    def _current_cue_discrim_weight(self) -> float:
        """Return the (possibly ramped) cue-discrimination loss weight.

        Stages 78 / 88 / 98 / 99 / 100 leave ``cue_discrim_weight`` at its
        default (0.0) so this method always returns 0.0 there, and the
        cue-discrimination branch in :meth:`compute_loss` never fires.
        """
        target = float(getattr(self.args, "cue_discrim_weight", 0.0))
        if target == 0.0:
            return 0.0
        warmup = int(getattr(self.args, "cue_discrim_warmup_steps", 0))
        if warmup <= 0:
            return target
        step = int(getattr(self.state, "global_step", 0))
        return target * min(1.0, max(0.0, step / float(warmup)))

    def _current_passthrough_weight(self) -> float:
        """Return the anti-passthrough hinge weight (no warmup; constant).

        Stages 78 / 88 / 98 / 99 / 100 leave ``passthrough_penalty_weight``
        at its default (0.0) so this returns 0.0 there and the penalty
        branch in :meth:`compute_loss` is fully bypassed.
        """
        return float(getattr(self.args, "passthrough_penalty_weight", 0.0))

    def _metrics_gather_mean(self, tensor: torch.Tensor) -> float:
        """Reduce a metric tensor to one float for ``self.log`` averaging.

        On a single process, avoid ``accelerator.gather_for_metrics`` (which
        calls distributed collectives). Those have been observed to segfault
        in some torchrun/world_size=1 + CUDA setups while the pin_memory
        thread is active.
        """
        x = tensor.detach()
        if x.numel() == 0:
            return 0.0
        if getattr(self.accelerator, "num_processes", 1) <= 1:
            return float(x.float().mean().cpu())
        return float(self.accelerator.gather_for_metrics(x).mean().item())

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ):
        """
        Optional ClearerVoice-Studio TSE-online warmup LR (``LambdaLR``), matching
        ``Solver._adjust_lr_warmup`` in upstream ``solver.py``.
        """
        if bool(getattr(self.args, "clearervoice_lr_scheduler", False)):
            optimizer = optimizer or self.optimizer
            if optimizer is None:
                raise ValueError("clearervoice_lr_scheduler requires an optimizer.")
            warmup_steps = int(getattr(self.args, "clearervoice_warmup_steps", 15000))
            if warmup_steps <= 0:
                raise ValueError("clearervoice_warmup_steps must be > 0.")
            logger.info(
                "LR: ClearerVoice TSE-online warmup "
                "(init_lr * (64**-0.5/0.001) * step * warmup**-1.5), "
                "clearervoice_warmup_steps=%s, peak learning_rate=%s",
                warmup_steps,
                self.args.learning_rate,
            )
            self.lr_scheduler = LambdaLR(optimizer, _clearervoice_lambda_lr_mult(warmup_steps))
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    # -- dataloaders -------------------------------------------------------

    def get_train_dataloader(self):
        if self._custom_train_dl is not None:
            return self._custom_train_dl
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self._custom_eval_dl is not None:
            return self._custom_eval_dl
        return super().get_eval_dataloader(eval_dataset)

    # -- loss --------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mix, cues, target = self._executor._extract_model_inputs(
            inputs, self.args.device,
        )

        if not self._logged_first_train_batch_shapes and self.is_world_process_zero():
            logger.info("%s", _format_collated_batch_shapes(inputs))
            dl_bs = int(getattr(self.args, "per_device_train_batch_size", 0) or 0)
            b_eff = int(mix.shape[0])
            ns_flat = inputs.get("num_speaker")
            per_mix_ns: list[int] | None = None
            if isinstance(ns_flat, (list, tuple)) and len(ns_flat) > 0:
                try:
                    per_mix_ns = _per_mix_num_speakers_from_collate_flat(
                        [int(x) for x in ns_flat]
                    )
                except (TypeError, ValueError):
                    per_mix_ns = None
            if per_mix_ns is not None and sum(per_mix_ns) == b_eff:
                logger.info(
                    "First train batch — DataLoader mixtures/step (per_device_train_batch_size)=%s; "
                    "per-mix num_speaker=%s (len=%s); after tse_collate_fn speaker-axis expand, "
                    "model batch dim B_eff=sum(num_speaker)=%s (= mix/target leading dim).",
                    dl_bs,
                    per_mix_ns,
                    len(per_mix_ns),
                    b_eff,
                )
            else:
                logger.info(
                    "First train batch — DataLoader mixtures/step=%s; model B_eff=%s; "
                    "collated inputs['num_speaker'] (first 32 entries)=%r "
                    "(expected: each mix repeated ns times with value ns; sum should equal B_eff).",
                    dl_bs,
                    b_eff,
                    list(ns_flat)[:32] if isinstance(ns_flat, (list, tuple)) else ns_flat,
                )
            cue_shapes = None if cues is None else [tuple(t.shape) for t in cues]
            logger.info(
                "First train batch → model forward: mix.shape=%s cues.shapes=%s target.shape=%s",
                tuple(mix.shape),
                cue_shapes,
                tuple(target.shape),
            )
            self._logged_first_train_batch_shapes = True

        # Stage-98+ opt-in: visual cue dropout (zero entire per-row cue tensor
        # with prob p during training only). Defaults to 0.0 (no-op) for
        # all existing stages; does not change behavior for stages 78 / 88.
        _ab_mode = (
            getattr(self.args, "visual_cue_ablation", "none") or "none"
        ).strip().lower()
        if cues is None:
            cues_for_fwd = None
        elif isinstance(cues, (list, tuple)) and len(cues) == 0:
            cues_for_fwd = None
        elif _ab_mode == "none":
            cues_for_fwd = cues
        else:
            cues_for_fwd = _apply_visual_cue_ablation(
                list(cues),
                inputs,
                mode=getattr(self.args, "visual_cue_ablation", "none"),
                scope=getattr(self.args, "visual_cue_ablation_scope", "both"),
                training=model.training,
                trainer=self,
            )
        cue_drop_p = float(getattr(self.args, "visual_cue_dropout", 0.0) or 0.0)
        if (
            cue_drop_p > 0.0
            and model.training
            and cues_for_fwd is not None
            and len(cues_for_fwd) > 0
        ):
            B = int(cues_for_fwd[0].shape[0])
            keep = (torch.rand(B, device=cues_for_fwd[0].device) >= cue_drop_p).to(
                cues_for_fwd[0].dtype
            )
            cues_for_fwd = [
                c * keep.view((B,) + (1,) * (c.ndim - 1)).to(c.dtype)
                for c in cues_for_fwd
            ]

        if cues_for_fwd is None:
            outputs = model(mix)
        else:
            outputs = model(mix, cues_for_fwd)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        loss = loss_function(
            outputs,
            target,
            mix,
            self.loss_type,
            mrstft_weight=self._current_mrstft_weight(),
        )

        # SI-SDR(est, target), SI-SDR(mix, target), and improvement (train logs).
        # Only computed on steps immediately before a logging boundary to avoid
        # multiple redundant FP32 SI-SDR passes + implicit CUDA→CPU syncs per step.
        _log_interval = int(getattr(self.args, "logging_steps", 100) or 100)
        _step = int(getattr(self.state, "global_step", 0))
        _is_log_step = _log_interval > 0 and (_step % _log_interval >= _log_interval - int(getattr(self.args, "gradient_accumulation_steps", 1)))
        if _is_log_step:
            try:
                with torch.no_grad():
                    sisdr_ot = _per_sample_sisdr_db(outputs[0].float(), target.float())
                    sisdr_mt = _per_sample_sisdr_db(mix.float(), target.float())
                    sisdr_imp = sisdr_ot - sisdr_mt
                self._metrics["sisdr_output_target"].append(
                    self._metrics_gather_mean(sisdr_ot))
                self._metrics["sisdr_mix_target"].append(
                    self._metrics_gather_mean(sisdr_mt))
                self._metrics["sisdr_improvement"].append(
                    self._metrics_gather_mean(sisdr_imp))
            except Exception:  # noqa: BLE001
                pass

        # Stage-101+ opt-in cue-discrimination loss. Defaults
        # (cue_discrim_weight == 0.0) skip this path entirely so all
        # earlier stages remain bit-for-bit unchanged.
        cd_w = self._current_cue_discrim_weight()
        if cd_w > 0.0:
            ns_flat = inputs.get("num_speaker")
            per_mix_ns: list[int] | None = None
            if isinstance(ns_flat, (list, tuple)) and len(ns_flat) > 0:
                try:
                    per_mix_ns = _per_mix_num_speakers_from_collate_flat(
                        [int(x) for x in ns_flat]
                    )
                except (TypeError, ValueError):
                    per_mix_ns = None
            B_eff = int(outputs[0].shape[0])
            if per_mix_ns is not None and sum(per_mix_ns) == B_eff:
                cd_temp = float(getattr(self.args, "cue_discrim_temperature", 5.0))
                cd_term = _cue_discrim_loss(
                    outputs[0], target, per_mix_ns, temperature=cd_temp,
                )
                loss = loss + cd_w * cd_term
                if _is_log_step:
                    try:
                        self._metrics["cue_discrim_loss"].append(
                            self._metrics_gather_mean(cd_term.detach())
                        )
                        self._metrics["cue_discrim_weight"].append(float(cd_w))
                    except Exception:
                        pass

        # Stage-101+ opt-in anti-passthrough hinge. Default weight 0.0
        # keeps stages 78/88/98/99/100 bit-for-bit unchanged. Computed
        # in float32 for log10 stability, regardless of bf16 autocast.
        pt_w = self._current_passthrough_weight()
        if pt_w > 0.0:
            pt_thr = float(getattr(self.args, "passthrough_penalty_threshold", 10.0))
            # _compute_per_sample_sisdr_loss returns -SI-SDR (auraloss
            # convention: lower = better). We need SI-SDR itself, hence
            # the negation.
            sisdr_out_mix = -_compute_per_sample_sisdr_loss(
                outputs[0].float(), mix.float()
            )                                                # (B,)
            pt_term = torch.relu(sisdr_out_mix - pt_thr).mean()
            loss = loss + pt_w * pt_term
            if _is_log_step:
                try:
                    self._metrics["passthrough_penalty"].append(
                        self._metrics_gather_mean(pt_term.detach())
                    )
                    self._metrics["sisdr_out_mix_mean"].append(
                        self._metrics_gather_mean(sisdr_out_mix.detach().float().mean())
                    )
                except Exception:
                    pass

        if _is_log_step and bool(getattr(self.args, "train_log_clearervoice_sisnri", False)):
            try:
                with torch.no_grad():
                    tgt_bt = _to_mono_bt(target.float())
                    est_bt = _to_mono_bt(outputs[0].float())
                    mix_bt = _to_mono_bt(mix.float())
                    b = int(tgt_bt.shape[0])
                    l = min(int(tgt_bt.shape[1]), int(est_bt.shape[1]), int(mix_bt.shape[1]))
                    if b > 0 and l > 0:
                        sisnr_est = _clearervoice_cal_sisnr_torch(
                            tgt_bt[:, :l], est_bt[:, :l]
                        )
                        sisnr_mix = _clearervoice_cal_sisnr_torch(
                            tgt_bt[:, :l], mix_bt[:, :l]
                        )
                        sisnri = sisnr_est - sisnr_mix
                        self._metrics["clearervoice_sisnri"].append(
                            self._metrics_gather_mean(sisnri.mean())
                        )
            except Exception:
                pass

        self._log_metrics(loss)
        return (loss,outputs) if return_outputs else loss

        
    def _log_metrics(self,loss):
        self._metrics["loss"].append(self._metrics_gather_mean(loss))
        #self._metrics["lr"].append(self.optimizer.param_groups[0]["lr"])
        self._metrics["epoch"].append(self.state.epoch)
        self._metrics["step"].append(self.state.global_step)
        self._metrics["batch_size"].append(self.args.per_device_train_batch_size)

    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics with averaging."""
        metrics = {key: sum(vals) / len(vals) for key, vals in self._metrics.items()}
        logs = {**logs, **metrics}

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)

        self._metrics.clear()
    # -- eval step ---------------------------------------------------------

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Must return ``(loss, logits, labels)``. A bare scalar tensor breaks
        ``evaluation_loop`` which does ``losses, logits, labels = ...`` and
        iterates over the return value.

        New (stage-100+, opt-in via flags; stages 78/88/98/99 unchanged):
          * ``--eval_loss_type`` non-empty → compute eval loss with
            :func:`loss_function` (e.g. ``SISDR_MRSTFT`` to make eval directly
            comparable to train).
          * ``--eval_extra_stats true`` → also accumulate per-sample SI-SDR
            loss into ``self._eval_sisdr_buffer`` for percentile reporting in
            :meth:`evaluate`.
          * Each validation pass always accumulates SI-SDR(est, target),
            SI-SDR(mix, target), and their difference into ``eval_*`` means.
          * ``--visual_cue_ablation`` + ``visual_cue_ablation_scope`` also apply here
            when scope includes ``eval`` (same mute/shuffle semantics as training).
        """
        model.eval()
        mix, cues, target = self._executor._extract_model_inputs(
            inputs, self.args.device,
        )
        _ab_mode = (
            getattr(self.args, "visual_cue_ablation", "none") or "none"
        ).strip().lower()
        if cues is None:
            cues_for_fwd = None
        elif isinstance(cues, (list, tuple)) and len(cues) == 0:
            cues_for_fwd = None
        elif _ab_mode == "none":
            cues_for_fwd = cues
        else:
            cues_for_fwd = _apply_visual_cue_ablation(
                list(cues),
                inputs,
                mode=getattr(self.args, "visual_cue_ablation", "none"),
                scope=getattr(self.args, "visual_cue_ablation_scope", "both"),
                training=False,
                trainer=self,
            )
        with torch.no_grad():
            if cues_for_fwd is None:
                outputs = model(mix)
            else:
                outputs = model(mix, cues_for_fwd)

            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            # Per-sample SI-SDR loss for diagnostic percentiles, only when
            # explicitly enabled. Append on CPU to bound VRAM growth.
            if self._eval_sisdr_buffer is not None:
                per_sample = _compute_per_sample_sisdr_loss(
                    outputs[0].float(), target.float()
                ).detach().cpu()
                self._eval_sisdr_buffer.append(per_sample)

            if self._eval_sisdr_target_ot_buf is not None:
                ot_b = _per_sample_sisdr_db(
                    outputs[0].float(), target.float(),
                ).detach().cpu()
                mt_b = _per_sample_sisdr_db(mix.float(), target.float()).detach().cpu()
                self._eval_sisdr_target_ot_buf.append(ot_b)
                self._eval_sisdr_target_mt_buf.append(mt_b)

            if (
                bool(getattr(self.args, "eval_clearervoice_metrics", False))
                and getattr(self, "_cv_eval_active", False)
                and self._cv_eval_acc is not None
            ):
                self._accumulate_clearervoice_val_metrics(mix, target, outputs[0])

            eval_loss_type = str(getattr(self.args, "eval_loss_type", "") or "").strip()
            if eval_loss_type:
                loss = loss_function(
                    outputs,
                    target,
                    mix,
                    eval_loss_type,
                    mrstft_weight=self._current_mrstft_weight(),
                )
            else:
                # Default / legacy path: SI-SDR loss aligned with normalized training path.
                loss = _mean_sisdr_loss_auraloss_normalized(outputs[0], target)
        return (loss, None, None)

    # -- evaluate (extra stats) -------------------------------------------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Wrap HF ``Trainer.evaluate`` to optionally inject per-sample SI-SDR
        statistics under ``<metric_key_prefix>_sisdr_loss_*``. When
        ``--eval_extra_stats`` is False (default) this is a transparent
        passthrough.

        Always aggregates utterance-mean SI-SDR between estimate and target,
        mixture and target, and their difference as ``*_sisdr_output_target``,
        ``*_sisdr_mix_target``, ``*_sisdr_improvement`` (DDP all-reduced).

        When ``--eval_clearervoice_metrics`` is true, also reports
        SI-SNRi / SDRi / PESQi / STOIi aligned with ClearerVoice-Studio
        ``Solver.evaluate`` (``solver.py`` around line 223).
        """
        extra = bool(getattr(self.args, "eval_extra_stats", False))
        cv_eval = bool(getattr(self.args, "eval_clearervoice_metrics", False))
        self._eval_sisdr_target_ot_buf = []
        self._eval_sisdr_target_mt_buf = []
        if extra:
            self._eval_sisdr_buffer = []
        if cv_eval:
            self._cv_eval_acc = {
                "sisnri": 0.0,
                "sdri": 0.0,
                "pesqi": 0.0,
                "stoii": 0.0,
                "n": 0.0,
                "n_pesq": 0.0,
                "n_stoi": 0.0,
            }
            self._cv_eval_active = True
        else:
            self._cv_eval_active = False
            self._cv_eval_acc = None

        buf_ot_list = None
        buf_mt_list = None
        try:
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            buf = self._eval_sisdr_buffer
            self._eval_sisdr_buffer = None
            buf_ot_list = self._eval_sisdr_target_ot_buf
            buf_mt_list = self._eval_sisdr_target_mt_buf
            self._eval_sisdr_target_ot_buf = None
            self._eval_sisdr_target_mt_buf = None
            self._cv_eval_active = False
            cv_acc = self._cv_eval_acc
            self._cv_eval_acc = None

        metrics = dict(metrics) if metrics is not None else {}

        if cv_eval and cv_acc is not None:
            if dist.is_available() and dist.is_initialized():
                vec = torch.tensor(
                    [
                        cv_acc["sisnri"],
                        cv_acc["sdri"],
                        cv_acc["pesqi"],
                        cv_acc["stoii"],
                        cv_acc["n"],
                        cv_acc["n_pesq"],
                        cv_acc["n_stoi"],
                    ],
                    dtype=torch.float64,
                    device=self.args.device,
                )
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)
                cv_acc["sisnri"] = float(vec[0].item())
                cv_acc["sdri"] = float(vec[1].item())
                cv_acc["pesqi"] = float(vec[2].item())
                cv_acc["stoii"] = float(vec[3].item())
                cv_acc["n"] = float(vec[4].item())
                cv_acc["n_pesq"] = float(vec[5].item())
                cv_acc["n_stoi"] = float(vec[6].item())

            n = cv_acc["n"]
            if n > 0:
                cv_stats = {
                    f"{metric_key_prefix}_clearervoice_sisnri": cv_acc["sisnri"] / n,
                    f"{metric_key_prefix}_clearervoice_sdri": cv_acc["sdri"] / n,
                }
                if cv_acc["n_pesq"] > 0:
                    cv_stats[f"{metric_key_prefix}_clearervoice_pesqi"] = (
                        cv_acc["pesqi"] / cv_acc["n_pesq"]
                    )
                if cv_acc["n_stoi"] > 0:
                    cv_stats[f"{metric_key_prefix}_clearervoice_stoii"] = (
                        cv_acc["stoii"] / cv_acc["n_stoi"]
                    )
                metrics.update(cv_stats)
                try:
                    self.log(cv_stats)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("eval_clearervoice_metrics: log() failed: %s", exc)

        try:
            if buf_ot_list and buf_mt_list and len(buf_ot_list) == len(buf_mt_list):
                ot_all = torch.cat(buf_ot_list).flatten().to(torch.float32)
                mt_all = torch.cat(buf_mt_list).flatten().to(torch.float32)
                if ot_all.numel() == mt_all.numel() and ot_all.numel() > 0:
                    dev = self.args.device
                    if not isinstance(dev, torch.device):
                        dev = torch.device(dev)
                    packed = torch.tensor(
                        [
                            float(ot_all.double().sum().item()),
                            float(mt_all.double().sum().item()),
                            float(ot_all.numel()),
                        ],
                        dtype=torch.float64,
                        device=dev,
                    )
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                    denom = packed[2].item()
                    if denom > 0:
                        mean_ot = float((packed[0] / denom).item())
                        mean_mt = float((packed[1] / denom).item())
                        tgt_stats = {
                            f"{metric_key_prefix}_sisdr_output_target": mean_ot,
                            f"{metric_key_prefix}_sisdr_mix_target": mean_mt,
                            f"{metric_key_prefix}_sisdr_improvement": mean_ot - mean_mt,
                        }
                        metrics.update(tgt_stats)
                        try:
                            self.log(tgt_stats)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "eval sisdr_output_target / mix_target / improvement: "
                                "log() failed: %s",
                                exc,
                            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "eval sisdr_output_target / mix_target / improvement failed: %s",
                exc,
            )

        if not extra or not buf:
            return metrics

        try:
            sisdr_loss = torch.cat(buf).flatten().to(torch.float32)
        except RuntimeError:
            return metrics

        if sisdr_loss.numel() == 0:
            return metrics

        # All percentiles are computed on -SI-SDR (auraloss convention: larger
        # value = worse separation). worst5pct_mean averages the largest 5%.
        n = int(sisdr_loss.numel())
        sorted_vals, _ = torch.sort(sisdr_loss)
        worst_k = max(1, n // 20)
        stats = {
            f"{metric_key_prefix}_sisdr_loss_mean": float(sisdr_loss.mean()),
            f"{metric_key_prefix}_sisdr_loss_median": float(sorted_vals[n // 2]),
            f"{metric_key_prefix}_sisdr_loss_p25": float(sorted_vals[max(0, n // 4 - 1)]),
            f"{metric_key_prefix}_sisdr_loss_p75": float(sorted_vals[min(n - 1, (3 * n) // 4)]),
            f"{metric_key_prefix}_sisdr_loss_p95": float(sorted_vals[min(n - 1, (95 * n) // 100)]),
            f"{metric_key_prefix}_sisdr_loss_worst5pct_mean": float(sorted_vals[-worst_k:].mean()),
            f"{metric_key_prefix}_sisdr_loss_n": float(n),
        }
        metrics.update(stats)
        try:
            self.log(stats)
        except Exception as exc:  # noqa: BLE001
            logger.warning("eval_extra_stats: log() failed: %s", exc)
        return metrics
     




_LOSS_FN_CACHE: dict[str, "torch.nn.Module"] = {}


def _get_cached_mrstft_module():
    """Cache MultiResolutionSTFTLoss across steps (it owns FFT buffers)."""
    mod = _LOSS_FN_CACHE.get("mrstft")
    if mod is None:
        mod = auraloss.freq.MultiResolutionSTFTLoss()
        _LOSS_FN_CACHE["mrstft"] = mod
    return mod


def _get_cached_sisdr_module():
    mod = _LOSS_FN_CACHE.get("sisdr")
    if mod is None:
        mod = auraloss.time.SISDRLoss()
        _LOSS_FN_CACHE["sisdr"] = mod
    return mod


def _normalize_audio_pair_3d(est: torch.Tensor, ref: torch.Tensor):
    """Return (est, ref) both shaped (B, C, T), broadcasting mono <-> multi-ch.

    Used by SISDR_MRSTFT (MRSTFT requires 3-D) and per-sample SI-SDR stats.
    """
    if est.dim() == 2:
        est = est.unsqueeze(1)
    if ref.dim() == 2:
        ref = ref.unsqueeze(1)
    if est.shape[1] != ref.shape[1]:
        if est.shape[1] == 1:
            est = est.expand(-1, ref.shape[1], -1)
        elif ref.shape[1] == 1:
            ref = ref.expand(-1, est.shape[1], -1)
    return est, ref


def _compute_per_sample_sisdr_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-row -SI-SDR (auraloss convention: lower = better) without reduction.

    Returns a 1-D tensor of length B (batch). Channels are averaged.
    Intended for diagnostic statistics (median / quantiles / worst-5%).
    """
    est, ref = _normalize_audio_pair_3d(est, ref)
    eps = 1e-8
    est_z = est - est.mean(dim=-1, keepdim=True)
    ref_z = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est_z * ref_z).sum(dim=-1) / (((ref_z ** 2).sum(dim=-1)) + eps)
    proj = ref_z * alpha.unsqueeze(-1)
    res = est_z - proj
    sisdr = 10 * torch.log10(
        (proj ** 2).sum(dim=-1) / ((res ** 2).sum(dim=-1) + eps) + eps
    )  # (B, C)
    return -sisdr.mean(dim=-1)  # (B,)


def _per_sample_sisdr_db(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-batch-row SI-SDR in dB between *est* and *ref* (scale-invariant).

    Higher is better. Uses the same projection definition as
    :func:`_compute_per_sample_sisdr_loss` (negated).
    """
    return -_compute_per_sample_sisdr_loss(est, ref)


def _to_mono_bt(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, T) or (B, T) → (B, T) float32."""
    x = x.float()
    if x.dim() == 3:
        return x[:, 0, :]
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected 2D/3D audio tensor, got shape {tuple(x.shape)}")


def _clearervoice_cal_sisnr_torch(
    source: torch.Tensor,
    estimate_source: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Batched ClearerVoice ``losses.metrics.cal_SISNR`` (SI-SNR in dB).

    ``source`` = clean target, ``estimate_source`` = network output or mixture
    (same argument order as upstream). Returns ``(B,)``.
    """
    assert source.shape == estimate_source.shape
    source = source - source.mean(dim=-1, keepdim=True)
    estimate_source = estimate_source - estimate_source.mean(dim=-1, keepdim=True)
    ref_energy = (source**2).sum(dim=-1).clamp_min(eps)
    proj = (source * estimate_source).sum(dim=-1, keepdim=True) * source / ref_energy.unsqueeze(-1)
    noise = estimate_source - proj
    ratio = (proj**2).sum(dim=-1) / ((noise**2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def _clearervoice_sdri_numpy(gt_1d: np.ndarray, est_1d: np.ndarray, mix_1d: np.ndarray) -> float:
    """One utterance: ``SDR(gt, est) - SDR(gt, mix)`` (ClearerVoice ``losses.metrics.SDR``)."""
    from mir_eval.separation import bss_eval_sources

    gt = np.asarray(gt_1d, dtype=np.float64).reshape(1, -1)
    est = np.asarray(est_1d, dtype=np.float64).reshape(1, -1)
    mix = np.asarray(mix_1d, dtype=np.float64).reshape(1, -1)
    sdr_e, _, _, _ = bss_eval_sources(gt, est)
    sdr_m, _, _, _ = bss_eval_sources(gt, mix)
    return float(np.asarray(sdr_e).reshape(-1)[0] - np.asarray(sdr_m).reshape(-1)[0])


def _cue_discrim_loss(
    outs: torch.Tensor,
    tgts: torch.Tensor,
    num_speakers_per_mix: list[int],
    temperature: float = 5.0,
) -> torch.Tensor:
    """InfoNCE-style cue discrimination loss (stage-101+ opt-in).

    ``tse_collate_fn`` repeats each mixture's audio ``ns`` times along the
    speaker axis: rows ``[k, k+1, ..., k+ns-1]`` of the batch share the same
    input mixture but differ only in the visual cue and the target waveform.
    A model that ignores the cue produces *identical* outputs for those rows,
    so all entries of the row-wise similarity matrix are equal and
    ``cross_entropy`` plateaus at ``log(ns)``. To minimize this loss the
    model is forced to make ``model(mix, cue_i)`` discriminably closer to
    ``target_i`` than to ``target_j`` for ``j != i`` — which is only possible
    if the cue is actually used.

    Cost: O(B_eff) total — for the typical ``num_speakers <= 4`` we evaluate
    at most ``4 * B_eff`` SI-SDR pairs.

    Notes:
      * Computed in float32 (log10 is unstable in bf16).
      * Only mixtures with ``ns >= 2`` contribute (1-spk samples skipped).
      * Returns a scalar requiring grad. If no mixture has ``ns >= 2``,
        returns a *connected* zero so backward is safe.
    """
    if outs.dim() == 2:
        outs = outs.unsqueeze(1)
    if tgts.dim() == 2:
        tgts = tgts.unsqueeze(1)
    outs_f = outs.float()
    tgts_f = tgts.float()
    losses: list[torch.Tensor] = []
    start = 0
    for ns in num_speakers_per_mix:
        if ns < 2:
            start += ns
            continue
        os_ = outs_f[start:start + ns]  # (ns, C, T)
        ts_ = tgts_f[start:start + ns]  # (ns, C, T)
        # Build (ns, ns) sim where sim[i, j] = -loss(os_[i], ts_[j]).
        # Higher = closer match. Pair-flatten so a single SI-SDR call covers
        # the whole matrix.
        oi = (
            os_.unsqueeze(1)
            .expand(ns, ns, *os_.shape[1:])
            .reshape(ns * ns, *os_.shape[1:])
        )
        tj = (
            ts_.unsqueeze(0)
            .expand(ns, ns, *ts_.shape[1:])
            .reshape(ns * ns, *ts_.shape[1:])
        )
        per = _compute_per_sample_sisdr_loss(oi, tj)  # (ns*ns,)
        sim = (-per).view(ns, ns) / float(max(temperature, 1e-6))
        labels = torch.arange(ns, device=sim.device)
        losses.append(torch.nn.functional.cross_entropy(sim, labels))
        start += ns
    if not losses:
        # Multiplying by 0 keeps the autograd connection so .backward()
        # doesn't complain about an unused parameter graph in DDP.
        return outs_f.sum() * 0.0
    return torch.stack(losses).mean()


def loss_function(outputs, target, mix, loss_type, mrstft_weight: float = 0.5):
    if loss_type == "SISDR":
        return _mean_sisdr_loss_auraloss_normalized(outputs[0], target)
    elif loss_type == "SISNR":
        return _mean_sisdr_loss_auraloss_normalized(outputs[0], target)
    elif loss_type == "SNR":
        snr_loss=auraloss.time.SNRLoss()
        return snr_loss(outputs[0], target).mean()
    elif loss_type == "OnlineAVCrossNetLoss":
        from wesep.utils.losses import OnlineAVCrossNetLoss
        onlineavcrossnet_loss=OnlineAVCrossNetLoss()
        return onlineavcrossnet_loss(outputs[0], target, mix)
    elif loss_type == "SISDR_MRSTFT":
        # Additive opt-in for stage 98+: SI-SDR + multi-resolution STFT magnitude.
        # Modules are cached so STFT buffers are not rebuilt every step.
        mrstft_mod = _get_cached_mrstft_module()
        dev = outputs[0].device
        if next(mrstft_mod.parameters(), torch.empty(0, device="cpu")).device != dev:
            mrstft_mod.to(dev)
        sdr_term = _mean_sisdr_loss_auraloss_normalized(outputs[0], target)
        if mrstft_weight == 0.0:
            return sdr_term
        est_t, ref_t = _normalize_audio_pair_3d(outputs[0], target)
        return sdr_term + mrstft_weight * mrstft_mod(est_t, ref_t).mean()
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
# ------------------------------------------------------------------
# Helpers for Trainer integration
# ------------------------------------------------------------------

class _LimitedDataLoader:
    """Wraps a DataLoader to yield at most *max_steps* batches per
    ``__iter__`` call.  Provides ``__len__`` so the Trainer can compute
    steps-per-epoch without requiring ``max_steps`` in TrainingArguments.
    """

    def __init__(self, dataloader, max_steps):
        self.dataloader = dataloader
        self.max_steps = max_steps
        self.dataset = dataloader.dataset

    def __getattr__(self, name):
        return getattr(self.dataloader, name)

    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if i >= self.max_steps:
                break
            yield batch

    def __len__(self):
        return self.max_steps


class _SetEpochCallbackMulti(TrainerCallback):
    """Call ``set_epoch`` on train/val :class:`OnlineMixIterableDataset` instances."""

    def __init__(self, *datasets):
        self.datasets = tuple(d for d in datasets if d is not None)

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + 1
        for d in self.datasets:
            if hasattr(d, "set_epoch"):
                d.set_epoch(epoch)


# ------------------------------------------------------------------
# CLI dataclasses for HfArgumentParser
# ------------------------------------------------------------------

@dataclass
class TSEOnlineDataArguments:
    """Online-mix **data** + paths. ``--model_config`` is model-only YAML (``model`` + ``model_args``)."""
    model_config: str = field(
        metadata={
            "help": "Model-only YAML path (model + model_args), e.g. confs/tse_bsrnn_visual_model.yaml.",
        },
    )

    mp4_dir_of_voxceleb2: str | None = field(
        default=None,
        metadata={
            "help": "MP4 root (VoxCeleb2 layout). Required when --use_voxceleb2.",
        },
    )
    use_voxceleb2: bool = field(
        default=False,
        metadata={"help": "Include VoxCeleb2 (split speakers by train_speaker_fraction)."},
    )
    use_lrs3: bool = field(
        default=True,
        metadata={"help": "Include LRS3 (split speakers by train_speaker_fraction)."},
    )
    use_chinese_lips: bool = field(
        default=True,
        metadata={
            "help": "Include Chinese_lips (train/val from dataset dirs; no random split).",
        },
    )
    lrs3_root: str = field(
        default=DEFAULT_LRS3_ROOT,
        metadata={"help": "LRS3 corpus root."},
    )
    lrs3_subsets: str = field(
        default="trainval",
        metadata={
            "help": "Comma-separated LRS3 subsets to scan (e.g. trainval or trainval,test).",
        },
    )
    chinese_lips_root: str = field(
        default=DEFAULT_CHINESE_LIPS_ROOT,
        metadata={"help": "Chinese_lips corpus root."},
    )
    chinese_lips_train_split: str = field(
        default="train",
        metadata={"help": "Chinese_lips top-level split name for **training** inventory only."},
    )
    chinese_lips_val_split: str = field(
        default="val",
        metadata={"help": "Chinese_lips top-level split name for **validation** inventory only."},
    )
    prefix_voxceleb2: str = field(
        default="vox2:",
        metadata={
            "help": "Optional extra prefix for VoxCeleb2 speaker keys when merging; "
            "empty uses built-in vox2: prefix.",
        },
    )

    # ---- Online mix / audio (no dataset YAML) ----
    resample_rate: int = field(default=16000, metadata={"help": "Target sample rate."})
    chunk_len: int = field(default=48000, metadata={"help": "Random chunk length (samples) after resample. per mix audio length."})
    whole_utt: bool = field(default=False, metadata={"help": "If set, disable random_chunk."})
    online_buffer_size: int = field(
        default=8,
        metadata={
            "help": "Number of single-speaker clips buffered before "
            "sample_speaker_group_without_repeat yields mixed bundles. "
            "Lower (e.g. 8) → fewer audio-file loads before the first mixture and faster "
            "time-to-first-batch; higher → more interference diversity in each buffer."
        },
    )
    reverb_prob: float = field(default=0.0, metadata={"help": "Reverb augmentation probability."})
    snr_range_low: float = field(default=-5.0, metadata={"help": "SNR mixing range low (dB)."})
    snr_range_high: float = field(default=10.0, metadata={"help": "SNR mixing range high (dB)."})
    gain_range_low: float = field(default=-12.0, metadata={"help": "Peak-norm gain range low (dB)."})
    gain_range_high: float = field(default=0.0, metadata={"help": "Peak-norm gain range high (dB)."})
    noise_prob: float = field(default=0.0, metadata={"help": "Probability to add MUSAN noise."})
    noise_lmdb_file: str | None = field(
        default='/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb',
        metadata={"help": "MUSAN LMDB directory (required if noise_prob > 0)."},
    )
    online_mix_deterministic: bool = field(
        default=False,
        metadata={
            "help": "If True, online mixture RNG does not depend on trainer epoch "
            "(same --seed → same mixture sequence across epochs). "
            "Per-rank / per-dataloader-worker streams still differ; use "
            "--dataloader_num_workers 0 for a single-process stream.",
        },
    )
    online_av_align: bool = field(
        default=False,
        metadata={
            "help": "Multi-corpus online: propagate random_chunk timings per speaker slot "
            "and crop each MP4 to match wav_spk{i} (processor_new + processor_visual_new). "
            "Enable explicitly for corrected lip–audio alignment (--online_av_align true). "
            "Default false preserves legacy grouping/visual decode.",
        },
    )
    #force_two_speaker_only: bool = field(
    #    default=False,
    #    metadata={
    #        "help": "If True, force exactly two speakers per mixture (distribution 0,1,0), "
    #        "paper-style 2-spk TSE without 1- or 4-spk buckets.",
    #    },
    #)
    #online_mix_clean_dry: bool = field(
    #    default=False,
    #    metadata={
    #        "help": "If True, disable MUSAN/reverb and set SNR/gain jitter to 0 dB "
    #        "(clean dry sum of two sources after timeline masking). "
    #        "Still uses random_chunk / timeline randomness unless you also set "
    #        "--whole_utt true and a fixed timeline (not implemented here).",
    #    },
    #)
    num_speakers_max: int = field(default=4, metadata={"help": "Max speakers per mixture."})
    num_speakers_distribution: str = field(
        default="0.1,0.75,0.15",
        metadata={"help": "Three comma-separated probs for 1 / 2 / 3+ speakers (favor 2-spk to save VRAM)."},
    )
    visual_max_frames: int = field(
        default=96,
        metadata={
            "help": "Cap decoded video time length T (and collate pad) for visual cues. "
            "0 = no cap. Lower values reduce VRAM (especially with 3–4 speakers per mix).",
        },
    )
    visual_resize: int = field(
        default=224,
        metadata={
            "help": "Resize every visual frame to H=W (pixels) before batching so mixed "
            "corpora (e.g. 224×224 face vs 720p frame) can torch.stack. "
            "0 = disable resize (single-resolution setups only).",
        },
    )
    visual_decode_cuda: bool = field(
        default=True,
        metadata={
            "help": "Decode MP4 frames with TorchCodec on GPU when CUDA is available "
            "(cuda:LOCAL_RANK under torchrun, else cuda:0). With "
            "--dataloader_num_workers > 0 this enables multiprocessing spawn workers "
            "so CUDA is legal in worker processes (adds startup cost vs fork). "
            "Set false to force CPU decoding (former default for workers)."
        },
    )
    sample_num_per_epoch: int = field(
        default=20000,
        metadata={"help": "Train mixtures per epoch."},
    )
    sample_num_per_epoch_val: int | None = field(
        default=5000,
        metadata={"help": "Val mixtures per epoch; default train*5000/20000 if unset."},
    )
    cue_visual_use: bool = field(default=True, metadata={"help": "Collate: use visual tensors."})
    cue_visual_required: bool = field(default=True, metadata={"help": "Collate: visual is required."})

    train_cues: str | None = field(
        default=None,
        metadata={
            "help": "Optional cues YAML for :func:`build_collect_keys`. "
            "If omitted, :func:`build_collect_keys_online` uses cue_visual_* flags.",
        },
    )
    tse_model: str | None = field(default=None, metadata={"help": "Override model YAML tse_model name."})
    visual_frontend: str | None = field(
        default=None,
        metadata={
            "help": "If muse or blaze (case-insensitive), override model YAML "
            "model_args.tse_model.visual.features: enable one frontend and disable the other.",
        },
    )
    blaze_visual_causal: str | None = field(
        default=None,
        metadata={
            "help": "If true or false, override model_args.tse_model.visual.features.blaze_visual.causal.",
        },
    )
    separator_causal: str | None = field(
        default=None,
        metadata={
            "help": "If true or false, override model_args.tse_model.separator.causal.",
        },
    )
    loss_type: str = field(
        default="SISDR",
        metadata={"help": "Loss type (SISDR, SISNR, SNR, OnlineAVCrossNetLoss, ...)."},
    )
    model_init_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Optional pretrained TSE checkpoint path (overrides model_init in YAML if both set)."},
    )
    num_speaker_pool: int = field(
        default=0,
        metadata={
            "help": "When <=0 uses all scanned speakers for "
            "train/val split; a positive value randomly subsamples that many "
            "speakers (smaller pool, less metadata in play).",
        },
    )
   
    skip_visual_decode: bool = field(
        default=False,
        metadata={"help": "If True, skip TorchCodec visual (debug only)."},
    )
    train_speaker_fraction: float = field(
        default=0.8,
        metadata={"help": "Fraction of pooled speakers for training (rest for val)."},
    )

@dataclass
class TSETrainingArguments(TrainingArguments):
    num_train_epochs: float = field(default=150, metadata={"help": "Total training epochs."})
    max_steps: int = field(default=-1, metadata={"help": "Overrides num_train_epochs if positive."})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy."})
    save_steps: int = field(default=100, metadata={"help": "Save every N steps."})
    save_total_limit: int = field(default=10, metadata={"help": "Max checkpoints to keep."})
    eval_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy."})
    eval_steps: int = field(default=500, metadata={"help": "Evaluate every N steps."})
    learning_rate: float = field(default=1e-3, metadata={"help": "Peak learning rate."})
    weight_decay: float = field(default=1e-4, metadata={"help": "Weight decay."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW."})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW."})
    warmup_steps: int = field(default=15000, metadata={"help": "Warmup steps."})
    max_grad_norm: float = field(default=5.0, metadata={"help": "Maximum gradient norm."})
    output_dir: str = field(default=None, metadata={"help": "Experiment output directory."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Batch size per device."})
    gradient_accumulation_steps: int = field(default=2, metadata={"help": "Gradient accumulation steps."})
    logging_steps: int = field(default=100, metadata={"help": "Log every N steps."})

    dataloader_drop_last: bool = field(default=True, metadata={"help": "DataLoader drop_last."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of workers for DataLoader."})
    # Do not duplicate HF ``dataloader_prefetch_factor`` here: we set prefetch_factor=1
    # inside ``build_dataloader_args_from_training_args`` when num_workers > 0 (shm-friendly).
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Pin memory."})
    load_best_model_at_end: bool = field(default=False, metadata={"help": "Load best model at end."})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Must be False for custom batch format."})
    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": "DDP find_unused_parameters."})
    bf16: bool = field(default=False, metadata={"help": "BF16 training."})
    report_to: None | str | list[str] = field(
        default="tensorboard",
        metadata={
            "help": "The list of integrations to report the results and logs to. Use 'all' for all installed integrations, 'none' for no integrations."
        },
    )
    clearervoice_lr_scheduler: bool = field(
        default=False,
        metadata={
            "help": "Use ClearerVoice-Studio TSE-online warmup LR from solver._adjust_lr_warmup: "
            "lr = learning_rate * (64**-0.5/0.001) * step_num * warmup**-1.5 for step_num in "
            "[1, warmup); then hold. Peak LR is --learning_rate. "
            "See https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction_online/solver.py "
            "When True, HF lr_scheduler_type / warmup_steps are ignored for the built scheduler."
        },
    )
    clearervoice_warmup_steps: int = field(
        default=15000,
        metadata={
            "help": "Warmup length for --clearervoice_lr_scheduler (default matches upstream solver)."
        },
    )
    clearervoice_plateau_halving: bool = field(
        default=False,
        metadata={
            "help": "Halve optimizer LR when eval_loss does not improve for "
            "--clearervoice_plateau_patience_evals consecutive evaluations "
            "(ClearerVoice-style; does not reload best weights)."
        },
    )
    clearervoice_plateau_patience_evals: int = field(
        default=5,
        metadata={"help": "Patience (in evaluation cycles) before LR *= 0.5."},
    )
    clearervoice_plateau_min_delta: float = field(
        default=0.0,
        metadata={"help": "Minimum eval_loss decrease to count as improvement."},
    )

    # ---- Stage-98+ opt-in regularizers (defaults: disabled, no-op for stages 78/88) ----
    visual_cue_dropout: float = field(
        default=0.0,
        metadata={
            "help": "Probability in [0,1) to zero an entire per-row visual cue tensor "
            "during training (per row in B_eff). 0.0 disables (default). "
            "Used to robustify against silent / occluded video and break "
            "over-reliance on visual frontend."
        },
    )
    visual_cue_ablation: str = field(
        default="none",
        metadata={
            "help": "Deterministic cue ablation for diagnostics: ``none`` (default), "
            "``mute`` (zeros all cues), ``shuffle_within_mixture`` (within each mixture "
            "block that shares ``wav_mix``, permute visual tensors across sibling "
            "target rows — breaks cue–target pairing). Applies where "
            "--visual_cue_ablation_scope says train / eval / both. Applied before "
            "forward; stochastic --visual_cue_dropout still runs afterward on mute "
            "(usually keep dropout 0 when using mute)."
        },
    )
    visual_cue_ablation_scope: str = field(
        default="both",
        metadata={
            "help": "Where ``visual_cue_ablation`` runs: ``train``, ``eval`` "
            "(validation ``prediction_step`` only), or ``both``. Default ``both``."
        },
    )

    # ---- Gradual visual-frontend unfreezing ----
    visual_unfreeze_mode: str = field(
        default="none",
        metadata={
            "help": "Visual frontend unfreezing strategy: 'none' (all frozen, "
            "default), 'metric' (unfreeze when an eval metric reaches a "
            "threshold), 'step' (unfreeze at a specified training step)."
        },
    )
    visual_unfreeze_metric: str = field(
        default="eval_sisdr_improvement",
        metadata={
            "help": "Eval metric key to watch when --visual_unfreeze_mode=metric. "
            "Default 'eval_sisdr_improvement' (higher = better)."
        },
    )
    visual_unfreeze_thresholds: str = field(
        default="",
        metadata={
            "help": "Comma-separated metric thresholds, one per unfreeze level. "
            "E.g. '1.0,3.0,5.0' means: level 1 when metric >= 1.0, level 2 "
            "when >= 3.0, level 3 when >= 5.0. If fewer thresholds than 5, "
            "higher levels stay frozen. Only used with --visual_unfreeze_mode=metric."
        },
    )
    visual_unfreeze_steps: str = field(
        default="",
        metadata={
            "help": "Comma-separated step numbers, one per unfreeze level. "
            "E.g. '10000,20000' means: level 1 at step 10000, level 2 at "
            "step 20000. Only used with --visual_unfreeze_mode=step."
        },
    )
    visual_unfreeze_lr_scale: float = field(
        default=0.1,
        metadata={
            "help": "LR multiplier for newly unfrozen visual-frontend parameters. "
            "Applied relative to the current base LR of param-group 0. "
            "Default 0.1 (unfrozen visual layers train at 1/10th the main LR)."
        },
    )

    use_ema: bool = field(
        default=False,
        metadata={
            "help": "Maintain an Exponential Moving Average of model weights and "
            "save it to <output_dir>/ema_model.pt at every save_steps and at "
            "train end. Disabled by default; does not affect the primary "
            "checkpoint stream."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={"help": "EMA decay (only used when --use_ema true)."},
    )

    # ---- Stage-100+ opt-in loss / eval diagnostics. Defaults preserve stage 98/99
    # behavior bit-for-bit (mrstft_weight=0.5 matches the previously hardcoded
    # 0.5 in loss_function::SISDR_MRSTFT; eval_loss_type=None preserves pure
    # SISDR eval; mrstft_warmup_steps=0 disables the warmup ramp; etc.).
    mrstft_weight: float = field(
        default=0.5,
        metadata={
            "help": "Weight on the MultiResolutionSTFT term inside the "
            "SISDR_MRSTFT loss: L = sisdr + mrstft_weight * mrstft. Default "
            "0.5 matches stage 98/99. Lower (e.g. 0.1-0.2) lets SI-SDR drive "
            "training; 0.0 reduces SISDR_MRSTFT to pure SISDR."
        },
    )
    mrstft_warmup_steps: int = field(
        default=0,
        metadata={
            "help": "If >0, linearly ramp the MRSTFT weight from 0 up to "
            "--mrstft_weight over this many optimizer steps (then hold). "
            "Useful when SI-SDR signal is weak early and MRSTFT dominates "
            "the gradient. Default 0 = no ramp (stage 98/99 behavior)."
        },
    )
    eval_loss_type: str = field(
        default="",
        metadata={
            "help": "If non-empty, use this loss key for eval as well as for "
            "extra apples-to-apples logging. Empty (default) keeps the "
            "existing pure-SISDR eval used by all earlier stages so their "
            "numbers stay comparable. Recommended for stage 100+: "
            "--eval_loss_type SISDR_MRSTFT."
        },
    )
    eval_extra_stats: bool = field(
        default=False,
        metadata={
            "help": "If true, accumulate per-sample SI-SDR loss during "
            "evaluation and report mean / median / p25 / p75 / p95 / worst-5%% "
            "as additional eval_* metrics. Default false (no extra log lines, "
            "stages 78/88/98/99 unchanged)."
        },
    )
    eval_clearervoice_metrics: bool = field(
        default=False,
        metadata={
            "help": "During validation, also compute SI-SNRi / SDRi / PESQi / STOIi "
            "using the same definitions as ClearerVoice-Studio "
            "train/target_speaker_extraction/solver.py evaluate() (~L223): "
            "torch SI-SNR on raw waveforms, mir_eval SDR improvement, PESQ/STOI "
            "on peak-normalized estimate vs raw mixture. "
            "Utterance-weighted means; DDP all-reduces sums. "
            "Requires pesq and pystoi."
        },
    )
    eval_audio_sr: int = field(
        default=16000,
        metadata={
            "help": "Sample rate passed to pesq / stoi when "
            "--eval_clearervoice_metrics is true (ClearerVoice ``args.audio_sr``)."
        },
    )
    train_log_clearervoice_sisnri: bool = field(
        default=False,
        metadata={
            "help": "If true, each train step logs ``clearervoice_sisnri``: batch-mean "
            "SI-NRi = cal_SISNR(target, est) - cal_SISNR(target, mix) (torch, same "
            "as ClearerVoice solver.py ~L223 and --eval_clearervoice_metrics). "
            "Only tensor math; negligible vs forward. Default false (no extra fields "
            "in train logs)."
        },
    )

    # ---- Stage-101+ opt-in cue-discrimination loss. Defaults preserve all
    # earlier stages bit-for-bit (cue_discrim_weight == 0.0 fully skips the
    # branch in compute_loss). The mechanism: tse_collate_fn places ``ns``
    # rows that share one input mixture but different cues / targets adjacent
    # in each batch. Among those rows we add an InfoNCE term forcing the
    # output for ``cue_i`` to match ``target_i`` better than ``target_j``
    # — the only way this is achievable is if the model actually uses the
    # cue. Diagnostics in stage 100 (logs/ablation_stage100_cue_mute.log)
    # showed the visual cue was being completely ignored after 6k steps.
    cue_discrim_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight on the cue-discrimination InfoNCE term added to "
            "the main reconstruction loss. 0.0 disables (default; stages "
            "78/88/98/99/100 behavior unchanged). Try 1.0 for stage 101."
        },
    )
    cue_discrim_warmup_steps: int = field(
        default=0,
        metadata={
            "help": "If >0, linearly ramp cue_discrim_weight from 0 up to "
            "the target over this many optimizer steps (then hold). "
            "Useful when the reconstruction model is still warming up and "
            "the cue branch should not dominate gradients early. 0 = no "
            "ramp, weight is constant from step 0."
        },
    )
    cue_discrim_temperature: float = field(
        default=5.0,
        metadata={
            "help": "Temperature divider applied to the (ns x ns) similarity "
            "matrix before cross-entropy. Smaller temperature = sharper "
            "penalty when wrong-cue output is closer to wrong target than "
            "to the correct one. SI-SDR ranges roughly +/- 30, so 5.0 keeps "
            "logits in roughly +/- 6 — a healthy softmax range."
        },
    )

    # ---- Stage-101+ opt-in anti-passthrough penalty. Stage-100 sanity-check
    # (logs + local/sanity_check_pipeline.py runs on 2026-04-29) showed the
    # model collapsed onto ``out ≈ mix`` (passthrough), making cue use
    # unnecessary. This term penalizes outputs that look too similar to the
    # input mixture. ReLU hinge so it is exactly 0 (free) until the model
    # actually approaches passthrough. Default weight 0.0 keeps stages
    # 78/88/98/99/100 unchanged.
    passthrough_penalty_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight on the anti-passthrough hinge "
            "max(SI-SDR(out, mix) - threshold, 0). 0.0 disables (default). "
            "Recommend 0.1 for stage 101 to break the out≈mix attractor."
        },
    )
    passthrough_penalty_threshold: float = field(
        default=10.0,
        metadata={
            "help": "SI-SDR threshold (dB) above which similarity to the "
            "input mixture is penalized. Set well below the typical "
            "SI-SDR(target, mix) of clean target/mix pairs (~+30 dB) so "
            "an honestly-separating model never trips the hinge, yet well "
            "above the SI-SDR(out, target) the model reaches at "
            "convergence (~ -1 to +5 dB), so genuine separation is not "
            "discouraged. Default 10 dB."
        },
    )


def _parse_cli_bool(s: str | None) -> bool | None:
    """Parse optional true/false CLI strings; None means do not override YAML."""
    if s is None:
        return None
    x = str(s).strip().lower()
    if x in ("1", "true", "yes", "y", "on"):
        return True
    if x in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(
        f"Expected a boolean string (true/false/yes/no/1/0), got {s!r}"
    )


def apply_tse_model_yaml_cli_overrides(
    data_args: TSEOnlineDataArguments, configs: dict
) -> None:
    """Mutates ``configs['model_args']['tse_model']`` from ``TSEOnlineDataArguments``."""
    tma = configs["model_args"].setdefault("tse_model", {})
    applied: list[str] = []

    if data_args.visual_frontend is not None:
        c = str(data_args.visual_frontend).strip().lower()
        if c not in ("muse", "blaze"):
            raise ValueError(
                f"--visual_frontend must be muse or blaze (got {data_args.visual_frontend!r}); "
                "omit the flag to keep model YAML settings."
            )
        feats = tma.setdefault("visual", {}).setdefault("features", {})
        muse = feats.setdefault("muse_visual", {})
        blaze = feats.setdefault("blaze_visual", {})
        muse["enabled"] = c == "muse"
        blaze["enabled"] = c == "blaze"
        applied.append(f"visual.features: muse_visual.enabled={muse['enabled']} "
                       f"blaze_visual.enabled={blaze['enabled']}")

    bc = _parse_cli_bool(data_args.blaze_visual_causal)
    if bc is not None:
        tma.setdefault("visual", {}).setdefault("features", {}).setdefault(
            "blaze_visual", {}
        )["causal"] = bc
        applied.append(f"visual.features.blaze_visual.causal={bc}")

    sc = _parse_cli_bool(data_args.separator_causal)
    if sc is not None:
        tma.setdefault("separator", {})["causal"] = sc
        applied.append(f"separator.causal={sc}")

    if applied:
        logger.info("CLI overrides on model YAML model_args.tse_model: %s", "; ".join(applied))


def build_dataloader_args_from_training_args(
    training_args: TSETrainingArguments,
    *,
    visual_decode_cuda: bool = True,
) -> dict:
    """Build ``DataLoader`` kwargs from ``TSETrainingArguments`` (no dataloader YAML).

    When ``num_workers > 0`` and **CUDA is available**, use **spawn** (not Linux ``fork``).
    The Hugging Face ``Trainer`` initializes the model on GPU before the DataLoader runs;
    forked worker processes must not inherit a CUDA-initialized parent or they can **deadlock**
    or crawl for a long time at ``0%%`` tqdm (``visual_decode_cuda`` false does not fix this).

    When ``visual_decode_cuda`` is additionally true, TorchCodec may decode inside workers
    on GPU; spawn is still required for the same parent-CUDA reason.
    """
    bs = training_args.per_device_train_batch_size
    if bs is None:
        bs = 1
    nw = int(getattr(training_args, "dataloader_num_workers", 0))
    da = {
        "batch_size": int(bs),
        "drop_last": bool(getattr(training_args, "dataloader_drop_last", True)),
        "num_workers": nw,
        "pin_memory": bool(getattr(training_args, "dataloader_pin_memory", True)),
    }
    if nw > 0:
        # Default 1 (not 2): workers × prefetch batches are staged via shared memory;
        # large wav + visual tensors in Docker/K8s often hit EAGAIN on /dev/shm (64MiB).
        da["prefetch_factor"] = 1
        # Keep worker processes across Trainer epochs to avoid fork/spawn gaps that
        # show up as brief GPU 0% in nvidia-smi when iteration restarts.
        da["persistent_workers"] = True
        if torch.cuda.is_available():
            da["multiprocessing_context"] = torch.multiprocessing.get_context(
                "spawn"
            )
    normalize_dataloader_args(da)
    return da


def _is_dist_rank0() -> bool:
    """Global rank 0 for torchrun/DDP (``RANK`` env). Single-process if unset."""
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


# ------------------------------------------------------------------
# Main training entry
# ------------------------------------------------------------------

def train() -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    
    # ---- model YAML: only ``model`` + ``model_args`` (optional ``model_init``) ----
    parser = HfArgumentParser((TSEOnlineDataArguments, TSETrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    cfg_path = data_args.model_config
    raw = load_yaml_config(cfg_path)
    
    for key in ("model", "model_args"):
        if key not in raw:
            raise ValueError(f"Model YAML must contain '{key}': {cfg_path}")
    configs = {"model": raw["model"], "model_args": raw["model_args"]}
    if "model_init" in raw:
        configs["model_init"] = raw["model_init"]

    apply_tse_model_yaml_cli_overrides(data_args, configs)

    dataloader_args = build_dataloader_args_from_training_args(
        training_args,
        visual_decode_cuda=(
            bool(getattr(data_args, "visual_decode_cuda", True))
            and not bool(getattr(data_args, "skip_visual_decode", False))
        ),
    )
    batch_size = dataloader_args["batch_size"]

    set_seed(training_args.seed)

    # ---- loss (TSEOnlineDataArguments only) ----
    criterion = parse_loss(data_args.loss_type)

    # ---- dataset & dataloader (dataset_args entirely from CLI) ----
    dataset_args = ensure_online_pipeline_defaults(
        build_dataset_args_from_tse_online_data_args(data_args)
    )
    if _is_dist_rank0():
        logger.info(
            "Online mix flags: deterministic=%s av_align=%s force_two_speaker=%s "
            "clean_dry=%s noise_prob=%s reverb_prob=%s snr_range=%s gain_range=%s dist=%s",
            dataset_args.get("online_mix_deterministic"),
            dataset_args.get("online_av_align", False),
            getattr(data_args, "force_two_speaker_only", False),
            getattr(data_args, "online_mix_clean_dry", False),
            dataset_args.get("noise_prob"),
            dataset_args.get("reverb_prob"),
            dataset_args.get("snr_conf", {}).get("range"),
            dataset_args.get("snr_conf", {}).get("gain"),
            dataset_args.get("num_speakers", {}).get("distribution"),
        )
        _ab_mode = getattr(data_args, "visual_cue_ablation", None) or "none"
        if str(_ab_mode).strip().lower() != "none":
            logger.info(
                "Visual cue ablation: mode=%s scope=%s "
                "(use mute vs none or shuffle_within_mixture to test cue usefulness)",
                _ab_mode,
                getattr(data_args, "visual_cue_ablation_scope", "both"),
            )

    train_inventory, val_inventory = build_train_val_merged_audio_visual_inventories(
        train_fraction=data_args.train_speaker_fraction,
        split_seed=training_args.seed,
        voxceleb2_mp4_dir=data_args.mp4_dir_of_voxceleb2,
        use_voxceleb2=data_args.use_voxceleb2,
        lrs3_root=data_args.lrs3_root,
        lrs3_subsets=_parse_lrs3_subsets(data_args.lrs3_subsets),
        use_lrs3=data_args.use_lrs3,
        chinese_lips_root=data_args.chinese_lips_root,
        use_chinese_lips=data_args.use_chinese_lips,
        chinese_lips_train_split=data_args.chinese_lips_train_split,
        chinese_lips_val_split=data_args.chinese_lips_val_split,
        prefix_voxceleb2=data_args.prefix_voxceleb2,
    )

    speakers_opt = None
    rng_train = random.Random(training_args.seed)
    rng_val = random.Random(training_args.seed + 100_003)
    train_speaker_ids = resolve_speaker_pool(
        train_inventory,
        rng_train,
        speakers=speakers_opt,
        num_speakers=data_args.num_speaker_pool,
    )
    val_speaker_ids = resolve_speaker_pool(
        val_inventory,
        rng_val,
        speakers=speakers_opt,
        num_speakers=data_args.num_speaker_pool,
    )
    if len(train_speaker_ids) < 2 or len(val_speaker_ids) < 2:
        raise ValueError(
            "After pooling, train and val each need >=2 speakers for mixing; "
            f"got train={len(train_speaker_ids)} val={len(val_speaker_ids)}"
        )

    train_inventory = subset_inventory(train_inventory, train_speaker_ids)
    val_inventory = subset_inventory(val_inventory, val_speaker_ids)

    train_dataset = OnlineMixIterableDataset(
        train_inventory,
        train_speaker_ids,
        dataset_args,
        seed=training_args.seed,
        with_visual_cue=True,
        skip_visual_decode=data_args.skip_visual_decode,
    )
    val_dataset = OnlineMixIterableDataset(
        val_inventory,
        val_speaker_ids,
        dataset_args,
        seed=training_args.seed + 7,
        with_visual_cue=True,
        skip_visual_decode=data_args.skip_visual_decode,
    )

    if data_args.train_cues:
        collect_keys = build_collect_keys(
            load_yaml(data_args.train_cues),
            dataset_args,
            BASE_COLLECT_KEYS,
        )
    else:
        collect_keys = build_collect_keys_online(
            dataset_args,
            BASE_COLLECT_KEYS,
            tse_model_name=configs["model"].get("tse_model"),
        )

    _vmf = dataset_args.get("visual_max_frames")

    # Keyword-only partial: DataLoader passes ``batch`` positionally; spawn pickling
    # forbids lambdas. (Do not use partial(fn, collect_keys, ...) — that binds
    # ``batch``'s slot wrong.)
    _collate = partial(
        tse_collate_fn,
        collect_keys=collect_keys,
        visual_max_frames=_vmf,
    )
    train_dl = DataLoader(
        train_dataset,
        **dataloader_args,
        collate_fn=_collate,
    )
    val_dl = DataLoader(
        val_dataset,
        **dataloader_args,
        collate_fn=_collate,
    )

    # ---- compute steps per epoch ----
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if int(dataset_args.get("sample_num_per_epoch", 0)) <= 0:
        raise ValueError("train_with_transformers_online requires --sample_num_per_epoch > 0.")
    sample_num_per_epoch = int(dataset_args["sample_num_per_epoch"])
    epoch_iter = sample_num_per_epoch // world_size // batch_size

    if int(dataset_args.get("sample_num_per_epoch_val", 0)) > 0:
        val_sample_num = int(dataset_args["sample_num_per_epoch_val"])
    else:
        val_sample_num = default_online_val_samples_per_epoch(sample_num_per_epoch)
    val_iter = val_sample_num // world_size // batch_size

    train_dataloader = _LimitedDataLoader(train_dl, epoch_iter)
    val_dataloader = _LimitedDataLoader(val_dl, val_iter)

    # ---- model ----
    if data_args.tse_model is not None:
        configs["model"]["tse_model"] = data_args.tse_model
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"]
    )

    init_ckpt = data_args.model_init_checkpoint
    if init_ckpt is None:
        init_ckpt = configs.get("model_init", {}).get("tse_model")
    if init_ckpt is not None:
        load_pretrained_model(model, init_ckpt)



    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"tse_model: {configs['model']['tse_model']}, "
                f"params: {num_params / 1e6:.2f}M")
    logger.info(
        "train: OnlineMixIterableDataset multi-source "
        f"use_voxceleb2={data_args.use_voxceleb2} use_lrs3={data_args.use_lrs3} "
        f"use_chinese_lips={data_args.use_chinese_lips} "
        f"mp4_dir_of_voxceleb2={data_args.mp4_dir_of_voxceleb2!r} "
        f"lrs3_root={data_args.lrs3_root!r} chinese_lips_root={data_args.chinese_lips_root!r} "
        f"train_speaker_fraction={data_args.train_speaker_fraction} "
        f"train_speakers={len(train_speaker_ids)} val_speakers={len(val_speaker_ids)} "
        f"skip_visual_decode={data_args.skip_visual_decode} "
        f"visual_resize={data_args.visual_resize} "
        f"visual_spatial_size={dataset_args.get('visual_spatial_size')!r} "
        f"visual_max_frames={dataset_args.get('visual_max_frames')!r}"
    )
    logger.info(
        "Note: collate expands along speakers, so effective rows per micro-batch ≈ "
        "sum of num_speaker per mixture; 3–4-speaker mixes use more VRAM than offline "
        "with the same per_device_train_batch_size. Use --visual_max_frames, "
        "--visual_resize (multi-corpus), --num_speakers_distribution / --num_speakers_max, "
        "or --bf16 to reduce memory."
    )
    logger.info(
        f"online val: train_mixtures/epoch={sample_num_per_epoch} "
        f"val_mixtures/epoch={val_sample_num} (default ratio 20k:5k unless overridden)"
    )
    logger.info(f"loss: {data_args.loss_type}")
    logger.info(f"epoch_iter: {epoch_iter}, val_iter: {val_iter}, "
                f"num_train_epochs: {training_args.num_train_epochs}")
    logger.info(f"world_size: {world_size}, batch_size/gpu: {batch_size}")
    _nw_dl = int(dataloader_args.get("num_workers", 0))
    _mpc = dataloader_args.get("multiprocessing_context")
    _spawn = (
        _mpc.get_start_method() if _mpc is not None else None
    )
    _vdec = bool(dataset_args.get("visual_decode_cuda", True)) and torch.cuda.is_available()
    if _nw_dl > 0:
        logger.info(
            "DataLoader num_workers=%s prefetch_factor=%s persistent_workers=%s "
            "worker_start_method=%s visual_torchcodec_cuda=%s "
            "(spawn avoids fork-after-CUDA in workers; tensors still use /dev/shm — "
            "RuntimeError shm → workers 0 or larger --shm-size)",
            _nw_dl,
            dataloader_args.get("prefetch_factor"),
            dataloader_args.get("persistent_workers"),
            repr(_spawn),
            _vdec,
        )
        #if _spawn == "spawn" and _nw_dl > 0:
        #    logger.info(
        #        "Note: spawn workers each perform a fresh Python import + CUDA/TorchCodec "
        #        "init — first training step after start can lag at 0%% tqdm for tens of "
        #        "minutes. For a faster cold start while keeping GPU decode in the trainer "
        #        "process, use --dataloader_num_workers 0 with --visual_decode_cuda true; "
        #        "raising num_workers trades that for parallel prefetch once workers finish "
        #        "initializing."
        #    )
    else:
        logger.info(
            "DataLoader num_workers=0 worker_start_method=n/a visual_torchcodec_cuda=%s "
            "(TorchCodec decode runs in training process; see --visual_decode_cuda)",
            _vdec,
        )
        if _vdec:
            logger.warning(
                "num_workers=0 + visual_decode_cuda=true decodes MP4 sequentially in "
                "the trainer process — on networked storage the first handful of mixtures "
                "can take tens of minutes. For prefetch parallelism increase "
                "--dataloader_num_workers (spawn workers pay import + TorchCodec/CUDA init "
                "up front). Optional: --visual_decode_cuda false moves decode off the GPU."
            )

    loss_type_str = data_args.loss_type
    if isinstance(loss_type_str, list):
        loss_type_str = loss_type_str[0]

    # ---- build TSETrainer ----
    callbacks: list[TrainerCallback] = [
        _SetEpochCallbackMulti(train_dataset, val_dataset),
    ]
    plateau_cb: ClearerVoicePlateauHalvingCallback | None = None
    if training_args.clearervoice_plateau_halving:
        plateau_cb = ClearerVoicePlateauHalvingCallback(
            patience_evals=training_args.clearervoice_plateau_patience_evals,
            min_delta=training_args.clearervoice_plateau_min_delta,
        )
        callbacks.append(plateau_cb)

    ema_cb: EMACallback | None = None
    if bool(getattr(training_args, "use_ema", False)):
        ema_cb = EMACallback(decay=float(training_args.ema_decay))
        callbacks.append(ema_cb)

    vf_unfreeze_cb: VisualUnfreezeCallback | None = None
    _vf_mode = getattr(training_args, "visual_unfreeze_mode", "none").strip().lower()
    if _vf_mode in ("metric", "step"):
        _vf_thresholds: list[float] = []
        _raw_thr = getattr(training_args, "visual_unfreeze_thresholds", "") or ""
        if _raw_thr.strip():
            _vf_thresholds = [float(x) for x in _raw_thr.split(",") if x.strip()]
        _vf_steps: list[int] = []
        _raw_stp = getattr(training_args, "visual_unfreeze_steps", "") or ""
        if _raw_stp.strip():
            _vf_steps = [int(x) for x in _raw_stp.split(",") if x.strip()]
        vf_unfreeze_cb = VisualUnfreezeCallback(
            mode=_vf_mode,
            metric=getattr(training_args, "visual_unfreeze_metric", "eval_sisdr_improvement"),
            thresholds=_vf_thresholds,
            steps=_vf_steps,
            lr_scale=float(getattr(training_args, "visual_unfreeze_lr_scale", 0.1)),
        )
        callbacks.append(vf_unfreeze_cb)
        logger.info(
            "VisualUnfreeze: mode=%s  metric=%s  thresholds=%s  steps=%s  lr_scale=%.4f",
            _vf_mode,
            training_args.visual_unfreeze_metric,
            _vf_thresholds,
            _vf_steps,
            training_args.visual_unfreeze_lr_scale,
        )

    trainer = TSETrainer(
        model=model,
        args=training_args,
        criterion=criterion,
        loss_type=loss_type_str,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )
    if plateau_cb is not None:
        plateau_cb.attach_trainer(trainer)
    if ema_cb is not None:
        ema_cb.attach_trainer(trainer)
    if vf_unfreeze_cb is not None:
        vf_unfreeze_cb.attach_trainer(trainer)

    _has_ckpt = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if _has_ckpt:
        # --- fast-forward: arm the dataset so the Trainer's skip_first_batches
        # consumes audio-only samples (no GPU video decode) ---
        from transformers.trainer_utils import get_last_checkpoint
        _ckpt_dir = get_last_checkpoint(training_args.output_dir)
        if _ckpt_dir is not None:
            _state_path = os.path.join(_ckpt_dir, "trainer_state.json")
            if os.path.isfile(_state_path):
                import json as _json
                with open(_state_path) as _f:
                    _ts = _json.load(_f)
                _global_step = int(_ts.get("global_step", 0))
                _grad_acc = int(getattr(training_args, "gradient_accumulation_steps", 1))
                _num_upd_per_epoch = math.ceil(epoch_iter / _grad_acc)
                _steps_in_epoch = (_global_step % _num_upd_per_epoch) * _grad_acc
                _skip_samples = _steps_in_epoch * batch_size
                if _skip_samples > 0:
                    train_dataset.set_fast_forward(_skip_samples, batch_size)
                    logger.info(
                        "[resume] checkpoint %s: global_step=%s, "
                        "steps_in_epoch=%s, skip_samples=%s "
                        "(audio-only fast-forward armed on train_dataset)",
                        _ckpt_dir, _global_step, _steps_in_epoch, _skip_samples,
                    )
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
