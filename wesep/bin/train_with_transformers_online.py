# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)
#               2026 Duo Ma (maduo@mycompany.com)
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
Like ``train_with_transformers.py``, but **train and validation** both use
:class:`wesep.dataset.online_dataset.OnlineMixIterableDataset` (MP4 inventory
→ online mixture + ``sample_fixed_visual_cue``), matching
``gen_online_mix_data_with_aug_without_repeat_speaker_with_visual_cue.py``.

Speakers from ``mp4_dir`` are split into **disjoint** train / val pools (default
80% / 20%). Validation mixture count per epoch defaults to ``train * 5000/20000``
unless set in YAML or via ``--sample_num_per_epoch_val``.

**YAML:** pass a single **model-only** file (``--model_config``) with ``model`` and
``model_args`` only — see ``confs/tse_bsrnn_visual_model.yaml``. Data pipeline,
DataLoader, and loss come from ``TSEOnlineDataArguments`` and ``TSETrainingArguments``.

**Cues / collate:** ``--cue_visual_*`` or ``--train_cues``.

Requires ``--sample_num_per_epoch`` > 0 (default 20000). At least **4** pooled speakers.
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
from typing import Optional
from collections import defaultdict
faulthandler.enable()
from contextlib import nullcontext
from pprint import pformat
import pathlib

import matplotlib.pyplot as plt
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
from wesep.dataset.online_dataset import (
    OnlineMixIterableDataset,
    build_dataset_args_from_tse_online_data_args,
    default_online_val_samples_per_epoch,
    ensure_online_pipeline_defaults,
    resolve_speaker_pool,
    scan_mp4_dir_of_voxceleb2,
    split_speaker_ids_train_val,
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

        if cues is None:
            outputs = model(mix)
        else:
            outputs = model(mix, cues)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
            loss = loss_function(outputs, target, mix, self.loss_type)

        self._log_metrics(loss)
        return (loss,outputs) if return_outputs else loss

        
    def _log_metrics(self,loss):
        self._metrics["loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())
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
        """
        model.eval()
        sisnr_loss=auraloss.time.SISDRLoss()
        mix, cues, target = self._executor._extract_model_inputs(
            inputs, self.args.device,
        )
        with torch.no_grad():
            if cues is None:
                outputs = model(mix)
            else:
                outputs = model(mix, cues)

            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            loss = sisnr_loss(outputs[0], target).mean()
        return (loss, None, None)
     




def loss_function(outputs, target, mix, loss_type):
    if loss_type == "SISDR":
        sisdr_loss=auraloss.time.SISDRLoss()
        return sisdr_loss(outputs[0], target).mean()
    elif loss_type == "SISNR":
        sisnr_loss=auraloss.time.SISDRLoss()
        return sisnr_loss(outputs[0], target).mean()
    elif loss_type == "SNR":
        snr_loss=auraloss.time.SNRLoss()
        return snr_loss(outputs[0], target).mean()
    elif loss_type == "OnlineAVCrossNetLoss":
        from wesep.utils.losses import OnlineAVCrossNetLoss
        onlineavcrossnet_loss=OnlineAVCrossNetLoss()
        return onlineavcrossnet_loss(outputs[0], target, mix)
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

    mp4_dir_of_voxceleb2: str = field(metadata={"help": "MP4 root (VoxCeleb2 layout) for online mixture train/val."})

    # ---- Online mix / audio (no dataset YAML) ----
    resample_rate: int = field(default=16000, metadata={"help": "Target sample rate."})
    chunk_len: int = field(default=48000, metadata={"help": "Random chunk length (samples) after resample. per mix audio length."})
    whole_utt: bool = field(default=False, metadata={"help": "If set, disable random_chunk."})
    online_buffer_size: int = field(
        default=8,
        metadata={
            "help": "Buffer for sample_speaker_group_without_repeat (see multi_data_online). "
            "Lower → faster first mixed sample."
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
    visual_decode_cuda: bool = field(
        default=True,
        metadata={
            "help": "Decode MP4 with TorchCodec on GPU when possible; with "
            "dataloader workers > 0 uses spawn so workers may use CUDA. "
            "Set false for CPU-only video decode."
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
    # Do not add dataloader_prefetch_factor here: HF TrainingArguments forbids setting it when
    # dataloader_num_workers == 0. We only pass prefetch_factor to torch DataLoader when workers > 0.
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


def build_dataloader_args_from_training_args(
    training_args: TSETrainingArguments,
    *,
    visual_decode_cuda: bool = True,
) -> dict:
    """Build ``DataLoader`` kwargs from ``TSETrainingArguments`` (no dataloader YAML).

    When ``num_workers > 0`` and CUDA is available, use **spawn** so workers are not forked
    from a CUDA-initialized training process (unsafe; can hang at 0%% tqdm).
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
        da["prefetch_factor"] = 2
        # Keep worker processes across Trainer epochs to avoid fork/spawn gaps that
        # show up as brief GPU 0% in nvidia-smi when iteration restarts.
        da["persistent_workers"] = True
        if torch.cuda.is_available():
            da["multiprocessing_context"] = torch.multiprocessing.get_context(
                "spawn"
            )
    normalize_dataloader_args(da)
    return da


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

    rng_pool = random.Random(training_args.seed)
    inventory = scan_mp4_dir_of_voxceleb2(data_args.mp4_dir_of_voxceleb2)
    speakers_opt = None
    #if data_args.speakers:
    #    speakers_opt = [
    #        s.strip() for s in data_args.speakers.split(",") if s.strip()
    #    ]
    speaker_ids = resolve_speaker_pool(
        inventory,
        rng_pool,
        speakers=speakers_opt,
        num_speakers=data_args.num_speaker_pool,
    )
    if len(speaker_ids) < 4:
        raise ValueError(
            "Online train/val split needs at least 4 pooled speakers "
            "(each of train and val needs >=2 speakers for mixing)."
        )

    tr_spk, va_spk = split_speaker_ids_train_val(
        speaker_ids,
        train_fraction=data_args.train_speaker_fraction,
        seed=training_args.seed,
    )
    train_inventory = subset_inventory(inventory, tr_spk)
    val_inventory = subset_inventory(inventory, va_spk)
    train_speaker_ids = tr_spk
    val_speaker_ids = va_spk
    # Subsets share clip list objects with ``inventory``; drop the full map so we
    # only retain the train/val dict shells (same clip metadata, one less large dict).
    del inventory

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
        f"train: OnlineMixIterableDataset mp4_dir_of_voxceleb2={data_args.mp4_dir_of_voxceleb2} "
        f"train_speakers={len(train_speaker_ids)} val_speakers={len(val_speaker_ids)} "
        f"skip_visual_decode={data_args.skip_visual_decode} "
        f"visual_max_frames={dataset_args.get('visual_max_frames')!r} "
        f"(CLI --visual_max_frames, 0 = no cap)"
    )
    logger.info(
        "Note: collate expands along speakers, so effective rows per micro-batch ≈ "
        "sum of num_speaker per mixture; 3–4-speaker mixes use more VRAM than offline "
        "with the same per_device_train_batch_size. Use --visual_max_frames, "
        "--num_speakers_distribution / --num_speakers_max, or --bf16 to reduce memory."
    )
    logger.info(
        f"online val: train_mixtures/epoch={sample_num_per_epoch} "
        f"val_mixtures/epoch={val_sample_num} (default ratio 20k:5k unless overridden)"
    )
    logger.info(f"loss: {data_args.loss_type}")
    logger.info(f"epoch_iter: {epoch_iter}, val_iter: {val_iter}, "
                f"num_train_epochs: {training_args.num_train_epochs}")
    logger.info(f"world_size: {world_size}, batch_size/gpu: {batch_size}")
    logger.info(
        f"DataLoader num_workers={dataloader_args.get('num_workers', 0)} "
        "(from TSETrainingArguments --dataloader_num_workers)"
    )

    loss_type_str = data_args.loss_type
    if isinstance(loss_type_str, list):
        loss_type_str = loss_type_str[0]

    # ---- build TSETrainer ----
    trainer = TSETrainer(
        model=model,
        args=training_args,
        criterion=criterion,
        loss_type=loss_type_str,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        eval_dataset=val_dataset,
        callbacks=[_SetEpochCallbackMulti(train_dataset, val_dataset)],
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
