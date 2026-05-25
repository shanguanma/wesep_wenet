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
Training script that mirrors ``train.py`` but uses:
  - Cosine / linear / constant LR schedules (with warmup) via CLI
  - Standard ``torch.optim`` optimizers configured via CLI
  - ``argparse`` CLI for all training hyper-parameters
  - YAML config for dataset / model definitions only

The distributed training loop, DDP wrapping, and checkpoint format are
identical to ``train.py``.  No extra dependencies beyond what ``train.py``
already uses.
"""

import argparse
import auraloss
import faulthandler
import logging
import math
import os
import re
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
    tse_collate_fn,
)
from wesep.dataset.dataset import Dataset
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


class _SetEpochCallback(TrainerCallback):
    """Call ``dataset.set_epoch`` at the beginning of every epoch so that
    the ``DistributedSampler`` inside the wesep ``Dataset`` re-shuffles."""

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + 1
        self.train_dataset.set_epoch(epoch)


# ------------------------------------------------------------------
# CLI dataclasses for HfArgumentParser
# ------------------------------------------------------------------

@dataclass
class TSEDataArguments:
    config: str = field(metadata={"help": "YAML config path (dataset_args, model/model_args, loss, dataloader_args)."})
    data_type: str = field(default="raw", metadata={"help": "Dataset type: raw or shard."})
    train_data: str = field(default=None, metadata={"help": "Training data list path."})
    val_data: str = field(default=None, metadata={"help": "Validation data list path."})
    train_samples: str = field(default=None, metadata={"help": "Training samples file (for counting epoch iterations)."})
    val_samples: str = field(default=None, metadata={"help": "Validation samples file."})
    train_cues: str = field(default=None, metadata={"help": "Training cues YAML path."})
    val_cues: str = field(default=None, metadata={"help": "Validation cues YAML path."})
    tse_model: str = field(default=None, metadata={"help": "TSE model name override."})
    loss_type: str = field(default=None, metadata={"help": "Loss type override."})
    #checkpoint: str = field(default=None, metadata={"help": "Resume from a Trainer checkpoint directory."})


@dataclass
class TSETrainingArguments(TrainingArguments):
    num_train_epochs: float = field(default=150, metadata={"help": "Total training epochs."})
    max_steps: int = field(default=-1, metadata={"help": "Overrides num_train_epochs if positive."})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy."})
    save_steps: int = field(default=100, metadata={"help": "Save every N steps."})
    save_total_limit: int = field(default=10, metadata={"help": "Max checkpoints to keep."})
    eval_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy."})
    eval_steps: int = field(default=100, metadata={"help": "Evaluate every N steps."})
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

# ------------------------------------------------------------------
# Main training entry
# ------------------------------------------------------------------

def train() -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    
    # ---- load YAML config ----
    parser = HfArgumentParser((TSEDataArguments, TSETrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    configs = load_yaml_config(data_args.config)
    for key in ("model", "model_args", "dataset_args", "dataloader_args"):
        if key not in configs:
            raise ValueError(f"YAML config must contain '{key}'")

    dataloader_args = configs["dataloader_args"]
    normalize_dataloader_args(dataloader_args)
    if training_args.per_device_train_batch_size is not None:
        dataloader_args["batch_size"] = training_args.per_device_train_batch_size
    batch_size = dataloader_args["batch_size"]

    # YAML often sets num_workers: 2 with prefetch_factor; worker processes use
    # POSIX shared memory (/dev/shm). Small shm (common in Docker/k8s) → bus error.
    # TrainingArguments.dataloader_num_workers (default 0) overrides YAML; use
    # --dataloader_num_workers N to re-enable workers when shm is large enough.
    dataloader_args["num_workers"] = int(
        getattr(training_args, "dataloader_num_workers", 0)
    )
    normalize_dataloader_args(dataloader_args)

    set_seed(training_args.seed)

    # ---- loss ----
    if data_args.loss_type is not None:
        configs["loss"] = data_args.loss_type
    else:
        configs["loss"] = configs["loss"]
        
    criterion_cfg = configs.get("loss", None)
    if criterion_cfg:
        criterion = parse_loss(criterion_cfg)
    else:
        criterion = [parse_loss("SISDR")]

    # ---- dataset & dataloader ----
    dataset_args = configs["dataset_args"]
    train_dataset = Dataset(
        data_args.data_type,
        data_args.train_data,
        dataset_args,
        state="train",
        repeat_dataset=configs.get("repeat_dataset", True),
        cues_yaml=data_args.train_cues,
    )
    val_dataset = Dataset(
        data_args.data_type,
        data_args.val_data,
        dataset_args,
        state="val",
        repeat_dataset=True,
        cues_yaml=data_args.val_cues,
    )
    train_collect_keys = build_collect_keys(
        load_yaml(data_args.train_cues), dataset_args, BASE_COLLECT_KEYS,
    )
    val_collect_keys = build_collect_keys(
        load_yaml(data_args.val_cues), dataset_args, BASE_COLLECT_KEYS,
    )
    train_dl = DataLoader(
        train_dataset,
        **dataloader_args,
        collate_fn=lambda batch: tse_collate_fn(batch, train_collect_keys),
    )
    val_dl = DataLoader(
        val_dataset,
        **dataloader_args,
        collate_fn=lambda batch: tse_collate_fn(batch, val_collect_keys),
    )

    # ---- compute steps per epoch ----
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if dataset_args.get("sample_num_per_epoch", 0) > 0:
        sample_num_per_epoch = dataset_args["sample_num_per_epoch"]
    else:
        with open(data_args.train_samples, "r", encoding="utf-8") as f:
            sample_num_per_epoch = sum(1 for _ in f)
    epoch_iter = sample_num_per_epoch // world_size // batch_size

    with open(data_args.val_samples, "r", encoding="utf-8") as f:
        val_sample_num = sum(1 for _ in f)
    val_iter = val_sample_num // world_size // batch_size

    train_dataloader = _LimitedDataLoader(train_dl, epoch_iter)
    val_dataloader = _LimitedDataLoader(val_dl, val_iter)

    # ---- model ----
    if data_args.tse_model is not None:
        configs["model"]["tse_model"] = data_args.tse_model
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"]
    )

    model_init_cfg = configs.get("model_init", {})
    if model_init_cfg.get("tse_model") is not None:
        load_pretrained_model(model, model_init_cfg["tse_model"])

    # ---- save config (rank 0 only) ----
    #if training_args.local_rank <= 0:
    #    os.makedirs(training_args.output_dir, exist_ok=True)
    #    merged = dict(configs)
    #    merged["cli_args"] = vars(data_args)
    #    with open(os.path.join(training_args.output_dir, "config.yaml"), "w") as f:
    #        f.write(yaml.dump(merged))

    # ---- logging ----
    logger = logging.getLogger(__name__)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"tse_model: {configs['model']['tse_model']}, "
                f"params: {num_params / 1e6:.2f}M")
    logger.info(f"loss: {configs.get('loss', 'SISDR')}")
    logger.info(f"epoch_iter: {epoch_iter}, val_iter: {val_iter}, "
                f"num_train_epochs: {training_args.num_train_epochs}")
    logger.info(f"world_size: {world_size}, batch_size/gpu: {batch_size}")
    logger.info(
        f"DataLoader num_workers={dataloader_args.get('num_workers', 0)} "
        "(from TrainingArguments; overrides YAML — use --dataloader_num_workers N if shm allows)"
    )

    # ---- resolve loss_type string for fallback ----
    loss_type_str = configs.get("loss", "SISDR")
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
        callbacks=[_SetEpochCallback(train_dataset)],
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
