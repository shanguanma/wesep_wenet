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
import faulthandler
import logging
import math
import os
import re

faulthandler.enable()
from contextlib import nullcontext
from pprint import pformat

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
#from transformers.utils import logging 

MAX_NUM_log_files = 100
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
#logger = logging.get_logger(__name__)

def load_yaml_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def normalize_dataloader_args(da: dict) -> None:
    """Drop keys invalid when num_workers == 0."""
    if int(da.get("num_workers", 0)) <= 0:
        da.pop("prefetch_factor", None)
        da.pop("persistent_workers", None)


# ------------------------------------------------------------------
# LR schedulers (pure PyTorch, no transformers dependency)
# ------------------------------------------------------------------

def _get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def _get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)


def _get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def _get_polynomial_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, power=1.0
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        lr_range = 1.0 - 0.0
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1.0 - (current_step - num_warmup_steps) / max(1, decay_steps)
        return lr_range * pct_remaining ** power
    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    """Create an LR scheduler by name. Same names as transformers.get_scheduler."""
    name = name.lower().replace("-", "_")
    if name in ("cosine", "cosine_with_warmup"):
        return _get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif name in ("linear", "linear_with_warmup"):
        return _get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif name in ("constant", "constant_with_warmup"):
        return _get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif name in ("polynomial", "polynomial_with_warmup"):
        return _get_polynomial_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            "Supported: cosine, linear, constant, polynomial"
        )


# ------------------------------------------------------------------
# Training / validation loops (same structure as Executor)
# ------------------------------------------------------------------

def train_one_epoch(
    dataloader,
    model,
    epoch_iter,
    optimizer,
    criterion,
    scheduler,
    scaler,
    epoch,
    enable_amp,
    logger,
    clip_grad,
    log_batch_interval,
    device,
    se_loss_weight,
    executor,
):
    model.train()
    losses = []

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_context = model.join
    else:
        model_context = nullcontext

    with model_context():
        for i, batch in enumerate(dataloader):
            mix, cues, target = executor._extract_model_inputs(batch, device)

            with torch.cuda.amp.autocast(enabled=enable_amp):
                if cues is None:
                    outputs = model(mix)
                else:
                    outputs = model(mix, cues)

                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]

                loss = 0.0
                for ii in range(len(criterion)):
                    for ji in range(len(se_loss_weight[0][ii])):
                        out_idx = se_loss_weight[0][ii][ji]
                        w = se_loss_weight[1][ii][ji]
                        loss = loss + w * (
                            criterion[ii](outputs[out_idx], target).mean()
                        )

            losses.append(loss.item())
            total_loss_avg = sum(losses) / len(losses)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradients(model, clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (i + 1) % log_batch_interval == 0:
                logger.info(
                    tp.row(
                        (
                            "TRAIN",
                            epoch,
                            i + 1,
                            total_loss_avg,
                            optimizer.param_groups[0]["lr"],
                        ),
                        width=10,
                        style="grid",
                    )
                )

            if (i + 1) == epoch_iter:
                break

    total_loss_avg = sum(losses) / len(losses)
    return total_loss_avg, 0


def validate(
    dataloader,
    model,
    val_iter,
    criterion,
    epoch,
    enable_amp,
    logger,
    log_batch_interval,
    device,
    executor,
):
    model.eval()
    losses = []
    import auraloss
    sisdr_loss=auraloss.time.SISDRLoss()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            mix, cues, target = executor._extract_model_inputs(batch, device)

            with torch.cuda.amp.autocast(enabled=enable_amp):
                if cues is None:
                    outputs = model(mix)
                else:
                    outputs = model(mix, cues)

                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                loss = sisdr_loss(outputs[0], target)
                #loss = criterion[0](outputs[0], target).mean()

            losses.append(loss.item())
            total_loss_avg = sum(losses) / len(losses)

            if (i + 1) % log_batch_interval == 0:
                logger.info(
                    tp.row(
                        ("VAL", epoch, i + 1, total_loss_avg, "-"),
                        width=10,
                        style="grid",
                    )
                )

            if (i + 1) == val_iter:
                break

    total_loss_avg = sum(losses) / len(losses)
    return total_loss_avg, 0

def loss_function(outputs, target, mix, loss_type):
    if loss_type == "SISDR":
        sisdr_loss=auraloss.time.SISDRLoss()
        return sisdr_loss(outputs[0], target)
    elif loss_type == "SISNR":
        sisnr_loss=auraloss.time.SISDRLoss()
        return sisnr_loss(outputs[0], target)
    elif loss_type == "SNR":
        snr_loss=auraloss.time.SNRLoss()
        return snr_loss(outputs[0], target)
    elif loss_type == "OnlineAVCrossNetLoss":
        from wesep.utils.losses import OnlineAVCrossNetLoss
        onlineavcrossnet_loss=OnlineAVCrossNetLoss()
        return onlineavcrossnet_loss(outputs[0], target, mix)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
# ------------------------------------------------------------------
# Main training entry
# ------------------------------------------------------------------

def train(cli_args: argparse.Namespace) -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    configs = load_yaml_config(cli_args.config)

    for key in ("model", "model_args", "dataset_args", "dataloader_args"):
        if key not in configs:
            raise ValueError(f"YAML config must contain '{key}'")

    dataloader_args = configs["dataloader_args"]
    normalize_dataloader_args(dataloader_args)

    if cli_args.batch_size is not None:
        dataloader_args["batch_size"] = cli_args.batch_size
    else:
        dataloader_args["batch_size"] = dataloader_args["batch_size"]

    checkpoint = cli_args.checkpoint
    if checkpoint is not None:
        checkpoint = os.path.realpath(checkpoint)
    find_unused_parameters = cli_args.find_unused_parameters

    # ---- distributed setup (same as train.py) ----
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus = [int(g) for g in cli_args.gpus.split(",")]
    gpu = gpus[rank]
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl")

    model_dir = os.path.join(cli_args.exp_dir, "models")
    logger = setup_logger(rank, cli_args.exp_dir, gpu, MAX_NUM_log_files)

    print("-------------------", dist.get_rank(), world_size)
    if world_size > 1:
        logger.info("training on multiple gpus, this gpu {}".format(gpu))

    if rank == 0:
        logger.info(f"exp_dir is: {cli_args.exp_dir}")
        logger.info("<== Config (train_md) ==>")
        logger.info(f"CLI args:\n{pformat(vars(cli_args))}")
        logger.info(f"YAML config:\n{pformat(configs)}")

    set_seed(cli_args.seed + rank)

    # ---- loss ----
    if cli_args.loss_type is not None:
        configs["loss"] = cli_args.loss_type
    else:
        configs["loss"] = configs["loss"]
        
    criterion_cfg = configs.get("loss", None)
    if criterion_cfg:
        criterion = parse_loss(criterion_cfg)
    else:
        criterion = [parse_loss("SISDR")]
    loss_args_cfg = configs.get("loss_args", {})
    loss_posi = loss_args_cfg.get("loss_posi", [[0]])
    loss_weight = loss_args_cfg.get("loss_weight", [[1.0]])
    loss_args = (loss_posi, loss_weight)

    
    # ---- dataset & dataloader (same as train.py) ----
    dataset_args = configs["dataset_args"]

    train_dataset = Dataset(
        cli_args.data_type,
        cli_args.train_data,
        dataset_args,
        state="train",
        repeat_dataset=configs.get("repeat_dataset", True),
        cues_yaml=cli_args.train_cues,
    )
    val_dataset = Dataset(
        cli_args.data_type,
        cli_args.val_data,
        dataset_args,
        state="val",
        repeat_dataset=True,
        cues_yaml=cli_args.val_cues,
    )
    train_collect_keys = build_collect_keys(
        load_yaml(cli_args.train_cues),
        dataset_args,
        BASE_COLLECT_KEYS,
    )
    val_collect_keys = build_collect_keys(
        load_yaml(cli_args.val_cues),
        dataset_args,
        BASE_COLLECT_KEYS,
    )
    train_dataloader = DataLoader(
        train_dataset,
        **dataloader_args,
        collate_fn=lambda batch: tse_collate_fn(batch, train_collect_keys),
    )
    val_dataloader = DataLoader(
        val_dataset,
        **dataloader_args,
        collate_fn=lambda batch: tse_collate_fn(batch, val_collect_keys),
    )

    batch_size = dataloader_args["batch_size"]
    if dataset_args.get("sample_num_per_epoch", 0) > 0:
        sample_num_per_epoch = dataset_args["sample_num_per_epoch"]
    else:
        with open(cli_args.train_samples, "r", encoding="utf-8") as f:
            sample_num_per_epoch = sum(1 for _ in f)
    epoch_iter = sample_num_per_epoch // world_size // batch_size

    with open(cli_args.val_samples, "r", encoding="utf-8") as f:
        val_sample_num = sum(1 for _ in f)
    val_iter = val_sample_num // world_size // batch_size

    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info("epoch iteration number: {}".format(epoch_iter))
        logger.info("val iteration number: {}".format(val_iter))

    # ---- model (same as train.py) ----
    logger.info("<== Model ==>")
    if cli_args.tse_model is not None:
        configs["model"]["tse_model"] = cli_args.tse_model
    else:
        configs["model"]["tse_model"] = configs["model"]["tse_model"]
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"]
    )
    num_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        logger.info("tse_model size: {:.2f} M".format(num_params / 1e6))
        for line in pformat(model).split("\n"):
            logger.info(line)

    # ddp_model (same as train.py)
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=find_unused_parameters
    )
    device = torch.device("cuda")

    if rank == 0:
        logger.info("<== TSE Model Loss ==>")
        logger.info("loss criterion is: {}".format(configs.get("loss", "SISDR")))

    # ---- HuggingFace optimizer ----
    optimizer = getattr(torch.optim, cli_args.optim_class)(
        ddp_model.parameters(),
        lr=cli_args.learning_rate,
        weight_decay=cli_args.weight_decay,
    )
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: {} (torch.optim)".format(cli_args.optim_class))

    # ---- HuggingFace LR scheduler ----
    total_steps = cli_args.num_epochs * epoch_iter
    warmup_steps = int(total_steps * cli_args.warmup_ratio)

    scheduler = create_scheduler(
        name=cli_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if rank == 0:
        logger.info("<== LR Scheduler ==>")
        logger.info(
            "scheduler is: {} (create_scheduler), "
            "warmup_steps={}, total_steps={}".format(
                cli_args.lr_scheduler_type, warmup_steps, total_steps
            )
        )

    # ---- load pretrained / checkpoint ----
    model_init_cfg = configs.get("model_init", {})
    if model_init_cfg.get("tse_model") is not None:
        logger.info(
            "Load initial model from {}".format(model_init_cfg["tse_model"])
        )
        load_pretrained_model(ddp_model, model_init_cfg["tse_model"])
    elif checkpoint is None:
        logger.info("Train model from scratch ...")

    for c in criterion:
        c = c.to(device)

    model_list = [ddp_model]
    optimizer_list = [optimizer]
    scheduler_list = [scheduler]
    scaler = torch.cuda.amp.GradScaler(enabled=cli_args.enable_amp)

    if checkpoint is not None:
        load_checkpoint(
            model_list, optimizer_list, scheduler_list, scaler, checkpoint
        )
        start_epoch = (
            int(re.findall(r"(?<=checkpoint_)\d*(?=.pt)", checkpoint)[0]) + 1
        )
        logger.info("Load checkpoint: {}".format(checkpoint))
    else:
        start_epoch = 1
    logger.info("start_epoch: {}".format(start_epoch))

    # ---- save config ----
    if rank == 0:
        merged = dict(configs)
        merged["cli_args"] = vars(cli_args)
        saved_path = os.path.join(cli_args.exp_dir, "config.yaml")
        with open(saved_path, "w") as f:
            f.write(yaml.dump(merged))

    # ---- training ----
    dist.barrier(device_ids=[gpu])
    if rank == 0:
        logger.info("<========== Training process ==========>")
        logger.info(
            "optim={}  lr={}  lr_scheduler={}  warmup_ratio={}".format(
                cli_args.optim_class,
                cli_args.learning_rate,
                cli_args.lr_scheduler_type,
                cli_args.warmup_ratio,
            )
        )
        header = ["Train/Val", "Epoch", "iter", "Loss", "LR"]
        for line in tp.header(header, width=10, style="grid").split("\n"):
            logger.info(line)
    dist.barrier(device_ids=[gpu])

    executor = Executor()

    train_losses = []
    val_losses = []
    for epoch in range(start_epoch, cli_args.num_epochs + 1):
        train_dataset.set_epoch(epoch)

        train_loss, _ = train_one_epoch(
            train_dataloader,
            ddp_model,
            epoch_iter,
            optimizer,
            criterion,
            scheduler,
            scaler,
            epoch=epoch,
            enable_amp=cli_args.enable_amp,
            logger=logger,
            clip_grad=cli_args.clip_grad,
            log_batch_interval=cli_args.log_batch_interval,
            device=device,
            se_loss_weight=loss_args,
            executor=executor,
        )

        val_loss, _ = validate(
            val_dataloader,
            ddp_model,
            val_iter,
            criterion,
            epoch=epoch,
            enable_amp=cli_args.enable_amp,
            logger=logger,
            log_batch_interval=cli_args.log_batch_interval,
            device=device,
            executor=executor,
        )

        if rank == 0:
            logger.info(
                "Epoch {} Train info train_loss {}".format(epoch, train_loss)
            )
            logger.info(
                "Epoch {} Val info val_loss {}".format(epoch, val_loss)
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            plt.figure()
            plt.title("Loss of Train and Validation")
            x = list(range(start_epoch, epoch + 1))
            plt.plot(x, train_losses, "b-", label="Train Loss", linewidth=0.8)
            plt.plot(
                x, val_losses, "c-", label="Validation Loss", linewidth=0.8
            )
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.xticks(range(start_epoch, epoch + 1, 1))
            plt.savefig(
                f"{cli_args.exp_dir}/{configs['model']['tse_model']}.png"
            )
            plt.close()

        if rank == 0:
            if (
                epoch % cli_args.save_epoch_interval == 0
                or epoch >= cli_args.num_epochs - cli_args.num_avg
            ):
                save_checkpoint(
                    model_list,
                    optimizer_list,
                    scheduler_list,
                    scaler,
                    os.path.join(
                        model_dir, "checkpoint_{}.pt".format(epoch)
                    ),
                )
                try:
                    os.symlink(
                        "checkpoint_{}.pt".format(epoch),
                        os.path.join(model_dir, "latest_checkpoint.pt"),
                    )
                except FileExistsError:
                    os.remove(
                        os.path.join(model_dir, "latest_checkpoint.pt")
                    )
                    os.symlink(
                        "checkpoint_{}.pt".format(epoch),
                        os.path.join(model_dir, "latest_checkpoint.pt"),
                    )

    if rank == 0:
        os.symlink(
            "checkpoint_{}.pt".format(cli_args.num_epochs),
            os.path.join(model_dir, "final_checkpoint.pt"),
        )
        logger.info(tp.bottom(len(header), width=10, style="grid"))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TSE training with HuggingFace LR scheduler. "
        "Same DDP loop as train.py, optimizer/scheduler from CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config path (keeps dataset_args, model/model_args, "
        "model_init, loss/loss_args, dataloader_args).",
    )

    # ---- Data paths ----
    p.add_argument(
        "--data_type", type=str, default="raw",
        help="Dataset type: raw or shard.",
    )
    p.add_argument(
        "--train_data", type=str, required=True,
        help="Training data list path.",
    )
    p.add_argument(
        "--val_data", type=str, required=True,
        help="Validation data list path.",
    )
    p.add_argument(
        "--train_samples", type=str, required=True,
        help="Training samples file (for counting epoch iterations).",
    )
    p.add_argument(
        "--val_samples", type=str, required=True,
        help="Validation samples file.",
    )
    p.add_argument(
        "--train_cues", type=str, required=True,
        help="Training cues YAML path.",
    )
    p.add_argument(
        "--val_cues", type=str, required=True,
        help="Validation cues YAML path.",
    )
    p.add_argument(
        "--tse_model", type=str, default=None,
        help="TSE model name. If not provided, use the model name in the config file.   ",
    )
    p.add_argument("--loss_type", type=str, default=None,
        help="Loss type. If not provided, use the loss type in the config file.   ",
    )
    # ---- Experiment / distributed ----
    p.add_argument(
        "--exp_dir", type=str, required=True,
        help="Experiment output directory.",
    )
    p.add_argument(
        "--num_epochs", type=int, default=150,
        help="Total training epochs.",
    )
    p.add_argument("--batch_size", type=int, default=4,
        help="Batch size. If not provided, use the batch size in the config file.   ",
    )

    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--gpus", type=str, default="0",
        help="GPU ids, e.g. '0' or '0,1'.",
    )

    # ---- Optimizer (torch.optim) ----
    p.add_argument(
        "--optim_class", type=str, default="AdamW",
        help="torch.optim class name (Adam, AdamW, SGD, etc.).",
    )
    p.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Peak learning rate.",
    )
    p.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay.",
    )

    # ---- LR scheduler ----
    p.add_argument(
        "--lr_scheduler_type", type=str, default="cosine",
        help="LR scheduler (cosine, linear, polynomial, constant).",
    )
    p.add_argument(
        "--warmup_ratio", type=float, default=0.0,
        help="Warmup ratio for LR scheduler.",
    )

    # ---- Misc training ----
    p.add_argument(
        "--clip_grad", type=float, default=5.0,
        help="Gradient clipping max norm.",
    )
    p.add_argument(
        "--enable_amp", action="store_true",
        help="Enable automatic mixed precision.",
    )
    p.add_argument(
        "--log_batch_interval", type=int, default=100,
        help="Log every N steps.",
    )
    p.add_argument(
        "--save_epoch_interval", type=int, default=1,
        help="Save checkpoint every N epochs.",
    )
    p.add_argument(
        "--num_avg", type=int, default=10,
        help="Always save checkpoints for last num_avg epochs.",
    )
    p.add_argument(
        "--find_unused_parameters", action="store_true",
        help="DDP find_unused_parameters flag.",
    )

    # ---- Resume ----
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Resume from wesep checkpoint .pt path.",
    )

    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
