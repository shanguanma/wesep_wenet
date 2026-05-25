# Copyright (c) 2026 Duo Ma (maduo@cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""
Cue-mute ablation for visual TSE checkpoints (read-only diagnostic).

Hypothesis under test
---------------------
After ~6k steps, stage-100 (TSE_BSRNN_VISUAL_v2 + SISDR_MRSTFT) sees per-sample
SI-SDR percentiles plateau (mean ≈ 0.6, every quartile static across 5 evals).
That is consistent with the model converging to a passthrough-ish local optimum
that ignores the visual cue.

This script directly tests it: runs the *same* online val pipeline as the train
script, twice — once with normal cues, once with all visual cues replaced by
zeros — and compares per-sample -SI-SDR (auraloss convention, lower = better)
distributions.

Decision rule
-------------
* Normal cue mean ≈ Zero cue mean → visual is being ignored. Stage 101 needs
  to force visual to be informative (cue-conditioning aux loss, freeze
  separator first, etc.).
* Normal cue mean << Zero cue mean → visual *is* used. The plateau then is
  about model capacity / data, not cue ignorance.

Design choices that matter
--------------------------
* Reuses :class:`TSEOnlineDataArguments` / :class:`TSETrainingArguments` and
  the val dataset construction from
  ``wesep.bin.train_with_transformers_multi_data_online`` so the val mixtures
  here are bit-for-bit identical to those produced by stage 100's
  ``trainer.evaluate()`` (same ``seed + 7`` for the val dataset, same
  ``OnlineMixIterableDataset``, same collate, same dataloader args).
* Loads either an EMA shadow snapshot (``ema_model.pt`` / ``ema_final.pt``)
  or a regular HF checkpoint (``checkpoint-XXX/``) via the unified
  :func:`wesep.utils.checkpoint.load_pretrained_model` adapter added in stage
  99 (handles native wesep dicts, raw torch state dicts, and safetensors).
* Read-only: never writes to ``output_dir``. ``Trainer`` is **not**
  instantiated, so no ``trainer_state.json`` etc. is touched.
* Single-GPU only: ablation does not need DDP, and avoiding the HF Trainer
  removes any chance of side effects on the live stage-100 run.

CLI shape
---------
The same flags you'd pass to stage 100, plus:

  --ema_ckpt PATH                    e.g. <exp_dir>/ema_model.pt
                                     (or omit to load the latest checkpoint-* under output_dir)
  --cue_modes normal,zero            comma-separated subset of {normal, zero}
  --num_eval_mixtures N              cap mixtures evaluated per mode
                                     (defaults to --sample_num_per_epoch_val).

Output
------
Per cue mode logs:
  count, mean, median, p25, p75, p95, worst5pct_mean   (all on -SI-SDR; lower = better)
Then prints the delta (zero - normal). A delta near 0 confirms cue ignorance.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# cuBLAS workaround: must be set BEFORE ``import torch``. Driver R580.65 +
# PyTorch cu128 has a broken cuBLASLt sgemm for >= 4D nn.Linear inputs;
# DISABLE_ADDMM_CUDA_LT routes around it. See local/cuda_probe.py.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

try:
    torch.backends.cuda.preferred_blas_library(backend="cublas")
except Exception:  # noqa: BLE001
    pass


def _install_linear_workaround() -> None:
    """Monkey-patch ``torch.nn.Linear.forward`` (see sanity_check_pipeline.py
    for full rationale): bypasses ``F.linear`` whose ``cublasSgemm`` /
    ``cublasLtMatmul`` paths are unstable on driver R580.65 + cu128 inside
    this particular model. ``x @ W.t() + b`` uses ``torch.matmul`` which is
    healthy on this stack."""
    if getattr(torch.nn.Linear, "_wesep_patched", False):
        return
    _orig_forward = torch.nn.Linear.forward

    def _patched_forward(self, input):  # type: ignore[no-redef]
        if input.is_cuda:
            x = input if input.is_contiguous() else input.contiguous()
            out = x @ self.weight.t()
            if self.bias is not None:
                out = out + self.bias
            return out
        return _orig_forward(self, input)

    torch.nn.Linear.forward = _patched_forward
    torch.nn.Linear._wesep_patched = True


_install_linear_workaround()

# Ensure the wesep_wenet repo root is importable when this file is launched from
# examples/visual/voxceleb2mix (PyHF setup.py is not always installed editable).
_THIS = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[4]  # examples/visual/voxceleb2mix/local/<file>
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import HfArgumentParser  # noqa: E402

from wesep.bin.train_with_transformers_multi_data_online import (  # noqa: E402
    TSEOnlineDataArguments,
    TSETrainingArguments,
    _compute_per_sample_sisdr_loss,
    _parse_lrs3_subsets,
    apply_tse_model_yaml_cli_overrides,
    build_dataloader_args_from_training_args,
)
from wesep.dataset.collate import (  # noqa: E402
    BASE_COLLECT_KEYS,
    build_collect_keys,
    build_collect_keys_online,
    tse_collate_fn,
)
from wesep.dataset.online_multi_dataset import (  # noqa: E402
    OnlineMixIterableDataset,
    build_dataset_args_from_tse_online_data_args,
    build_train_val_merged_audio_visual_inventories,
    ensure_online_pipeline_defaults,
    resolve_speaker_pool,
    subset_inventory,
)
from wesep.models import get_model  # noqa: E402
from wesep.utils.checkpoint import load_pretrained_model  # noqa: E402
from wesep.utils.executor import Executor  # noqa: E402
from wesep.utils.file_utils import load_yaml  # noqa: E402
from wesep.utils.utils import set_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ablation_cue_mute")


@dataclass
class AblationArguments:
    """Ablation-only arguments. Everything else is reused from stage 100."""

    ema_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to an EMA shadow snapshot (``ema_model.pt`` / "
            "``ema_final.pt``). If omitted, falls back to the latest "
            "``checkpoint-*`` directory under ``output_dir`` and loads its "
            "``model.safetensors`` via load_pretrained_model.",
        },
    )
    cue_modes: str = field(
        default="normal,zero",
        metadata={
            "help": "Comma-separated subset of {normal,zero}. ``zero`` "
            "replaces every cue tensor with zeros before model.forward; "
            "``normal`` runs the real cue.",
        },
    )
    num_eval_mixtures: int = field(
        default=0,
        metadata={
            "help": "Cap mixtures evaluated per mode (0 = use "
            "``--sample_num_per_epoch_val``).",
        },
    )
    print_first_batch_shapes: bool = field(
        default=True,
        metadata={"help": "Log the first val batch shapes for sanity."},
    )


def _resolve_checkpoint(output_dir: str, ema_ckpt: Optional[str]) -> str:
    if ema_ckpt:
        p = pathlib.Path(ema_ckpt)
        if not p.exists():
            raise FileNotFoundError(f"--ema_ckpt does not exist: {ema_ckpt}")
        return str(p)
    odir = pathlib.Path(output_dir)
    cands = sorted(
        (d for d in odir.glob("checkpoint-*") if d.is_dir()),
        key=lambda d: int(d.name.rsplit("-", 1)[-1]) if d.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    if not cands:
        raise FileNotFoundError(
            f"No --ema_ckpt given and no checkpoint-* under {output_dir}",
        )
    cand = cands[-1]
    sft = cand / "model.safetensors"
    bin_ = cand / "pytorch_model.bin"
    if sft.exists():
        return str(sft)
    if bin_.exists():
        return str(bin_)
    raise FileNotFoundError(f"No model weights file found inside {cand}")


def _build_val_dataloader(
    data_args: TSEOnlineDataArguments,
    training_args: TSETrainingArguments,
    configs: dict,
):
    """Mirror exactly how the train script builds val_dataset / val_dl."""
    dataset_args = ensure_online_pipeline_defaults(
        build_dataset_args_from_tse_online_data_args(data_args)
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

    import random

    rng_val = random.Random(training_args.seed + 100_003)
    val_speaker_ids = resolve_speaker_pool(
        val_inventory,
        rng_val,
        speakers=None,
        num_speakers=data_args.num_speaker_pool,
    )
    if len(val_speaker_ids) < 2:
        raise ValueError(
            f"Need >=2 val speakers after pooling; got {len(val_speaker_ids)}"
        )
    val_inventory = subset_inventory(val_inventory, val_speaker_ids)

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
    dataloader_args = build_dataloader_args_from_training_args(training_args)

    val_dl = DataLoader(
        val_dataset,
        **dataloader_args,
        collate_fn=lambda batch, ck=collect_keys, v=_vmf: tse_collate_fn(
            batch, ck, visual_max_frames=v
        ),
    )

    val_sample_num = int(dataset_args.get("sample_num_per_epoch_val") or 0)
    return val_dl, val_sample_num, dataloader_args["batch_size"]


def _zero_cues(cues: Optional[List[torch.Tensor]]) -> Optional[List[torch.Tensor]]:
    if cues is None:
        return None
    return [torch.zeros_like(c) for c in cues]


def _percentile_stats(losses: torch.Tensor) -> dict:
    losses = losses.flatten().to(torch.float32)
    n = int(losses.numel())
    if n == 0:
        return {"n": 0}
    sorted_vals, _ = torch.sort(losses)

    def _q(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(sorted_vals[idx].item())

    worst5 = sorted_vals[max(0, n - max(1, int(round(0.05 * n)))) :]
    return {
        "n": n,
        "mean": float(sorted_vals.mean().item()),
        "median": _q(0.5),
        "p25": _q(0.25),
        "p75": _q(0.75),
        "p95": _q(0.95),
        "worst5pct_mean": float(worst5.mean().item()),
        "min": float(sorted_vals[0].item()),
        "max": float(sorted_vals[-1].item()),
    }


def _fmt_stats(name: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"[{name}] no samples"
    # NB: lower = better (auraloss convention: -SI-SDR).
    return (
        f"[{name}] n={s['n']} "
        f"mean={s['mean']:+.3f}  median={s['median']:+.3f}  "
        f"p25={s['p25']:+.3f}  p75={s['p75']:+.3f}  p95={s['p95']:+.3f}  "
        f"worst5%mean={s['worst5pct_mean']:+.3f}  "
        f"[range {s['min']:+.2f} .. {s['max']:+.2f}]"
    )


def _run_one_pass(
    model: torch.nn.Module,
    val_dl: DataLoader,
    executor: Executor,
    device: torch.device,
    cue_mode: str,
    n_target_mixtures: int,
    batch_size: int,
    bf16: bool,
    print_first: bool,
) -> torch.Tensor:
    """Iterate val_dl until n_target_mixtures rows have been processed; collect
    per-sample -SI-SDR loss. Returns a 1-D tensor on CPU.

    NOTE: ``val_dl`` yields collated *speaker-axis-expanded* batches, i.e. each
    DL step returns ``B_eff = sum(num_speaker)`` rows but the underlying
    OnlineMixIterableDataset only consumed ``batch_size`` mixtures. We stop
    after ``ceil(n_target_mixtures / batch_size)`` DL steps so the effective
    val budget matches stage 100's evaluator (which is also ``val_iter`` DL
    steps).
    """
    if cue_mode not in ("normal", "zero"):
        raise ValueError(f"Unknown cue_mode {cue_mode!r}")

    n_dl_steps = max(1, (n_target_mixtures + batch_size - 1) // batch_size)
    logger.info(
        "Pass [%s]: target ≈%d mixtures (%d DL steps × bs=%d).",
        cue_mode,
        n_target_mixtures,
        n_dl_steps,
        batch_size,
    )

    model.eval()
    losses_chunks: list[torch.Tensor] = []

    autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": bf16}

    step = 0
    last_log = 0
    it = iter(val_dl)
    while step < n_dl_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(val_dl)
            batch = next(it)
        mix, cues, target = executor._extract_model_inputs(batch, device)
        if cue_mode == "zero":
            cues = _zero_cues(cues)
        if print_first and step == 0:
            cue_shape = None if cues is None else [tuple(c.shape) for c in cues]
            logger.info(
                "Pass [%s]: first batch mix=%s target=%s cue_shapes=%s",
                cue_mode,
                tuple(mix.shape),
                tuple(target.shape),
                cue_shape,
            )

        with torch.no_grad(), torch.amp.autocast(**autocast_kwargs):
            outputs = model(mix) if cues is None else model(mix, cues)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            est = outputs[0]
            per_sample = _compute_per_sample_sisdr_loss(
                est.float(), target.float()
            ).detach().cpu()
        losses_chunks.append(per_sample)

        step += 1
        # Periodic progress log: every ~10% of the budget.
        if step - last_log >= max(1, n_dl_steps // 10):
            running = torch.cat(losses_chunks)
            logger.info(
                "  [%s] step %d/%d  rows=%d  running mean=%.3f median=%.3f",
                cue_mode,
                step,
                n_dl_steps,
                int(running.numel()),
                float(running.mean().item()),
                float(torch.median(running).item()),
            )
            last_log = step

    return torch.cat(losses_chunks) if losses_chunks else torch.empty(0)


def main() -> None:
    parser = HfArgumentParser((TSEOnlineDataArguments, TSETrainingArguments, AblationArguments))
    data_args, training_args, ablation_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # ---- model config ----
    cfg_path = data_args.model_config
    raw = load_yaml(cfg_path)
    for k in ("model", "model_args"):
        if k not in raw:
            raise ValueError(f"Model YAML must contain '{k}': {cfg_path}")
    configs = {"model": raw["model"], "model_args": raw["model_args"]}
    apply_tse_model_yaml_cli_overrides(data_args, configs)

    # ---- model build ----
    if data_args.tse_model is not None:
        configs["model"]["tse_model"] = data_args.tse_model
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"]
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "tse_model=%s  params=%.2fM",
        configs["model"]["tse_model"],
        n_params / 1e6,
    )

    # ---- weights ----
    ckpt_path = _resolve_checkpoint(training_args.output_dir, ablation_args.ema_ckpt)
    logger.info("Loading weights from: %s", ckpt_path)
    if ckpt_path.endswith(".pt") and os.path.basename(ckpt_path).startswith("ema"):
        # EMA snapshot is a plain CPU state_dict saved via torch.save(self._shadow).
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "models" in sd:
            sd = sd["models"][0]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            "EMA load: missing=%d unexpected=%d", len(missing), len(unexpected),
        )
        if missing or unexpected:
            logger.warning(
                "  first missing=%s ; first unexpected=%s",
                missing[:3], unexpected[:3],
            )
    else:
        load_pretrained_model(model, ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ---- val dataloader ----
    val_dl, val_sample_num, batch_size = _build_val_dataloader(
        data_args, training_args, configs
    )

    n_target = int(ablation_args.num_eval_mixtures or 0) or int(val_sample_num) or 5000
    logger.info(
        "Eval budget per mode: n_target_mixtures=%d (val pool=%d).",
        n_target, val_sample_num,
    )

    modes = [m.strip() for m in (ablation_args.cue_modes or "").split(",") if m.strip()]
    if not modes:
        raise ValueError("--cue_modes must contain at least one of {normal,zero}.")

    executor = Executor()
    results: dict[str, dict] = {}
    for mode in modes:
        losses = _run_one_pass(
            model=model,
            val_dl=val_dl,
            executor=executor,
            device=device,
            cue_mode=mode,
            n_target_mixtures=n_target,
            batch_size=batch_size,
            bf16=bool(training_args.bf16),
            print_first=bool(ablation_args.print_first_batch_shapes),
        )
        s = _percentile_stats(losses)
        results[mode] = s
        logger.info(_fmt_stats(mode, s))

    if "normal" in results and "zero" in results:
        n = results["normal"]
        z = results["zero"]
        if n.get("n", 0) > 0 and z.get("n", 0) > 0:
            logger.info("─" * 78)
            for k in ("mean", "median", "p25", "p75", "p95", "worst5pct_mean"):
                logger.info(
                    "Δ(zero - normal)  %s = %+.3f   (zero=%+.3f, normal=%+.3f)",
                    k, z[k] - n[k], z[k], n[k],
                )
            logger.info("─" * 78)
            mean_gap = z["mean"] - n["mean"]
            if abs(mean_gap) < 0.5:
                verdict = (
                    "Visual cue is being IGNORED (zeroing cues changes mean SI-SDR "
                    "by <0.5 dB). Stage-101 should force visual to matter."
                )
            elif mean_gap > 2.0:
                verdict = (
                    "Visual cue is BEING USED (zeroing cues degrades mean SI-SDR by "
                    f"{mean_gap:+.2f} dB). The plateau is not about cue ignorance."
                )
            else:
                verdict = (
                    "Visual cue has SOME effect but it's weak (Δ in [0.5, 2.0] dB)."
                )
            logger.info("Verdict: %s", verdict)


if __name__ == "__main__":
    main()
