# Copyright (c) 2026 Duo Ma (maduo@cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0.

"""
Evaluate a trained visual-TSE checkpoint on **official held-out test folders**
(LRS3 ``test`` subset and Chinese_lips ``test/test`` layout).

Unlike training-time ``val``, which merges trainval + speaker_fraction splits,
this script builds a **single flat inventory** per corpus via
``build_merged_audio_visual_inventory`` — only speakers/clips under the test
directories participate. Online mixtures are drawn exclusively from that pool.

Examples::

    cd examples/visual/voxceleb2mix
    CUDA_VISIBLE_DEVICES=0 ./local/run_eval_test_corpora.sh \\
      /maduo/exp/.../ema_model.pt 3000

CLI reuses ``TSEOnlineDataArguments`` / ``TSETrainingArguments`` for model YAML,
collate keys, audio/video dims (same as stage 102). Extra flags select which
test corpora to score (comma-separated):

    * ``lrs3``       — scan ``<lrs3_root>/test/`` only
    * ``chinese_lips`` — scan ``<chinese_lips_root>/test/test/*/FACE``
    * ``merged``     — pool LRS3 test + Chinese_lips test speakers in one mixer

Default: ``lrs3,chinese_lips`` (two separate reports). Add ``merged`` or use
``all`` for all three.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field

# cuBLAS workaround (same rationale as ablation_cue_mute.py).
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

_THIS = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import HfArgumentParser  # noqa: E402

from wesep.bin.train_with_transformers_multi_data_online import (  # noqa: E402
    TSEOnlineDataArguments,
    TSETrainingArguments,
    _compute_per_sample_sisdr_loss,
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
    build_merged_audio_visual_inventory,
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
logger = logging.getLogger("eval_test_corpora")


@dataclass
class EvalTestCorporaArguments:
    """Extra CLI only for this script."""

    ema_ckpt: str | None = field(
        default=None,
        metadata={
            "help": "Weights path (EMA ``.pt`` or checkpoint ``model.safetensors``). "
            "If omitted, latest checkpoint-* under output_dir (training_args).",
        },
    )
    eval_targets: str = field(
        default="lrs3,chinese_lips",
        metadata={
            "help": "Comma-separated: lrs3, chinese_lips, merged, or all "
            "(all = lrs3 + chinese_lips + merged).",
        },
    )
    num_eval_mixtures: int = field(
        default=3000,
        metadata={
            "help": "How many online mixtures to evaluate per target "
            "(same counting as trainer val / ablation).",
        },
    )
    eval_seed_offset: int = field(
        default=424242,
        metadata={
            "help": "OnlineMixIterableDataset seed = training_args.seed + this "
            "(keep disjoint from train val seed+7 if desired).",
        },
    )
    eval_noise_prob: float = field(
        default=0.0,
        metadata={
            "help": "Override dataset noise_prob for test-only eval (default 0 = no MUSAN).",
        },
    )
    print_first_batch_shapes: bool = field(default=True)


def _resolve_checkpoint(output_dir: str, ema_ckpt: str | None) -> str:
    if ema_ckpt:
        p = pathlib.Path(ema_ckpt)
        if not p.exists():
            raise FileNotFoundError(f"--ema_ckpt does not exist: {ema_ckpt}")
        return str(p)
    odir = pathlib.Path(output_dir)
    cands = sorted(
        (d for d in odir.glob("checkpoint-*") if d.is_dir()),
        key=lambda d: int(d.name.rsplit("-", 1)[-1])
        if d.name.rsplit("-", 1)[-1].isdigit()
        else -1,
    )
    if not cands:
        raise FileNotFoundError(
            f"No --ema_ckpt given and no checkpoint-* under {output_dir}",
        )
    cand = cands[-1]
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = cand / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No model weights file found inside {cand}")


def _parse_eval_targets(s: str) -> list[str]:
    raw = [x.strip().lower() for x in (s or "").split(",") if x.strip()]
    if not raw:
        raise ValueError("--eval_targets is empty")
    if "all" in raw:
        return ["lrs3", "chinese_lips", "merged"]
    allowed = {"lrs3", "chinese_lips", "merged"}
    bad = set(raw) - allowed
    if bad:
        raise ValueError(f"Unknown eval_targets entries {bad}; allowed {allowed}")
    return raw


def _inventory_for_target(
    target: str,
    data_args: TSEOnlineDataArguments,
) -> dict[str, list[dict]]:
    """Return merged MP4 inventory using **test splits only**."""
    root_lrs3 = data_args.lrs3_root
    root_cn = data_args.chinese_lips_root

    if target == "lrs3":
        return build_merged_audio_visual_inventory(
            use_voxceleb2=False,
            use_lrs3=True,
            lrs3_root=root_lrs3,
            lrs3_subsets=("test",),
            use_chinese_lips=False,
        )
    if target == "chinese_lips":
        return build_merged_audio_visual_inventory(
            use_voxceleb2=False,
            use_lrs3=False,
            use_chinese_lips=True,
            chinese_lips_root=root_cn,
            chinese_lips_splits=("test",),
        )
    if target == "merged":
        return build_merged_audio_visual_inventory(
            use_voxceleb2=False,
            use_lrs3=True,
            lrs3_root=root_lrs3,
            lrs3_subsets=("test",),
            use_chinese_lips=True,
            chinese_lips_root=root_cn,
            chinese_lips_splits=("test",),
        )
    raise ValueError(f"unknown target {target!r}")


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
    return (
        f"[{name}] n={s['n']}  "
        f"mean={s['mean']:+.4f}  median={s['median']:+.4f}  "
        f"p25={s['p25']:+.4f}  p75={s['p75']:+.4f}  p95={s['p95']:+.4f}  "
        f"worst5%mean={s['worst5pct_mean']:+.4f}  "
        f"[min..max {s['min']:+.2f} .. {s['max']:+.2f}]  "
        f"(metrics are -SI-SDR loss; lower = better separation)"
    )


def _run_eval_pass(
    model: torch.nn.Module,
    val_dl: DataLoader,
    executor: Executor,
    device: torch.device,
    n_target_mixtures: int,
    batch_size: int,
    bf16: bool,
    print_first: bool,
) -> torch.Tensor:
    """Collect per-row -SI-SDR on ``n_target_mixtures`` mixtures."""
    n_dl_steps = max(1, (n_target_mixtures + batch_size - 1) // batch_size)
    logger.info(
        "Eval pass: ~%d mixtures (%d dataloader steps × bs=%d).",
        n_target_mixtures,
        n_dl_steps,
        batch_size,
    )

    model.eval()
    chunks: list[torch.Tensor] = []
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
        if print_first and step == 0:
            cue_shape = None if cues is None else [tuple(c.shape) for c in cues]
            logger.info(
                "First batch mix=%s target=%s cue_shapes=%s",
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
        chunks.append(per_sample)

        step += 1
        if step - last_log >= max(1, n_dl_steps // 10):
            running = torch.cat(chunks)
            logger.info(
                "  step %d/%d  rows=%d  running mean=%.4f median=%.4f",
                step,
                n_dl_steps,
                int(running.numel()),
                float(running.mean().item()),
                float(torch.median(running).item()),
            )
            last_log = step

    return torch.cat(chunks) if chunks else torch.empty(0)


def _build_dataloader_for_inventory(
    inventory: dict[str, list[dict]],
    data_args: TSEOnlineDataArguments,
    training_args: TSETrainingArguments,
    configs: dict,
    dataset_seed: int,
    noise_prob_override: float,
) -> tuple[DataLoader, int]:
    dataset_args = ensure_online_pipeline_defaults(
        build_dataset_args_from_tse_online_data_args(data_args)
    )
    dataset_args["noise_prob"] = float(noise_prob_override)

    import random

    rng = random.Random(int(dataset_seed))
    speaker_ids = resolve_speaker_pool(
        inventory,
        rng,
        speakers=None,
        num_speakers=data_args.num_speaker_pool,
    )
    if len(speaker_ids) < 2:
        raise ValueError(
            f"Test inventory needs >=2 speakers for mixing; got {len(speaker_ids)}",
        )
    inventory = subset_inventory(inventory, speaker_ids)

    dataset = OnlineMixIterableDataset(
        inventory,
        speaker_ids,
        dataset_args,
        seed=int(dataset_seed),
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
        dataset,
        **dataloader_args,
        collate_fn=lambda batch, ck=collect_keys, v=_vmf: tse_collate_fn(
            batch, ck, visual_max_frames=v
        ),
    )
    bs = int(dataloader_args["batch_size"])
    return val_dl, bs


def main() -> None:
    parser = HfArgumentParser(
        (TSEOnlineDataArguments, TSETrainingArguments, EvalTestCorporaArguments)
    )
    data_args, training_args, eval_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    cfg_path = data_args.model_config
    raw = load_yaml(cfg_path)
    for k in ("model", "model_args"):
        if k not in raw:
            raise ValueError(f"Model YAML must contain '{k}': {cfg_path}")
    configs = {"model": raw["model"], "model_args": raw["model_args"]}
    apply_tse_model_yaml_cli_overrides(data_args, configs)

    if data_args.tse_model is not None:
        configs["model"]["tse_model"] = data_args.tse_model
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"]
    )
    logger.info(
        "tse_model=%s  params=%.2fM",
        configs["model"]["tse_model"],
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    ckpt_path = _resolve_checkpoint(training_args.output_dir, eval_args.ema_ckpt)
    logger.info("Loading weights from: %s", ckpt_path)
    if ckpt_path.endswith(".pt") and os.path.basename(ckpt_path).startswith("ema"):
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "models" in sd:
            sd = sd["models"][0]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info("EMA load: missing=%d unexpected=%d", len(missing), len(unexpected))
    else:
        load_pretrained_model(model, ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    targets = _parse_eval_targets(eval_args.eval_targets)
    n_mix = int(eval_args.num_eval_mixtures)
    seed_base = int(training_args.seed) + int(eval_args.eval_seed_offset)

    executor = Executor()
    logger.info(
        "Test roots: lrs3_root=%r (subset=test)  chinese_lips_root=%r (split=test/test)",
        data_args.lrs3_root,
        data_args.chinese_lips_root,
    )
    logger.info(
        "Eval targets=%s  num_eval_mixtures=%d  eval_noise_prob=%s  dataset_seed=%s",
        targets,
        n_mix,
        eval_args.eval_noise_prob,
        seed_base,
    )

    for tgt in targets:
        logger.info("=" * 78)
        logger.info("CORPUS TARGET: %s", tgt.upper())
        logger.info("=" * 78)
        inv = _inventory_for_target(tgt, data_args)
        n_spk = len(inv)
        n_clip = sum(len(v) for v in inv.values())
        logger.info("Inventory: %d speakers, %d clips", n_spk, n_clip)

        seed_extra = {"lrs3": 11, "chinese_lips": 17, "merged": 23}[tgt]
        val_dl, batch_size = _build_dataloader_for_inventory(
            inv,
            data_args,
            training_args,
            configs,
            dataset_seed=seed_base + seed_extra,
            noise_prob_override=eval_args.eval_noise_prob,
        )

        losses = _run_eval_pass(
            model=model,
            val_dl=val_dl,
            executor=executor,
            device=device,
            n_target_mixtures=n_mix,
            batch_size=batch_size,
            bf16=bool(training_args.bf16),
            print_first=bool(eval_args.print_first_batch_shapes),
        )
        stats = _percentile_stats(losses)
        logger.info(_fmt_stats(f"test:{tgt}", stats))
        # SI-SDR in dB ≈ -mean(loss) for interpretation
        if stats.get("n", 0) > 0:
            logger.info(
                "  ⇒ approximate SI-SDR (negated loss convention): "
                "mean≈%+.2f dB  median≈%+.2f dB",
                -stats["mean"],
                -stats["median"],
            )


if __name__ == "__main__":
    main()
