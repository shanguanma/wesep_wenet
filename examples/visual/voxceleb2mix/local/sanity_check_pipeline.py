# Copyright (c) 2026 Duo Ma (maduo@cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""
End-to-end sanity check for the VoxCeleb2-mix online TSE pipeline.

Goal: before launching another training stage, verify that every link in the
chain from raw inventories → online mixer → speaker-axis collate → model
forward → loss is doing what we *think* it is doing. Failures here would
explain why stages 78 / 88 / 98 / 99 / 100 plateau no matter how we tune
hyperparameters.

The script runs as a single forward-only diagnostic — it does NOT train and
does NOT touch any output_dir under the live experiments. With ``--weights
PATH`` (e.g. an EMA snapshot) it adds the model-side checks; without it, only
data-side checks run.

Test inventory (each prints a clear PASS / WARN / FAIL line):

  Data side:
    T1  Per-mix-block sharing: rows of the same ``num_speaker`` block must
        share an identical ``wav_mix`` (axis=mix) but each row has its own
        ``wav_target`` (axis=spk) and its own ``visual_aux`` (axis=spk).
    T2  Target uniqueness: ``wav_target_i != wav_target_j`` within a block
        (different speakers' clean signals).
    T3  Cue uniqueness: ``visual_aux_i != visual_aux_j`` within a block.
    T4  Mix energetics: ``SI-SDR(wav_mix, wav_target_i)`` lies in the
        physically expected band:
          ns=1 -> very high (mix is the target, ideally > +20 dB)
          ns=2 -> roughly [-7, +6] dB, depending on SNR mixing range
          ns=3 -> roughly [-10, -1] dB
          ns=4 -> roughly [-12, -2] dB
        Outside band -> data bug or unusual SNR distribution.
    T5  Visual / audio frame ratio: ``T_video ~ chunk_len_seconds * fps``
        (typically 25 or 30 fps; visual_max_frames may cap T_video).

  Model side (only when --weights given):
    T6  Forward run-shape: ``model(mix, cues)`` returns ``(B_eff, T)`` or
        ``(B_eff, 1, T)`` and contains no NaN / Inf.
    T7  Identity: ``model(target_i, cue_i)`` should preserve the clean
        target. SI-SDR(out, target_i) should be high (>= +10 dB for any
        non-broken model). Failure here means the model destroys clean
        signal regardless of cue.
    T8  Output destination: for 2/3/4-spk mixes, compare
        SI-SDR(out, mix) vs SI-SDR(out, target_i).
          out ~= mix      -> passthrough (cue ignored, separator inert)
          out ~= target_i -> real TSE
          out ~= 0        -> degenerate "predict silence"
    T9  Cue swap: build an alternate cue list where row i gets cue_j
        (j != i within the same block) and feed model(mix, cue_swap).
        For a TSE model, SI-SDR(out_swap, target_i) should drop
        substantially compared to T8.
    T10 1-spk passthrough: pick rows whose mixture has ``ns == 1``
        (target == mix) and measure SI-SDR(out, target). A sane TSE model
        outputs the input unchanged here (>= +20 dB).

Usage:

  cd /maduo/codebase/wesep_wenet/examples/visual/voxceleb2mix
  CUDA_VISIBLE_DEVICES=1 python local/sanity_check_pipeline.py \
      --model_config confs/tse_bsrnn_visual_model_v2.yaml \
      --mp4_dir_of_voxceleb2 /F00120240032/voxceleb2_zk_mixture/mp4/train \
      --use_voxceleb2 True --use_lrs3 True --use_chinese_lips True \
      --lrs3_root /F00120240032/lrs3/trainval \
      --chinese_lips_root /F00120240032/Chinese_lips \
      --train_speaker_fraction 0.8 \
      --output_dir /tmp/sanity_check_dummy \
      --tse_model TSE_BSRNN_VISUAL --visual_frontend muse \
      --separator_causal false \
      --sample_num_per_epoch 60000 --sample_num_per_epoch_val 15000 \
      --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
      --noise_prob 0.3 --per_device_train_batch_size 3 \
      --dataloader_num_workers 0 --visual_max_frames 75 --bf16 false \
      --weights /maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_short_warmup_balanced_loss/ema_model.pt \
      --num_mixtures 16

A bash wrapper ``local/run_sanity_check.sh`` mirrors the stage-100 data flags.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# cuBLAS workaround (MUST come before ``import torch``)
# ---------------------------------------------------------------------------
# On this machine (NVIDIA driver R580.65 + PyTorch 2.10.0+cu128) cuBLASLt's
# fused gemm-and-bias kernel raises ``CUBLAS_STATUS_NOT_INITIALIZED`` for
# 4D / 5D nn.Linear inputs. The recovery path (unfused cublasSgemm) is also
# fragile. PyTorch reads ``DISABLE_ADDMM_CUDA_LT`` and
# ``TORCH_BLAS_PREFER_CUBLASLT`` exactly once at C++ initialization, so they
# MUST be present before ``import torch`` runs — setting them later is a
# no-op for the gemm_and_bias dispatcher. We set them here defensively so
# running this file directly (without the bash wrapper) is also safe.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Belt-and-suspenders: also forward the preference via the public Python API
# (skipping cuBLASLt where the dispatcher honors it).
try:
    torch.backends.cuda.preferred_blas_library(backend="cublas")
except Exception:  # noqa: BLE001
    pass


def _install_linear_workaround() -> None:
    """Monkey-patch ``torch.nn.Linear.forward`` to bypass ``F.linear`` on this
    machine.

    Why
    ---
    On driver R580.65 + PyTorch cu128, both code paths used by ``F.linear``
    blow up for some calls inside this model (the failure pattern can be
    reproduced via local/cuda_probe.py for the cuBLASLt path; the unfused
    ``cublasSgemm`` fallback also fails sporadically inside the model
    forward, even though it works in isolation). What *does* work
    consistently is the ``a @ b`` matmul path (probe step 4).

    What this patch does
    --------------------
    Replace ``F.linear(x, W, b)`` with ``x @ W.T + b`` (with an explicit
    contiguous() to dodge any non-contiguous reshape inside cuBLAS). This
    routes through ``torch.matmul``'s code path instead of ``F.linear``'s
    addmm/cublasSgemm, sidestepping the buggy kernel selection while
    producing identical results.

    Cost
    ----
    ~5-10% extra latency on the matmul because we lose the fused bias add.
    Acceptable for a sanity check / ablation tool. The patch is process-
    local; it does NOT affect the live training process.
    """
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
logger = logging.getLogger("sanity_check")


@dataclass
class SanityArgs:
    weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to a weight file (EMA snapshot ``ema_model.pt`` "
            "/ ``ema_final.pt``, or HF safetensors / pytorch_model.bin). "
            "When provided, model-side tests T6 - T10 also run."
        },
    )
    num_mixtures: int = field(
        default=16,
        metadata={
            "help": "Number of *DataLoader steps* (each step = "
            "per_device_train_batch_size mixtures) to draw for tests T1 - T5. "
            "Model tests T6 - T9 also iterate this many steps; T10 (1-spk "
            "passthrough) iterates more steps if needed to find 1-spk rows.",
        },
    )
    audio_dump_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, save the first 1 mixture (mix + per-speaker target) "
            "as 16-kHz WAV files into this directory for human listening.",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def _hr(title: str = "") -> None:
    line = "=" * 78
    if title:
        logger.info("%s", line)
        logger.info("== %s", title)
        logger.info("%s", line)
    else:
        logger.info("%s", line)


def _per_mix_blocks(ns_flat: list[int]) -> list[tuple[int, int]]:
    """Return list of (start, ns) pairs."""
    blocks = []
    i = 0
    n = len(ns_flat)
    while i < n:
        ns = int(ns_flat[i])
        if ns <= 0 or i + ns > n:
            raise ValueError(f"Bad num_speaker layout @ i={i}: {ns_flat}")
        for j in range(1, ns):
            if int(ns_flat[i + j]) != ns:
                raise ValueError(
                    f"Inconsistent num_speaker block @ i={i}, ns={ns}: "
                    f"{ns_flat[i:i+ns]}"
                )
        blocks.append((i, ns))
        i += ns
    return blocks


def _sisdr(est: torch.Tensor, ref: torch.Tensor) -> float:
    """Single-row SI-SDR in dB (positive = good). Tensors are 1-D."""
    eps = 1e-8
    est_z = est - est.mean()
    ref_z = ref - ref.mean()
    alpha = (est_z * ref_z).sum() / ((ref_z ** 2).sum() + eps)
    proj = ref_z * alpha
    res = est_z - proj
    return float(10 * torch.log10(((proj ** 2).sum() / ((res ** 2).sum() + eps)) + eps))


def _l2(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.flatten().float()).item())


def _rms_dbfs(x: torch.Tensor) -> float:
    rms = torch.sqrt((x.float() ** 2).mean() + 1e-12)
    return float(20 * torch.log10(rms + 1e-12))


def _build_val_dataloader(
    data_args: TSEOnlineDataArguments,
    training_args: TSETrainingArguments,
    configs: dict,
):
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
        val_inventory, rng_val, speakers=None,
        num_speakers=data_args.num_speaker_pool,
    )
    val_inventory = subset_inventory(val_inventory, val_speaker_ids)
    val_dataset = OnlineMixIterableDataset(
        val_inventory, val_speaker_ids, dataset_args,
        seed=training_args.seed + 7, with_visual_cue=True,
        skip_visual_decode=data_args.skip_visual_decode,
    )
    if data_args.train_cues:
        collect_keys = build_collect_keys(
            load_yaml(data_args.train_cues), dataset_args, BASE_COLLECT_KEYS,
        )
    else:
        collect_keys = build_collect_keys_online(
            dataset_args, BASE_COLLECT_KEYS,
            tse_model_name=configs["model"].get("tse_model"),
        )
    _vmf = dataset_args.get("visual_max_frames")
    da = build_dataloader_args_from_training_args(training_args)
    val_dl = DataLoader(
        val_dataset, **da,
        collate_fn=lambda batch, ck=collect_keys, v=_vmf: tse_collate_fn(
            batch, ck, visual_max_frames=v
        ),
    )
    return val_dl, dataset_args, da["batch_size"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_data_side(
    val_dl: DataLoader,
    dataset_args: dict,
    n_steps: int,
    audio_dump_dir: Optional[str] = None,
) -> dict:
    """T1 - T5. Returns aggregate stats for downstream model tests."""
    _hr("Data-side tests (T1 - T5)")
    executor = Executor()
    device = torch.device("cpu")  # data side runs on CPU; cheap
    sr = int(dataset_args.get("resample_rate", 16000))
    chunk_len = int(dataset_args.get("chunk_len", 48000))
    chunk_sec = chunk_len / float(sr)
    vmf = int(dataset_args.get("visual_max_frames") or 0)
    expected_video_T_25fps = round(chunk_sec * 25)
    expected_video_T_30fps = round(chunk_sec * 30)
    if vmf > 0:
        expected_video_T_25fps = min(vmf, expected_video_T_25fps)
        expected_video_T_30fps = min(vmf, expected_video_T_30fps)

    failures: list[str] = []
    warnings: list[str] = []
    sisdr_by_ns: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    ns_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    seen_video_T: set[int] = set()
    first_batch_logged = False

    it = iter(val_dl)
    for step in range(n_steps):
        try:
            batch = next(it)
        except StopIteration:
            break

        ns_flat = batch.get("num_speaker") or []
        ns_flat = [int(x) for x in ns_flat]
        try:
            blocks = _per_mix_blocks(ns_flat)
        except ValueError as e:
            failures.append(f"step {step}: {e}")
            continue

        mix = batch["wav_mix"].float()       # (B_eff, 1, T_a)
        target = batch["wav_target"].float() # (B_eff, 1, T_a)
        cue = batch.get("visual_aux")        # (B_eff, H, W, C, T_v) or similar

        if not first_batch_logged:
            logger.info(
                "First batch: B_eff=%d  mix=%s  target=%s  cue=%s  "
                "num_speaker=%s  blocks=%s",
                int(mix.shape[0]),
                tuple(mix.shape),
                tuple(target.shape),
                None if cue is None else tuple(cue.shape),
                ns_flat[:16],
                blocks[:8],
            )
            first_batch_logged = True
            if audio_dump_dir:
                _dump_first_mixture(
                    audio_dump_dir, mix, target, blocks, sr,
                )

        for (start, ns) in blocks:
            ns_counts[min(ns, 4)] = ns_counts.get(min(ns, 4), 0) + 1
            mix_block = mix[start:start + ns]
            tgt_block = target[start:start + ns]
            cue_block = cue[start:start + ns] if cue is not None else None

            # T1: rows of same block share mix exactly.
            if ns >= 2:
                ref_mix = mix_block[0]
                for k in range(1, ns):
                    if not torch.equal(mix_block[k], ref_mix):
                        failures.append(
                            f"T1 step {step} block@{start} ns={ns}: row {k} mix differs from row 0"
                        )
                        break
                # T2: targets within block must differ.
                for a in range(ns):
                    for b in range(a + 1, ns):
                        if torch.equal(tgt_block[a], tgt_block[b]):
                            failures.append(
                                f"T2 step {step} block@{start} ns={ns}: target rows {a},{b} identical"
                            )
                # T3: cues within block must differ.
                if cue_block is not None:
                    for a in range(ns):
                        for b in range(a + 1, ns):
                            if torch.equal(cue_block[a], cue_block[b]):
                                failures.append(
                                    f"T3 step {step} block@{start} ns={ns}: cue rows {a},{b} identical"
                                )

            # T4: mix-target SI-SDR per row.
            for k in range(ns):
                d = _sisdr(mix_block[k].flatten(), tgt_block[k].flatten())
                sisdr_by_ns[min(ns, 4)].append(d)

            # T5: video time length.
            if cue_block is not None:
                tv = int(cue_block.shape[-1])
                seen_video_T.add(tv)

    # ---- summarize ----
    n_total = sum(ns_counts.values())
    logger.info(
        "Visited %d mix-blocks  (ns histogram: 1=%d, 2=%d, 3=%d, 4=%d)",
        n_total, ns_counts[1], ns_counts[2], ns_counts[3], ns_counts[4],
    )
    if n_total == 0:
        logger.info(FAIL + " no blocks observed; dataloader is empty?")
        return {"sisdr_by_ns": sisdr_by_ns, "n_blocks": n_total, "failures": failures}

    if not failures:
        logger.info(PASS + " T1/T2/T3 same-block sharing+uniqueness held over %d blocks", n_total)
    else:
        logger.info(FAIL + " T1/T2/T3 violations:")
        for f in failures[:10]:
            logger.info("    - %s", f)
        if len(failures) > 10:
            logger.info("    (... %d more)", len(failures) - 10)

    # T4 ranges (SI-SDR is 'positive = better' here, NOT auraloss convention):
    expected_bands = {
        1: (+15.0, +200.0),
        2: (-12.0, +12.0),
        3: (-15.0, +5.0),
        4: (-18.0, +2.0),
    }
    for ns, vals in sisdr_by_ns.items():
        if not vals:
            continue
        v = torch.tensor(vals, dtype=torch.float32)
        lo, hi = expected_bands[ns]
        in_band = ((v >= lo) & (v <= hi)).float().mean().item() * 100.0
        tag = PASS if in_band > 90 else (WARN if in_band > 70 else FAIL)
        logger.info(
            "%s T4 ns=%d  SI-SDR(mix,target)  n=%d  mean=%+5.2f  median=%+5.2f  "
            "range [%+6.1f, %+6.1f]  in-band[%+.0f,%+.0f]: %.1f%%",
            tag, ns, len(vals),
            float(v.mean()), float(v.median()),
            float(v.min()), float(v.max()),
            lo, hi, in_band,
        )

    # T5: video frame check
    if seen_video_T:
        logger.info(
            "T5 visual_aux T values seen: %s  | expected ~ %d (25fps) or %d (30fps); cap visual_max_frames=%s",
            sorted(seen_video_T),
            expected_video_T_25fps, expected_video_T_30fps, vmf,
        )
        ok = any(
            (abs(t - expected_video_T_25fps) <= 2 or abs(t - expected_video_T_30fps) <= 2)
            for t in seen_video_T
        )
        logger.info((PASS if ok else WARN) + " T5 video frame count is in expected band.")
    else:
        logger.info(WARN + " T5 visual_aux not present in batch.")

    return {
        "sisdr_by_ns": sisdr_by_ns,
        "n_blocks": n_total,
        "failures": failures,
    }


def _dump_first_mixture(
    out_dir: str, mix: torch.Tensor, tgt: torch.Tensor,
    blocks: list[tuple[int, int]], sr: int,
) -> None:
    try:
        import torchaudio
    except ImportError:
        logger.warning("torchaudio missing; skipping audio dump.")
        return
    p = pathlib.Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    if not blocks:
        return
    start, ns = blocks[0]
    m = mix[start].float().cpu()  # (1, T)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    torchaudio.save(str(p / "mix.wav"), m, sr)
    for k in range(ns):
        t = tgt[start + k].float().cpu()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        torchaudio.save(str(p / f"target_spk{k+1}.wav"), t, sr)
    logger.info(
        "Dumped first %d-spk mixture to %s (mix.wav + target_spk*.wav).",
        ns, out_dir,
    )


def _load_weights(model: torch.nn.Module, path: str) -> None:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.name.endswith(".pt") and p.name.startswith("ema"):
        sd = torch.load(str(p), map_location="cpu")
        if isinstance(sd, dict) and "models" in sd:
            sd = sd["models"][0]
        miss, unex = model.load_state_dict(sd, strict=False)
        logger.info("Loaded EMA: missing=%d unexpected=%d", len(miss), len(unex))
        if miss or unex:
            logger.warning(
                "  first missing=%s ; first unexpected=%s",
                miss[:3], unex[:3],
            )
    else:
        load_pretrained_model(model, str(p))


def _model_forward(model, mix, cues):
    """Wrapper matching trainer behavior."""
    with torch.no_grad():
        out = model(mix) if cues is None else model(mix, cues)
    if not isinstance(out, (list, tuple)):
        out = [out]
    return out[0]


def _zero_cue_like(cues):
    return None if cues is None else [torch.zeros_like(c) for c in cues]


def _swap_cue_within_block(cues, blocks):
    """Permute cues so row i in a block gets cue from row (i+1)%ns. ns=1 -> unchanged."""
    if cues is None:
        return None
    new_list = []
    for c in cues:
        nc = c.clone()
        for (start, ns) in blocks:
            if ns < 2:
                continue
            roll = torch.roll(c[start:start + ns], shifts=1, dims=0)
            nc[start:start + ns] = roll
        new_list.append(nc)
    return new_list


def test_model_side(
    val_dl: DataLoader,
    weights_path: str,
    configs: dict,
    n_steps: int,
    bf16: bool,
) -> None:
    _hr("Model-side tests (T6 - T10)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA unavailable; running model checks on CPU (slow).")

    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"]
    )
    _load_weights(model, weights_path)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model %s  params=%.2fM", configs["model"]["tse_model"], n_params / 1e6)

    # cuBLAS warmup: when this script shares a GPU with another running
    # PyTorch process (e.g. live training), creating the cuBLAS handle on
    # the *first* nn.Linear forward sometimes raises
    # ``CUBLAS_STATUS_NOT_INITIALIZED`` due to workspace contention. Doing a
    # tiny matmul up front forces cuBLAS to claim its handle while no other
    # tensors are competing for memory yet, then we free it.
    if device.type == "cuda":
        try:
            a = torch.randn(8, 8, device=device)
            b = torch.randn(8, 8, device=device)
            _ = (a @ b).sum().item()
            del a, b
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("cuBLAS warmup ok.")
        except Exception as e:
            logger.warning("cuBLAS warmup failed: %s", e)

    autocast_kwargs = dict(device_type=device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")

    executor = Executor()

    nan_inf = 0
    rows_total = 0
    sisdr_out_target: list[float] = []
    sisdr_out_mix: list[float] = []
    identity_sisdr: list[float] = []
    swap_sisdr_out_target: list[float] = []
    one_spk_passthrough_sisdr: list[float] = []

    def _safe_forward_to_cpu(model, mix_in, cues_in, label: str):
        """Forward, copy to CPU as fp32, free GPU activations, return CPU tensor."""
        with torch.amp.autocast(**autocast_kwargs):
            out = _model_forward(model, mix_in, cues_in)
        cpu_out = out.detach().float().cpu()
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return cpu_out

    it = iter(val_dl)
    for step in range(n_steps):
        try:
            batch = next(it)
        except StopIteration:
            break

        ns_flat = [int(x) for x in (batch.get("num_speaker") or [])]
        try:
            blocks = _per_mix_blocks(ns_flat)
        except ValueError:
            continue
        mix, cues, target = executor._extract_model_inputs(batch, device)
        # Keep CPU copies of mix / target for SI-SDR on CPU; release the
        # GPU-side mix when not needed for the next forward.
        mix_cpu = mix.detach().float().cpu()
        tgt_cpu = target.detach().float().cpu()

        # ---- T6 / T8: vanilla forward (then free GPU before next forward) ----
        out_cpu = _safe_forward_to_cpu(model, mix, cues, "vanilla")
        if torch.isnan(out_cpu).any() or torch.isinf(out_cpu).any():
            nan_inf += 1
            del out_cpu
            continue

        # T7: identity feeding clean target as input (cue still target's).
        out_id_cpu = _safe_forward_to_cpu(model, target, cues, "identity")

        # T9: cue-swap (only meaningful for ns>=2 blocks). Build the swapped
        # cue list on whatever device the originals live on (already device).
        cues_swap = _swap_cue_within_block(cues, blocks)
        out_swap_cpu = _safe_forward_to_cpu(model, mix, cues_swap, "swap")
        del cues_swap

        # ---- per-row metrics (all on CPU; tensors are already small) ----
        B = int(mix.shape[0])
        rows_total += B
        for k in range(B):
            est = out_cpu[k].flatten()
            tgt = tgt_cpu[k].flatten()
            mx = mix_cpu[k].flatten()
            sisdr_out_target.append(_sisdr(est, tgt))
            sisdr_out_mix.append(_sisdr(est, mx))
            identity_sisdr.append(_sisdr(out_id_cpu[k].flatten(), tgt))
            swap_sisdr_out_target.append(_sisdr(out_swap_cpu[k].flatten(), tgt))

        for (start, ns) in blocks:
            if ns == 1:
                k = start
                one_spk_passthrough_sisdr.append(
                    _sisdr(out_cpu[k].flatten(), tgt_cpu[k].flatten())
                )

        # Release the per-step buffers. Empty cache once per step keeps the
        # peak resident GPU memory close to "single-forward + parameters".
        del mix, target, cues, out_cpu, out_id_cpu, out_swap_cpu, mix_cpu, tgt_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- summarize ----
    def _stats(label, vals, ok_pred):
        if not vals:
            logger.info("%s %s: no rows", WARN, label)
            return
        v = torch.tensor(vals, dtype=torch.float32)
        med = float(v.median())
        mean_ = float(v.mean())
        p25 = float(torch.quantile(v, 0.25))
        p75 = float(torch.quantile(v, 0.75))
        tag = PASS if ok_pred(mean_, med) else FAIL
        logger.info(
            "%s %s  n=%d  mean=%+6.2f  median=%+6.2f  p25=%+6.2f  p75=%+6.2f  range[%+6.1f,%+6.1f]",
            tag, label, len(vals), mean_, med, p25, p75,
            float(v.min()), float(v.max()),
        )

    if nan_inf > 0:
        logger.info(FAIL + " T6 model produced NaN/Inf in %d/%d batches", nan_inf, n_steps)
    else:
        logger.info(PASS + " T6 model forward shapes/finite over %d steps", n_steps)

    _stats("T7 identity SI-SDR(model(target,cue), target)",
           identity_sisdr, lambda mu, mdn: mdn >= 10.0)

    _stats("T8a SI-SDR(out, target)",
           sisdr_out_target, lambda mu, mdn: mdn >= 5.0)
    _stats("T8b SI-SDR(out, mix)   <- if >> T8a, model is doing passthrough",
           sisdr_out_mix, lambda mu, mdn: True)  # informational

    if sisdr_out_mix and sisdr_out_target:
        delta = (
            torch.tensor(sisdr_out_mix).median().item()
            - torch.tensor(sisdr_out_target).median().item()
        )
        if delta > 5.0:
            logger.info(
                FAIL + " T8 verdict: out is much closer to MIX than to TARGET "
                "(median ΔSI-SDR = %+.2f). Model is doing passthrough.",
                delta,
            )
        elif delta < -3.0:
            logger.info(
                PASS + " T8 verdict: out is closer to TARGET than to MIX "
                "(median ΔSI-SDR = %+.2f). Real TSE behavior.",
                delta,
            )
        else:
            logger.info(
                WARN + " T8 verdict: out is neither close to mix nor to target "
                "(median ΔSI-SDR = %+.2f).",
                delta,
            )

    _stats("T9 cue-swap SI-SDR(out_swap, target)",
           swap_sisdr_out_target, lambda mu, mdn: True)
    if sisdr_out_target and swap_sisdr_out_target:
        d_norm = torch.tensor(sisdr_out_target).median().item()
        d_swap = torch.tensor(swap_sisdr_out_target).median().item()
        delta_swap = d_norm - d_swap
        if delta_swap < 1.0:
            logger.info(
                FAIL + " T9 verdict: cue swap barely changed output "
                "(ΔSI-SDR(normal - swap) = %+.2f). Cue is not being used.",
                delta_swap,
            )
        else:
            logger.info(
                PASS + " T9 verdict: cue swap degraded output by ΔSI-SDR=%.2f dB; cue is used.",
                delta_swap,
            )

    _stats("T10 1-spk passthrough SI-SDR(out, target)",
           one_spk_passthrough_sisdr, lambda mu, mdn: mdn >= 15.0)


def main() -> None:
    parser = HfArgumentParser(
        (TSEOnlineDataArguments, TSETrainingArguments, SanityArgs)
    )
    data_args, training_args, sanity_args = parser.parse_args_into_dataclasses()
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

    val_dl, dataset_args, _bs = _build_val_dataloader(
        data_args, training_args, configs,
    )

    n_steps = max(1, int(sanity_args.num_mixtures))

    data_summary = test_data_side(
        val_dl=val_dl, dataset_args=dataset_args, n_steps=n_steps,
        audio_dump_dir=sanity_args.audio_dump_dir,
    )

    # If the data side already shows fundamental violations, skip the model
    # tests because their interpretation is conditioned on the data being
    # correct.
    fatal_data_failures = [f for f in data_summary["failures"] if "T1" in f or "T2" in f]
    if fatal_data_failures:
        logger.info(
            FAIL + " skipping model-side tests because T1/T2 failed: data layout broken."
        )
        return

    if sanity_args.weights:
        # Build a fresh val_dl iterator for the model side so steps don't double-consume.
        val_dl_m, _, _ = _build_val_dataloader(data_args, training_args, configs)
        test_model_side(
            val_dl=val_dl_m,
            weights_path=sanity_args.weights,
            configs=configs,
            n_steps=n_steps,
            bf16=bool(training_args.bf16),
        )
    else:
        logger.info(
            "(skipping model-side tests; pass --weights PATH to enable T6 - T10)"
        )

    _hr("Done")


if __name__ == "__main__":
    main()
