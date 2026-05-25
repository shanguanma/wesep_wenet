# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

"""
Online mixture + optional visual cue (TorchCodec), **multi-source inventory** fork.

This module is a **self-contained copy** of ``online_dataset.py`` from the same
directory, extended with LRS3 / Chinese_lips scanners and merge helpers so
multi-corpus experiments do not touch single-dataset training code.

Pipeline (aligned with ``gen_online_mix_data_with_aug_without_repeat_speaker_with_visual_cue.py``):

  ``parse_raw_single_spk`` → ``resample`` → ``random_chunk`` → **speaker grouping**
  → ``apply_timeline`` → ``add_reverb`` → ``snr_mixer`` → (optional ``add_noise``) → **visual decode**.

  ``dataset_args[\"online_av_align\"]`` selects grouping / decode:

  * ``True``: ``processor_new`` + ``processor_visual_new`` (lip window matches audio chunk).

  * ``False`` (default): legacy ``processor`` + ``processor_visual``.

Multi-source helpers:

- **LRS3** ``<root>/<subset>/<youtube_video_id>/*.mp4``
- **Chinese_lips** ``<root>/<split>/<split>/<speaker_id>/FACE/*.mp4``

Train / val inventories: use :func:`build_train_val_merged_audio_visual_inventories`.
VoxCeleb2 and LRS3 have no official dev split — speakers are partitioned by
``train_fraction`` (same idea as :func:`split_speaker_ids_train_val`).
Chinese_lips uses the dataset's own ``train`` / ``val`` directories (no random
split). Final train (resp. val) inventory merges all training (resp. dev)
sources.

For a single flat merged list without train/val discipline, see
:func:`build_merged_audio_visual_inventory`.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
import time
import re
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from wesep.dataset import processor
from wesep.dataset import processor_new
from wesep.dataset import processor_visual
from wesep.dataset import processor_visual_new

logger = logging.getLogger(__name__)

# Reuse k* ids in a large ring so ``key_to_mp4`` stays bounded during infinite training.
_KEY_RING = 1 << 20


def scan_mp4_dir_of_voxceleb2(mp4_dir: str) -> dict[str, list[dict]]:
    mp4_root = Path(mp4_dir)
    if not mp4_root.is_dir():
        raise FileNotFoundError(f"mp4_dir does not exist: {mp4_dir}")

    inventory: dict[str, list[dict]] = {}
    for spk_dir in sorted(mp4_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        clips = []
        for vid_dir in sorted(spk_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            video_id = vid_dir.name
            for mp4_file in sorted(vid_dir.glob("*.mp4")):
                clips.append({
                    "video_id": video_id,
                    "clip_id": mp4_file.stem,
                    "path": str(mp4_file),
                })
        if clips:
            inventory[spk_id] = clips

    n_clips = sum(len(v) for v in inventory.values())
    logger.info(
        "[online_multi_dataset] VoxCeleb2-style scan: %s speakers, %s clips",
        len(inventory),
        n_clips,
    )
    return inventory


def iter_single_spk_json(
    inventory: dict[str, list[dict]],
    speaker_ids: list[str],
    rng: random.Random,
    key_to_mp4: dict[str, str],
) -> Iterator[dict[str, Any]]:
    """Yield dicts compatible with :func:`processor.parse_raw_single_spk` input."""
    n = 0
    while True:
        spk = rng.choice(speaker_ids)
        clip = rng.choice(inventory[spk])
        kid = f"k{(n % _KEY_RING):08d}"
        n += 1
        key_to_mp4[kid] = clip["path"]
        yield {
            "key": kid,
            "spk": [spk],
            "src": {spk: [clip["path"]]},
        }


def parse_mix_key_ids(mix_key: str) -> list[str]:
    """Parse ``mix_<k00000001>_<k00000002>_...`` into ordered clip-id tokens."""
    if not mix_key.startswith("mix_"):
        raise ValueError(f"Unexpected mixture key (no mix_ prefix): {mix_key}")
    rest = mix_key[len("mix_") :]
    if not rest:
        raise ValueError(f"Empty mixture key after mix_: {mix_key}")
    parts = rest.split("_")
    for p in parts:
        if not re.fullmatch(r"k\d+", p):
            raise ValueError(
                f"Unexpected segment in mixture key {mix_key!r}: {p!r} "
                "(expected k + digits, e.g. k00000001)"
            )
    return parts


def mp4_paths_for_mix_key(mix_key: str, key_to_mp4: dict[str, str]) -> list[str]:
    """Ordered mp4 paths for each slot in the mixture (same order as ``k*`` ids)."""
    kids = parse_mix_key_ids(mix_key)
    return [key_to_mp4[k] for k in kids]


def default_timeline_conf() -> dict[str, Any]:
    """Default timeline config (same as the visual-cue generation script)."""
    return {
        "two_speaker": {
            "overlap_ratio": [0.5, 1.0],
            "overlap_position": {
                "head": 0.3,
                "middle": 0.4,
                "tail": 0.3,
            },
            "middle_mode": {
                "crossing": 0.6,
                "containment": 0.4,
            },
        },
        "extra_speaker_activity": [0.1, 0.8],
        "silence": {
            "allow": False,
            "head_tail_ratio": [0.0, 0.1],
        },
    }


def build_dataset_args(
    *,
    resample_rate: int,
    chunk_len: int,
    whole_utt: bool,
    online_buffer_size: int,
    timeline_conf: dict | None,
    reverb_prob: float,
    reverb_conf: dict | None,
    snr_conf: dict,
    noise_prob: float,
    noise_lmdb_file: str | None,
) -> dict[str, Any]:
    """Defaults for online mixing; merge YAML ``dataset_args`` on top in training."""
    return {
        "resample_rate": resample_rate,
        "chunk_len": chunk_len,
        "whole_utt": whole_utt,
        "num_speakers": {
            "distribution": [0.1, 0.75, 0.15],
            "max_speakers": 4,
        },
        "online_buffer_size": online_buffer_size,
        "timeline": timeline_conf,
        "reverb_prob": reverb_prob,
        "reverb_conf": reverb_conf,
        "snr_conf": snr_conf,
        "noise_prob": noise_prob,
        "noise_lmdb_file": noise_lmdb_file,
    }


def build_dataset_args_from_tse_online_data_args(o: Any) -> dict[str, Any]:
    """
    Build a full ``dataset_args`` dict for online training **without** YAML.

    Expects an object with attributes matching
    ``TSEOnlineDataArguments`` in ``train_with_transformers_online.py`` (duck-typed).
    """
    dist_s = getattr(o, "num_speakers_distribution", "0.1,0.75,0.15")
    parts = [float(x.strip()) for x in str(dist_s).split(",")]
    if len(parts) != 3:
        raise ValueError(
            "num_speakers_distribution must be three comma-separated floats, e.g. 0.1,0.7,0.2"
        )
    #if bool(getattr(o, "force_two_speaker_only", False)):
    #    parts = [0.0, 1.0, 0.0]

    snr_lo = float(getattr(o, "snr_range_low", -5.0))
    snr_hi = float(getattr(o, "snr_range_high", 10.0))
    gain_lo = float(getattr(o, "gain_range_low", -12.0))
    gain_hi = float(getattr(o, "gain_range_high", 0.0))
    #if bool(getattr(o, "online_mix_clean_dry", False)):
    #    snr_lo = snr_hi = 0.0
    #    gain_lo = gain_hi = 0.0

    snr_conf = {
        "range": [snr_lo, snr_hi],
        "gain": [gain_lo, gain_hi],
    }
    da: dict[str, Any] = {
        "resample_rate": int(getattr(o, "resample_rate", 16000)),
        "chunk_len": int(getattr(o, "chunk_len", 48000)),
        "whole_utt": bool(getattr(o, "whole_utt", False)),
        "online_buffer_size": int(getattr(o, "online_buffer_size", 8)),
        "num_speakers": {
            "distribution": parts,
            "max_speakers": int(getattr(o, "num_speakers_max", 4)),
        },
        "timeline": None,
        "reverb_prob": float(getattr(o, "reverb_prob", 0.0)),
        "reverb_conf": None,
        "snr_conf": snr_conf,
        "noise_prob": float(getattr(o, "noise_prob", 0.0)),
        "noise_lmdb_file": getattr(o, "noise_lmdb_file", None),
        "online_mix_deterministic": bool(getattr(o, "online_mix_deterministic", False)),
        "sample_num_per_epoch": int(getattr(o, "sample_num_per_epoch", 20000)),
        "cues": {
            "visual": {
                "use": bool(getattr(o, "cue_visual_use", True)),
                "required": bool(getattr(o, "cue_visual_required", True)),
            }
        },
    }
    #if bool(getattr(o, "online_mix_clean_dry", False)):
    #    da["reverb_prob"] = 0.0
    #    da["noise_prob"] = 0.0
    #if bool(getattr(o, "force_two_speaker_only", False)):
    #    da["num_speakers"]["max_speakers"] = 2
    snv = getattr(o, "sample_num_per_epoch_val", None)
    if snv is not None:
        da["sample_num_per_epoch_val"] = int(snv)
    vm = int(getattr(o, "visual_max_frames", 96))
    da["visual_max_frames"] = vm if vm > 0 else None
    vr = int(getattr(o, "visual_resize", 0))
    da["visual_spatial_size"] = (vr, vr) if vr > 0 else None
    da["visual_decode_cuda"] = bool(getattr(o, "visual_decode_cuda", True))
    da["online_av_align"] = bool(getattr(o, "online_av_align", False))
    return da


def ensure_online_pipeline_defaults(dataset_args: dict[str, Any]) -> dict[str, Any]:
    """Apply defaults expected by :func:`run_online_mix_pipeline` if YAML omits them."""
    out = dict(dataset_args)
    if out.get("timeline") is None:
        out["timeline"] = default_timeline_conf()
    out.setdefault(
        "num_speakers",
        {"distribution": [0.1, 0.7, 0.2], "max_speakers": 4},
    )
    out.setdefault("online_buffer_size", 8)
    out.setdefault("reverb_prob", 0.0)
    out.setdefault("noise_prob", 0.0)
    out.setdefault("online_av_align", False)
    return out


def run_online_mix_pipeline(
    single_spk_iter: Iterator[dict[str, Any]],
    dataset_args: dict[str, Any],
    rng: random.Random,
) -> Iterator[dict[str, Any]]:
    """
    Same chain as ``build_audio_base_layer`` (train, online) + ``build_mix_layer``
    without ``shuffle`` / ``filter_len``.

    When ``dataset_args.get(\"online_av_align\", False)`` is True, grouping uses
    :func:`processor_new.sample_speaker_group_without_repeat` so each slot keeps
    ``chunk_ratio_spk{i}`` for lip–audio alignment. Otherwise uses legacy
    :func:`processor.sample_speaker_group_without_repeat`.
    """
    data = processor.parse_raw_single_spk(single_spk_iter)
    data = processor.resample(data, dataset_args["resample_rate"])

    if not dataset_args.get("whole_utt", False):
        data = processor.random_chunk(data, dataset_args["chunk_len"], rng)

    if bool(dataset_args.get("online_av_align", False)):
        data = processor_new.sample_speaker_group_without_repeat(
            data,
            dataset_args["num_speakers"],
            dataset_args["online_buffer_size"],
            dataset_args.get("timeline"),
            rng,
        )
    else:
        data = processor.sample_speaker_group_without_repeat(
            data,
            dataset_args["num_speakers"],
            dataset_args["online_buffer_size"],
            dataset_args.get("timeline"),
            rng,
        )
    data = processor.apply_timeline(data)
    data = processor.add_reverb(
        data,
        dataset_args.get("reverb_prob", 0),
        dataset_args.get("reverb_conf"),
        rng,
    )
    data = processor.snr_mixer(
        data,
        dataset_args.get("snr_conf"),
        rng,
    )

    if dataset_args.get("noise_prob", 0) and dataset_args.get("noise_lmdb_file"):
        data = processor.add_noise(
            data,
            dataset_args["noise_lmdb_file"],
            dataset_args["noise_prob"],
        )

    return data


def build_visual_resource_slice(
    sample: dict[str, Any],
    key_to_mp4: dict[str, str],
) -> dict[str, list[dict]]:
    """One-sample slice of ``visual.json`` for TorchCodec cue lookup (``mix_spk_id``)."""
    mix_key = sample["key"]
    n = int(sample["num_speaker"])
    kids = parse_mix_key_ids(mix_key)
    if len(kids) != n:
        raise ValueError(
            f"num_speaker={n} but mix key has {len(kids)} tokens: {mix_key!r}"
        )
    rd: dict[str, list[dict]] = {}
    for i in range(1, n + 1):
        spk_i = sample[f"spk{i}"]
        lk = f"{mix_key}::{spk_i}"
        mp4_path = key_to_mp4[kids[i - 1]]
        rd[lk] = [{
            "utt_id": f"{Path(mp4_path).parent.name}_{Path(mp4_path).stem}",
            "path": mp4_path,
        }]
    return rd


def apply_visual_cue_stream(
    mix_iter: Iterator[dict[str, Any]],
    key_to_mp4: dict[str, str],
    *,
    skip_visual_decode: bool,
    max_visual_time_frames: int | None = None,
    visual_spatial_size: tuple[int, int] | None = None,
    visual_decode_cuda: bool = True,
    online_av_align: bool = False,
) -> Iterator[dict[str, Any]]:
    """Mirror ``build_cue_layer`` → visual / fixed / ``mix_spk_id`` (training parity).

    Visual cue JSON is passed in-memory to TorchCodec (no per-sample tempfile on
    NFS — major latency win over writing/reading a temp ``.json`` per mixture).

    When ``online_av_align`` is True, decoding uses ``chunk_ratio_spk{i}`` plus
    :func:`~processor_visual_new.sample_fixed_visual_cue`; otherwise legacy
    :func:`~processor_visual.sample_fixed_visual_cue`.
    """
    if skip_visual_decode:
        yield from mix_iter
        return

    _timing = os.environ.get("WESPE_LOG_DATASET_TIMING", "").lower() in (
        "1",
        "true",
        "yes",
    )
    _t0 = time.perf_counter() if _timing else 0.0
    _seen_w: set[int] = set()

    for sample in mix_iter:
        rd = build_visual_resource_slice(sample, key_to_mp4)
        cue_fn = (
            processor_visual_new.sample_fixed_visual_cue
            if online_av_align
            else processor_visual.sample_fixed_visual_cue
        )
        gen = cue_fn(
            iter([sample]),
            resource_path=None,
            key_field="mix_spk_id",
            scope="speaker",
            required=True,
            max_visual_time_frames=max_visual_time_frames,
            spatial_size_hw=visual_spatial_size,
            visual_decode_cuda=visual_decode_cuda,
            spk_resource=rd,
        )
        enriched = next(gen)
        if _timing:
            wi = get_worker_info()
            wid = wi.id if wi is not None else -1
            if wid not in _seen_w:
                _seen_w.add(wid)
                logger.info(
                    "[dataset_timing] worker_id=%s first sample with video decode "
                    "ready after %.2fs (key=%r visual_decode_cuda=%s)",
                    wid,
                    time.perf_counter() - _t0,
                    enriched.get("key"),
                    visual_decode_cuda,
                )
        yield enriched


class OnlineMixIterableDataset(IterableDataset):
    """
    Infinite train iterator: MP4 inventory → online mix → optional visual decode.

    ``set_epoch(epoch)`` updates the RNG so each epoch uses a different stream
    (same pattern as :class:`DataList`), **unless** ``dataset_args["online_mix_deterministic"]``
    is True — then the mixture stream depends only on ``seed`` / rank / dataloader worker id,
    so **repeated runs with the same ``--seed`` see the same mixture sequence** (use
    ``--dataloader_num_workers 0`` if you need a single global stream without worker splits).

    Supports fast resume via :meth:`set_fast_forward`: during checkpoint
    resumption the HF Trainer's ``skip_first_batches`` still runs, but the
    first *N* samples are produced with the audio-only pipeline (no GPU
    video decode), making the skip orders-of-magnitude faster.
    """

    def __init__(
        self,
        inventory: dict[str, list[dict]],
        speaker_ids: list[str],
        dataset_args: dict[str, Any],
        *,
        seed: int = 42,
        with_visual_cue: bool = True,
        skip_visual_decode: bool = False,
    ) -> None:
        super().__init__()
        if len(speaker_ids) < 2:
            raise ValueError("OnlineMixIterableDataset needs at least 2 speaker IDs.")
        self.inventory = inventory
        self.speaker_ids = list(speaker_ids)
        self.dataset_args = dataset_args
        self.seed = int(seed)
        self.with_visual_cue = with_visual_cue
        self.skip_visual_decode = skip_visual_decode
        self.max_visual_time_frames: int | None = dataset_args.get(
            "visual_max_frames")
        self.visual_spatial_size: tuple[int, int] | None = dataset_args.get(
            "visual_spatial_size")
        self._online_mix_deterministic = bool(
            dataset_args.get("online_mix_deterministic", False))
        self._epoch = 0
        # Per-worker counters, populated inside __iter__ for state_dict().
        self._worker_rng: dict[int, random.Random] = {}
        self._worker_sample_count: dict[int, int] = {}
        # Fast-forward budget: total samples to produce with audio-only
        # pipeline (no visual decode) before switching to full mode.
        # Set via set_fast_forward() before the DataLoader workers start.
        self._ff_total_samples: int = 0
        self._ff_batch_size: int = 1
        self._ff_armed: bool = False

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        # After the resumed epoch completes, the next set_epoch clears
        # fast-forward so subsequent epochs run the full pipeline.
        if self._ff_armed:
            self._ff_armed = False
        else:
            self._ff_total_samples = 0

    # ---- fast-resume helpers ------------------------------------------------

    def set_fast_forward(self, total_skip_samples: int, batch_size: int = 1) -> None:
        """Tell the dataset to produce *total_skip_samples* lightweight samples
        (audio pipeline only, dummy visual) before switching to the full
        pipeline.  Call **before** the DataLoader workers start (i.e. before
        ``trainer.train``).

        The HF Trainer's ``skip_first_batches`` will consume and discard these
        lightweight batches, giving the same correct resume semantics —
        but orders-of-magnitude faster because GPU video decode is skipped.
        """
        self._ff_total_samples = max(int(total_skip_samples), 0)
        self._ff_batch_size = max(int(batch_size), 1)
        self._ff_armed = self._ff_total_samples > 0
        if self._ff_total_samples > 0:
            logger.info(
                "[OnlineMixIterableDataset] fast-forward armed: %s samples "
                "(batch_size=%s) will use audio-only pipeline",
                self._ff_total_samples, self._ff_batch_size,
            )

    # ---- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        num_workers = max(1, worker.num_workers if worker is not None else 1)

        if self._online_mix_deterministic:
            rng = random.Random(self.seed + rank * 17_711 + wid * 1_009)
        else:
            rng = random.Random(
                self.seed + self._epoch * 1_000_003 + rank * 17_711 + wid * 1_009
            )

        self._worker_rng[wid] = rng
        self._worker_sample_count[wid] = 0

        key_to_mp4: dict[str, str] = {}
        single_iter = iter_single_spk_json(
            self.inventory,
            self.speaker_ids,
            rng,
            key_to_mp4,
        )
        mix_iter = run_online_mix_pipeline(single_iter, self.dataset_args, rng)

        # ---- Phase 1: fast-forward (audio-only, dummy visual) ----
        # Floor division: guarantee total dummies ≤ skip budget so no dummy
        # sample can leak past skip_first_batches into a real training batch.
        # Any shortfall (at most num_workers-1 samples) is covered by real
        # samples that get discarded during the skip — tiny cost, full safety.
        ff_per_worker = self._ff_total_samples // num_workers if self._ff_total_samples > 0 else 0
        # Clear so persistent workers don't re-apply on the next epoch.
        self._ff_total_samples = 0

        if ff_per_worker > 0 and self.with_visual_cue:
            _h, _w = self.visual_spatial_size or (96, 96)
            _t0 = time.perf_counter()
            _skipped = 0
            for sample in mix_iter:
                if _skipped >= ff_per_worker:
                    mix_iter = itertools.chain([sample], mix_iter)
                    break
                ns = int(sample.get("num_speaker", 2))
                for si in range(1, ns + 1):
                    sample[f"visual_spk{si}"] = torch.zeros(3, _h, _w, 1)
                self._worker_sample_count[wid] += 1
                _skipped += 1
                yield sample
            _elapsed = time.perf_counter() - _t0
            logger.info(
                "[fast-forward] worker %s: produced %s audio-only samples "
                "in %.1fs (skipping video decode)",
                wid, _skipped, _elapsed,
            )

        # ---- Phase 2: full pipeline (audio + visual decode) ----
        if self.with_visual_cue:
            mix_iter = apply_visual_cue_stream(
                mix_iter,
                key_to_mp4,
                skip_visual_decode=self.skip_visual_decode,
                max_visual_time_frames=self.max_visual_time_frames,
                visual_spatial_size=self.visual_spatial_size,
                visual_decode_cuda=bool(
                    self.dataset_args.get("visual_decode_cuda", True)),
                online_av_align=bool(
                    self.dataset_args.get("online_av_align", False)),
            )
        for sample in mix_iter:
            self._worker_sample_count[wid] += 1
            yield sample


def resolve_speaker_pool(
    inventory: dict[str, list[dict]],
    rng: random.Random,
    *,
    speakers: list[str] | None,
    num_speakers: int,
) -> list[str]:
    """Pick speaker IDs (same semantics as the generation CLI).

    If ``speakers`` is omitted: ``num_speakers <= 0`` or ``>= len(inventory)``
    returns **all** speaker IDs in sorted order (no random subsample, no extra
    list copies beyond one key list). Otherwise returns a random subset of size
    ``min(num_speakers, len(inventory))``.
    """
    if speakers:
        missing = [s for s in speakers if s not in inventory]
        if missing:
            raise ValueError(f"Unknown speakers: {missing}")
        return list(speakers)
    all_spk = sorted(inventory.keys())
    n_req = int(num_speakers)
    if n_req <= 0 or n_req >= len(all_spk):
        return list(all_spk)
    return rng.sample(all_spk, n_req)


def subset_inventory(
    inventory: dict[str, list[dict]],
    speaker_ids: list[str],
) -> dict[str, list[dict]]:
    """Restrict MP4 inventory to the given speakers (order preserved)."""
    missing = [s for s in speaker_ids if s not in inventory]
    if missing:
        raise ValueError(f"Speakers not in inventory: {missing}")
    return {s: inventory[s] for s in speaker_ids}


def split_speaker_ids_train_val(
    speaker_ids: list[str],
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Disjoint train / val speaker pools (default 80% / 20%, i.e. 4:1 by count).

    Both sides need at least 2 speakers so :class:`OnlineMixIterableDataset`
    can form mixtures. Requires ``len(speaker_ids) >= 4``.
    """
    n = len(speaker_ids)
    if n < 4:
        raise ValueError(
            f"Need at least 4 speakers in the pool to split train/val "
            f"(>=2 each); got {n}"
        )
    tf = float(train_fraction)
    if not (0.0 < tf < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    rng = random.Random(int(seed))
    perm = sorted(speaker_ids)
    rng.shuffle(perm)

    n_train = int(round(n * tf))
    n_train = max(2, min(n_train, n - 2))
    train_ids = perm[:n_train]
    val_ids = perm[n_train:]
    assert len(train_ids) >= 2 and len(val_ids) >= 2
    return train_ids, val_ids


def default_online_val_samples_per_epoch(train_samples: int) -> int:
    """Match offline 20k train / 5k val → ratio 1:4 val vs train."""
    t = int(train_samples)
    return max(1, int(round(t * 5000 / 20000)))


# ---------------------------------------------------------------------------
# Multi-corpus inventory (LRS3, Chinese_lips) + merge
# ---------------------------------------------------------------------------

# Defaults for common cluster paths (override in callers).
DEFAULT_LRS3_ROOT = "/F00120240032/lrs3"
DEFAULT_CHINESE_LIPS_ROOT = "/F00120240032/Chinese_lips"

PREFIX_VOXCELEB2 = "vox2"
PREFIX_LRS3 = "lrs3:"
PREFIX_CHINESE_LIPS = "cnlips:"


def _resolve_lrs3_subset_directory(root: Path, subset: str) -> Path | None:
    """
    Canonical layout: ``root/<subset>/<youtube_id>/*.mp4`` with ``root`` = corpus root.

    Also accepts ``root`` already pointing at ``.../<subset>`` (shell scripts often set
    ``--lrs3_root .../lrs3/trainval``); in that case do **not** append ``subset`` again.
    """
    candidate = root / subset
    if candidate.is_dir():
        return candidate
    if root.name == subset and root.is_dir():
        logger.info(
            "[online_multi_dataset] LRS3: lrs3_root ends with subset %r — scanning "
            "%s directly (avoid .../%s/%s)",
            subset,
            root,
            subset,
            subset,
        )
        return root
    return None


def scan_mp4_dir_of_lrs3(
    lrs3_root: str,
    *,
    subsets: tuple[str, ...] = ("trainval",),
    require_existing: bool = False,
) -> dict[str, list[dict]]:
    """
    Scan LRS3 MP4 clips into the same inventory schema as VoxCeleb2-style scans.

    Expected layout::

        <lrs3_root>/<subset>/<youtube_video_id>/*.mp4

    If ``lrs3_root`` is already ``.../<subset>`` (e.g. ``.../lrs3/trainval``), pass
    ``subsets=("trainval",)`` — that directory is scanned without doubling the name.

    Each ``youtube_video_id`` directory is one speaker bucket (same speaker within
    the clips). Typical ``subset`` values: ``trainval``, ``test``.
    """
    root = Path(lrs3_root)
    if not root.is_dir():
        raise FileNotFoundError(f"lrs3_root does not exist: {lrs3_root}")

    inventory: dict[str, list[dict]] = {}
    for subset in subsets:
        sub_root = _resolve_lrs3_subset_directory(root, subset)
        if sub_root is None:
            tried = root / subset
            msg = (
                f"LRS3 subset directory missing: {tried} "
                f"(if your data live under .../lrs3/trainval, set --lrs3_root to "
                f"the corpus root .../lrs3, or pass .../lrs3/trainval and keep "
                f"--lrs3_subsets trainval)"
            )
            if require_existing:
                raise FileNotFoundError(msg)
            logger.warning("[online_multi_dataset] %s", msg)
            continue

        for vid_dir in sorted(sub_root.iterdir()):
            if not vid_dir.is_dir():
                continue
            video_id = vid_dir.name
            clips: list[dict] = []
            for mp4_file in sorted(vid_dir.glob("*.mp4")):
                clips.append({
                    "video_id": video_id,
                    "clip_id": mp4_file.stem,
                    "path": str(mp4_file),
                })
            if clips:
                if video_id in inventory:
                    inventory[video_id].extend(clips)
                else:
                    inventory[video_id] = clips

    n_clips = sum(len(v) for v in inventory.values())
    logger.info(
        "[online_multi_dataset] LRS3: subsets=%s -> %s speakers, %s clips (%s)",
        subsets,
        len(inventory),
        n_clips,
        lrs3_root,
    )
    return inventory


def scan_mp4_dir_of_chinese_lips(
    chinese_lips_root: str,
    *,
    splits: tuple[str, ...] = ("train",),
    face_subdir: str = "FACE",
    require_existing: bool = False,
) -> dict[str, list[dict]]:
    """
    Scan Chinese_lips face-crop MP4s into the VoxCeleb2-compatible inventory schema.

    Expected layout::

        <root>/<split>/<split>/<speaker_id>/<face_subdir>/*.mp4

    Example::

        .../Chinese_lips/train/train/096_27_F_JKYS/FACE/*.mp4
        .../Chinese_lips/val/val/155_54_F_ZX/FACE/*.mp4

    Parameters
    ----------
    splits:
        Which top-level splits to include (each tuple element is scanned and
        merged into one inventory). Use ``("train", "val")`` to pool train+val.
    face_subdir:
        Subdirectory name under each speaker (default ``FACE``).
    """
    root = Path(chinese_lips_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"chinese_lips_root does not exist: {chinese_lips_root}")

    inventory: dict[str, list[dict]] = {}
    for split in splits:
        base = root / split / split
        if not base.is_dir():
            msg = f"Chinese_lips split directory missing: {base}"
            if require_existing:
                raise FileNotFoundError(msg)
            logger.warning("[online_multi_dataset] %s", msg)
            continue

        for spk_dir in sorted(base.iterdir()):
            if not spk_dir.is_dir():
                continue
            spk_id = spk_dir.name
            face_dir = spk_dir / face_subdir
            if not face_dir.is_dir():
                continue
            clips: list[dict] = []
            for mp4_file in sorted(face_dir.glob("*.mp4")):
                clips.append({
                    "video_id": face_subdir,
                    "clip_id": mp4_file.stem,
                    "path": str(mp4_file),
                })
            if not clips:
                continue
            if spk_id in inventory:
                inventory[spk_id].extend(clips)
            else:
                inventory[spk_id] = clips

    n_clips = sum(len(v) for v in inventory.values())
    logger.info(
        "[online_multi_dataset] Chinese_lips: splits=%s -> %s speakers, %s clips (%s)",
        splits,
        len(inventory),
        n_clips,
        root,
    )
    return inventory


def prefix_speaker_ids(
    inventory: dict[str, list[dict]],
    prefix: str,
) -> dict[str, list[dict]]:
    """Prefix every speaker key so multiple corpora can be merged safely."""
    if not prefix:
        return dict(inventory)
    if not prefix.endswith(":"):
        prefix = f"{prefix}:"
    return {f"{prefix}{k}": v for k, v in inventory.items()}


def merge_inventories(
    *parts: dict[str, list[dict]],
    strict: bool = True,
) -> dict[str, list[dict]]:
    """Merge inventories left-to-right; optionally enforce disjoint speaker ids."""
    out: dict[str, list[dict]] = {}
    for p in parts:
        dup = set(out) & set(p)
        if dup and strict:
            sample = sorted(dup)[:8]
            raise ValueError(
                "merge_inventories: duplicate speaker ids "
                f"(showing up to 8): {sample}"
            )
        out.update(p)
    return out


def build_train_val_merged_audio_visual_inventories(
    *,
    train_fraction: float = 0.8,
    split_seed: int = 42,
    voxceleb2_mp4_dir: str | None = None,
    use_voxceleb2: bool = False,
    lrs3_root: str | None = DEFAULT_LRS3_ROOT,
    lrs3_subsets: tuple[str, ...] = ("trainval",),
    use_lrs3: bool = True,
    chinese_lips_root: str | None = DEFAULT_CHINESE_LIPS_ROOT,
    use_chinese_lips: bool = True,
    chinese_lips_train_split: str = "train",
    chinese_lips_val_split: str = "val",
    prefix_voxceleb2: str = "",
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """
    Build merged **train** and **validation** inventories for online mixing.

    - **VoxCeleb2 / LRS3:** full scan, then :func:`split_speaker_ids_train_val`
      per corpus (same ``train_fraction`` / ``split_seed``). Train and val
      speaker sets are disjoint within each corpus.
    - **Chinese_lips:** no random split; ``chinese_lips_train_split`` (e.g.
      ``train``) only contributes to the train inventory, and
      ``chinese_lips_val_split`` (e.g. ``val``) only to the val inventory.

    Speaker keys are prefixed with :data:`PREFIX_VOXCELEB2` / :data:`PREFIX_LRS3` /
    :data:`PREFIX_CHINESE_LIPS` so merged dicts stay collision-free.

    Raises
    ------
    ValueError
        If a ratio-split corpus has fewer than 4 speaker buckets, or if no
        sources are enabled, or merged train/val inventories are empty.
    """
    chunks_tr: list[dict[str, list[dict]]] = []
    chunks_va: list[dict[str, list[dict]]] = []

    vox_pfx = prefix_voxceleb2.strip() if prefix_voxceleb2.strip() else PREFIX_VOXCELEB2

    if use_voxceleb2:
        if not voxceleb2_mp4_dir:
            raise ValueError("use_voxceleb2=True requires voxceleb2_mp4_dir")
        inv_full = scan_mp4_dir_of_voxceleb2(voxceleb2_mp4_dir)
        all_spk = sorted(inv_full.keys())
        if len(all_spk) < 4:
            raise ValueError(
                "VoxCeleb2: need at least 4 speaker buckets for train/val split "
                f"(same as split_speaker_ids_train_val); got {len(all_spk)}"
            )
        tr_ids, va_ids = split_speaker_ids_train_val(
            all_spk,
            train_fraction=train_fraction,
            seed=split_seed,
        )
        chunks_tr.append(
            prefix_speaker_ids(subset_inventory(inv_full, tr_ids), vox_pfx))
        chunks_va.append(
            prefix_speaker_ids(subset_inventory(inv_full, va_ids), vox_pfx))

    if use_lrs3 and lrs3_root:
        inv_full = scan_mp4_dir_of_lrs3(lrs3_root, subsets=lrs3_subsets)
        all_spk = sorted(inv_full.keys())
        if len(all_spk) < 4:
            raise ValueError(
                "LRS3: need at least 4 speaker buckets for train/val split; "
                f"got {len(all_spk)}"
            )
        tr_ids, va_ids = split_speaker_ids_train_val(
            all_spk,
            train_fraction=train_fraction,
            seed=split_seed,
        )
        chunks_tr.append(
            prefix_speaker_ids(
                subset_inventory(inv_full, tr_ids),
                PREFIX_LRS3.rstrip(":"),
            ))
        chunks_va.append(
            prefix_speaker_ids(
                subset_inventory(inv_full, va_ids),
                PREFIX_LRS3.rstrip(":"),
            ))

    if use_chinese_lips and chinese_lips_root:
        inv_tr = scan_mp4_dir_of_chinese_lips(
            chinese_lips_root,
            splits=(chinese_lips_train_split,),
        )
        inv_va = scan_mp4_dir_of_chinese_lips(
            chinese_lips_root,
            splits=(chinese_lips_val_split,),
        )
        if not inv_tr:
            logger.warning(
                "[online_multi_dataset] Chinese_lips train inventory empty "
                "(split=%s)",
                chinese_lips_train_split,
            )
        if not inv_va:
            logger.warning(
                "[online_multi_dataset] Chinese_lips val inventory empty "
                "(split=%s)",
                chinese_lips_val_split,
            )
        chunks_tr.append(
            prefix_speaker_ids(inv_tr, PREFIX_CHINESE_LIPS.rstrip(":")))
        chunks_va.append(
            prefix_speaker_ids(inv_va, PREFIX_CHINESE_LIPS.rstrip(":")))

    if not chunks_tr or not chunks_va:
        raise ValueError(
            "build_train_val_merged_audio_visual_inventories: no sources "
            "enabled or one side is empty (check use_* flags and roots)."
        )

    merged_tr = merge_inventories(*chunks_tr)
    merged_va = merge_inventories(*chunks_va)
    if len(merged_tr) < 2 or len(merged_va) < 2:
        raise ValueError(
            "After merge, train and val inventories each need at least 2 "
            f"speakers for mixing; got train={len(merged_tr)} val={len(merged_va)}"
        )

    logger.info(
        "[online_multi_dataset] merged train inventory: %s speakers, %s clips",
        len(merged_tr),
        sum(len(v) for v in merged_tr.values()),
    )
    logger.info(
        "[online_multi_dataset] merged val inventory: %s speakers, %s clips",
        len(merged_va),
        sum(len(v) for v in merged_va.values()),
    )
    return merged_tr, merged_va


def build_merged_audio_visual_inventory(
    *,
    voxceleb2_mp4_dir: str | None = None,
    lrs3_root: str | None = DEFAULT_LRS3_ROOT,
    lrs3_subsets: tuple[str, ...] = ("trainval",),
    chinese_lips_root: str | None = DEFAULT_CHINESE_LIPS_ROOT,
    chinese_lips_splits: tuple[str, ...] = ("train",),
    prefix_voxceleb2: str = "",
    use_lrs3: bool = True,
    use_chinese_lips: bool = True,
    use_voxceleb2: bool = False,
) -> dict[str, list[dict]]:
    """
    Build one **flat** combined inventory (no train/val separation).

    Prefer :func:`build_train_val_merged_audio_visual_inventories` for training
    with a proper validation set.

    Speaker ids from LRS3 and Chinese_lips are prefixed with
    :data:`PREFIX_LRS3` and :data:`PREFIX_CHINESE_LIPS` to avoid collisions.
    Optional VoxCeleb2 scan uses ``prefix_voxceleb2`` when non-empty; otherwise
    :data:`PREFIX_VOXCELEB2` should be passed explicitly if merging with other
    corpora.

    Parameters
    ----------
    use_voxceleb2:
        If True, ``voxceleb2_mp4_dir`` must be set and scanned.
    use_lrs3 / use_chinese_lips:
        Set False to skip a source (and ignore its root).
    """
    chunks: list[dict[str, list[dict]]] = []

    if use_voxceleb2:
        if not voxceleb2_mp4_dir:
            raise ValueError("use_voxceleb2=True requires voxceleb2_mp4_dir")
        inv = scan_mp4_dir_of_voxceleb2(voxceleb2_mp4_dir)
        vox_pfx = prefix_voxceleb2.strip() if prefix_voxceleb2.strip() else PREFIX_VOXCELEB2
        inv = prefix_speaker_ids(inv, vox_pfx)
        chunks.append(inv)

    if use_lrs3 and lrs3_root:
        inv = scan_mp4_dir_of_lrs3(lrs3_root, subsets=lrs3_subsets)
        chunks.append(prefix_speaker_ids(inv, PREFIX_LRS3.rstrip(":")))

    if use_chinese_lips and chinese_lips_root:
        inv = scan_mp4_dir_of_chinese_lips(
            chinese_lips_root,
            splits=chinese_lips_splits,
        )
        chunks.append(prefix_speaker_ids(inv, PREFIX_CHINESE_LIPS.rstrip(":")))

    if not chunks:
        raise ValueError(
            "build_merged_audio_visual_inventory: no sources enabled / all roots empty"
        )

    merged = merge_inventories(*chunks)
    logger.info(
        "[online_multi_dataset] merged inventory: %s speakers, %s clips",
        len(merged),
        sum(len(v) for v in merged.values()),
    )
    return merged


__all__ = [
    "DEFAULT_CHINESE_LIPS_ROOT",
    "DEFAULT_LRS3_ROOT",
    "PREFIX_CHINESE_LIPS",
    "PREFIX_LRS3",
    "PREFIX_VOXCELEB2",
    "OnlineMixIterableDataset",
    "apply_visual_cue_stream",
    "build_dataset_args",
    "build_dataset_args_from_tse_online_data_args",
    "build_merged_audio_visual_inventory",
    "build_train_val_merged_audio_visual_inventories",
    "build_visual_resource_slice",
    "default_online_val_samples_per_epoch",
    "default_timeline_conf",
    "ensure_online_pipeline_defaults",
    "iter_single_spk_json",
    "merge_inventories",
    "mp4_paths_for_mix_key",
    "parse_mix_key_ids",
    "prefix_speaker_ids",
    "resolve_speaker_pool",
    "run_online_mix_pipeline",
    "scan_mp4_dir_of_chinese_lips",
    "scan_mp4_dir_of_lrs3",
    "scan_mp4_dir_of_voxceleb2",
    "split_speaker_ids_train_val",
    "subset_inventory",
]
