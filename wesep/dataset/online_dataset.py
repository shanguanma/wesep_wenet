# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

"""
Online mixture + optional visual cue (TorchCodec) for training or data generation.

Pipeline (aligned with ``gen_online_mix_data_with_aug_without_repeat_speaker_with_visual_cue.py``):

  ``parse_raw_single_spk`` → ``resample`` → ``random_chunk`` →
  ``sample_speaker_group_without_repeat`` → ``apply_timeline`` → ``add_reverb`` →
  ``snr_mixer`` → (optional ``add_noise``) →
  ``processor_visual.sample_fixed_visual_cue`` (fixed / ``mix_spk_id``)

Use :class:`OnlineMixIterableDataset` in training scripts; use the helpers from
data generation CLIs without duplicating logic.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from wesep.dataset import processor
from wesep.dataset import processor_visual

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
    logger.info(f"[online_dataset] Scanned {len(inventory)} speakers, {n_clips} clips")
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
    snr_conf = {
        "range": [
            float(getattr(o, "snr_range_low", -5.0)),
            float(getattr(o, "snr_range_high", 10.0)),
        ],
        "gain": [
            float(getattr(o, "gain_range_low", -12.0)),
            float(getattr(o, "gain_range_high", 0.0)),
        ],
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
        "sample_num_per_epoch": int(getattr(o, "sample_num_per_epoch", 20000)),
        "cues": {
            "visual": {
                "use": bool(getattr(o, "cue_visual_use", True)),
                "required": bool(getattr(o, "cue_visual_required", True)),
            }
        },
    }
    snv = getattr(o, "sample_num_per_epoch_val", None)
    if snv is not None:
        da["sample_num_per_epoch_val"] = int(snv)
    vm = int(getattr(o, "visual_max_frames", 96))
    da["visual_max_frames"] = vm if vm > 0 else None
    vr = int(getattr(o, "visual_resize", 0))
    da["visual_spatial_size"] = (vr, vr) if vr > 0 else None
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
    return out


def run_online_mix_pipeline(
    single_spk_iter: Iterator[dict[str, Any]],
    dataset_args: dict[str, Any],
    rng: random.Random,
) -> Iterator[dict[str, Any]]:
    """
    Same chain as ``build_audio_base_layer`` (train, online) + ``build_mix_layer``
    without ``shuffle`` / ``filter_len``, but using
    :func:`processor.sample_speaker_group_without_repeat`.
    """
    data = processor.parse_raw_single_spk(single_spk_iter)
    data = processor.resample(data, dataset_args["resample_rate"])

    if not dataset_args.get("whole_utt", False):
        data = processor.random_chunk(data, dataset_args["chunk_len"], rng)

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
    """One-sample slice of ``visual.json`` for :func:`sample_fixed_visual_cue` (``mix_spk_id``)."""
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
) -> Iterator[dict[str, Any]]:
    """Mirror ``build_cue_layer`` → visual / fixed / ``mix_spk_id`` (training parity).

    Cue metadata is passed in-memory (no tempfile per sample).
    """
    if skip_visual_decode:
        yield from mix_iter
        return

    for sample in mix_iter:
        rd = build_visual_resource_slice(sample, key_to_mp4)
        gen = processor_visual.sample_fixed_visual_cue(
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
        yield enriched


class OnlineMixIterableDataset(IterableDataset):
    """
    Infinite train iterator: MP4 inventory → online mix → optional visual decode.

    ``set_epoch(epoch)`` updates the RNG so each epoch uses a different stream
    (same pattern as :class:`DataList`).
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
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        # Separate streams per rank / worker / epoch
        rng = random.Random(
            self.seed + self._epoch * 1_000_003 + rank * 17_711 + wid * 1_009
        )
        key_to_mp4: dict[str, str] = {}
        single_iter = iter_single_spk_json(
            self.inventory,
            self.speaker_ids,
            rng,
            key_to_mp4,
        )
        mix_iter = run_online_mix_pipeline(single_iter, self.dataset_args, rng)
        if self.with_visual_cue:
            mix_iter = apply_visual_cue_stream(
                mix_iter,
                key_to_mp4,
                skip_visual_decode=self.skip_visual_decode,
                max_visual_time_frames=self.max_visual_time_frames,
                visual_spatial_size=self.visual_spatial_size,
                visual_decode_cuda=bool(
                    self.dataset_args.get("visual_decode_cuda", True)),
            )
        yield from mix_iter


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
