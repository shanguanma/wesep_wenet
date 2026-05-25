#!/usr/bin/env python3
"""
Generate 2-speaker mixture data using the **same online-mix pipeline** as training:
``wesep.dataset.processor`` (see ``dataset.py`` → ``build_mix_layer``).

Pipeline (mirrors ``Dataset`` with ``online_mix: true``):
  ``parse_raw_single_spk`` → ``resample`` → ``random_chunk`` →
  ``sample_speaker_group_without_repeat`` → ``apply_timeline`` → ``add_reverb`` → ``snr_mixer``
  (optional ``add_noise`` if ``noise_prob > 0``).

Single-speaker lines point to VoxCeleb2 **mp4** paths (audio decoded like training).

**Note:** ``sample_speaker_group_without_repeat`` buffers ``online_buffer_size`` single-speaker
samples before the first mixture is emitted. Use a small buffer (e.g. 8–32) for
quick tests.

Examples
--------
# Minimal test (small buffer, no reverb, no extra noise)
python gen_online_mix_data2.py \
    --mp4_dir /F00120240032/voxceleb2_zk_mixture/mp4/train \
    --output_dir ./online_mix2_test \
    --num_mixtures 10 \
    --online_buffer_size 8 \
    --num_speakers 4

# Closer to training config (3 s chunks @ 16 kHz, SNR range, optional reverb)
python gen_online_mix_data2.py \
    --mp4_dir /F00120240032/voxceleb2_zk_mixture/mp4/train \
    --output_dir ./online_mix2 \
    --num_mixtures 10 \
    --online_buffer_size 8 \
    --chunk_len 48000 \
    --reverb_prob 0.0 \
    --snr_range -5 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import torch
import torchaudio
import yaml

from wesep.dataset import processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VoxCeleb2 mp4 scan (same layout as gen_online_mix_data.py)
# ---------------------------------------------------------------------------

def scan_mp4_dir(mp4_dir: str) -> dict[str, list[dict]]:
    mp4_root = Path(mp4_dir)
    if not mp4_root.is_dir():
        raise FileNotFoundError(f"mp4_dir does not exist: {mp4_dir}")

    inventory: dict[str, list[dict]] = {}
    for spk_dir in sorted(mp4_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name # 
        #print(f"spk_id: {spk_id}")
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
    logger.info(f"Scanned {len(inventory)} speakers, {n_clips} clips")
    return inventory


# ---------------------------------------------------------------------------
# Single-speaker stream + sidecar for visual.json (keys: k00000001, …)
# ---------------------------------------------------------------------------

def iter_single_spk_json(
    inventory: dict[str, list[dict]],
    speaker_ids: list[str],
    rng: random.Random,
    key_to_mp4: dict[str, str],
):
    """Yield dicts compatible with ``parse_raw_single_spk`` input."""
    n = 0
    while True:
        spk = rng.choice(speaker_ids)
        clip = rng.choice(inventory[spk])
        kid = f"k{n:08d}"
        n += 1
        key_to_mp4[kid] = clip["path"]
        yield {
            "key": kid,
            "spk": [spk],
            "src": {spk: [clip["path"]]},
        }


_MIX_KEY_RE = re.compile(r"^mix_(k\d+)_(k\d+)$")


def parse_mix_key_for_visual(mix_key: str, key_to_mp4: dict[str, str]) -> dict[str, str]:
    """Map mixture key to per-speaker mp4 paths using sidecar lookup."""
    m = _MIX_KEY_RE.match(mix_key)
    if not m:
        raise ValueError(f"Unexpected mixture key format: {mix_key}")
    k1, k2 = m.group(1), m.group(2)
    return {
        "spk1_mp4": key_to_mp4[k1],
        "spk2_mp4": key_to_mp4[k2],
    }


# ---------------------------------------------------------------------------
# Online-mix chain (same order as dataset.build_mix_layer + base layers)
# ---------------------------------------------------------------------------

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
) -> dict:
    """Always 2 speakers: distribution [0, 1, 0] → ``sample_num_speakers`` returns 2."""
    return {
        "resample_rate": resample_rate,
        "chunk_len": chunk_len,
        "whole_utt": whole_utt,
        "num_speakers": {
            "distribution": [0.0, 1.0, 0.0],
        "online_buffer_size": online_buffer_size,
        "timeline": timeline_conf,
        "reverb_prob": reverb_prob,
        "reverb_conf": reverb_conf,
        "snr_conf": snr_conf,
        "noise_prob": noise_prob,
        "noise_lmdb_file": noise_lmdb_file,
    }


def run_online_mix_pipeline(
    single_spk_iter,
    dataset_args: dict,
    rng: random.Random,
):
    """
    Equivalent to ``build_audio_base_layer`` (train, online) + ``build_mix_layer``
    without ``shuffle`` / ``filter_len``.
    """
    data = processor.parse_raw_single_spk(single_spk_iter)
    data = processor.resample(data, dataset_args["resample_rate"])

    if not dataset_args.get("whole_utt", False):
        data = processor.random_chunk(data, dataset_args["chunk_len"])

    data = processor.sample_speaker_group_without_repeat(
        data,
        dataset_args["num_speakers"],
        dataset_args["online_buffer_size"],
        dataset_args.get("timeline"),
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


# ---------------------------------------------------------------------------
# Write outputs (samples.jsonl, cues, wavs)
# ---------------------------------------------------------------------------

def safe_filename(key: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", key)[:200]


def write_manifests_and_audio(
    mix_iter,
    output_dir: Path,
    key_to_mp4: dict[str, str],
    num_mixtures: int,
    *,
    skip_test: bool,
):
    mix_dir = output_dir / "mix"
    s1_dir = output_dir / "s1"
    s2_dir = output_dir / "s2"
    cues_dir = output_dir / "cues"
    for d in (mix_dir, s1_dir, s2_dir, cues_dir):
        d.mkdir(parents=True, exist_ok=True)

    samples = []
    visual_index: dict[str, list[dict]] = {}
    count = 0

    for sample in mix_iter:
        print(f"sample: {sample}")
        if count >= num_mixtures:
            break

        mix_key = sample["key"]
        spk1 = sample["spk1"]
        spk2 = sample["spk2"]
        sr = int(sample["sample_rate"])
        wav_mix = sample["wav_mix"]
        wav_s1 = sample["wav_spk1"]
        wav_s2 = sample["wav_spk2"]

        stem = safe_filename(mix_key)
        mix_path = mix_dir / f"{stem}.wav"
        p1 = s1_dir / f"{stem}.wav"
        p2 = s2_dir / f"{stem}.wav"

        torchaudio.save(str(mix_path), wav_mix.cpu(), sr)
        torchaudio.save(str(p1), wav_s1.cpu(), sr)
        torchaudio.save(str(p2), wav_s2.cpu(), sr)

        samples.append({
            "key": mix_key,
            "spk": [spk1, spk2],
            "mix": {"default": [str(mix_path)]},
            "src": {
                spk1: [str(p1)],
                spk2: [str(p2)],
            },
        })

        paths = parse_mix_key_for_visual(mix_key, key_to_mp4)
        mp4_a = paths["spk1_mp4"]
        mp4_b = paths["spk2_mp4"]
        visual_index[f"{mix_key}::{spk1}"] = [{
            "utt_id": f"{Path(mp4_a).parent.name}_{Path(mp4_a).stem}",
            "path": mp4_a,
        }]
        visual_index[f"{mix_key}::{spk2}"] = [{
            "utt_id": f"{Path(mp4_b).parent.name}_{Path(mp4_b).stem}",
            "path": mp4_b,
        }]

        count += 1
        if count % max(1, num_mixtures // 10) == 0:
            logger.info(f"  wrote {count}/{num_mixtures} mixtures")

    jsonl_path = output_dir / "samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {jsonl_path} ({len(samples)} lines)")

    raw_list = output_dir / "raw.list"
    if raw_list.exists() or raw_list.is_symlink():
        raw_list.unlink()
    raw_list.symlink_to("samples.jsonl")

    visual_json = cues_dir / "visual.json"
    with open(visual_json, "w", encoding="utf-8") as f:
        json.dump(visual_index, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {visual_json} ({len(visual_index)} entries)")

    cues_yaml = output_dir / "cues.yaml"
    cues_conf = {
        "cues": {
            "visual": {
                "type": "raw_mp4",
                "guaranteed": True,
                "scope": "speaker",
                "policy": {
                    "type": "fixed",
                    "key": "mix_spk_id",
                    "resource": str(visual_json.resolve()),
                },
            }
        }
    }
    with open(cues_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cues_conf, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Wrote {cues_yaml}")

    if not skip_test and samples:
        _smoke_test(samples[0], visual_index)


def _smoke_test(first: dict, visual_index: dict):
    mix_path = first["mix"]["default"][0]
    w, sr = torchaudio.load(mix_path)
    logger.info(f"[smoke] mix ok: shape={tuple(w.shape)}, sr={sr}")
    k = first["key"]
    for spk in first["spk"]:
        vk = f"{k}::{spk}"
        assert vk in visual_index, vk
    logger.info("[smoke] visual keys ok")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Online mix data via wesep.dataset.processor (same as training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mp4_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_mixtures", type=int, default=20)
    p.add_argument("--speakers", type=str, nargs="+", default=None)
    p.add_argument("--num_speakers", type=int, default=8,
                   help="Random speakers count when --speakers not set")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--resample_rate", type=int, default=16000)
    p.add_argument("--chunk_len", type=int, default=48000,
                   help="Samples per chunk after resample (e.g. 3 s @ 16 kHz)")
    p.add_argument("--whole_utt", action="store_true",
                   help="Disable random_chunk (use full decoded utterance)")

    p.add_argument("--online_buffer_size", type=int, default=32,
                   help="sample_speaker_group_without_repeat shuffle buffer; first outputs "
                        "after this many single-speaker samples")

    p.add_argument("--reverb_prob", type=float, default=0.0)
    p.add_argument("--reverb_conf", type=str, default=None,
                   help="Optional YAML merged into reverb_conf (keys override)")
    p.add_argument("--snr_range", type=float, nargs=2, default=[-5.0, 10.0],
                   metavar=("LOW", "HIGH"))
    p.add_argument("--gain_range", type=float, nargs=2, default=[-12.0, 0.0],
                   metavar=("LOW", "HIGH"))

    p.add_argument("--noise_prob", type=float, default=0.0)
    p.add_argument("--noise_conf", type=str, default=None,
                   help="Optional YAML merged into noise_conf (keys override)")
    p.add_argument("--noise_lmdb_file", type=str, default=None)

    p.add_argument("--dataset_config", type=str, default=None,
                   help="Optional YAML merged into dataset_args (keys override)")
    p.add_argument("--timeline_conf", type=str, default=None,
                   help="Optional YAML merged into timeline_conf (keys override)")
    
    p.add_argument("--skip_test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    inventory = scan_mp4_dir(args.mp4_dir)
    if args.speakers:
        speaker_ids = args.speakers
        missing = [s for s in speaker_ids if s not in inventory]
        if missing:
            raise ValueError(f"Unknown speakers: {missing}")
    else:
        all_spk = sorted(inventory.keys())
        n = min(args.num_speakers, len(all_spk))
        speaker_ids = rng.sample(all_spk, n)

    if len(speaker_ids) < 2:
        raise ValueError("Need at least 2 speakers for 2-speaker mixture.")

    logger.info(f"Using {len(speaker_ids)} speakers")

    key_to_mp4: dict[str, str] = {}

    snr_conf = {
        "range": [args.snr_range[0], args.snr_range[1]],
        "gain": [args.gain_range[0], args.gain_range[1]],
    }

    dataset_args = build_dataset_args(
        resample_rate=args.resample_rate,
        chunk_len=args.chunk_len,
        whole_utt=args.whole_utt,
        online_buffer_size=args.online_buffer_size,
        timeline_conf=None,
        reverb_prob=args.reverb_prob,
        reverb_conf=None, # if reverb_prob > 0, default reverb_conf will be used
        snr_conf=snr_conf,
        noise_prob=args.noise_prob,
        noise_lmdb_file=args.noise_lmdb_file,
    )

    if args.dataset_config:
        with open(args.dataset_config, encoding="utf-8") as f:
            extra = yaml.safe_load(f)
        if extra:
            dataset_args.update(extra)
            logger.info(f"Merged dataset_config from {args.dataset_config}")

    single_iter = iter_single_spk_json(
        inventory, speaker_ids, rng, key_to_mp4,
    )
    mix_iter = run_online_mix_pipeline(single_iter, dataset_args, rng)

    out = Path(args.output_dir)
    write_manifests_and_audio(
        mix_iter,
        out,
        key_to_mp4,
        args.num_mixtures,
        skip_test=args.skip_test,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
