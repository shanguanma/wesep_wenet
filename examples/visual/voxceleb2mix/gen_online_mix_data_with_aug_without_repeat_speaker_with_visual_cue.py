#!/usr/bin/env python3
"""
Generate mixture data with the **same online-mix + visual cue pipeline** as training:

- ``dataset.py`` order: ``build_mix_layer`` then ``build_cue_layer`` (visual).

Pipeline (see ``wesep.dataset.online_dataset``):

  ``parse_raw_single_spk`` → ``resample`` → ``random_chunk`` →
  ``sample_speaker_group_without_repeat`` → ``apply_timeline`` → ``add_reverb`` →
  ``snr_mixer`` → (optional ``add_noise``) →
  **``processor_visual.sample_fixed_visual_cue``** (same as ``cues.yaml`` visual / fixed / mix_spk_id)

Then write wav + ``samples.jsonl`` + ``cues/visual.json`` + ``cues.yaml``.
Video tensors are attached in-memory for training parity; they are **not** written to JSONL.

Use ``--skip_visual_decode`` to only write manifests (faster, no TorchCodec pass).

Examples

python gen_online_mix_data_with_aug_without_repeat_speaker_with_visual_cue.py \
    --mp4_dir /F00120240032/voxceleb2_zk_mixture/mp4/train \
    --output_dir ./online_mix2_with_aug_without_repeat_speaker_with_visual_cue \
    --num_mixtures 5 \
    --online_buffer_size 4 \
    --chunk_len 24000 \
    --reverb_prob 0.5 \
    --snr_range -5 10 \
    --noise_prob 0.5 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --skip_test \
    --skip_visual_decode
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import tempfile
from pathlib import Path

import torch
import torchaudio
import yaml

from wesep.dataset import processor_visual
from wesep.dataset.online_dataset import (
    apply_visual_cue_stream,
    build_dataset_args,
    build_visual_resource_slice,
    default_timeline_conf,
    iter_single_spk_json,
    mp4_paths_for_mix_key,
    parse_mix_key_ids,
    resolve_speaker_pool,
    run_online_mix_pipeline,
    scan_mp4_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_filename(key: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", key)[:200]


def write_manifests_and_audio(
    mix_iter,
    output_dir: Path,
    key_to_mp4: dict[str, str],
    num_mixtures: int,
    *,
    skip_test: bool,
    log_visual_shapes: bool = True,
):
    mix_dir = output_dir / "mix"
    cues_dir = output_dir / "cues"
    mix_dir.mkdir(parents=True, exist_ok=True)
    cues_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    visual_index: dict[str, list[dict]] = {}
    count = 0

    for sample in mix_iter:
        if count >= num_mixtures:
            break

        if (
            log_visual_shapes
            and any(k.startswith("visual_") for k in sample)
        ):
            for vk in sorted(k for k in sample if k.startswith("visual_")):
                t = sample[vk]
                logger.info(
                    f"[visual cue] {vk} shape={tuple(t.shape)} dtype={t.dtype}"
                )

        mix_key = sample["key"]
        num_speaker = int(sample["num_speaker"])
        if num_speaker < 1:
            raise ValueError(f"Invalid num_speaker: {num_speaker}")

        kids = parse_mix_key_ids(mix_key)
        if len(kids) != num_speaker:
            raise ValueError(
                f"num_speaker={num_speaker} but mix key has {len(kids)} "
                f"clip tokens: {mix_key!r}"
            )

        sr = int(sample["sample_rate"])
        wav_mix = sample["wav_mix"]

        stem = safe_filename(mix_key)
        mix_path = mix_dir / f"{stem}.wav"
        torchaudio.save(str(mix_path), wav_mix.cpu(), sr)

        spk_ids: list[str] = []
        src_map: dict[str, list[str]] = {}
        mp4_paths = mp4_paths_for_mix_key(mix_key, key_to_mp4)

        for i in range(1, num_speaker + 1):
            spk_i = sample[f"spk{i}"]
            wav_i = sample[f"wav_spk{i}"]
            spk_ids.append(spk_i)

            sdir = output_dir / f"s{i}"
            sdir.mkdir(parents=True, exist_ok=True)
            pi = sdir / f"{stem}.wav"
            torchaudio.save(str(pi), wav_i.cpu(), sr)
            src_map[spk_i] = [str(pi)]

            mp4_i = mp4_paths[i - 1]
            visual_index[f"{mix_key}::{spk_i}"] = [{
                "utt_id": f"{Path(mp4_i).parent.name}_{Path(mp4_i).stem}",
                "path": mp4_i,
            }]

        samples.append({
            "key": mix_key,
            "spk": spk_ids,
            "mix": {"default": [str(mix_path)]},
            "src": src_map,
        })

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Online mix data via wesep.dataset.processor (same as training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mp4_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_mixtures", type=int, default=20)
    p.add_argument("--speakers", type=str, nargs="+", default=None)
    p.add_argument(
        "--num_speakers",
        type=int,
        default=8,
        help="Pool size when --speakers unset; <=0 or >=#speakers uses all scanned speakers.",
    )
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

    p.add_argument(
        "--skip_visual_decode",
        action="store_true",
        help="Skip TorchCodec visual pass (only write wav + manifest; faster).",
    )
    p.add_argument("--skip_test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    inventory = scan_mp4_dir(args.mp4_dir)
    speaker_ids = resolve_speaker_pool(
        inventory,
        rng,
        speakers=args.speakers,
        num_speakers=args.num_speakers,
    )

    if len(speaker_ids) < 2:
        raise ValueError("Need at least 2 speakers for mixture.")

    logger.info(f"Using {len(speaker_ids)} speakers")

    key_to_mp4: dict[str, str] = {}

    snr_conf = {
        "range": [args.snr_range[0], args.snr_range[1]],
        "gain": [args.gain_range[0], args.gain_range[1]],
    }
    timeline_conf = default_timeline_conf()

    dataset_args = build_dataset_args(
        resample_rate=args.resample_rate,
        chunk_len=args.chunk_len,
        whole_utt=args.whole_utt,
        online_buffer_size=args.online_buffer_size,
        timeline_conf=timeline_conf,
        reverb_prob=args.reverb_prob,
        reverb_conf=None,
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

    if args.timeline_conf:
        with open(args.timeline_conf, encoding="utf-8") as f:
            textra = yaml.safe_load(f)
        if textra:
            dataset_args["timeline"] = textra
            logger.info(f"Merged timeline_conf from {args.timeline_conf}")

    max_mix = int(dataset_args.get("num_speakers", {}).get("max_speakers", 4))
    if len(speaker_ids) < max_mix:
        logger.warning(
            "Speaker pool has %d IDs but max_speakers=%d in config: each "
            "mixture is capped by min(requested, distinct IDs in buffer). "
            "Prefer --num_speakers >= %d (or --speakers with at least %d IDs).",
            len(speaker_ids),
            max_mix,
            max_mix,
            max_mix,
        )

    single_iter = iter_single_spk_json(
        inventory, speaker_ids, rng, key_to_mp4,
    )
    mix_iter = run_online_mix_pipeline(single_iter, dataset_args, rng)
    mix_iter = apply_visual_cue_stream(
        mix_iter,
        key_to_mp4,
        skip_visual_decode=args.skip_visual_decode,
    )

    out = Path(args.output_dir)
    write_manifests_and_audio(
        mix_iter,
        out,
        key_to_mp4,
        args.num_mixtures,
        skip_test=args.skip_test,
        log_visual_shapes=not args.skip_visual_decode,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
