#!/usr/bin/env python3
"""Quick standalone eval: compare SI-SDR with real vs muted visual cues.

Loads checkpoint directly, creates a small val set, runs two forward passes
(normal cues vs zeroed cues). No Trainer, no resume overhead.

Usage:
    cd /maduo/codebase/wesep_wenet/examples/visual/voxceleb2mix
    python quick_ablation_eval.py \
        --ckpt /maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage111_av_align/checkpoint-6000 \
        --num_batches 50
"""

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from wesep.models import get_model
from wesep.dataset.online_multi_dataset import (
    OnlineMixIterableDataset,
    build_train_val_merged_audio_visual_inventories,
    ensure_online_pipeline_defaults,
    default_timeline_conf,
)
from wesep.dataset.collate import tse_collate_fn, build_collect_keys_online, BASE_COLLECT_KEYS
from torch.utils.data import DataLoader
from functools import partial
from safetensors.torch import load_file


def si_sdr_db(est, ref):
    """Per-sample SI-SDR in dB. est, ref: (B, 1, T) or (B, T)."""
    if est.dim() == 3:
        est = est[:, 0, :]
    if ref.dim() == 3:
        ref = ref[:, 0, :]
    eps = 1e-8
    est_z = est - est.mean(dim=-1, keepdim=True)
    ref_z = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est_z * ref_z).sum(dim=-1) / (ref_z.pow(2).sum(dim=-1) + eps)
    proj = ref_z * alpha.unsqueeze(-1)
    res = est_z - proj
    return 10 * torch.log10(proj.pow(2).sum(-1) / (res.pow(2).sum(-1) + eps) + eps)


def _build_collect_keys(dataset_args):
    return build_collect_keys_online(
        dataset_args,
        BASE_COLLECT_KEYS,
        tse_model_name="TSE_BSRNN_VISUAL",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage111_av_align/checkpoint-6000",
    )
    parser.add_argument(
        "--model_config",
        default="confs/tse_bsrnn_visual_model_v2.yaml",
    )
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--visual_max_frames", type=int, default=75)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- load model config ----
    with open(args.model_config) as f:
        configs = yaml.safe_load(f)
    model_cls = get_model(configs["model"]["tse_model"])
    model = model_cls(configs["model_args"]["tse_model"])
    print(f"Model: {configs['model']['tse_model']}, "
          f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ---- load checkpoint weights ----
    ckpt_path = os.path.join(args.ckpt, "model.safetensors")
    if os.path.isfile(ckpt_path):
        state_dict = load_file(ckpt_path, device="cpu")
    else:
        pt_path = os.path.join(args.ckpt, "pytorch_model.bin")
        state_dict = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"Loaded weights from {args.ckpt}")

    # ---- build val dataset ----
    dataset_args = ensure_online_pipeline_defaults({
        "resample_rate": 16000,
        "chunk_len": 48000,
        "whole_utt": False,
        "online_buffer_size": 8,
        "num_speakers": {
            "distribution": [0.0, 1.0, 0.0],
            "max_speakers": 2,
        },
        "timeline": default_timeline_conf(),
        "reverb_prob": 0.0,
        "reverb_conf": None,
        "snr_conf": {"range": [-5.0, 10.0], "gain": [-12.0, 0.0]},
        "noise_prob": 0.0,
        "noise_lmdb_file": "/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb",
        "online_mix_deterministic": True,
        "online_av_align": True,
        "visual_max_frames": args.visual_max_frames,
        "visual_spatial_size": (224, 224),
        "visual_decode_cuda": True,
        "cues": {"visual": {"use": True, "required": True}},
    })

    mp4_dir = "/F00120240032/voxceleb2_zk_mixture/mp4/train"
    lrs3_root = "/F00120240032/lrs3/trainval"

    _train_inv, val_inventory = build_train_val_merged_audio_visual_inventories(
        train_fraction=0.8,
        split_seed=42,
        voxceleb2_mp4_dir=mp4_dir,
        use_voxceleb2=True,
        lrs3_root=lrs3_root,
        use_lrs3=True,
        use_chinese_lips=False,
    )
    del _train_inv
    val_speaker_ids = set(val_inventory.keys())

    val_dataset = OnlineMixIterableDataset(
        val_inventory,
        val_speaker_ids,
        dataset_args,
        seed=42 + 7,
        with_visual_cue=True,
        skip_visual_decode=False,
    )

    collect_keys = _build_collect_keys(dataset_args)
    collate_fn = partial(
        tse_collate_fn,
        collect_keys=collect_keys,
        visual_max_frames=args.visual_max_frames,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ---- run eval: normal vs muted ----
    sisdr_normal_all = []
    sisdr_muted_all = []
    sisdr_mix_all = []

    print(f"\nRunning {args.num_batches} batches (bs={args.batch_size}) ...")
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_batches:
                break

            wav_mix = batch["wav_mix"].float().to(device)
            wav_target = batch["wav_target"].float().to(device)
            visual_aux = batch["visual_aux"].float().to(device)

            cues_normal = [visual_aux]

            cues_muted = [torch.zeros_like(visual_aux)]

            # forward with normal cues
            out_normal = model(wav_mix, cues_normal)
            if out_normal.dim() == 2:
                out_normal = out_normal.unsqueeze(1)

            # forward with muted cues
            out_muted = model(wav_mix, cues_muted)
            if out_muted.dim() == 2:
                out_muted = out_muted.unsqueeze(1)

            # SI-SDR
            sisdr_n = si_sdr_db(out_normal.float(), wav_target.float())
            sisdr_m = si_sdr_db(out_muted.float(), wav_target.float())
            sisdr_mix = si_sdr_db(wav_mix.float(), wav_target.float())

            sisdr_normal_all.append(sisdr_n.cpu())
            sisdr_muted_all.append(sisdr_m.cpu())
            sisdr_mix_all.append(sisdr_mix.cpu())

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                print(f"  batch {i + 1}/{args.num_batches} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # ---- aggregate ----
    sisdr_normal = torch.cat(sisdr_normal_all)
    sisdr_muted = torch.cat(sisdr_muted_all)
    sisdr_mix = torch.cat(sisdr_mix_all)

    imp_normal = sisdr_normal - sisdr_mix
    imp_muted = sisdr_muted - sisdr_mix

    n = len(sisdr_normal)
    print(f"\n{'='*60}")
    print(f"  Visual Cue Ablation Results  ({n} samples)")
    print(f"{'='*60}")
    print(f"  SI-SDR (mix→target):           {sisdr_mix.mean():.2f} dB")
    print(f"  SI-SDR (normal cues→target):   {sisdr_normal.mean():.2f} dB")
    print(f"  SI-SDR (muted cues→target):    {sisdr_muted.mean():.2f} dB")
    print(f"{'─'*60}")
    print(f"  SI-SDRi (normal cues):         {imp_normal.mean():.2f} dB")
    print(f"  SI-SDRi (muted cues):          {imp_muted.mean():.2f} dB")
    print(f"  Delta (normal - muted):        {(imp_normal - imp_muted).mean():.2f} dB")
    print(f"{'='*60}")

    if imp_normal.mean() > imp_muted.mean() + 0.5:
        print("  >> Visual cues ARE helping (+{:.2f} dB over muted)".format(
            (imp_normal.mean() - imp_muted.mean()).item()))
    elif abs(imp_normal.mean() - imp_muted.mean()) < 0.5:
        print("  >> Visual cues have MINIMAL effect (delta < 0.5 dB)")
    else:
        print("  >> Visual cues may be HURTING ({:.2f} dB vs muted)".format(
            (imp_normal.mean() - imp_muted.mean()).item()))


if __name__ == "__main__":
    main()
