# Copyright (c) 2026 Duo Ma (maduo@cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0.

"""
Evaluate **SI-SNR only** between enhanced (estimated) and reference WAVs.

This mirrors the SI-SNR definition in ``wesep.utils.score.cal_SISNR`` used by
``wesep.bin.score`` / ``tools/score.sh``, but skips STOI/PESQ/DNSMOS/BSS metrics.

Typical layout after ``local/infer_test_export_wav.py`` (stage 103)::

    {export_dir}/audio/spk1.scp      # enhanced
    {export_dir}/ref_dset/single.wav.scp

Examples::

    # One corpus (depends only on numpy + soundfile)
    python examples/visual/voxceleb2mix/local/eval_si_snr_only.py \\
      --export_dir /path/to/stage103_test_export_300utts/lrs3

    # All corpora under an export root
    python .../eval_si_snr_only.py \\
      --export_parent /path/to/stage103_test_export_300utts \\
      --corpora lrs3,chinese_lips,merged

    # Explicit SCPs
    python .../eval_si_snr_only.py \\
      --ref_scp /path/to/single.wav.scp \\
      --est_scp /path/to/spk1.scp
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Iterable

import numpy as np
import soundfile as sf

def cal_SISNR(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    """Same as ``wesep.utils.score.cal_SISNR`` (scale-invariant SNR, dB)."""
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    assert len(est) == len(ref)
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)

    t = np.sum(est_zm * ref_zm) * ref_zm / (np.linalg.norm(ref_zm) ** 2 + eps)
    return float(
        20 * np.log10(eps + np.linalg.norm(t) / (np.linalg.norm(est_zm - t) + eps))
    )


def _mono_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _keys_in_order(scp_path: pathlib.Path) -> list[str]:
    keys: list[str] = []
    with scp_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            keys.append(line.split(maxsplit=1)[0])
    return keys


def _wav_path_dict(scp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    d: dict[str, pathlib.Path] = {}
    with scp_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, path_str = line.split(maxsplit=1)
            d[key] = pathlib.Path(path_str)
    return d


def _load_mono_float64(wav_path: pathlib.Path) -> np.ndarray:
    data, _sr = sf.read(str(wav_path), dtype="float64", always_2d=False)
    return _mono_1d(data)


def eval_si_snr_pair(
    ref_paths: dict[str, pathlib.Path],
    est_paths: dict[str, pathlib.Path],
    keys: Iterable[str],
) -> tuple[list[str], list[float]]:
    scores_k: list[str] = []
    scores_v: list[float] = []
    ref_k = set(ref_paths)
    est_k = set(est_paths)
    if ref_k != est_k:
        raise ValueError(
            "ref_scp and est_scp key mismatch: "
            f"only_in_est={len(est_k - ref_k)} only_in_ref={len(ref_k - est_k)}"
        )

    for key in keys:
        if key not in ref_paths:
            raise KeyError(f"Missing key in ref scp: {key}")
        ref_a = _load_mono_float64(ref_paths[key])
        est_a = _load_mono_float64(est_paths[key])
        n = min(ref_a.shape[0], est_a.shape[0])
        if n == 0:
            raise ValueError(f"Empty audio for key={key}")
        ref_a = ref_a[:n]
        est_a = est_a[:n]
        scores_k.append(key)
        scores_v.append(float(cal_SISNR(est_a, ref_a)))
    return scores_k, scores_v


def _run_one(
    ref_scp: pathlib.Path,
    est_scp: pathlib.Path,
    per_utt_tsv: pathlib.Path | None,
) -> tuple[float, float, float, float]:
    keys = _keys_in_order(ref_scp)
    ref_paths = _wav_path_dict(ref_scp)
    est_paths = _wav_path_dict(est_scp)
    ks, vs = eval_si_snr_pair(ref_paths, est_paths, keys)
    arr = np.array(vs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    vmin = float(arr.min())
    vmax = float(arr.max())

    if per_utt_tsv is not None:
        per_utt_tsv.parent.mkdir(parents=True, exist_ok=True)
        with per_utt_tsv.open("w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp, delimiter="\t")
            w.writerow(["key", "si_snr"])
            for k, v in zip(ks, vs):
                w.writerow([k, f"{v:.6f}"])

    return mean, std, vmin, vmax


def main() -> None:
    p = argparse.ArgumentParser(description="SI-SNR-only evaluation for TSE exports.")
    p.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="Corpus directory containing audio/spk1.scp and ref_dset/single.wav.scp",
    )
    p.add_argument(
        "--export_parent",
        type=str,
        default=None,
        help="Parent of per-corpus folders (used with --corpora).",
    )
    p.add_argument(
        "--corpora",
        type=str,
        default="lrs3,chinese_lips,merged",
        help="Comma-separated corpus names under --export_parent.",
    )
    p.add_argument("--ref_scp", type=str, default=None)
    p.add_argument("--est_scp", type=str, default=None)
    p.add_argument(
        "--per_utt_tsv",
        type=str,
        default=None,
        help="If set, write per-utterance SI-SNR (TSV). With multiple corpora, "
        "suffix _<corpus>.tsv is appended.",
    )
    args = p.parse_args()

    jobs: list[tuple[str, pathlib.Path, pathlib.Path]] = []

    if args.export_dir:
        root = pathlib.Path(args.export_dir).resolve()
        jobs.append(
            (
                root.name,
                root / "ref_dset" / "single.wav.scp",
                root / "audio" / "spk1.scp",
            )
        )
    if args.export_parent:
        parent = pathlib.Path(args.export_parent).resolve()
        for name in [x.strip() for x in args.corpora.split(",") if x.strip()]:
            root = parent / name
            jobs.append(
                (
                    name,
                    root / "ref_dset" / "single.wav.scp",
                    root / "audio" / "spk1.scp",
                )
            )
    if args.ref_scp and args.est_scp:
        jobs.append(
            (
                "custom",
                pathlib.Path(args.ref_scp).resolve(),
                pathlib.Path(args.est_scp).resolve(),
            )
        )

    if not jobs:
        p.error("Specify --export_dir and/or --export_parent, or both --ref_scp and --est_scp")

    # De-duplicate if user passed overlapping modes
    seen: set[tuple[str, str]] = set()
    unique_jobs: list[tuple[str, pathlib.Path, pathlib.Path]] = []
    for label, ref_p, est_p in jobs:
        sig = (str(ref_p), str(est_p))
        if sig in seen:
            continue
        seen.add(sig)
        unique_jobs.append((label, ref_p, est_p))

    grand_sum = 0.0
    grand_n = 0

    for label, ref_p, est_p in unique_jobs:
        if not ref_p.is_file():
            print(f"[skip] {label}: missing ref_scp {ref_p}", file=sys.stderr)
            continue
        if not est_p.is_file():
            print(f"[skip] {label}: missing est_scp {est_p}", file=sys.stderr)
            continue

        per_tsv: pathlib.Path | None = None
        if args.per_utt_tsv:
            base = pathlib.Path(args.per_utt_tsv)
            if len(unique_jobs) > 1:
                per_tsv = base.with_name(base.stem + f"_{label}" + base.suffix)
            else:
                per_tsv = base

        mean, std, vmin, vmax = _run_one(ref_p, est_p, per_tsv)
        n = sum(1 for _ in _keys_in_order(ref_p))
        grand_sum += mean * n
        grand_n += n

        print(
            f"[{label}] utterances={n}  SI-SNR_mean={mean:.4f} dB  "
            f"std={std:.4f}  min={vmin:.4f}  max={vmax:.4f}"
        )

    if grand_n > 1 and len(unique_jobs) > 1:
        print(f"[ALL weighted by utt count] SI-SNR_mean={grand_sum / grand_n:.4f} dB")


if __name__ == "__main__":
    main()
