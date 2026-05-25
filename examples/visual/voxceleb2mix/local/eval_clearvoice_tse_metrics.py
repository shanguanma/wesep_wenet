# Copyright (c) 2026 Duo Ma (maduo@cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0.

"""
Objective evaluation aligned with **ClearerVoice-Studio** TSE ``evaluate()`` —
see ``train/target_speaker_extraction/solver.py`` (SI-SNRi, SDRi, PESQi, STOIi).

References:
  https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction/solver.py
  https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction/losses/metrics.py

Requires **per-utterance mixture WAVs** (same keys as enhanced/ref). These are written by
``local/infer_test_export_wav.py`` as ``{corpus}/mix/mix.scp`` + ``mix/wavs/*.wav``.

Typical usage (repo root or any cwd)::

    python examples/visual/voxceleb2mix/local/eval_clearvoice_tse_metrics.py \\
      --export_parent /path/to/stage103_test_export_300utts \\
      --corpora lrs3,chinese_lips
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import soundfile as sf


def cal_SISNR_clearvoice(source: np.ndarray, estimate_source: np.ndarray, eps: float = 1e-6) -> float:
    """Port of ClearerVoice ``losses.metrics.cal_SISNR`` (numpy, mono).

    Args:
        source: clean target ``s`` (same naming as upstream).
        estimate_source: network output ``s'`` or mixture for the second term.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    estimate_source = np.asarray(estimate_source, dtype=np.float64).reshape(-1)
    assert source.shape == estimate_source.shape
    source = source - np.mean(source)
    estimate_source = estimate_source - np.mean(estimate_source)
    ref_energy = np.sum(source**2) + eps
    proj = np.sum(source * estimate_source) * source / ref_energy
    noise = estimate_source - proj
    ratio = np.sum(proj**2) / (np.sum(noise**2) + eps)
    return float(10 * np.log10(ratio + eps))


def SDR_clearvoice(gt: np.ndarray, est: np.ndarray) -> float:
    """Port of ClearerVoice ``losses.metrics.SDR`` (single-source)."""
    from mir_eval.separation import bss_eval_sources

    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    sdr, _, _, _ = bss_eval_sources(gt[np.newaxis, :], est[np.newaxis, :])
    return float(np.asarray(sdr).reshape(-1)[0])


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


@dataclass
class UttScores:
    sisnr_est: float
    sisnr_mix: float
    sisnri: float
    sdri: float | None
    pesqi: float | None
    stoii: float | None


def _pesq_wb_safe(sr: int, ref: np.ndarray, deg: np.ndarray) -> float | None:
    try:
        from pesq import PesqError, pesq
    except ImportError:
        return None
    try:
        score = pesq(sr, ref.astype(np.float64), deg.astype(np.float64), mode="wb", on_error=PesqError.RETURN_VALUES)
    except Exception:
        return None
    if score == PesqError.NO_UTTERANCES_DETECTED:
        return None
    return float(score)


def _stoi_safe(sr: int, ref: np.ndarray, deg: np.ndarray) -> float | None:
    try:
        from pystoi.stoi import stoi
    except ImportError:
        return None
    try:
        return float(stoi(ref, deg, fs_sig=sr, extended=False))
    except Exception:
        return None


def eval_one_corpus(
    *,
    ref_paths: dict[str, pathlib.Path],
    est_paths: dict[str, pathlib.Path],
    mix_paths: dict[str, pathlib.Path],
    keys: Iterable[str],
    sample_rate: int,
    compute_sdri: bool,
    compute_pesq: bool,
    compute_stoi: bool,
) -> tuple[list[str], list[UttScores]]:
    ref_k = set(ref_paths)
    est_k = set(est_paths)
    mix_k = set(mix_paths)
    if ref_k != est_k or ref_k != mix_k:
        raise ValueError(
            "scp key mismatch among ref/est/mix: "
            f"|ref|={len(ref_k)} |est|={len(est_k)} |mix|={len(mix_k)}"
        )

    out_k: list[str] = []
    out_s: list[UttScores] = []

    for key in keys:
        ref_a = _load_mono_float64(ref_paths[key])
        est_a = _load_mono_float64(est_paths[key])
        mix_a = _load_mono_float64(mix_paths[key])
        n = min(ref_a.shape[0], est_a.shape[0], mix_a.shape[0])
        if n == 0:
            raise ValueError(f"Empty audio for key={key}")
        ref_a = ref_a[:n]
        est_a = est_a[:n]
        mix_a = mix_a[:n]

        # --- SI-SNR / SI-SNRi (ClearerVoice solver.evaluate, lines ~223–224)
        sisnr_est = cal_SISNR_clearvoice(ref_a, est_a)
        sisnr_mix = cal_SISNR_clearvoice(ref_a, mix_a)
        sisnri = sisnr_est - sisnr_mix

        sdri: float | None = None
        if compute_sdri:
            sdri = float(SDR_clearvoice(ref_a, est_a) - SDR_clearvoice(ref_a, mix_a))

        pesqi: float | None = None
        if compute_pesq and sample_rate == 16000:
            est_peak = np.max(np.abs(est_a))
            scale = float(est_peak + 1e-12)
            est_n = est_a / scale
            pr_e = _pesq_wb_safe(sample_rate, ref_a, est_n)
            pr_m = _pesq_wb_safe(sample_rate, ref_a, mix_a)
            if pr_e is not None and pr_m is not None:
                pesqi = pr_e - pr_m

        stoii: float | None = None
        if compute_stoi:
            sr_e = _stoi_safe(sample_rate, ref_a, est_a)
            sr_m = _stoi_safe(sample_rate, ref_a, mix_a)
            if sr_e is not None and sr_m is not None:
                stoii = sr_e - sr_m

        out_k.append(key)
        out_s.append(
            UttScores(
                sisnr_est=sisnr_est,
                sisnr_mix=sisnr_mix,
                sisnri=sisnri,
                sdri=sdri,
                pesqi=pesqi,
                stoii=stoii,
            )
        )

    return out_k, out_s


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def _run_export_dir(
    corpus_root: pathlib.Path,
    sample_rate: int,
    compute_sdri: bool,
    compute_pesq: bool,
    compute_stoi: bool,
    per_utt_tsv: pathlib.Path | None,
    label: str,
) -> tuple[dict[str, tuple[float, float]], list[float]]:
    """Returns ``(metric_summary, per_utt_si_snri_values)``."""
    ref_scp = corpus_root / "ref_dset" / "single.wav.scp"
    est_scp = corpus_root / "audio" / "spk1.scp"
    mix_scp = corpus_root / "mix" / "mix.scp"
    if not mix_scp.is_file():
        print(
            f"[skip] {label}: missing mixture scp {mix_scp} "
            "(re-run stage 103 export after updating infer_test_export_wav.py).",
            file=sys.stderr,
        )
        return {}, []

    keys = _keys_in_order(ref_scp)
    ref_paths = _wav_path_dict(ref_scp)
    est_paths = _wav_path_dict(est_scp)
    mix_paths = _wav_path_dict(mix_scp)

    ks, scores = eval_one_corpus(
        ref_paths=ref_paths,
        est_paths=est_paths,
        mix_paths=mix_paths,
        keys=keys,
        sample_rate=sample_rate,
        compute_sdri=compute_sdri,
        compute_pesq=compute_pesq,
        compute_stoi=compute_stoi,
    )

    n = len(scores)
    sisnr_est_m, sisnr_est_s = _mean_std([s.sisnr_est for s in scores])
    sisnr_mix_m, sisnr_mix_s = _mean_std([s.sisnr_mix for s in scores])
    sisnri_m, sisnri_s = _mean_std([s.sisnri for s in scores])

    report: dict[str, tuple[float, float]] = {
        "SI_SNR_est": (sisnr_est_m, sisnr_est_s),
        "SI_SNR_mix": (sisnr_mix_m, sisnr_mix_s),
        "SI_SNRi": (sisnri_m, sisnri_s),
    }

    sd_vals = [s.sdri for s in scores if s.sdri is not None]
    if sd_vals:
        m, s = _mean_std(sd_vals)
        report["SDRi"] = (m, s)

    pq_vals = [s.pesqi for s in scores if s.pesqi is not None]
    if pq_vals:
        m, s = _mean_std(pq_vals)
        report["PESQi"] = (m, s)

    st_vals = [s.stoii for s in scores if s.stoii is not None]
    if st_vals:
        m, s = _mean_std(st_vals)
        report["STOIi"] = (m, s)

    line = (
        f"[{label}] utterances={n}  "
        f"SI-SNR(est)={sisnr_est_m:.4f}±{sisnr_est_s:.4f} dB  "
        f"SI-SNR(mix)={sisnr_mix_m:.4f}±{sisnr_mix_s:.4f} dB  "
        f"SI-SNRi={sisnri_m:.4f}±{sisnri_s:.4f} dB"
    )
    if "SDRi" in report:
        m, s = report["SDRi"]
        line += f"  SDRi={m:.4f}±{s:.4f} dB"
    if "PESQi" in report:
        m, s = report["PESQi"]
        line += f"  PESQi={m:.4f}±{s:.4f}"
    if "STOIi" in report:
        m, s = report["STOIi"]
        line += f"  STOIi={m:.4f}±{s:.4f}"
    print(line)

    if per_utt_tsv is not None:
        per_utt_tsv.parent.mkdir(parents=True, exist_ok=True)
        with per_utt_tsv.open("w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp, delimiter="\t")
            header = ["key", "si_snr_est", "si_snr_mix", "si_snri", "sdri", "pesqi", "stoii"]
            w.writerow(header)
            for k, sc in zip(ks, scores):
                w.writerow(
                    [
                        k,
                        f"{sc.sisnr_est:.6f}",
                        f"{sc.sisnr_mix:.6f}",
                        f"{sc.sisnri:.6f}",
                        "" if sc.sdri is None else f"{sc.sdri:.6f}",
                        "" if sc.pesqi is None else f"{sc.pesqi:.6f}",
                        "" if sc.stoii is None else f"{sc.stoii:.6f}",
                    ]
                )

    return report, [s.sisnri for s in scores]


def main() -> None:
    p = argparse.ArgumentParser(
        description="ClearerVoice-style TSE metrics (need mixture WAVs from stage 103 export).",
    )
    p.add_argument("--export_dir", type=str, default=None)
    p.add_argument("--export_parent", type=str, default=None)
    p.add_argument(
        "--corpora",
        type=str,
        default="lrs3,chinese_lips,merged",
        help="Comma-separated corpus names under --export_parent.",
    )
    p.add_argument("--ref_scp", type=str, default=None)
    p.add_argument("--est_scp", type=str, default=None)
    p.add_argument("--mix_scp", type=str, default=None)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--no_sdri", action="store_true", help="Skip SDRi (mir_eval).")
    p.add_argument("--no_pesq", action="store_true")
    p.add_argument("--no_stoi", action="store_true")
    p.add_argument("--per_utt_tsv", type=str, default=None)
    args = p.parse_args()

    compute_sdri = not args.no_sdri
    compute_pesq = not args.no_pesq
    compute_stoi = not args.no_stoi

    if compute_sdri:
        try:
            import mir_eval.separation  # noqa: F401
        except ImportError:
            print(
                "[warn] mir_eval not installed — skipping SDRi "
                "(install with: pip install mir_eval).",
                file=sys.stderr,
            )
            compute_sdri = False

    if args.ref_scp and args.est_scp and args.mix_scp:
        ref_p = pathlib.Path(args.ref_scp).resolve()
        est_p = pathlib.Path(args.est_scp).resolve()
        mix_p = pathlib.Path(args.mix_scp).resolve()
        keys = _keys_in_order(ref_p)
        ref_paths = _wav_path_dict(ref_p)
        est_paths = _wav_path_dict(est_p)
        mix_paths = _wav_path_dict(mix_p)
        _, scores = eval_one_corpus(
            ref_paths=ref_paths,
            est_paths=est_paths,
            mix_paths=mix_paths,
            keys=keys,
            sample_rate=args.sample_rate,
            compute_sdri=compute_sdri,
            compute_pesq=compute_pesq,
            compute_stoi=compute_stoi,
        )
        sisnr_est_m, _ = _mean_std([s.sisnr_est for s in scores])
        sisnr_mix_m, _ = _mean_std([s.sisnr_mix for s in scores])
        sisnri_m, sisnri_s = _mean_std([s.sisnri for s in scores])
        line = (
            f"[custom] utterances={len(scores)}  "
            f"SI-SNR(est)={sisnr_est_m:.4f} dB  "
            f"SI-SNR(mix)={sisnr_mix_m:.4f} dB  "
            f"SI-SNRi={sisnri_m:.4f}±{sisnri_s:.4f} dB"
        )
        sd_vals = [s.sdri for s in scores if s.sdri is not None]
        if sd_vals:
            m, s = _mean_std(sd_vals)
            line += f"  SDRi={m:.4f}±{s:.4f} dB"
        pq_vals = [s.pesqi for s in scores if s.pesqi is not None]
        if pq_vals:
            m, s = _mean_std(pq_vals)
            line += f"  PESQi={m:.4f}±{s:.4f}"
        st_vals = [s.stoii for s in scores if s.stoii is not None]
        if st_vals:
            m, s = _mean_std(st_vals)
            line += f"  STOIi={m:.4f}±{s:.4f}"
        print(line)
        return

    jobs: list[tuple[str, pathlib.Path]] = []
    if args.export_dir:
        root = pathlib.Path(args.export_dir).resolve()
        jobs.append((root.name, root))
    if args.export_parent:
        parent = pathlib.Path(args.export_parent).resolve()
        for name in [x.strip() for x in args.corpora.split(",") if x.strip()]:
            jobs.append((name, parent / name))

    if not jobs:
        p.error("Need --export_dir or --export_parent (or --ref_scp/--est_scp/--mix_scp).")

    seen_paths: set[str] = set()
    ordered_unique: list[tuple[str, pathlib.Path]] = []
    for label, root in jobs:
        sig = str(root.resolve())
        if sig in seen_paths:
            continue
        seen_paths.add(sig)
        ordered_unique.append((label, root))

    grand_sisnri: list[float] = []
    for label, root in ordered_unique:
        per_tsv = None
        if args.per_utt_tsv:
            base = pathlib.Path(args.per_utt_tsv)
            per_tsv = (
                base.with_name(base.stem + f"_{label}" + base.suffix)
                if len(ordered_unique) > 1
                else base
            )

        rep, sisnri_utts = _run_export_dir(
            root,
            args.sample_rate,
            compute_sdri,
            compute_pesq,
            compute_stoi,
            per_tsv,
            label,
        )
        if sisnri_utts:
            grand_sisnri.extend(sisnri_utts)

    if len(ordered_unique) > 1 and len(grand_sisnri) > 0:
        gmean = float(np.mean(np.array(grand_sisnri, dtype=np.float64)))
        print(f"[ALL utterance-weighted] SI-SNRi_mean={gmean:.4f} dB")


if __name__ == "__main__":
    main()
