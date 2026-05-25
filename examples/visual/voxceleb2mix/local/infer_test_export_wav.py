# Copyright (c) 2026 Duo Ma (maduo@cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0.

"""
Export **enhanced** (estimated) and **reference** mono WAVs for official test
inventories (LRS3 test / Chinese_lips test / merged), plus Kaldi-style SCPs so
``tools/score.sh`` (stage 6) can run unchanged.

Writes per corpus under ``{export_root}/{target}/``::

    audio/*.wav
    audio/spk1.scp      # enhanced — consumed as ``--exp_dir``/audio/spk1.scp
    ref_dset/single.wav.scp
    ref_dset/wavs/*.wav # references aligned key-for-key with spk1.scp
    mix/wavs/*.wav      # mixture input (same keys) — for SI-SNRi / ClearerVoice-style eval
    mix/mix.scp

Then invoke (from repo root)::

    ./tools/score.sh --dset {export_root}/{target}/ref_dset \\
      --exp_dir {export_root}/{target} --fs 16k ...
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

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
_LOCAL_DIR = str(_THIS.parent)
for _p in (_LOCAL_DIR, str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import HfArgumentParser  # noqa: E402

from eval_test_corpora import (  # noqa: E402
    EvalTestCorporaArguments,
    _build_dataloader_for_inventory,
    _inventory_for_target,
    _parse_eval_targets,
    _resolve_checkpoint,
)
from wesep.bin.train_with_transformers_multi_data_online import (  # noqa: E402
    TSEOnlineDataArguments,
    TSETrainingArguments,
    apply_tse_model_yaml_cli_overrides,
)
from wesep.dataset.online_multi_dataset import (  # noqa: E402
    ensure_online_pipeline_defaults,
    build_dataset_args_from_tse_online_data_args,
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
logger = logging.getLogger("infer_test_export_wav")


def _mono_1d(x: np.ndarray) -> np.ndarray:
    """Flatten one utterance to shape ``[T]`` (handles ``[T]``, ``[1, T]``, ``[1,1,T]``)."""
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    return a


@dataclass
class InferTestExportExtraArguments:
    export_root: str | None = field(
        default=None,
        metadata={
            "help": "Parent directory for per-corpus outputs. "
            "Default: <output_dir>/stage103_test_export",
        },
    )


def _sample_rate_hz(data_args: TSEOnlineDataArguments) -> int:
    dataset_args = ensure_online_pipeline_defaults(
        build_dataset_args_from_tse_online_data_args(data_args)
    )
    return int(dataset_args.get("resample_rate", 16000))


def _run_export_pass(
    *,
    model: torch.nn.Module,
    val_dl: DataLoader,
    executor: Executor,
    device: torch.device,
    n_target_mixtures: int,
    batch_size: int,
    bf16: bool,
    sample_rate: int,
    audio_dir: pathlib.Path,
    ref_wav_dir: pathlib.Path,
    mix_wav_dir: pathlib.Path,
    key_prefix: str,
    print_first: bool,
) -> int:
    """Save enhanced + reference + mixture WAVs; return number of utterances written."""
    n_dl_steps = max(1, (n_target_mixtures + batch_size - 1) // batch_size)
    logger.info(
        "Export pass: up to %d utterances (~%d steps × bs=%d).",
        n_target_mixtures,
        n_dl_steps,
        batch_size,
    )

    audio_dir.mkdir(parents=True, exist_ok=True)
    ref_wav_dir.mkdir(parents=True, exist_ok=True)
    mix_wav_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": bf16}

    spk1_lines: list[str] = []
    single_lines: list[str] = []
    mix_lines: list[str] = []

    step = 0
    saved = 0
    it = iter(val_dl)
    while step < n_dl_steps and saved < n_target_mixtures:
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

        est32 = est.float().cpu().numpy()
        tgt32 = target.float().cpu().numpy()
        mix32 = mix.float().cpu().numpy()
        bsz = est32.shape[0]

        for bi in range(bsz):
            if saved >= n_target_mixtures:
                break
            key = f"{key_prefix}_{saved:08d}"
            e = _mono_1d(est32[bi])
            r = _mono_1d(tgt32[bi])
            m = _mono_1d(mix32[bi])
            min_len = min(e.shape[0], r.shape[0], m.shape[0])
            e = e[:min_len]
            r = r[:min_len]
            m = m[:min_len]

            enh_path = audio_dir / f"{key}.wav"
            ref_path = ref_wav_dir / f"{key}.wav"
            mix_path = mix_wav_dir / f"{key}.wav"
            sf.write(str(enh_path), e, sample_rate)
            sf.write(str(ref_path), r, sample_rate)
            sf.write(str(mix_path), m, sample_rate)

            spk1_lines.append(f"{key} {enh_path.resolve()}\n")
            single_lines.append(f"{key} {ref_path.resolve()}\n")
            mix_lines.append(f"{key} {mix_path.resolve()}\n")
            saved += 1

        step += 1

    spk_scp = audio_dir / "spk1.scp"
    ref_scp = ref_wav_dir.parent / "single.wav.scp"
    mix_scp = mix_wav_dir.parent / "mix.scp"
    spk_scp.write_text("".join(spk1_lines), encoding="utf-8")
    ref_scp.write_text("".join(single_lines), encoding="utf-8")
    mix_scp.write_text("".join(mix_lines), encoding="utf-8")

    logger.info("Wrote %d utterances under %s", saved, audio_dir.parent)
    return saved


def main() -> None:
    parser = HfArgumentParser(
        (
            TSEOnlineDataArguments,
            TSETrainingArguments,
            EvalTestCorporaArguments,
            InferTestExportExtraArguments,
        )
    )
    data_args, training_args, eval_args, export_args = parser.parse_args_into_dataclasses()

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

    export_root = export_args.export_root or os.path.join(
        training_args.output_dir,
        "stage103_test_export",
    )
    export_root_p = pathlib.Path(export_root)
    export_root_p.mkdir(parents=True, exist_ok=True)

    sr = _sample_rate_hz(data_args)
    if sr != 16000:
        logger.warning(
            "Dataset resample_rate=%d; tools/score.sh defaults expect 16 kHz refs.",
            sr,
        )

    targets = _parse_eval_targets(eval_args.eval_targets)
    n_mix = int(eval_args.num_eval_mixtures)
    seed_base = int(training_args.seed) + int(eval_args.eval_seed_offset)

    executor = Executor()
    logger.info(
        "export_root=%s  eval_targets=%s  num_utts=%d  sr=%d",
        export_root,
        targets,
        n_mix,
        sr,
    )

    for tgt in targets:
        logger.info("=" * 78)
        logger.info("EXPORT CORPUS: %s", tgt.upper())
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

        tgt_root = export_root_p / tgt
        audio_dir = tgt_root / "audio"
        ref_dset = tgt_root / "ref_dset"
        ref_wav_dir = ref_dset / "wavs"
        mix_wav_dir = tgt_root / "mix" / "wavs"

        key_prefix = f"st103_{tgt}"
        _run_export_pass(
            model=model,
            val_dl=val_dl,
            executor=executor,
            device=device,
            n_target_mixtures=n_mix,
            batch_size=batch_size,
            bf16=bool(training_args.bf16),
            sample_rate=sr,
            audio_dir=audio_dir,
            ref_wav_dir=ref_wav_dir,
            mix_wav_dir=mix_wav_dir,
            key_prefix=key_prefix,
            print_first=bool(eval_args.print_first_batch_shapes),
        )

        logger.info(
            "Score this corpus with:\n"
            "  ./tools/score.sh --dset %s --exp_dir %s --fs 16k ...",
            ref_dset,
            tgt_root,
        )


if __name__ == "__main__":
    main()
