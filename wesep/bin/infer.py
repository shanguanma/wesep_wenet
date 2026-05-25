from __future__ import print_function

import os
import re
import time

import numpy as np

# Toolkit CUDA on LD_LIBRARY_PATH shadows PyTorch's bundled cuBLAS and triggers
# CUBLAS_STATUS_NOT_INITIALIZED in nn.Linear (same fix as run_md_sribd.sh stage 3).
_SYSTEM_CUDA_LD_PREFIX = re.compile(
    r"^(/maduo/software/cuda[0-9]+\.[0-9]+\.[0-9]+|/usr/local/cuda)")


def _sanitize_ld_library_path_for_torch():
    lp = os.environ.get("LD_LIBRARY_PATH", "")
    if not lp:
        return
    kept = [
        p for p in lp.split(":") if p and not _SYSTEM_CUDA_LD_PREFIX.match(p)
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(kept)


_sanitize_ld_library_path_for_torch()

# Honor shell overrides (e.g. run_md_sribd.sh exports CUDA_LAUNCH_BLOCKING=0).
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


def _wav_np_1d(arr):
    """Mono waveform as shape (T,) for SI-SNR / soundfile."""
    x = np.squeeze(np.asarray(arr))
    if x.ndim != 1:
        raise ValueError(
            "infer expects mono waveform [T] or [1, T] per row; got shape "
            + str(np.asarray(arr).shape))
    return x


import fire
import soundfile
import torch
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (
    BASE_COLLECT_KEYS,
    build_collect_keys,
    tse_collate_fn,
    AUX_KEY_MAP,
)
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.file_utils import load_yaml
from wesep.utils.utils import (
    generate_enahnced_scp,
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)


def _resolve_infer_checkpoint(configs):
    """Path to weights for inference.

    Prefer explicit ``configs['checkpoint']`` (CLI ``--checkpoint``). If absent,
    search common locations under ``exp_dir`` so ``run_md_sribd.sh`` stage 5
    can omit ``--checkpoint`` when only the default averaged model exists.
    """
    ckpt = configs.get("checkpoint")
    if ckpt:
        path = os.path.expanduser(os.path.expandvars(str(ckpt)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    exp_dir = configs.get("exp_dir") or "."
    candidates = [
        os.path.join(exp_dir, "models", "avg_best_model.pt"),
        os.path.join(exp_dir, "avg_best_model.pt"),
        os.path.join(exp_dir, "models", "latest_checkpoint.pt"),
        os.path.join(exp_dir, "ema_model.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise ValueError(
        "infer: missing checkpoint — pass --checkpoint /path/to.pt, or place one of "
        f"avg_best_model.pt / latest_checkpoint.pt / ema_model.pt under {exp_dir!r}. "
        f"Tried: {candidates}"
    )


def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_cnt = 0
    accept_cnt = 0

    configs = parse_config_or_kwargs(config, **kwargs)
    sign_save_wav = configs.get(
        "save_wav", True)  # Control if save the extracted speech as .wav

    rank = 0
    set_seed(configs["seed"] + rank)
    raw_gpu = configs["gpus"]
    try:
        gpu = int(raw_gpu)
    except (TypeError, ValueError):
        gpu = -1
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    sample_rate = configs.get("fs", None)
    if sample_rate is None or sample_rate == "16k":
        sample_rate = 16000
    else:
        sample_rate = 8000

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    model_path = _resolve_infer_checkpoint(configs)
    load_pretrained_model(model, model_path)

    logger = get_logger(configs["exp_dir"], "infer.log")
    logger.info("Load checkpoint from {}".format(model_path))
    save_audio_dir = os.path.join(configs["exp_dir"], "audio")
    if sign_save_wav:
        if not os.path.exists(save_audio_dir):
            try:
                os.makedirs(save_audio_dir)
                print(f"Directory {save_audio_dir} created successfully.")
            except OSError as e:
                print(f"Error creating directory {save_audio_dir}: {e}")
        else:
            print(f"Directory {save_audio_dir} already exists.")
    else:
        print("Do NOT save the results in wav.")

    model = model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        # Force CUDA context + cuBLAS handle creation before first forward
        # (avoids sporadic CUBLAS_STATUS_NOT_INITIALIZED on first GEMM).
        with torch.cuda.device(device):
            _warm = torch.zeros(32, 32, device=device, dtype=torch.float32)
            _ = _warm @ _warm
        torch.cuda.synchronize(device)

    configs["dataset_args"]["whole_utt"] = True
    test_dataset = Dataset(
        configs["data_type"],
        configs["test_data"],
        configs["dataset_args"],
        state="test",
        repeat_dataset=configs.get("repeat_dataset", False),
        cues_yaml=configs.get("test_cues", None),
    )
    test_collect_keys = build_collect_keys(
        load_yaml(configs["test_cues"]),
        configs["dataset_args"],
        BASE_COLLECT_KEYS,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=lambda batch: tse_collate_fn(batch, test_collect_keys))

    with open(configs["test_data"], "r", encoding="utf-8") as f:
        test_iter = sum(1 for _ in f)
    logger.info("test number: {}".format(test_iter))

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            mix, cues, target = extract_model_inputs(batch, device)
            spk = batch["spk"]
            key = batch["key"]

            if cues is None:
                outputs = model(mix)
            else:
                outputs = model(mix, cues)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if torch.min(outputs.max(dim=1).values) > 0:
                outputs = ((outputs /
                            abs(outputs).max(dim=1, keepdim=True)[0] *
                            0.9).cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()

            ref = target.cpu().numpy()
            ests = np.asarray(outputs)
            mix_np = mix.cpu().numpy()

            # tse_collate_fn stacks one row per target speaker (wav_mix duplicated).
            n_rows = int(ref.shape[0])
            if ests.ndim == 1:
                ests = ests.reshape(1, -1)
            if ests.shape[0] != n_rows or mix_np.shape[0] != n_rows:
                raise ValueError(
                    "infer batch shape mismatch -- ref {}, estimates {}, mix {}".format(
                        ref.shape, ests.shape, mix_np.shape))

            wav_utt_tag = total_cnt + 1
            if sign_save_wav:
                for si in range(n_rows):
                    wf = os.path.join(
                        save_audio_dir,
                        f"Utt{wav_utt_tag}-{key[si]}-T{spk[si]}.wav",
                    )
                    soundfile.write(wf, _wav_np_1d(ests[si]), sample_rate)

            for si in range(n_rows):
                ei = np.asarray(ests[si])
                ri = np.asarray(ref[si])
                mi = np.asarray(mix_np[si])
                min_len = min(ei.shape[-1], ri.shape[-1], mi.shape[-1])
                ei = _wav_np_1d(ei[..., :min_len])
                ri = _wav_np_1d(ri[..., :min_len])
                mi = _wav_np_1d(mi[..., :min_len])
                sisnr, delta = cal_SISNRi(ei, ri, mi)

                logger.info(
                    "Num={} | Utt={} | Target speaker={} | SI-SNR={:.2f} | SI-SNRi={:.2f}"
                    .format(total_cnt + 1, key[si], spk[si], sisnr, delta))
                total_SISNR += sisnr
                total_SISNRi += delta
                total_cnt += 1
                if delta > 1:
                    accept_cnt += 1

        end = time.time()
    # generate the scp file of the enhanced speech for scoring
    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info("Time Elapsed: {:.1f}s".format(end - start))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info(
        "Acceptance rate of Utterances with SI-SDRi > 1 dB: {:.2f}".format(
            accept_cnt / total_cnt * 100))


def extract_model_inputs(batch, device):
    """
        Build model inputs from collated batch.

        Args:
            batch: dict from tse_collate_fn
            device: torch.device

        Returns:
            mix:    Tensor [B, 1, T]
            cues:   list[Tensor] or None
            target: Tensor [B, 1, T]
        """
    if "wav_mix" not in batch:
        raise RuntimeError("[executor] Missing required key: wav_mix")
    if "wav_target" not in batch:
        raise RuntimeError("[executor] Missing required key: wav_target")

    mix = batch["wav_mix"].float().to(device)
    target = batch["wav_target"].float().to(device)

    cues = []
    for k in list(AUX_KEY_MAP.values()):
        if k in batch and batch[k] is not None:
            cues.append(batch[k].float().to(device))

    if len(cues) == 0:
        cues = None

    return mix, cues, target


if __name__ == "__main__":
    fire.Fire(infer)
