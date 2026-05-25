# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info

from wesep.utils.file_utils import load_json

# conda install ffmpeg=7.0 -c conda-forge
# pip install torchcodec==0.7 --index-url=https://download.pytorch.org/whl/cu128
from torchcodec.decoders import VideoDecoder

DeviceLike = Union[str, torch.device]


def _to_torch_device(device: Optional[DeviceLike]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device) if isinstance(device, str) else device


def _dataloader_worker_safe_decode_device(
    decode_device: Optional[DeviceLike],
    *,
    allow_cuda_in_worker: bool,
) -> Optional[DeviceLike]:
    """
    Historical default: forbid CUDA inside DataLoader workers (Linux ``fork``
    after parent touched CUDA breaks many setups), so decoding fell back to CPU.

    When ``allow_cuda_in_worker`` is True, the caller must have constructed the
    DataLoader with ``multiprocessing_context="spawn"`` so each worker can init
    CUDA safely; then we keep ``decode_device`` (typically ``cuda:LOCAL_RANK``).
    """
    if get_worker_info() is None:
        return decode_device
    if allow_cuda_in_worker:
        return decode_device
    return None


def _default_decode_device_for_process() -> Optional[torch.device]:
    """
    Pick a GPU for TorchCodec video decode when ``LOCAL_RANK`` is set (``torchrun``)
    or fall back to ``cuda:0`` for plain ``python`` single-process runs.

    Returns ``None`` when CUDA is unavailable (CPU decode).
    """
    if not torch.cuda.is_available():
        return None
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        try:
            return torch.device("cuda", int(lr))
        except (ValueError, RuntimeError):
            return torch.device("cuda", 0)
    return torch.device("cuda", 0)


def read_video_with_torchcodec(
    video_path: str,
    pts_unit: str = "sec",
    decode_device: Optional[DeviceLike] = None,
    frame_device: Optional[DeviceLike] = "cpu",
    audio_sec: Optional[float] = None,
):
    """
    使用 TorchCodec 的 VideoDecoder 模拟 torchvision.io.read_video 的功能。

    decode_device 指定 NVDEC/FFmpeg 解码所在设备；多卡 DDP 时应对每个进程传入该 rank
    对应的 GPU（例如 torch.device(\"cuda\", local_rank)），避免所有进程挤在同一张卡上解码。

    Args:
        video_path (str): 视频文件路径。
        pts_unit (str): 时间戳单位，TorchCodec 默认为 "sec" (秒)，此参数仅用于接口对齐。
        decode_device: 解码器使用的设备；None 表示 CPU 解码。
        frame_device: 解码后帧张量放置的设备；默认 \"cpu\"，便于与 DataLoader / 音频张量一致。
            若设为 None，则帧保留在 decode_device 上。
        audio_sec: 若给定，只解码约 audio_sec 秒对应的帧数，避免 enrollment 视频远长于
            当前 mix 时整段解码占满 worker 内存。

    Returns:
        tuple: (video_frames, audio_frames, info)
            - video_frames (torch.Tensor): [T, H, W, C] uint8，与 torchvision.io.read_video 一致。
            - audio_frames: TorchCodec 暂不支持音频解码，返回 None。
            - info (dict): 含 'video_fps' (float)。
    """
    _ = pts_unit
    dev = _to_torch_device(decode_device)
    decoder = VideoDecoder(video_path, seek_mode="exact", device=dev)
    fps = float(decoder.metadata.average_fps or 0.0)
    if fps <= 0.0 or math.isnan(fps):
        fps = 25.0

    if audio_sec is not None and audio_sec > 0:
        max_frames = max(1, int(round(audio_sec * fps)))
        frame_batch = decoder[:max_frames]
    else:
        frame_batch = decoder[:]
    del decoder
    # NCHW on decode device
    video_nchw = frame_batch.data
    del frame_batch
    # Align with torchvision read_video: THWC uint8
    video = video_nchw.permute(0, 2, 3, 1).contiguous()
    del video_nchw

    if frame_device is None:
        out_dev = dev
    else:
        out_dev = _to_torch_device(frame_device)
    if video.device != out_dev:
        video = video.to(out_dev)

    info = {"video_fps": fps}

    return video, None, info


def resize_video_thwc_float(
    video: torch.Tensor,
    size_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Resize ``video`` shaped ``[T, H, W, C]`` float to ``(H_out, W_out)``.

    Used to unify resolutions across corpora (e.g. face crops vs full-frame HD)
    before batching with ``torch.stack``.
    """
    h, w = int(size_hw[0]), int(size_hw[1])
    if video.shape[1] == h and video.shape[2] == w:
        return video
    # [T, H, W, C] -> [T, C, H, W]
    x = video.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def _wav_num_samples(wav: torch.Tensor) -> int:
    """Time-domain length for ``wav_mix`` / ``wav_spk*`` (e.g. ``[1, T]`` or ``[T]``).

    Note: ``len(wav)`` on a 2D tensor returns the **batch/channel** dim (often 1),
    not ``T``. Using ``shape[-1]`` matches training mixtures.
    """
    if wav.dim() == 0:
        return 1
    return int(wav.shape[-1])


def _build_lookup_key(sample, spk_slot, key_field):
    """
    Build lookup key for speaker cue resource.

    key_field semantics:
      - "spk_id"       -> use sample[spk_slot]
      - "mix_spk_id"   -> use f"{sample['key']}::{sample[spk_slot]}"
    """
    if key_field == "spk_id":
        return sample[spk_slot]

    elif key_field == "mix_spk_id":
        mix_key = sample.get("key", None)
        if mix_key is None:
            raise KeyError("sample missing 'key' for mix_spk_id cue")
        return f"{mix_key}::{sample[spk_slot]}"

    else:
        raise ValueError(f"Unsupported key_field for speaker cue: {key_field}")


# module-level cache (per worker process)
_SPK_RESOURCE_CACHE = {}


def _get_spk_resource(resource_path):
    """
    Lazy-load and cache speaker cue resources.

    Cache is keyed by resource_path to avoid train/val or
    multi-dataset cross-contamination.
    """
    if resource_path not in _SPK_RESOURCE_CACHE:
        _SPK_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    return _SPK_RESOURCE_CACHE[resource_path]


def sample_fixed_visual_cue(
    data,
    resource_path=None,
    key_field="mix_spk_id",
    scope="speaker",
    required=True,
    decode_device: Optional[DeviceLike] = None,
    frame_device: Optional[DeviceLike] = "cpu",
    use_local_rank_decode_device: bool = True,
    max_visual_time_frames: Optional[int] = None,
    spatial_size_hw: Optional[tuple[int, int]] = None,
    visual_decode_cuda: bool = True,
    spk_resource: Optional[dict] = None,
):
    if scope not in ("speaker", "utterance"):
        raise ValueError(f"Unsupported scope: {scope}")

    if not visual_decode_cuda:
        decode_device = None
        use_local_rank_decode_device = False
    elif decode_device is None and use_local_rank_decode_device:
        decode_device = _default_decode_device_for_process()
    decode_device = _dataloader_worker_safe_decode_device(
        decode_device, allow_cuda_in_worker=bool(visual_decode_cuda)
    )
    # Optional: force CPU decode if NVDEC is unstable (uncomment on broken drivers).
    # decode_device = None

    if spk_resource is not None:
        cue_map = spk_resource
    elif resource_path is not None:
        cue_map = _get_spk_resource(resource_path)
    else:
        raise ValueError("sample_fixed_visual_cue: pass resource_path or spk_resource")

    for sample in data:

        spk_slots = [k for k in sample.keys() if k.startswith("spk")]

        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        if scope == "utterance":
            spk_slots = [spk_slots[0]]

        for slot in spk_slots:
            lookup_key = _build_lookup_key(sample, slot, key_field)

            if lookup_key not in cue_map:
                if required:
                    raise KeyError(f"fixed visual cue not found: {lookup_key}")
                continue

            items = cue_map[lookup_key]
            if not items:
                if required:
                    raise RuntimeError(f"empty fixed visual cue: {lookup_key}")
                continue

            enroll_item = items[0]
            video_path = enroll_item["path"]

            if "chunk_ratio" in sample:
                audio_sec = sample["chunk_ratio"]["orig_len"] / sample[
                    "sample_rate"]
            else:
                audio_sec = (
                    _wav_num_samples(sample["wav_mix"]) / sample["sample_rate"]
                )

            try:
                video, _, info = read_video_with_torchcodec(
                    video_path,
                    pts_unit="sec",
                    decode_device=decode_device,
                    frame_device=frame_device,
                    audio_sec=audio_sec,
                )
                fps = info["video_fps"]
            except Exception as e:
                logging.warning(f"Failed to read video: {video_path}, err={e}")
                if required:
                    raise
                continue

            # video: [T, H, W, C] uint8
            if video.numel() == 0:
                if required:
                    raise RuntimeError(f"Empty video: {video_path}")
                continue

            # Align video with audio length by truncation or repetition
            # (Frames are already capped by audio_sec in read_video_with_torchcodec.)
            video_sec = video.shape[0] / fps
            if video_sec < audio_sec:
                need_sec = audio_sec - video_sec
                need_frames = int(round(need_sec * fps))

                last_frame = video[-1:].repeat(need_frames, 1, 1, 1)
                video = torch.cat([video, last_frame], dim=0)
            elif video_sec > audio_sec:
                max_frames = max(1, int(round(audio_sec * fps)))
                video = video[:max_frames].contiguous()

            # Apply chunk-level cropping if specified
            if "chunk_ratio" in sample:
                ratio = sample["chunk_ratio"]
                start_ratio = ratio["start_ratio"]
                end_ratio = ratio["end_ratio"]

                F = video.shape[0]

                start_f = int(start_ratio * F)
                end_f = int(end_ratio * F)

                start_f = max(0, min(start_f, F))
                end_f = max(start_f + 1, min(end_f, F))

                video = video[start_f:end_f]

            if max_visual_time_frames is not None and video.shape[0] > int(
                    max_visual_time_frames):
                video = video[: int(max_visual_time_frames)].contiguous()

            # Normalize to [0, 1] float (div_ avoids a second float buffer)
            video = video.float().div_(255.0)

            if spatial_size_hw is not None:
                video = resize_video_thwc_float(video, spatial_size_hw)

            # Convert to [H, W, C, T]
            video = video.permute(1, 2, 3, 0)

            sample[f"visual_{slot}"] = video

        # utterance-level cue: copy from spk1 to all spk slots
        if scope == "utterance":
            emb = sample[f"visual_{spk_slots[0]}"]
            for slot in [k for k in sample.keys() if k.startswith("spk")]:
                sample[f"visual_{slot}"] = emb
        yield sample
