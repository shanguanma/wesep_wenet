# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0
"""Fixed visual cues with **per-speaker temporal alignment** for online mixtures.

:class:`processor_visual.sample_fixed_visual_cue` relied on optional top-level
``chunk_ratio``. After mixing, that field was absent while ``wav_spk{i}`` stayed
aligned to random intra-file chunks. This module consumes ``chunk_ratio_spk{i}``
emitted by :func:`processor_new.sample_speaker_group_without_repeat`.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from wesep.dataset.processor_visual import (  # reuse decode / geo helpers
    DeviceLike,
    _build_lookup_key,
    _dataloader_worker_safe_decode_device,
    _default_decode_device_for_process,
    _get_spk_resource,
    read_video_with_torchcodec,
    resize_video_thwc_float,
    _wav_num_samples,
)

logger = logging.getLogger(__name__)


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
    """Same contract as ``processor_visual.sample_fixed_visual_cue``.

    Prefer ``chunk_ratio_spk{i}`` (slot ``\"spki\"`` ↔ key ``chunk_ratio_spki``).

    Fallback order per slot:

    #. Sample-level legacy ``chunk_ratio`` (offline / single-speaker parity).
    #. Decode ``len(wav_spki)/sr`` seconds from MP4 offset 0 (**imprecise** if
       ``random_chunk`` was used upstream without propagating ratios).
    """
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

    if spk_resource is not None:
        cue_map = spk_resource
    elif resource_path is not None:
        cue_map = _get_spk_resource(resource_path)
    else:
        raise ValueError(
            "sample_fixed_visual_cue: pass resource_path or spk_resource"
        )

    for sample in data:

        spk_slots = sorted(
            [k for k in sample.keys() if k.startswith("spk")],
            key=lambda s: (
                int("".join(filter(str.isdigit, s))) if any(
                    str.isdigit(ch) for ch in s
                ) else s
            ),
        )

        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        if scope == "utterance":
            spk_slots = [spk_slots[0]]

        sr = float(sample["sample_rate"])

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

            ratio = sample.get(f"chunk_ratio_{slot}")
            if ratio is None and isinstance(sample.get("chunk_ratio"), dict):
                ratio = sample.get("chunk_ratio")

            wav_key = f"wav_{slot}"

            if isinstance(ratio, dict):
                orig_len = ratio.get("orig_len")
                if isinstance(orig_len, torch.Tensor):
                    audio_sec = float(orig_len.item()) / sr
                else:
                    try:
                        audio_sec = float(orig_len) / sr
                    except (TypeError, ValueError):
                        audio_sec = _wav_num_samples(sample["wav_mix"]) / sr
            else:
                if wav_key in sample:
                    audio_sec = _wav_num_samples(sample[wav_key]) / sr
                else:
                    audio_sec = _wav_num_samples(sample["wav_mix"]) / sr

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
                logger.warning(
                    "Failed to read video: %s, err=%s", video_path, e
                )
                if required:
                    raise
                continue

            if video.numel() == 0:
                if required:
                    raise RuntimeError(f"Empty video: {video_path}")
                continue

            video_sec = video.shape[0] / fps
            if video_sec < audio_sec:
                need_sec = audio_sec - video_sec
                need_frames = int(round(need_sec * fps))
                last_frame = video[-1:].repeat(need_frames, 1, 1, 1)
                video = torch.cat([video, last_frame], dim=0)
            elif video_sec > audio_sec:
                max_frames = max(1, int(round(audio_sec * fps)))
                video = video[:max_frames].contiguous()

            if isinstance(ratio, dict):
                start_ratio = float(ratio["start_ratio"])
                end_ratio = float(ratio["end_ratio"])

                ff = video.shape[0]

                start_f = int(start_ratio * ff)
                end_f = int(end_ratio * ff)

                start_f = max(0, min(start_f, ff))
                end_f = max(start_f + 1, min(end_f, ff))

                video = video[start_f:end_f]

            if max_visual_time_frames is not None and video.shape[0] > int(
                    max_visual_time_frames):
                video = video[: int(max_visual_time_frames)].contiguous()

            video = video.float().div_(255.0)

            if spatial_size_hw is not None:
                video = resize_video_thwc_float(video, spatial_size_hw)

            video = video.permute(1, 2, 3, 0)

            sample[f"visual_{slot}"] = video

        if scope == "utterance":
            emb = sample[f"visual_{spk_slots[0]}"]
            for sl in [k for k in sample.keys() if k.startswith("spk")]:
                sample[f"visual_{sl}"] = emb
        yield sample
