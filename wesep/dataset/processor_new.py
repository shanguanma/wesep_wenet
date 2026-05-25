# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0
"""Online-mix grouping variants that preserve **per-slot audio chunk alignment**.

:class:`processor.sample_speaker_group_without_repeat` rebuilds mixture dicts
without copying ``chunk_ratio``, so downstream video decode starts at time 0
while ``wav_spk{i}`` may come from random interior chunks — see
:class:`processor_visual_new.sample_fixed_visual_cue`.

This module keeps legacy :mod:`processor` untouched; trainers that need aligned
lip video should route through ``sample_speaker_group_without_repeat``
from **here**.
"""

from __future__ import annotations

import random

from wesep.dataset.timeline import (
    parse_overlap_ratio,
    parse_timeline,
    sample_num_speakers,
    timeline_generator,
)


def _freeze_chunk_ratio(maybe):
    """Shallow-copy ``chunk_ratio`` dict from ``random_chunk``."""
    return dict(maybe) if isinstance(maybe, dict) else None


def sample_speaker_group_without_repeat(
    data,
    num_speakers=None,
    shuffle_size=1000,
    timeline_conf=None,
    rng=random,
):
    """Same as ``wesep.dataset.processor.sample_speaker_group_without_repeat``,
    plus ``chunk_ratio_spk{i}`` copied from each constituent single-speaker row.

    Keys mirror speaker slots ``spk1`` … ``spkK`` so
    ``processor_visual_new.sample_fixed_visual_cue`` can crop each MP4 segment
    to the same temporal window as ``wav_spk{i}``.
    """
    assert num_speakers is not None

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            rng.shuffle(buf)
            max_distinct_spk = len({s["spk"] for s in buf})
            for x in buf:
                num_speaker = sample_num_speakers(num_speakers, rng)
                num_speaker = min(num_speaker, max_distinct_spk)
                if timeline_conf is not None:
                    timeline, overlap_ratio = timeline_generator(
                        timeline_conf,
                        num_speaker,
                        rng,
                    )
                else:
                    timeline = [
                        {"speaker": i, "start": 0.0, "end": 1.0}
                        for i in range(num_speaker)
                    ]
                    overlap_ratio = {"overlap_ratio": 1.0}

                cur_spk = x["spk"]
                example = {
                    "key": x["key"],
                    "wav_spk1": x["wav"],
                    "spk1": x["spk"],
                    "sample_rate": x["sample_rate"],
                    "num_speaker": num_speaker,
                    "overlap_ratio_2spk": parse_overlap_ratio(overlap_ratio),
                }
                cr1 = _freeze_chunk_ratio(x.get("chunk_ratio"))
                if cr1 is not None:
                    example["chunk_ratio_spk1"] = cr1

                example["timeline_spk1"] = parse_timeline(
                    [t for t in timeline if t["speaker"] == 0])
                key = "mix_" + x["key"]
                used_spk_ids = {cur_spk}
                for slot in range(2, num_speaker + 1):
                    candidates = [
                        s for s in buf if s["spk"] not in used_spk_ids
                    ]
                    interference = rng.choice(candidates)
                    used_spk_ids.add(interference["spk"])
                    key = key + "_" + interference["key"]
                    example["timeline_spk" + str(slot)] = parse_timeline([
                        t for t in timeline if t["speaker"] == (slot - 1)
                    ])
                    example["wav_spk" + str(slot)] = interference["wav"]
                    example["spk" + str(slot)] = interference["spk"]
                    cro = _freeze_chunk_ratio(interference.get("chunk_ratio"))
                    if cro is not None:
                        example[f"chunk_ratio_spk{slot}"] = cro
                example["key"] = key
                yield example

            buf = []

    rng.shuffle(buf)
    unique_spk = list({s["spk"] for s in buf})
    K = len(unique_spk)
    for x in buf:
        num_speaker = sample_num_speakers(num_speakers, rng)
        num_speaker = min(num_speaker, K)
        if timeline_conf is not None:
            timeline, overlap_ratio = timeline_generator(
                timeline_conf,
                num_speaker,
                rng,
            )
        else:
            timeline = [
                {"speaker": i, "start": 0.0, "end": 1.0}
                for i in range(num_speaker)
            ]
            overlap_ratio = {"overlap_ratio": 1.0}

        cur_spk = x["spk"]
        example = {
            "key": x["key"],
            "wav_spk1": x["wav"],
            "spk1": x["spk"],
            "sample_rate": x["sample_rate"],
            "num_speaker": num_speaker,
            "overlap_ratio_2spk": parse_overlap_ratio(overlap_ratio),
        }
        cr1 = _freeze_chunk_ratio(x.get("chunk_ratio"))
        if cr1 is not None:
            example["chunk_ratio_spk1"] = cr1

        example["timeline_spk1"] = parse_timeline([
            t for t in timeline if t["speaker"] == 0
        ])
        key = "mix_" + x["key"]
        used_spk_ids = {cur_spk}
        for slot in range(2, num_speaker + 1):
            candidates = [s for s in buf if s["spk"] not in used_spk_ids]
            interference = rng.choice(candidates)
            used_spk_ids.add(interference["spk"])
            key = key + "_" + interference["key"]
            example["timeline_spk" + str(slot)] = parse_timeline([
                t for t in timeline if t["speaker"] == (slot - 1)
            ])
            example["wav_spk" + str(slot)] = interference["wav"]
            example["spk" + str(slot)] = interference["spk"]
            cro = _freeze_chunk_ratio(interference.get("chunk_ratio"))
            if cro is not None:
                example[f"chunk_ratio_spk{slot}"] = cro
        example["key"] = key
        yield example
