# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np

import torch

BASE_COLLECT_KEYS = {
    # ===== Metadata =====
    "spk": {
        "source": "dataset",
        "key_tpl": "spk{}",
        "axis": "spk",
        "required": True,
        "align": None,
        "as_tensor": False,
    },
    "key": {
        "source": "dataset",
        "axis": "mix",
        "required": True,
        "align": None,
        "as_tensor": False,
    },
    "num_speaker": {
        "source": "dataset",
        "axis": "mix",
        "required": True,
        "align": None,
        "as_tensor": False,
    },

    # ===== Model features =====
    "wav_mix": {
        "source": "dataset",
        "axis": "mix",
        "required": True,
        "align": "max",
        "as_tensor": True,
    },
    "wav_target": {
        "source": "dataset",
        "key_tpl": "wav_spk{}",
        "axis": "spk",
        "required": True,
        "align": "max",
        "as_tensor": True,
    },

    # ===== Audio cue =====
    "audio_aux": {
        "source": "dataset",
        "key_tpl": "audio_spk{}",
        "axis": "spk",
        "required": False,
        "align": "min",
        "as_tensor": True,
    },

    # ===== Spatial cue (placeholder) =====
    "spatial_aux": {
        "source": "dataset",
        "key_tpl": "spatial_spk{}",
        "axis": "spk",
        "required": False,
        "align": "max",
    },

    # ===== Visual cue (placeholder) =====
    "visual_aux": {
        "source": "dataset",
        "key_tpl": "visual_spk{}",
        "axis": "spk",
        "required": False,
        "align": "max",
    },

    # ===== Textual cue (placeholder) =====
    "textual_aux": {
        "source": "dataset",
        "key_tpl": "textual_spk{}",
        "axis": "spk",
        "required": False,
        "align": None,
    },
}

AUX_KEY_MAP = {
    "audio": "audio_aux",
    "spatial": "spatial_aux",
    "visual": "visual_aux",
    "textual": "textual_aux",
}


def build_collect_keys(cues_conf, train_conf, base_table):
    """
    Args:
        cues_conf: dict loaded from cues.yaml["cues"]
        train_conf: model conf (expects dataset_args.cues)
        base_table: BASE_COLLECT_KEYS

    Returns:
        collect_keys: dict for tse_collate_fn
    """
    collect_keys = {}
    if "cues" in cues_conf:
        cues_conf = cues_conf["cues"]

    # ---- 1) Fixed Keys (always needed) ----
    for k in ["wav_mix", "wav_target", "spk", "key", "num_speaker"]:
        if k not in base_table:
            raise KeyError(f"[collect_keys] base_table missing fixed key: {k}")
        collect_keys[k] = dict(base_table[k])

    # ---- 2) Training-side cue wishes ----
    want_cues = train_conf.get("cues", {})

    # ---- 3) Iterate over training wishes (authoritative) ----
    for cue_name, want in want_cues.items():
        use = want.get("use", False)
        if not use:
            continue

        # ---- dataset must be able to provide this cue ----
        cue_cfg = cues_conf.get(cue_name)
        if cue_cfg is None:
            raise RuntimeError(
                f"[collect_keys] Training requires cue '{cue_name}', "
                f"but dataset cues.yaml does not provide it.")

        scope = cue_cfg.get("scope")
        if scope != "speaker":
            # for now we only collate speaker-level cues
            continue

        # ---- cue_name is the modality ----
        modality = cue_name
        if modality not in AUX_KEY_MAP:
            raise RuntimeError(
                f"[collect_keys] Unknown cue modality '{modality}'. "
                f"Known: {list(AUX_KEY_MAP.keys())}")

        aux_key = AUX_KEY_MAP[modality]

        if aux_key not in base_table:
            raise KeyError(
                f"[collect_keys] base_table missing aux key spec: {aux_key}")

        spec = dict(base_table[aux_key])

        # ---- Consistency check ----
        if want.get("required",
                    False) and not cue_cfg.get("guaranteed", False):
            raise RuntimeError(
                f"[collect_keys] Training requires cue '{cue_name}' to be present "
                f"in every sample, but dataset cues.yaml marks it as optional."
            )

        # ---- Final required logic for collate ----
        # collate only cares about training behavior
        spec["required"] = want["required"]

        collect_keys[aux_key] = spec

    return collect_keys


def build_collect_keys_online(
    train_conf: dict,
    base_table: dict | None = None,
    *,
    tse_model_name: str | None = None,
) -> dict:
    """
    Build ``collect_keys`` without a separate ``cues.yaml`` file (e.g. online-mix training).

    Offline training uses a cues YAML so :func:`build_collect_keys` can cross-check
    ``dataset_args.cues`` against file-backed cue definitions. For online pipelines
    that attach tensors in-process (e.g. ``visual_spk*``), only the **modality list**
    and **required** flags from ``dataset_args["cues"]`` matter for collation.

    This function synthesizes a minimal ``cues`` dict (``scope: speaker``,
    ``guaranteed`` aligned with ``required``) and calls :func:`build_collect_keys`.

    If ``dataset_args.cues`` has no ``use: true`` entry and ``tse_model_name`` contains
    ``"VISUAL"`` (case-insensitive), defaults to a single visual cue (use + required).

    Raises:
        ValueError: if no usable cue configuration can be inferred.
    """
    base_table = base_table or BASE_COLLECT_KEYS
    cues = dict(train_conf.get("cues") or {})
    tc = {**train_conf, "cues": cues}

    def _any_used() -> bool:
        return any(bool(w.get("use")) for w in cues.values())

    if not _any_used():
        if tse_model_name and "VISUAL" in tse_model_name.upper():
            cues["visual"] = {"use": True, "required": True}
        else:
            raise ValueError(
                "[collect_keys] Online mode without train_cues: set dataset_args.cues in "
                "the experiment YAML (e.g. visual: {use: true, required: true}), or use a "
                "visual TSE model name, or pass --train_cues to a cues YAML file."
            )
    tc["cues"] = cues

    synthetic = {"cues": {}}
    for cue_name, want in cues.items():
        if not want.get("use", False):
            continue
        synthetic["cues"][cue_name] = {
            "scope": "speaker",
            "guaranteed": bool(want.get("required", False)),
        }
    return build_collect_keys(synthetic, tc, base_table)


def _to_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(f"[collate] Unsupported type: {type(x)}")


def _pad_or_crop_to_len(x, target_len):
    # assume time dim is last
    cur_len = x.shape[-1]
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[..., :target_len]
    # pad
    pad_shape = list(x.shape)
    pad_shape[-1] = target_len - cur_len
    pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def tse_collate_fn(batch, collect_keys, visual_max_frames: Optional[int] = None):
    """
    Speaker-axis collate:
      - each mix generates num_speaker batch samples
      - mix-axis features are copied
      - speaker-axis features are indexed
      - metadata is also copied

    Args:
        batch: list[dict]
        collect_keys: dict
        visual_max_frames: If set, cap the **last** dimension of ``visual_aux``
            after ``align`` (usually video time ``T`` for shape ``[..., T]``).
            Reduces GPU memory when batches pad to a very long clip.

    Returns:
        new_batch: dict
    """

    if len(batch) == 0:
        return {}

    # ---- 1) get num_speaker per mix ----
    num_speakers = []
    for s in batch:
        if "num_speaker" not in s:
            raise RuntimeError(
                f"[collate] Missing required key 'num_speaker' in sample: {s.get('key')}"
            )
        num_speakers.append(int(s["num_speaker"]))

    new_batch = {}

    # ---- 2) main loop over keys ----
    for out_key, spec in collect_keys.items():
        key_tpl = spec.get("key_tpl", None)
        axis = spec.get("axis", "mix")
        align = spec.get("align", None)
        required = spec.get("required", True)
        as_tensor = spec.get("as_tensor", True)

        flat_vals = []  # final batch samples
        flat_lens = []  # time lengths (for align)

        ref_tensor = None
        # ---- 2.1) scan batch ----
        for bidx, s in enumerate(batch):
            ns = num_speakers[bidx]

            if not as_tensor:
                # ---------- metadata ----------
                if key_tpl is None:
                    if out_key not in s:
                        if required:
                            raise RuntimeError(
                                f"[collate] Missing required key '{out_key}' in sample: {s.get('key')}"
                            )
                        v = None
                    else:
                        v = s[out_key]

                    # copy to each speaker-sample
                    for _ in range(ns):
                        flat_vals.append(v)

                else:
                    # speaker-indexed metadata
                    for i in range(1, ns + 1):
                        k = key_tpl.format(i)
                        if k not in s:
                            if required:
                                raise RuntimeError(
                                    f"[collate] Missing required key '{k}' in sample: {s.get('key')}"
                                )
                            v = None
                        else:
                            v = s[k]
                        flat_vals.append(v)

                continue

            # ---------- feature ----------
            if axis == "mix":
                # single feature → copy ns times
                if out_key not in s:
                    if required:
                        raise RuntimeError(
                            f"[collate] Missing required key '{out_key}' in sample: {s.get('key')}"
                        )
                    x = None
                else:
                    x = _to_tensor(s[out_key])

                for _ in range(ns):
                    flat_vals.append(x)
                    if x is not None:
                        flat_lens.append(x.shape[-1])

            elif axis == "spk":
                # speaker-indexed feature
                for i in range(1, ns + 1):
                    k = key_tpl.format(i)
                    if k not in s:
                        if required:
                            raise RuntimeError(
                                f"[collate] Missing required key '{k}' in sample: {s.get('key')}"
                            )
                        x = None
                    else:
                        x = _to_tensor(s[k])

                    flat_vals.append(x)
                    if x is not None:
                        flat_lens.append(x.shape[-1])

            else:
                raise ValueError(f"[collate] Unknown axis: {axis}")

            if x is not None and ref_tensor is None:
                ref_tensor = x

        # ---- 2.2) metadata done ----
        if not as_tensor:
            new_batch[out_key] = flat_vals
            continue

        # ---- 2.3) determine target length ----
        if align is None:
            raise ValueError(
                f"[collate] Transfer '{out_key}' into Tensor needs an approach of the alignment."
            )
        else:
            if len(flat_lens) == 0:
                # no real value in entire batch
                target_len = 512
            elif align == "max":
                target_len = max(flat_lens)
            elif align == "min":
                target_len = min(flat_lens)
            else:
                raise ValueError(f"[collate] Unknown align mode: {align}")

            if out_key == "visual_aux" and visual_max_frames is not None:
                target_len = min(target_len, int(visual_max_frames))

        # ---- 2.4) pad / crop / fill ----
        out_feats = []

        for idx, x in enumerate(flat_vals):
            if x is None:
                if required:
                    raise RuntimeError(
                        f"[collate] Required feature '{out_key}' missing for batch sample {idx}"
                    )
                if ref_tensor is None:
                    raise RuntimeError(
                        f"[collate] Cannot infer fallback shape for '{out_key}': "
                        "no real sample exists in batch.")
                # fallback
                x = torch.zeros(ref_tensor.shape,
                                dtype=ref_tensor.dtype,
                                device=ref_tensor.device)

            x = _pad_or_crop_to_len(x, target_len)

            out_feats.append(x)

        # ---- 2.5) stack batch dimension ----
        new_batch[out_key] = torch.stack(out_feats, dim=0)
    return new_batch
