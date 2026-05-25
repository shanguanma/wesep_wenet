import pathlib
from typing import List, Optional

import torch

from wesep.utils.schedulers import BaseClass


def _load_states_for_pretrained(path: str) -> dict:
    """Return a dict matching the wesep checkpoint contract
    (``{"models": [state_dict, ...], ...}``) regardless of whether ``path``
    points to:

    1. A wesep checkpoint produced by :func:`save_checkpoint`
       (``{"models": [...], "optimizers": [...], ...}``). Returned as-is, so
       behavior for stages 1–88 is bit-for-bit unchanged.
    2. A HuggingFace ``model.safetensors`` file. Wrapped into
       ``{"models": [state_dict]}`` so generator-style consumers work.
    3. A raw ``state_dict`` saved via ``torch.save(state_dict, ...)``
       (e.g. HuggingFace ``pytorch_model.bin``). Wrapped the same way.

    Only the new (2) and (3) branches are additive; the wesep branch (1) is
    detected by the presence of a top-level ``"models"`` list and short-
    circuits before any reformatting.
    """
    p = pathlib.Path(path)
    if p.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading a .safetensors checkpoint requires the 'safetensors' "
                "package; install via `pip install safetensors`."
            ) from exc
        sd = load_file(str(p))
        return {"models": [sd]}

    obj = torch.load(str(p), map_location="cpu", weights_only=False)

    # (1) Native wesep format — leave untouched so existing call sites
    # (load_checkpoint, downstream key access like states["optimizers"]) keep
    # seeing the exact same object.
    if isinstance(obj, dict) and "models" in obj and isinstance(obj["models"], list):
        return obj

    # (3) Raw state_dict (every value is a Tensor). Wrap it.
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return {"models": [obj]}

    raise ValueError(
        f"Unrecognized checkpoint format at {p}: expected a wesep checkpoint "
        f"({{'models': [...]}}) , a safetensors file, or a raw state_dict, "
        f"but top-level object was {type(obj).__name__}"
    )


def load_pretrained_model(model: torch.nn.Module,
                          path: str,
                          type: str = "generator"):
    assert type in ["generator", "discriminator"]
    states = _load_states_for_pretrained(path)
    if type == "generator":
        state = states["models"][0]
    else:
        assert len(states["models"]) == 2
        state = states["models"][1]

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state)
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


def load_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
    only_model: bool = False,
    mode: str = "all",
):
    assert mode in ["all", "generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if mode == "generator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][0]],
            [states["optimizers"][0]],
            [states["schedulers"][0]],
        )
    elif mode == "discriminator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][1]],
            [states["optimizers"][1]],
            [states["schedulers"][1]],
        )
    else:
        model_state, optimizer_state, scheduler_state = (
            states["models"],
            states["optimizers"],
            states["schedulers"],
        )

    for model, state in zip(models, model_state):
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state, strict=False)
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
    if not only_model:
        for optimizer, state in zip(optimizers, optimizer_state):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, scheduler_state):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is not None:
                scaler.load_state_dict(states["scaler"])


def save_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
):
    if isinstance(models[0], torch.nn.DataParallel):
        state_dict = [model.module.state_dict() for model in models]
    elif isinstance(models[0], torch.nn.parallel.DistributedDataParallel):
        state_dict = [model.module.state_dict() for model in models]
    else:
        state_dict = [model.state_dict() for model in models]
    torch.save(
        {
            "models":
            state_dict,
            "optimizers": [o.state_dict() for o in optimizers],
            "schedulers":
            [s.state_dict() if s is not None else None for s in schedulers],
            "scaler":
            scaler.state_dict() if scaler is not None else None,
        },
        path,
    )
