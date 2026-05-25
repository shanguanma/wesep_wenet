# Copyright (c) 2026 Duo Ma (maduo@cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""
Minimal CUDA / cuBLAS probe for narrowing down ``CUBLAS_STATUS_NOT_INITIALIZED``.

Run this BEFORE any heavier scripts when cuBLAS is misbehaving. It performs a
series of escalating tests so you can see exactly which step dies:

  1. Print env: torch.version, torch.version.cuda, cuDNN, NVCC info, driver.
  2. Tiny tensor: ``torch.zeros(1).cuda()``  (CUDA context creation only).
  3. Tiny matmul: 8x8 sgemm  (forces cuBLAS handle allocation).
  4. Large matmul: 1024x1024 sgemm (cuBLASLt path, similar to nn.Linear).
  5. nn.Linear on 2D, 3D, 4D, 5D inputs (matches the model's tensor ranks).
  6. nn.Linear with bf16 autocast (matches stage 100 training path).

The first failing step tells you where to look:

  Fail at step 2 → CUDA driver / context broken (driver vs torch toolkit).
  Fail at step 3 → cuBLAS handle creation broken (env / library mismatch).
  Fail at step 4 → cuBLASLt issue (often fixed by ``CUBLAS_WORKSPACE_CONFIG``).
  Fail at step 5 (4D / 5D only) → ``nn.Linear`` reshape bug for that rank.
  Fail at step 6 → bf16 / autocast incompatibility.

Usage:

  cd /maduo/codebase/wesep_wenet/examples/visual/voxceleb2mix
  CUDA_VISIBLE_DEVICES=0 python local/cuda_probe.py

You can also force synchronous CUDA errors for clearer tracebacks::

  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python local/cuda_probe.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback

# IMPORTANT: cuBLAS workaround env vars must be set BEFORE ``import torch``.
# PyTorch reads them once at C++ initialization. Setting them in Python after
# import has no effect on the gemm_and_bias dispatcher.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _hr(title: str = "") -> None:
    print("=" * 78, flush=True)
    if title:
        print("== " + title, flush=True)
        print("=" * 78, flush=True)


def _step(idx: int, name: str) -> None:
    print(f"\n--- step {idx}: {name} ---", flush=True)


def _ok(idx: int, name: str) -> None:
    print(f"[OK]   step {idx}: {name}", flush=True)


def _fail(idx: int, name: str, err: BaseException) -> None:
    print(f"[FAIL] step {idx}: {name}", flush=True)
    print("        type:", type(err).__name__, flush=True)
    print("        msg :", str(err).splitlines()[0] if str(err) else "(no message)", flush=True)
    traceback.print_exc()


def main() -> int:
    _hr("CUDA / cuBLAS probe")
    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"prefix: {sys.prefix}", flush=True)
    for k in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "LD_LIBRARY_PATH",
    ):
        print(f"env {k}={os.environ.get(k, '<unset>')}", flush=True)

    # -----------------------------------------------------------------
    # Step 1: env reporting
    # -----------------------------------------------------------------
    _step(1, "report env (no CUDA usage yet)")
    try:
        import torch  # noqa: E402

        # Honor the same cuBLAS workaround that the production scripts use.
        # If it's already broken at import we still want to *report* it; the
        # individual steps below will tell us where things actually fall over.
        try:
            torch.backends.cuda.preferred_blas_library(backend="cublas")
            print("torch.backends.cuda.preferred_blas_library = cublas (forced)", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"preferred_blas_library API unavailable: {_e}", flush=True)

        print(f"torch.__version__       = {torch.__version__}", flush=True)
        print(f"torch.version.cuda      = {torch.version.cuda}", flush=True)
        print(f"torch.backends.cudnn    = {torch.backends.cudnn.version()}", flush=True)
        print(f"torch.cuda.is_available = {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"torch.cuda.device_count = {torch.cuda.device_count()}", flush=True)
            for i in range(torch.cuda.device_count()):
                print(f"  cuda[{i}] = {torch.cuda.get_device_name(i)} "
                      f"cap={torch.cuda.get_device_capability(i)}", flush=True)
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,driver_version,memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            print("nvidia-smi:", out.stdout.strip() or out.stderr.strip(), flush=True)
        except Exception as e:  # noqa: BLE001
            print("nvidia-smi probe failed:", e, flush=True)
        _ok(1, "report env")
    except Exception as e:  # noqa: BLE001
        _fail(1, "report env", e)
        return 1

    if not torch.cuda.is_available():
        print("\n[WARN] CUDA unavailable; cannot probe cuBLAS.", flush=True)
        return 0

    device = torch.device("cuda")

    # -----------------------------------------------------------------
    # Step 2: CUDA context creation (no math yet)
    # -----------------------------------------------------------------
    _step(2, "tiny tensor on GPU (CUDA context only, no cuBLAS)")
    try:
        a = torch.zeros(1, device=device)
        torch.cuda.synchronize()
        del a
        _ok(2, "context creation")
    except Exception as e:  # noqa: BLE001
        _fail(2, "context creation", e)
        return 2

    # -----------------------------------------------------------------
    # Step 3: small sgemm (cuBLAS handle creation)
    # -----------------------------------------------------------------
    _step(3, "8x8 fp32 matmul (forces cuBLAS handle alloc)")
    try:
        a = torch.randn(8, 8, device=device)
        b = torch.randn(8, 8, device=device)
        c = a @ b
        torch.cuda.synchronize()
        s = float(c.sum().item())
        print(f"        sum={s:.4f}", flush=True)
        del a, b, c
        _ok(3, "small sgemm")
    except Exception as e:  # noqa: BLE001
        _fail(3, "small sgemm", e)
        return 3

    # -----------------------------------------------------------------
    # Step 4: large sgemm (cuBLASLt path)
    # -----------------------------------------------------------------
    _step(4, "1024x1024 fp32 matmul (cuBLASLt path used by nn.Linear)")
    try:
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = a @ b
        torch.cuda.synchronize()
        s = float(c.sum().item())
        print(f"        sum={s:.2f}  shape={tuple(c.shape)}", flush=True)
        del a, b, c
        torch.cuda.empty_cache()
        _ok(4, "large sgemm")
    except Exception as e:  # noqa: BLE001
        _fail(4, "large sgemm", e)
        return 4

    # -----------------------------------------------------------------
    # Step 5: nn.Linear on 2D / 3D / 4D / 5D inputs
    # -----------------------------------------------------------------
    import torch.nn as nn

    in_f, out_f = 256, 128
    lin = nn.Linear(in_f, out_f).to(device)
    test_shapes = {
        "2D (B, F)":          (32, in_f),
        "3D (B, T, F)":       (4, 75, in_f),
        "4D (B, band, T, F)": (4, 6, 75, in_f),
        "5D (B, S, band, T, F)": (4, 2, 6, 75, in_f),  # speech.py:100 path
    }
    for label, shape in test_shapes.items():
        _step(5, f"nn.Linear({in_f}->{out_f}) on {label} = {shape}")
        try:
            x = torch.randn(*shape, device=device)
            y = lin(x)
            torch.cuda.synchronize()
            print(f"        out shape={tuple(y.shape)}  finite={bool(torch.isfinite(y).all().item())}", flush=True)
            del x, y
            torch.cuda.empty_cache()
            _ok(5, label)
        except Exception as e:  # noqa: BLE001
            _fail(5, label, e)
            return 5

    # -----------------------------------------------------------------
    # Step 5b: nn.Linear on NON-contiguous 4D input (matches speech.py:100
    # which does ``y = torch.transpose(y, 2, 3); self.fc(y)``).
    # -----------------------------------------------------------------
    _step(51, "nn.Linear on NON-contiguous 4D (transpose-before-Linear, like speech.py:100)")
    try:
        # Make a (B, band, F, T) tensor and transpose to (B, band, T, F) so the
        # last dim is the in_features, but underlying storage is permuted.
        x_raw = torch.randn(4, 6, in_f, 75, device=device)
        x = torch.transpose(x_raw, 2, 3)  # (4, 6, 75, in_f) but non-contig
        assert not x.is_contiguous(), "expected non-contiguous"
        y = lin(x)
        torch.cuda.synchronize()
        print(f"        non-contig out shape={tuple(y.shape)}  finite={bool(torch.isfinite(y).all().item())}", flush=True)
        del x_raw, x, y
        torch.cuda.empty_cache()
        _ok(51, "non-contiguous 4D nn.Linear")
    except Exception as e:  # noqa: BLE001
        _fail(51, "non-contiguous 4D nn.Linear", e)
        # IMPORTANT: do not return; we still want step 6 to finish so we have
        # the full picture before deciding on a workaround.
        print("    -> this is the suspected failure mode; the model takes this path.", flush=True)

    # -----------------------------------------------------------------
    # Step 6: bf16 autocast nn.Linear
    # -----------------------------------------------------------------
    _step(6, "nn.Linear under bf16 autocast")
    try:
        x = torch.randn(4, 6, 75, in_f, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = lin(x)
        torch.cuda.synchronize()
        print(f"        bf16 out shape={tuple(y.shape)} dtype={y.dtype}", flush=True)
        del x, y
        torch.cuda.empty_cache()
        _ok(6, "bf16 autocast")
    except Exception as e:  # noqa: BLE001
        _fail(6, "bf16 autocast", e)
        return 6

    print("\n[OK] all CUDA / cuBLAS probes passed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
