#!/usr/bin/env bash
# Cue-mute ablation for the live stage-100 run.
#
# Goal: confirm whether the model has been ignoring the visual cue. Runs the
# stage-100 val pipeline twice on the same mixtures — first with normal cues,
# then with all visual cues replaced by zeros — and compares per-sample SI-SDR
# percentiles.
#
# This is read-only: it does NOT touch ${exp_dir}/checkpoint-*, optimizer.pt,
# trainer_state.json, or ema_model.pt. Safe to run while stage 100 is still
# training.
#
# Usage:
#   cd /maduo/codebase/wesep_wenet/examples/visual/voxceleb2mix
#   nohup ./local/run_ablation_stage100.sh \
#     > logs/ablation_stage100_cue_mute.log 2>&1 &
#
# Optional first arg: weights path (default: ${exp_dir}/ema_model.pt).
# Optional second arg: # of val mixtures (default: 6000 ≈ 1.2 epochs of stage-100 val).
set -euo pipefail

cd "$(dirname "$0")/.."

exp_dir=${EXP_DIR:-/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_short_warmup_balanced_loss}
ema_ckpt=${1:-${exp_dir}/ema_model.pt}
num_mixtures=${2:-6000}

if [ ! -e "$ema_ckpt" ]; then
  echo "[ablation] ERROR: weights not found: $ema_ckpt" >&2
  echo "           pass an explicit path as arg 1, or check that EMA has been written." >&2
  exit 1
fi

data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train

# Use a *separate* output_dir so HfArgumentParser doesn't try to write into
# the live stage-100 dir. HF Trainer is NOT instantiated here; this dir stays
# empty (or holds nothing but the logs HfArgumentParser may print).
ablation_out=${ABLATION_OUT_DIR:-${exp_dir}/ablation_cue_mute}
mkdir -p "$ablation_out"

export TENSORBOARD_LOGGING_DIR=$ablation_out/logs   # unused by this script, kept for parity
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# === Critical CUDA library path fix ===
# Strip CUDA-toolkit dirs out of LD_LIBRARY_PATH so PyTorch's RPATH (cu128)
# wins. A stray /maduo/software/cuda12.6.0/lib* in LD_LIBRARY_PATH causes
# a second libcublas (cu126) to be dlopen-ed mid-forward, which throws
# CUBLAS_STATUS_NOT_INITIALIZED on subsequent sgemm calls.
#
# NOTE: ``grep -Ev`` returns exit code 1 when *every* input line matches the
# pattern (nothing left to print). Combined with ``set -e -o pipefail`` that
# would silently kill this script before any output. ``|| true`` makes the
# pipeline benign.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _new_ldpath=$(printf '%s\n' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | { grep -Ev '^(/maduo/software/cuda[0-9]+\.[0-9]+\.[0-9]+|/usr/local/cuda)' || true; } \
        | paste -sd ':' -)
    LD_LIBRARY_PATH=$_new_ldpath
    unset _new_ldpath
fi
export LD_LIBRARY_PATH

# cuBLAS workaround for driver R580 + cu128 PyTorch (see local/cuda_probe.py).
export TORCH_BLAS_PREFER_CUBLASLT=${TORCH_BLAS_PREFER_CUBLASLT:-0}
export DISABLE_ADDMM_CUDA_LT=${DISABLE_ADDMM_CUDA_LT:-1}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "--- LD_LIBRARY_PATH after sanitization:"
echo "$LD_LIBRARY_PATH"
echo "---"

# IMPORTANT: every flag below mirrors stage 100 except where noted, so the val
# dataset (seed = seed + 7, same online mixer, same speaker pool, same
# collate) reproduces the val seen by trainer.evaluate().
python local/ablation_cue_mute.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir "$ablation_out" \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --per_device_train_batch_size 3 \
    --dataloader_num_workers 2 \
    --visual_max_frames 75 \
    --bf16 true \
    --seed 42 \
    --ema_ckpt "$ema_ckpt" \
    --cue_modes "normal,zero" \
    --num_eval_mixtures "$num_mixtures"
