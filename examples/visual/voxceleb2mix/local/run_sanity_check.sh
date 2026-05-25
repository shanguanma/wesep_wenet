#!/usr/bin/env bash
# End-to-end sanity check on the online-mix VoxCeleb2/LRS3/Chinese_lips
# pipeline. Mirrors stage 100's data flags so the val mixtures inspected
# here are exactly the ones the trainer would see.
#
# Read-only: writes nothing into any live experiment dir. Optional
# --audio_dump_dir saves the first mixture's wavs for human listening.
#
# Usage:
#   cd /maduo/codebase/wesep_wenet/examples/visual/voxceleb2mix
#   ./local/run_sanity_check.sh                                    # data-side only
#   ./local/run_sanity_check.sh                                  \
#       /maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_short_warmup_balanced_loss/ema_model.pt
#                                                                   # + model side
#   ./local/run_sanity_check.sh /path/to/weights.pt 32 /tmp/dump   # 32 steps, dump audio
set -euo pipefail

cd "$(dirname "$0")/.."

weights=${1:-}
num_mixtures=${2:-16}
audio_dump_dir=${3:-}

data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train

# Throwaway dir; the script never writes here, but HF parser requires the flag.
sanity_out=${SANITY_OUT_DIR:-/tmp/wesep_sanity_check}
mkdir -p "$sanity_out"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# === Critical CUDA library path fix ===
# Your shell env has /maduo/software/cuda12.6.0/lib* in LD_LIBRARY_PATH which
# loads CUDA 12.6 libcublas before PyTorch's bundled cu128 libcublas (loaded
# via RPATH). When the model forward triggers a second dlopen (e.g. on the
# first Linear after STFT/Conv ops), the wrong libcublas version gets pulled
# in and ``cublasSgemm`` raises ``CUBLAS_STATUS_NOT_INITIALIZED``.
#
# We strip every CUDA-toolkit dir from LD_LIBRARY_PATH so PyTorch's RPATH
# wins and only one libcublas (cu128, the one PyTorch was built against)
# stays loaded. Other userland paths (TensorRT, etc.) are kept.
#
# NOTE: ``grep -Ev`` returns exit code 1 when *every* input line matches the
# pattern (i.e. nothing left to print). Combined with ``set -e -o pipefail``
# that would silently kill this script. ``|| true`` makes the pipeline benign.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _new_ldpath=$(printf '%s\n' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | { grep -Ev '^(/maduo/software/cuda[0-9]+\.[0-9]+\.[0-9]+|/usr/local/cuda)' || true; } \
        | paste -sd ':' -)
    LD_LIBRARY_PATH=$_new_ldpath
    unset _new_ldpath
fi
export LD_LIBRARY_PATH

# cuBLAS workarounds — see local/cuda_probe.py for full diagnosis.
# Driver R580.65 + PyTorch cu128 has a broken cuBLASLt sgemm path for >= 4D
# nn.Linear inputs. We disable cuBLASLt entirely so PyTorch falls back to the
# regular cuBLAS handle that does work on this machine. Both env vars are
# read by PyTorch's ATen at import time, so they MUST be exported before
# Python is invoked.
export TORCH_BLAS_PREFER_CUBLASLT=${TORCH_BLAS_PREFER_CUBLASLT:-0}
export DISABLE_ADDMM_CUDA_LT=${DISABLE_ADDMM_CUDA_LT:-1}
# Bound cuBLAS workspace + use expandable allocator (helps when sharing GPU).
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "--- LD_LIBRARY_PATH after sanitization:"
echo "$LD_LIBRARY_PATH"
echo "---"

extra_flags=()
if [ -n "$weights" ]; then
  extra_flags+=( --weights "$weights" )
fi
if [ -n "$audio_dump_dir" ]; then
  extra_flags+=( --audio_dump_dir "$audio_dump_dir" )
fi
num_gpus=1
#export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
 local/sanity_check_pipeline.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir "$sanity_out" \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --per_device_train_batch_size 1 \
    --dataloader_num_workers 0 \
    --visual_max_frames 75 \
    --bf16 false \
    --seed 42 \
    --num_mixtures "$num_mixtures" \
    "${extra_flags[@]}"
