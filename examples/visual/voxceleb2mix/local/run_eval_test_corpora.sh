#!/usr/bin/env bash
# Evaluate a checkpoint on **official test folders** only:
#   LRS3:       /F00120240032/lrs3/test/
#   Chinese_lips: /F00120240032/Chinese_lips/test/test/<speaker>/FACE/*.mp4
#
# Usage:
#   cd /maduo/codebase/wesep_wenet/examples/visual/voxceleb2mix
#   CUDA_VISIBLE_DEVICES=0 ./local/run_eval_test_corpora.sh \
#     /maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2/ema_model.pt 3000
#
# Optional env:
#   EVAL_TARGETS=lrs3,chinese_lips,merged   (default: lrs3,chinese_lips)
#
set -euo pipefail

cd "$(dirname "$0")/.."

ema_ckpt=${1:?usage: $0 /path/to/ema_model.pt [num_mixtures]}
num_mixtures=${2:-3000}

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _new_ldpath=$(printf '%s\n' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | { grep -Ev '^(/maduo/software/cuda[0-9]+\.[0-9]+\.[0-9]+|/usr/local/cuda)' || true; } \
        | paste -sd ':' -)
    LD_LIBRARY_PATH=$_new_ldpath
    unset _new_ldpath
fi
export LD_LIBRARY_PATH

exp_dir=${EXP_DIR:-/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2}
out_dir=${OUTPUT_DIR:-/tmp/wesep_eval_test_corpora}
mkdir -p "$out_dir"

targets=${EVAL_TARGETS:-lrs3,chinese_lips}

python local/eval_test_corpora.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --output_dir "$out_dir" \
    --lrs3_root /F00120240032/lrs3 \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --visual_frontend muse \
    --separator_causal false \
    --noise_prob 0.0 \
    --per_device_train_batch_size 3 \
    --dataloader_num_workers 2 \
    --visual_max_frames 75 \
    --bf16 true \
    --seed 42 \
    --ema_ckpt "$ema_ckpt" \
    --eval_targets "$targets" \
    --num_eval_mixtures "$num_mixtures" \
    --eval_noise_prob 0.0
