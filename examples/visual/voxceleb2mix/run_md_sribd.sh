#!/bin/bash
. ./path_sribd.sh || exit 1

# General configuration
stage=1
stop_stage=1000
# 查看当前限制
#ulimit -n
# 设置为 65535 或更高
#ulimit -n 65535
# file_system 模式下，每个 Tensor 块都会占用一个文件句柄。视频数据 Batch 很大时，默认的 1024 限制瞬间就会被突破
. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  # Data preparation related
data=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data
mkdir -p $data
noise_type="clean"
data_type="raw" # shard/raw
# it is from hlt_sz 4090D compute cluster, /share/workspace/shared_datasets/speechdata/02_voxceleb2/01_voxceleb2_zk/{mixture,mp4}
mix_data_path=/F00120240032/voxceleb2_zk_mixture/mixture/ #/YourPATH/voxceleb2/mixture

  ./local/prepare_data.sh --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 2
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${data_type}" = "shard" ]; then
  echo "Making shards from samples.jsonl ..."
  data=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data
  data=${data}/${noise_type}
  for dset in train val test; do
    python tools/make_shards_from_samples.py \
      --samples ${data}/${dset}/samples.jsonl \
      --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      ${data}/${dset}/shards \
      ${data}/${dset}/shard.list
  done
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  data=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data
  data_type="raw" # shard/raw
  noise_type="clean"
  data=${data}/${noise_type}
  # Training related
  gpus="[0]"
  config=confs/tse_bsrnn_visual_sribd.yaml
  exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VIS
  if [ -z "${config}" ] && [ -f "${exp_dir}/config.yaml" ]; then
  config="${exp_dir}/config.yaml"
  fi
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  # TSE model initialization related
  checkpoint=
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi
  train_script=wesep/bin/train.py
  export OMP_NUM_THREADS=8
  # Drop conflicting CUDA libs from LD_LIBRARY_PATH so PyTorch uses its bundled cuBLAS
  # (avoids ``CUBLAS_STATUS_NOT_INITIALIZED`` in nn.Linear / fusion layers).
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _new_ldpath=$(printf '%s\n' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | { grep -Ev '^(/maduo/software/cuda[0-9]+\.[0-9]+\.[0-9]+|/usr/local/cuda)' || true; } \
        | paste -sd ':' -)
    LD_LIBRARY_PATH=$_new_ldpath
    unset _new_ldpath
  fi
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    ${train_script} --config $config \
    --exp_dir ${exp_dir} \
    --gpus $gpus \
    --num_avg ${num_avg} \
    --data_type "${data_type}" \
    --train_data ${data}/train/${data_type}.list \
    --train_cues ${data}/train/cues.yaml \
    --train_samples ${data}/train/samples.jsonl \
    --val_data ${data}/val/${data_type}.list \
    --val_cues ${data}/val/cues.yaml \
    --val_samples ${data}/val/samples.jsonl \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  # Model average related
  exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VIS
  num_avg=5
  avg_model=$exp_dir/models/avg_best_model.pt
  python wesep/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg} \
    --mode best \
    --epochs "19,20,21,22,23"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint="${exp_dir}/models/avg_best_model.pt"
  fi
  echo "Start inferencing ..."
  data=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data
  data_type="raw" # shard/raw
  noise_type="clean"
  fs=16k
  data=${data}/${noise_type}
  exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VIS
  save_results=True
  config=confs/tse_bsrnn_visual_sribd.yaml
  export CUDA_LAUNCH_BLOCKING=0
  # Same as stage 3: system CUDA libs on LD_LIBRARY_PATH break PyTorch cuBLAS.
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _new_ldpath=$(printf '%s\n' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | { grep -Ev '^(/maduo/software/cuda[0-9]+\.[0-9]+\.[0-9]+|/usr/local/cuda)' || true; } \
        | paste -sd ':' -)
    LD_LIBRARY_PATH=$_new_ldpath
    unset _new_ldpath
  fi
  python wesep/bin/infer.py --config $config \
    --fs ${fs} \
    --gpus 0 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
#[ INFO : 2026-05-13 19:25:10,166 ] - Num=5996 | Utt=test_test_id01000_w-heJCr7jto_00112_0_test_id05202_H98u0ss2EcQ_00072_-7.602846687264531_4.608 | Target speaker=id05202 | SI-SNR=5.00 | SI-SNRi=-2.94
#[ INFO : 2026-05-13 19:25:11,884 ] - Num=5997 | Utt=test_test_id06816_TO7RXdwBz0Q_00104_0_test_id07494_g-q_pbChriU_00264_9.340327387473689_5.248 | Target speaker=id06816 | SI-SNR=5.59 | SI-SNRi=-5.07
#[ INFO : 2026-05-13 19:25:11,885 ] - Num=5998 | Utt=test_test_id06816_TO7RXdwBz0Q_00104_0_test_id07494_g-q_pbChriU_00264_9.340327387473689_5.248 | Target speaker=id07494 | SI-SNR=-12.02 | SI-SNRi=-1.37
#[ INFO : 2026-05-13 19:25:13,137 ] - Num=5999 | Utt=test_test_id01509_9rfQsvJ0sVc_00109_0_test_id05459_3TI6dVmEwzw_00008_9.477101421540237_4.032 | Target speaker=id01509 | SI-SNR=5.96 | SI-SNRi=-3.39
#[ INFO : 2026-05-13 19:25:13,137 ] - Num=6000 | Utt=test_test_id01509_9rfQsvJ0sVc_00109_0_test_id05459_3TI6dVmEwzw_00008_9.477101421540237_4.032 | Target speaker=id05459 | SI-SNR=-12.50 | SI-SNRi=-3.15
#[ INFO : 2026-05-13 19:25:14,730 ] - Time Elapsed: 4705.6s
#[ INFO : 2026-05-13 19:25:14,731 ] - Average SI-SNR: -2.67
#[ INFO : 2026-05-13 19:25:14,731 ] - Average SI-SNRi: -2.67
#[ INFO : 2026-05-13 19:25:14,731 ] - Acceptance rate of Utterances with SI-SDRi > 1 dB: 5.77

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Start scoring ..."
  data=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data
  gpus="[0]"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  # Inferencing and scoring related
  use_pesq=true
  use_dnsmos=true
  dnsmos_use_gpu=true
  fs=16k
  exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VIS
  ./tools/score.sh --dset "${data}/test" \
    --exp_dir "${exp_dir}" \
    --fs ${fs} \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_use_gpu "${dnsmos_use_gpu}" \
    --n_gpu "${num_gpus}"
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
   echo "Prepare wav.scp for musan ..."
   real_data=/F00120240032/musan
   dest_dir=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data
   mkdir -p $dest_dir/musan
  find -L ${real_data} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >$dest_dir/musan/wav.scp
  # Convert all musan data to LMDB
  echo "conver musan data to LMDB ..."
  python tools/make_lmdb.py ${dest_dir}/musan/wav.scp ${dest_dir}/musan/lmdb

fi
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Start training with musan..."
  data=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data
  data_type="raw" # shard/raw
  noise_type="clean"
  data=${data}/${noise_type}
  # Training related
  gpus="[0]"
  config=confs/tse_bsrnn_visual_with_musan_sribd.yaml
  exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VIS_with_musan
  if [ -z "${config}" ] && [ -f "${exp_dir}/config.yaml" ]; then
  config="${exp_dir}/config.yaml"
  fi
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  # TSE model initialization related
  checkpoint=
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi
  train_script=wesep/bin/train.py
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    ${train_script} --config $config \
    --exp_dir ${exp_dir} \
    --gpus $gpus \
    --num_avg ${num_avg} \
    --data_type "${data_type}" \
    --train_data ${data}/train/${data_type}.list \
    --train_cues ${data}/train/cues.yaml \
    --train_samples ${data}/train/samples.jsonl \
    --val_data ${data}/val/${data_type}.list \
    --val_cues ${data}/val/cues.yaml \
    --val_samples ${data}/val/samples.jsonl \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
data=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/clean
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VIS_with_musan_cosine
num_gpus=1
# TSE model initialization related
  checkpoint=
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi

export OMP_NUM_THREADS=8
export TRANSFORMERS_VERBOSITY=debug
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_md.py \
    --config confs/tse_bsrnn_visual_with_musan_sribd.yaml \
    --data_type raw \
    --train_data $data/train/raw.list \
    --val_data $data/val/raw.list \
    --train_samples $data/train/samples.jsonl \
    --val_samples $data/val/samples.jsonl \
    --train_cues $data/train/cues.yaml \
    --val_cues $data/val/cues.yaml \
    --exp_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL'\
    --loss_type 'SISDR'\
    --gpus "0" \
    --num_epochs 150 \
    --batch_size 4\
    --optim_class AdamW \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --clip_grad 5.0 \
    --seed 42 \
    ${checkpoint:+--checkpoint $checkpoint}
fi

# it use real 
if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ]; then
data=/maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/clean
exp_dir=/maduo/exp/wesep_wenet/online_avcrossnet_mamba_with_musan_with
num_gpus=1
# TSE model initialization related
  checkpoint=
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers.py \
    --config confs/tse_bsrnn_visual_with_musan_sribd.yaml \
    --data_type raw \
    --train_data $data/train/raw.list \
    --val_data $data/val/raw.list \
    --train_samples $data/train/samples.jsonl \
    --val_samples $data/val/samples.jsonl \
    --train_cues $data/train/cues.yaml \
    --val_cues $data/val/cues.yaml \
    --output_dir $exp_dir \
    --tse_model 'TSE_ONLINE_AVCROSSNET_MAMBA_VISUAL'\
    --loss_type 'OnlineAVCrossNetLoss'\
    --num_train_epochs 150 \
    --per_device_train_batch_size 3\
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 100 
fi

# Online mix + visual (train_with_transformers_online.py — not train_with_transformers.py)
if [ ${stage} -le 38 ] && [ ${stop_stage} -ge 38 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/online_avcrossnet_mamba_with_musan_online_mix
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_online.py \
    --model_config confs/tse_bsrnn_visual_model.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --output_dir $exp_dir \
    --tse_model 'TSE_ONLINE_AVCROSSNET_MAMBA_VISUAL' \
    --loss_type 'OnlineAVCrossNetLoss' \
    --sample_num_per_epoch 20000 \
    --sample_num_per_epoch_val 5000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --num_train_epochs 150 \
    --per_device_train_batch_size 3 \
    --dataloader_num_workers 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 500\
    --bf16 true\
    --visual_max_frames 75
# about 60GB cuda memory
fi

# Online mix + visual (train_with_transformers_online.py — not train_with_transformers.py)
if [ ${stage} -le 48 ] && [ ${stop_stage} -ge 48 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_with_musan_online_mix
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_online.py \
    --model_config confs/tse_bsrnn_visual_model.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR' \
    --sample_num_per_epoch 20000 \
    --sample_num_per_epoch_val 5000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --num_train_epochs 150 \
    --per_device_train_batch_size 3 \
    --dataloader_num_workers 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 500\
    --bf16 true\
    --visual_max_frames 75
# about 30GB cuda memory
fi

if [ ${stage} -le 58 ] && [ ${stop_stage} -ge 58 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_with_musan_online_mix_multi_data
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR' \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --num_train_epochs 150 \
    --per_device_train_batch_size 3 \
    --dataloader_num_workers 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 500\
    --bf16 true\
    --visual_max_frames 75
# about 23GB cuda memory
fi
if [ ${stage} -le 68 ] && [ ${stop_stage} -ge 68 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_with_musan_online_mix_multi_data_bs6
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR' \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --num_train_epochs 150 \
    --per_device_train_batch_size 6 \
    --dataloader_num_workers 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 500\
    --bf16 true\
    --visual_max_frames 75
# about 32GB cuda memory 
fi

if [ ${stage} -le 78 ] && [ ${stop_stage} -ge 78 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_with_musan_online_mix_multi_data_bs6_blaze_visual_casual_seperator_casual
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR' \
    --visual_frontend "blaze" \
    --blaze_visual_causal true \
    --separator_causal true \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --num_train_epochs 150 \
    --per_device_train_batch_size 6 \
    --dataloader_num_workers 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 500\
    --bf16 true\
    --visual_max_frames 75
fi

# Stage 88 (vs 78): same data / Blaze+causal model, but schedule tuned for SI-SDR plateau.
# Rationale: TSETrainingArguments defaults warmup_steps=15000 and (with default linear
# schedule) constant LR after warmup—often fine early but weak late refinement. Cosine
# decay + shorter warmup + slightly lower peak LR is a low-risk first knob to try.
# eval_steps 1000 cuts validation wall time vs 500 (val is expensive in this pipeline).
# Run: ./run_md_sribd.sh --stage 88 --stop-stage 88
if [ ${stage} -le 88 ] && [ ${stop_stage} -ge 88 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_bs6_blaze_causal_clearervoice_plateau_halving
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR' \
    --visual_frontend "blaze" \
    --blaze_visual_causal true \
    --separator_causal true \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --num_train_epochs 150 \
    --per_device_train_batch_size 6 \
    --dataloader_num_workers 2 \
    --learning_rate 0.001 \
    --clearervoice_lr_scheduler true \
    --clearervoice_warmup_steps 15000 \
    --clearervoice_plateau_halving true \
    --clearervoice_plateau_patience_evals 5 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi

# ============================================================================
# Stage 98 — fresh "v2" run that targets the eval_loss plateau seen in stages 78/88.
# ----------------------------------------------------------------------------
# Hypothesis (see logs/run_md_sribd_stage{78_1,88}.log + historical 48/58/68):
#   Eval plateaus at ~34.36 in stage 78 and regresses to ~35.0 in stage 88
#   primarily due to (a) causal masking on visual + separator, (b) bs=6 collate
#   expansion changing per-step optimization dynamics relative to stages 48/58
#   (eval ~27.3 / 27.5 with bs=3, non-causal), and (c) peak LR=1e-3 in stage 88
#   being unstable for this BSRNN+visual setup.
# Mitigation (this stage):
#   * Use NEW model YAML confs/tse_bsrnn_visual_model_v2.yaml (muse_visual non-causal,
#     separator non-causal). Original YAML and stage 78/88 unchanged.
#   * Keep memory budget by using bs=3 + grad-accum 2 (effective bs ≈ 6 again).
#   * Use new SISDR_MRSTFT loss (registered in wesep/utils/losses.py and
#     handled in train_with_transformers_multi_data_online.loss_function).
#   * Cosine LR with warmup_ratio=5%, peak LR=2e-4. NO ClearerVoice plateau halving.
#   * EMA on, decay 0.999 — saved separately to <output_dir>/ema_model.pt.
#   * Visual cue dropout 0.1 to break over-reliance on visual frontend.
# Output goes to a NEW exp_dir; this stage does not touch any earlier exp_dir.
# Run: nohup ./run_md_sribd.sh --stage 98 --stop-stage 98 \
#        > logs/run_md_sribd_stage98.log 2>&1 &
if [ ${stage} -le 98 ] && [ ${stop_stage} -ge 98 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_muse_noncausal_mrstft_bs3_warmcos
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --num_train_epochs 150 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --visual_cue_dropout 0.1 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi

# ============================================================================
# Stage 99 — warm-start ablation. Copies (does NOT move) the best stage 68
# checkpoint into a new exp_dir, then fine-tunes with the v2 schedule above.
# Purpose: separate "model/data limit" from "schedule/loss limit" — if stage 99
# also plateaus near stage 68 with the new loss/schedule, the ceiling is
# data-limited; if it improves, the original schedule was leaving SI-SDR on
# the table.
# Run: nohup ./run_md_sribd.sh --stage 99 --stop-stage 99 \
#        > logs/run_md_sribd_stage99.log 2>&1 &
if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
src_exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_with_musan_online_mix_multi_data_bs6
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_warmstart_from_stage68_mrstft_cos
num_gpus=1
mkdir -p $exp_dir/seed_init
# Copy (not move/symlink) the source checkpoint so the original stage-68
# checkpoint stream is untouched.
src_ckpt=""
if [ -f "$src_exp_dir/avg_best_model.pt" ]; then
  src_ckpt="$src_exp_dir/avg_best_model.pt"
elif [ -f "$src_exp_dir/models/avg_best_model.pt" ]; then
  src_ckpt="$src_exp_dir/models/avg_best_model.pt"
else
  # Fall back to the most recent HF Trainer checkpoint dir's pytorch_model.bin / model.safetensors.
  latest_ckpt_dir=$(ls -1dt $src_exp_dir/checkpoint-* 2>/dev/null | head -n1)
  if [ -n "$latest_ckpt_dir" ]; then
    if [ -f "$latest_ckpt_dir/pytorch_model.bin" ]; then
      src_ckpt="$latest_ckpt_dir/pytorch_model.bin"
    elif [ -f "$latest_ckpt_dir/model.safetensors" ]; then
      src_ckpt="$latest_ckpt_dir/model.safetensors"
    fi
  fi
fi
if [ -z "$src_ckpt" ]; then
  echo "stage 99: no source checkpoint found under $src_exp_dir; aborting." >&2
  exit 1
fi
cp -n "$src_ckpt" "$exp_dir/seed_init/$(basename $src_ckpt)"
warmstart_ckpt="$exp_dir/seed_init/$(basename $src_ckpt)"
# load_pretrained_model in wesep/utils/checkpoint.py natively supports
# wesep / safetensors / raw-state_dict formats, so any of model.safetensors,
# pytorch_model.bin or a wesep avg_best_model.pt works as-is here.
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --model_init_checkpoint "$warmstart_ckpt" \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --num_train_epochs 60 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --visual_cue_dropout 0.1 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi

# ============================================================================
# Stage 100 — diagnostic-friendly v2 run that addresses the "train down, eval
# flat" symptom seen in stages 98/99.
# ----------------------------------------------------------------------------
# Changes vs stage 98 (each opt-in flag defaults to no-op so stages 78/88/98/99
# remain bit-for-bit reproducible):
#   * --warmup_steps 5000 (replaces warmup_ratio=0.05). At 150 epochs * 10000
#     opt-steps the previous warmup_ratio meant 75k warmup steps; we observed
#     LR was still only 4.3e-5 at step 16k, so the eval plateau was largely
#     LR-starved. 5k warmup hits peak LR in ~0.5 epoch.
#   * --max_steps 200000 + --num_train_epochs 200000 (cosine completes within
#     ~20 epochs of training instead of stretching across 1.5M steps).
#   * --eval_loss_type SISDR_MRSTFT — eval and train now use the same loss
#     formula so eval_loss is directly comparable to the train curve.
#   * --eval_extra_stats true — also logs eval_sisdr_loss_{mean,median,p25,
#     p75,p95,worst5pct_mean} so we can see whether eval is dominated by a
#     long tail of failures.
#   * --mrstft_weight 0.2 + --mrstft_warmup_steps 10000 — MRSTFT term ramps
#     from 0 to 0.2 over the first 10k optimizer steps so SI-SDR drives
#     learning early; once SI-SDR is meaningful, MRSTFT helps refine spectra.
#   * --learning_rate 1.5e-4 (between stage 98's 2e-4 and stage 99's 1e-4).
#   * --visual_cue_dropout 0.2 (stronger, per analysis).
# Run:
#   nohup ./run_md_sribd.sh --stage 100 --stop-stage 100 \
#     > logs/run_md_sribd_stage100.log 2>&1 &
if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_short_warmup_balanced_loss
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 5000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.2 \
    --mrstft_warmup_steps 10000 \
    --eval_loss_type SISDR_MRSTFT \
    --eval_extra_stats true \
    --visual_cue_dropout 0.2 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi

# ============================================================================
# Stage 101 — force the visual cue to be informative.
# ----------------------------------------------------------------------------
# Why this stage exists
#   The cue-mute ablation on the stage-100 EMA snapshot
#   (logs/ablation_stage100_cue_mute.log) showed that replacing every visual
#   cue with zeros changes mean SI-SDR by +0.006 dB and every percentile by
#   <0.12 dB, i.e. the model was completely ignoring the visual frontend.
#   Continuing to train stage 100 cannot cure this on its own — the model is
#   already at a "passthrough"-ish local optimum.
#
# What changes vs stage 100 (each new flag defaults to a no-op so stages
# 78 / 88 / 98 / 99 / 100 remain bit-for-bit reproducible):
#   * --cue_discrim_weight 2.0 (sanity-check verdict was "cue completely
#     ignored", so we double the original 1.0) + --cue_discrim_warmup_steps
#     500 (was 2000 — bringing the term in earlier reduces the time the
#     model can spend in the passthrough basin) + --cue_discrim_temperature
#     2.0 (was 5.0 — sharper softmax, larger gradient when wrong-cue output
#     is closer to wrong target):
#       Adds an InfoNCE term over the (ns x ns) similarity matrix of the
#       ``ns`` adjacent rows of each mixture (which share the same input
#       audio and differ only in cue / target). Forces the model to make
#       model(mix, cue_i) closer to target_i than to any target_j; this is
#       only achievable if the cue is actually used by the separator.
#   * --passthrough_penalty_weight 0.1 + --passthrough_penalty_threshold 10:
#       NEW. Hinge ReLU(SI-SDR(out, mix) - 10dB). Sanity check
#       (logs/sanity_check_pipeline.*) showed median SI-SDR(out, mix) = +32.6
#       dB at stage-100 convergence, i.e. the network is doing pure
#       passthrough. This term is exactly 0 (free) until SI-SDR(out, mix)
#       exceeds the threshold; only then does it start pushing the output
#       *away* from the input mixture. A correctly-separating model never
#       trips this hinge because SI-SDR(target_i, mix) for ns >= 2 is
#       typically below 0 dB on the main speaker channel.
#   * --visual_cue_dropout 0.0 (was 0.2 in stage 100):
#       stage-100 dropout was actively pushing the model toward "ignore the
#       cue". With cue-discrim now penalizing cue-ignorance we want the cue
#       always present.
#   * --loss_type SISDR + --mrstft_weight 0.0 + --eval_loss_type SISDR:
#       remove the MRSTFT term — it was the #1 reason stage 98 collapsed
#       into phase-blind solutions, and stage 100's improved schedule still
#       did not help cue use. Pure SI-SDR + cue-discrim + anti-passthrough
#       is the cleanest test of the hypothesis that visual TSE will work
#       once cue-use is forced and passthrough is blocked.
#   * Output dir is fresh — must train from scratch because stage-100 init
#       weights have already converged to "cue-ignore" basin.
# Run:
#   nohup ./run_md_sribd.sh --stage 101 --stop-stage 101 \
#     > logs/run_md_sribd_stage101.log 2>&1 &
if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 5000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.0 \
    --mrstft_warmup_steps 0 \
    --cue_discrim_weight 2.0 \
    --cue_discrim_warmup_steps 500 \
    --cue_discrim_temperature 2.0 \
    --passthrough_penalty_weight 0.1 \
    --passthrough_penalty_threshold 10.0 \
    --eval_loss_type SISDR \
    --eval_extra_stats true \
    --visual_cue_dropout 0.0 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi


# ============================================================================
# Stage 102 — recover from stage-101 collapse with a gentler cue-discrim recipe.
# ----------------------------------------------------------------------------
# Why this stage exists
#   Stage 101 (logs/run_md_sribd_stage101.log) ran 17k steps with
#   cue_discrim_weight=2.0, warmup=500, temperature=2.0, pure SISDR loss.
#   The cue_discrim term saturated before the model had time to learn basic
#   reconstruction; by step 1k the model was stuck in an *anti-aligned*
#   regime — cue_discrim_loss settled at ~1.4 (above the log(ns)~1.06
#   uniform baseline) and eval_sisdr_loss_mean oscillated in [2.6, 5.2]
#   for 16 evaluations with no downward trend. Net result: stage 101 was
#   ~4 dB worse than stage 100's sanity-check median SI-SDR.
#
# What changes vs stage 101 (each new flag is opt-in; stages 78/88/98/99/100
# remain bit-for-bit reproducible because all the new flags default to no-op
# values when omitted):
#   * --loss_type SISDR_MRSTFT + --mrstft_weight 0.2 + --eval_loss_type
#     SISDR_MRSTFT:
#       Restore an MRSTFT magnitude term during early training. Pure SISDR
#       has near-zero gradient when out and target are uncorrelated (the
#       common stage-0 state), and stage 101 confirmed that without the
#       MRSTFT scaffold the model never establishes a reconstruction
#       baseline. mrstft_weight=0.2 is much smaller than the 0.5 that drove
#       stage 98 into a phase-blind solution but large enough to provide
#       a stable energy gradient when SISDR is uninformative.
#   * --cue_discrim_weight 0.3 (was 2.0):
#       Massively reduce cue_discrim's grip. SI-SDR + MRSTFT now drive most
#       of the gradient; cue_discrim only nudges the model toward using
#       the cue once reconstruction is reasonable.
#   * --cue_discrim_warmup_steps 15000 (was 500):
#       Give the reconstruction loss ~1 epoch of (60k mixtures / 6
#       effective per step) ≈ 10k steps to learn before cue_discrim
#       reaches its full weight. This is the single most important change
#       — stage 101's 500-step ramp was the root cause of the anti-
#       alignment trap.
#   * --cue_discrim_temperature 8.0 (was 2.0):
#       Soften the softmax. With per-pair SI-SDR in roughly +/- 20 dB,
#       temperature 8.0 keeps logits in +/- 2.5 — softmax is responsive
#       but not saturated, so noise-direction gradients no longer dominate.
#   * --passthrough_penalty_threshold 5.0 (was 10.0):
#       Stage 101 showed sisdr_out_mix_mean clusters at 3-5 dB; the 10 dB
#       hinge basically never engaged. Lowering to 5 dB lets the penalty
#       bite earlier if the model starts drifting toward mix.
#       --passthrough_penalty_weight stays at 0.1 (no harm if not engaged).
#   * Output dir is fresh (..._cue_discrim_v2) — must train from scratch
#     because the stage-101 EMA already encodes the anti-aligned local
#     minimum.
#
# Run:
#   nohup bash./run_md_sribd.sh --stage 102 --stop-stage 102 \
#     > logs/run_md_sribd_stage102.log 2>&1 &
#
# Early stop signals to watch:
#   * step  ~3k: eval_sisdr_loss_mean should be < 2 (stage 101 stayed > 3)
#   * step  ~5k: cue_discrim_loss ~ log(2.5) ≈ 1.0 (still in ramp; slightly
#                below the 1.06 baseline is fine)
#   * step ~15k: cue_discrim_loss drops below log(2) = 0.69 once weight
#                reaches 0.3 — this is the canary that the cue is being
#                used in the *correct* direction.
#   * step ~25k: eval_sisdr_loss_mean < 0 (i.e. SI-SDR(out, target) > 0 dB)
#   If by ~30k cue_discrim_loss is still >= log(2), the InfoNCE formulation
#   is the issue, not the schedule, and we should switch to a margin/hinge
#   cue loss in a follow-up stage.
if [ ${stage} -le 102 ] && [ ${stop_stage} -ge 102 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips True \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3 \
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 5000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.2 \
    --mrstft_warmup_steps 0 \
    --cue_discrim_weight 0.3 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.1 \
    --passthrough_penalty_threshold 5.0 \
    --eval_loss_type SISDR_MRSTFT \
    --eval_extra_stats true \
    --visual_cue_dropout 0.0 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi



# ============================================================================
# Stage 103 — test-set WAV export + objective scoring (mirrors stage 5 + 6).
# ----------------------------------------------------------------------------
# Stage 5 runs legacy ``wesep/bin/infer.py`` on a fixed ``data/clean/test``
# list and writes ``${exp_dir}/audio/*.wav`` + ``audio/spk1.scp``. Stage 6
# runs ``./tools/score.sh --dset .../test --exp_dir ...``.
#
# For the **visual online** pipeline (stage 102), test evaluation uses
# ``local/eval_test_corpora.py`` on official LRS3 / Chinese_lips **test**
# folders only. Stage 103 extends that path by exporting **estimated** and
# **reference** mono WAVs plus aligned SCPs, then reuses ``tools/score.sh``
# unchanged (same key alignment requirement as stage 6).
#
# Outputs per corpus under ``${STAGE103_EXPORT_ROOT:-${exp_dir}/stage103_test_export}/${corpus}/``::
#   audio/*.wav + audio/spk1.scp        → passed as --exp_dir to score.sh
#   ref_dset/single.wav.scp + wavs/     → passed as --dset to score.sh
#   mix/mix.scp + mix/wavs/*.wav        → mixture (for ClearerVoice-style SI-SNRi in stage 106)
#
# Environment overrides (optional):
#   STAGE103_CKPT          default: ${exp_dir}/ema_model.pt
#   STAGE103_EXPORT_ROOT   default: ${exp_dir}/stage103_test_export
#   STAGE103_NUM_MIXTURES  default: 3000
#   STAGE103_EVAL_TARGETS  default: lrs3,chinese_lips  (see eval_test_corpora)
#
# Run (from any cwd; script resolves repo paths):
#   bash examples/visual/voxceleb2mix/run_md_sribd.sh --stage 103 --stop-stage 103 \\
#     > logs/run_md_sribd_stage103.log 2>&1 &
if [ ${stage} -le 103 ] && [ ${stop_stage} -ge 103 ]; then
  echo "Stage 103: export test-set estimated WAVs + run tools/score.sh ..."
  _MD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _WENET_ROOT="$(cd "${_MD_SCRIPT_DIR}/../../.." && pwd)"
  cd "${_WENET_ROOT}" || exit 1
  export PYTHONPATH="${_WENET_ROOT}:${PYTHONPATH:-}"

  exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2
  ema_ckpt=${STAGE103_CKPT:-${exp_dir}/ema_model.pt}
  export_root=${STAGE103_EXPORT_ROOT:-${exp_dir}/stage103_test_export_300utts}
  num_mixtures=${STAGE103_NUM_MIXTURES:-300}
  eval_targets=${STAGE103_EVAL_TARGETS:-lrs3,chinese_lips}

  export OMP_NUM_THREADS=8
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _new_ldpath=$(printf '%s\n' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | { grep -Ev '^(/maduo/software/cuda[0-9]+\.[0-9]+\.[0-9]+|/usr/local/cuda)' || true; } \
        | paste -sd ':' -)
    LD_LIBRARY_PATH=$_new_ldpath
    unset _new_ldpath
  fi

  python "${_MD_SCRIPT_DIR}/local/infer_test_export_wav.py" \
    --model_config "${_MD_SCRIPT_DIR}/confs/tse_bsrnn_visual_model_v2.yaml" \
    --output_dir "${exp_dir}" \
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
    --ema_ckpt "${ema_ckpt}" \
    --export_root "${export_root}" \
    --eval_targets "${eval_targets}" \
    --num_eval_mixtures "${num_mixtures}" \
    --eval_noise_prob 0.0
fi
if [ ${stage} -le 104 ] && [ ${stop_stage} -ge 104 ]; then
  # --------------------------------------------------------------------------
  # Multi-CPU parallelism on a single GPU machine
  #
  # * STOI / SI-SDR / SAR / ... run inside each ``wesep.bin.score`` worker on CPU.
  # * Parallelism = number of ``run.pl`` workers (~ ``score_nj`` shards), capped
  #   by ``run.pl`` (CPU cores when not using ``--gpu``; see tools/run.pl).
  #
  # Recommended for maximum CPU throughput with one GPU:
  #   STAGE104_DNSMOS_USE_GPU=false \\
  #   STAGE104_SCORE_NJ=8 \\
  #   STAGE104_OMP_NUM_THREADS=1 \\
  #   bash run_md_sribd.sh --stage 104 --stop-stage 104
  # DNSMOS then uses ONNX CPU EP (slower per utterance but workers scale with cores).
  #
  # If you keep DNSMOS on GPU, raise ``STAGE104_SCORE_NJ`` moderately (e.g. 8–16);
  # workers share the card (device index modulo); avoid huge ``score_nj`` if VRAM is tight.
  # --------------------------------------------------------------------------
  use_pesq=true
  use_dnsmos=true
  dnsmos_use_gpu=${STAGE104_DNSMOS_USE_GPU:-true}
  fs=16k
  gpus="[0]"
  num_gpus=$(echo "$gpus" | awk -F ',' '{print NF}')
  # Parallel scoring shards (see tools/score.sh score_nj / split_scp + run.pl --max-jobs-run).
  score_nj=${STAGE104_SCORE_NJ:-16}
  if [ -n "${STAGE104_OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS="${STAGE104_OMP_NUM_THREADS}"
    export MKL_NUM_THREADS="${STAGE104_OMP_NUM_THREADS}"
    export NUMEXPR_NUM_THREADS="${STAGE104_OMP_NUM_THREADS}"
  fi
  exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2
  export_root=${STAGE103_EXPORT_ROOT:-${exp_dir}/stage103_test_export_300utts}
  for _corpus in lrs3 chinese_lips merged; do
    _dset="${export_root}/${_corpus}/ref_dset"
    _inf_exp="${export_root}/${_corpus}"
    if [ ! -s "${_inf_exp}/audio/spk1.scp" ]; then
      echo "Stage 104: skip scoring for '${_corpus}' (no ${_inf_exp}/audio/spk1.scp)."
      continue
    fi
    echo "Stage 104: scoring ${_corpus}  --dset ${_dset}  --exp_dir ${_inf_exp}"
    ./tools/score.sh --dset "${_dset}" \
      --exp_dir "${_inf_exp}" \
      --fs ${fs} \
      --use_pesq "${use_pesq}" \
      --use_dnsmos "${use_dnsmos}" \
      --dnsmos_use_gpu "${dnsmos_use_gpu}" \
      --n_gpu "${num_gpus}" \
      --score_nj "${score_nj}"
  done
fi
if [ ${stage} -le 105 ] && [ ${stop_stage} -ge 105 ]; then
# show the result
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2
export_root=${exp_dir}/stage103_test_export_300utts
for _corpus in lrs3 chinese_lips merged; do
    _dset="${export_root}/${_corpus}/ref_dset"
    _inf_exp="${export_root}/${_corpus}"
    if [ ! -s "${_inf_exp}/audio/spk1.scp" ]; then
      echo "Stage 104: skip scoring for '${_corpus}' (no ${_inf_exp}/audio/spk1.scp)."
      continue
    fi

  _dir="${_inf_exp}/scoring"
  ./tools/show_enh_score.sh "${_dir}/../.." > \
    "${_dir}/../../RESULTS.md"
done
fi
# Stage 106 — ClearerVoice-Studio aligned metrics (SI-SNRi, SDRi, PESQi, STOIi).
# Reference: https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction/solver.py
# Requires mixture WAVs from stage 103 (``mix/mix.scp``). Re-run stage 103 if missing.
# Optional env: STAGE106_CORPORA (default lrs3,chinese_lips,merged).
if [ ${stage} -le 106 ] && [ ${stop_stage} -ge 106 ]; then
  echo "Stage 106: ClearerVoice-style evaluation (SI-SNRi / SDRi / PESQi / STOIi) ..."
  _MD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _WENET_ROOT="$(cd "${_MD_SCRIPT_DIR}/../../.." && pwd)"
  cd "${_WENET_ROOT}" || exit 1
  exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2
  export_root=${STAGE103_EXPORT_ROOT:-${exp_dir}/stage103_test_export_300utts}
  corpora106=${STAGE106_CORPORA:-lrs3,chinese_lips,merged}
  python3 "${_MD_SCRIPT_DIR}/local/eval_clearvoice_tse_metrics.py" \
    --export_parent "${export_root}" \
    --corpora "${corpora106}" \
    --sample_rate 16000
fi
#bash run_md_sribd.sh --stage 106 --stop-stage 106
#Stage 106: ClearerVoice-style evaluation (SI-SNRi / SDRi / PESQi / STOIi) ...
#/maduo/miniconda3/envs/wesep_py310_cu126/lib/python3.10/site-packages/pystoi/stoi.py:66: RuntimeWarning: Not enough STFT frames to compute intermediate intelligibility measure after removing silent frames. Returning 1e-5. Please check you wav files
#  warnings.warn('Not enough STFT frames to compute intermediate '
#[lrs3] utterances=300  SI-SNR(est)=2.5555±8.8807 dB  SI-SNR(mix)=1.9471±17.2146 dB  SI-SNRi=0.6084±11.9592 dB  SDRi=-5.8402±48.1435 dB  PESQi=0.0607±0.3087  STOIi=0.0256±0.1009
#[chinese_lips] utterances=300  SI-SNR(est)=-0.1770±11.5180 dB  SI-SNR(mix)=3.0566±19.5073 dB  SI-SNRi=-3.2335±10.8002 dB  SDRi=-12.5626±53.1724 dB  PESQi=-0.0131±0.1645  STOIi=-0.0397±0.0883
#[skip] merged: missing mixture scp /maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2/stage103_test_export_300utts/merged/mix/mix.scp (re-run stage 103 export after updating infer_test_export_wav.py).
#[ALL utterance-weighted] SI-SNRi_mean=-1.3126 dB


# ============================================================================
# Stage 107 — 2-speaker-only, VoxCeleb2+LRS3 (no Chinese_lips), clean dry mix.
# ----------------------------------------------------------------------------
# Data (vs generic multi-speaker pipeline):
#   * **Exactly 2 speakers** per mixture (``--force_two_speaker_only`` /
#     ``num_speakers_max=2`` / distribution 0,1,0 inside the trainer).
#   * **Chinese_lips off** — only VoxCeleb2 + LRS3 inventories.
#   * **No MUSAN noise, no RIR / reverb** — ``--online_mix_clean_dry`` zeros
#     SNR/gain jitter + forces ``noise_prob`` / ``reverb_prob`` to 0 in dataset args;
#     CLI still sets ``--noise_prob 0 --reverb_prob 0`` for clarity.
#
# Runtime: match **throughput knobs from stage 102** (logged fast path):
#   ``dataloader_num_workers=2``, ``prefetch_factor=1`` (trainer), ``visual_resize=224``,
#   ``visual_max_frames=75``, ``visual_decode_cuda=true``.
#
# Note: CUDA + ``num_workers>0`` uses **spawn** workers. The tqdm line can sit at ``0/200000``
# for **many minutes** while each worker imports the stack and initializes CUDA/TorchCodec — this
# matches the trainer log line about spawn + first-step lag; it is **not** a 2-spk vs 4-spk logic bug.
#
# If ``ulimit -n`` stays at **1024** after the raise attempt below, ``torch.multiprocessing``
# ``file_system`` sharing + large video batches can **deadlock or crawl**; fix the hard/soft nofile
# limit in the shell/K8s/container, or temporarily use ``--dataloader_num_workers 0``.
#
# Diagnose long stalls at 0/200000: run with
#   WESPE_LOG_DATASET_TIMING=1 bash run_md_sribd.sh --stage 107 --stop-stage 107
# Look for ``[dataset_timing]`` — if audio+first decode takes minutes, try
# ``--visual_decode_cuda false`` (CPU decode in workers, avoids NVDEC contention on cuda:0).
#
# Run:
#   bash run_md_sribd.sh --stage 107 --stop-stage 107 \\
#     > logs/run_md_sribd_stage107.log 2>&1 &
if [ ${stage} -le 107 ] && [ ${stop_stage} -ge 107 ]; then
#echo "[stage107] open files (soft) before ulimit: $(ulimit -n)"
#if ! ulimit -n 65535 2>/dev/null; then
#  echo "[stage107] WARN: ulimit -n 65535 failed (need higher hard limit or root). Current: $(ulimit -n)" >&2
#fi
#echo "[stage107] open files (soft) after ulimit: $(ulimit -n)"
#_cur=$(ulimit -n)
#if [[ "$_cur" =~ ^[0-9]+$ ]] && (( _cur < 8192 )); then
#  echo "[stage107] WARN: fd limit $_cur is low for workers+file_system; expect slow/hung first batch. Raise nofile or use --dataloader_num_workers 0." >&2
#fi
# file_system 模式下，每个 Tensor 块都会占用一个文件句柄；训练数据与 video batch 大时 1024 不够
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_two_spk_clean_det
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --train_speaker_fraction 0.9 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 15000 \
    --force_two_speaker_only true \
    --num_speakers_max 2 \
    --num_speakers_distribution 0,1,0 \
    --online_mix_clean_dry true \
    --online_mix_deterministic false \
    --noise_prob 0 \
    --reverb_prob 0 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --dataloader_num_workers 2 \
    --visual_decode_cuda true \
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 5000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.2 \
    --mrstft_warmup_steps 0 \
    --cue_discrim_weight 0.3 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.1 \
    --passthrough_penalty_threshold 5.0 \
    --eval_loss_type SISDR_MRSTFT \
    --eval_clearervoice_metrics false \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri false \
    --eval_extra_stats true \
    --visual_cue_dropout 0.0 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75 \
    --visual_resize 224
fi

# Stage 108 — reproduction of pre-fix online AV behaviour (lip decode not tied to audio chunk windows).
#   Use --online_av_align false. See stage 109 for the aligned pipeline.
# Run: ... --stage 108 --stop-stage 108 ...
if [ ${stage} -le 108 ] && [ ${stop_stage} -ge 108 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_v2_wo_chinese_lip
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 300 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.0 \
    --online_mix_deterministic true \
    --num_speakers_max 2\
    --num_speakers_distribution 0.1,0.9,0\
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 5000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.2 \
    --mrstft_warmup_steps 0 \
    --cue_discrim_weight 0.3 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.1 \
    --passthrough_penalty_threshold 5.0 \
    --eval_loss_type SISDR_MRSTFT \
    --eval_extra_stats true \
    --eval_clearervoice_metrics true \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri true \
    --visual_cue_dropout 0.0 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75 \
    --online_av_align false
fi

# Stage 109 — lip–audio aligned online mix pipeline (recommended for visual TSE).
# Uses --online_av_align true: processor_new + processor_visual_new inside
# wesep/dataset/online_multi_dataset.py (cropped MP4 timelines match wav_spk{i} chunks).
# Run: nohup bash run_md_sribd.sh --stage 109 --stop-stage 109 \\
#        > logs/run_md_sribd_stage109.log 2>&1 &
if [ ${stage} -le 109 ] && [ ${stop_stage} -ge 109 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_v2_cue_discrim_stage109_av_align
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 300 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.0 \
    --online_mix_deterministic true \
    --online_av_align true \
    --num_speakers_max 2\
    --num_speakers_distribution 0.1,0.9,0\
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 5000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.2 \
    --mrstft_warmup_steps 0 \
    --cue_discrim_weight 0.3 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.1 \
    --passthrough_penalty_threshold 5.0 \
    --eval_loss_type SISDR_MRSTFT \
    --eval_extra_stats true \
    --eval_clearervoice_metrics true \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri true \
    --visual_cue_dropout 0.0 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi

if [ ${stage} -le 110 ] && [ ${stop_stage} -ge 110 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage109_av_align
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 300 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.0 \
    --online_mix_deterministic true \
    --online_av_align true \
    --num_speakers_max 2\
    --num_speakers_distribution 0.1,0.9,0\
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 5000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.0 \
    --mrstft_warmup_steps 0 \
    --cue_discrim_weight 0.0 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.1 \
    --passthrough_penalty_threshold 5.0 \
    --eval_loss_type SISDR \
    --eval_extra_stats true \
    --eval_clearervoice_metrics true \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri true \
    --visual_cue_dropout 0.0 \
    --use_ema true \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi
if [ ${stage} -le 111 ] && [ ${stop_stage} -ge 111 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage111_av_align
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 300 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.0 \
    --online_mix_deterministic true \
    --online_av_align true \
    --num_speakers_max 2\
    --num_speakers_distribution 0,1,0\
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 2 \
    --learning_rate 1.0e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 3000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.0 \
    --mrstft_warmup_steps 0 \
    --cue_discrim_weight 0.001 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.2 \
    --passthrough_penalty_threshold 3.0 \
    --eval_loss_type SISDR \
    --eval_extra_stats true \
    --eval_clearervoice_metrics true \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri true \
    --visual_cue_dropout 0.0 \
    --use_ema false \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi



if [ ${stage} -le 112 ] && [ ${stop_stage} -ge 112 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage111_av_align
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 300 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.0 \
    --online_mix_deterministic true \
    --online_av_align true \
    --num_speakers_max 2\
    --num_speakers_distribution 0,1,0\
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 2 \
    --learning_rate 1.0e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 3000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.0 \
    --mrstft_warmup_steps 0 \
    --cue_discrim_weight 0.001 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.2 \
    --passthrough_penalty_threshold 3.0 \
    --eval_loss_type SISDR \
    --eval_extra_stats true \
    --eval_clearervoice_metrics true \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri true \
    --visual_cue_dropout 0.0 \
    --use_ema false \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75 \
    --visual_cue_ablation mute \
    --visual_cue_ablation_scope eval
fi



if [ ${stage} -le 114 ] && [ ${stop_stage} -ge 114 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage111_av_align_debug2
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 3000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.0 \
    --online_mix_deterministic true \
    --online_av_align true \
    --num_speakers_max 2\
    --num_speakers_distribution 0,1,0\
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 2 \
    --learning_rate 1.0e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 3000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.1 \
    --mrstft_warmup_steps 5000 \
    --cue_discrim_weight 0.001 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.5 \
    --passthrough_penalty_threshold 0.0 \
    --eval_loss_type SISDR \
    --eval_extra_stats true \
    --eval_clearervoice_metrics true \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri true \
    --visual_cue_dropout 0.15 \
    --use_ema false \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi



if [ ${stage} -le 115 ] && [ ${stop_stage} -ge 115 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage111_av_align_big_batch_lr
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 3000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.0 \
    --online_mix_deterministic true \
    --online_av_align true \
    --num_speakers_max 2\
    --num_speakers_distribution 0,1,0\
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 2 \
    --learning_rate 3.0e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 3000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.1 \
    --mrstft_warmup_steps 5000 \
    --cue_discrim_weight 0.001 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.5 \
    --passthrough_penalty_threshold 0.0 \
    --eval_loss_type SISDR \
    --eval_extra_stats true \
    --eval_clearervoice_metrics true \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri true \
    --visual_cue_dropout 0.15 \
    --use_ema false \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi


if [ ${stage} -le 116 ] && [ ${stop_stage} -ge 116 ]; then
data_path_of_voxceleb2=/F00120240032/voxceleb2_zk_mixture/mp4/train
exp_dir=/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage111_av_align_big_batch_lr_with_noise_reverb
num_gpus=1
export TENSORBOARD_LOGGING_DIR=$exp_dir/logs
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_with_transformers_multi_data_online.py \
    --model_config confs/tse_bsrnn_visual_model_v2.yaml \
    --mp4_dir_of_voxceleb2 $data_path_of_voxceleb2 \
    --use_voxceleb2 True \
    --use_lrs3 True \
    --use_chinese_lips False \
    --lrs3_root /F00120240032/lrs3/trainval \
    --chinese_lips_root /F00120240032/Chinese_lips \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal false \
    --sample_num_per_epoch 60000 \
    --sample_num_per_epoch_val 3000 \
    --noise_lmdb_file /maduo/exp/wesep_wenet/datasets/voxceleb2mix/data/musan/lmdb \
    --noise_prob 0.3\
    --reverb_prob 0.2\
    --online_mix_deterministic true \
    --online_av_align true \
    --num_speakers_max 2\
    --num_speakers_distribution 0,1,0\
    --max_steps 200000 \
    --num_train_epochs 200000 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 2 \
    --learning_rate 3.0e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 3000 \
    --weight_decay 0.01 \
    --max_grad_norm 5.0 \
    --mrstft_weight 0.1 \
    --mrstft_warmup_steps 5000 \
    --cue_discrim_weight 0.001 \
    --cue_discrim_warmup_steps 15000 \
    --cue_discrim_temperature 8.0 \
    --passthrough_penalty_weight 0.5 \
    --passthrough_penalty_threshold 0.0 \
    --eval_loss_type SISDR \
    --eval_extra_stats true \
    --eval_clearervoice_metrics true \
    --eval_audio_sr 16000 \
    --train_log_clearervoice_sisnri true \
    --visual_cue_dropout 0.15 \
    --use_ema false \
    --ema_decay 0.999 \
    --logging_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 true \
    --visual_max_frames 75
fi

