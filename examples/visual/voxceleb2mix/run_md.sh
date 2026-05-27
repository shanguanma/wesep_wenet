#!/bin/bash

# Copyright 2026 Ke Zhang (kylezhang1118@gmail.com)
#
. ./path.sh || exit 1

# General configuration
stage=1
stop_stage=1000

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  # Data preparation related
data=/share/workspace3/maduo/datasets/voxceleb2mix/data
mkdir -p $data
noise_type="clean"
data_type="raw" # shard/raw
mix_data_path=/share/workspace/shared_datasets/speechdata/02_voxceleb2/01_voxceleb2_zk/mixture #/YourPATH/voxceleb2/mixture

  ./local/prepare_data.sh --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 2
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${data_type}" = "shard" ]; then
  echo "Making shards from samples.jsonl ..."
  data=/share/workspace3/maduo/datasets/voxceleb2mix/data
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
  data=/share/workspace3/maduo/datasets/voxceleb2mix/data
  data_type="raw" # shard/raw
  noise_type="clean"
  data=${data}/${noise_type}
  # Training related
  gpus="[0]"
  config=confs/tse_bsrnn_visual.yaml
  exp_dir=/share/workspace3/maduo/exp/TSE_BSRNN_VIS
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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  # Model average related
  num_avg=10
  avg_model=$exp_dir/models/avg_best_model.pt
  python wesep/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg} \
    --mode best \
    --epochs "138,141"
fi
if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint="${exp_dir}/models/avg_best_model.pt"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Start inferencing ..."
  data=/share/workspace3/maduo/datasets/voxceleb2mix/data
  data_type="raw" # shard/raw
  noise_type="clean"
  data=${data}/${noise_type}
  exp_dir=/share/workspace3/maduo/exp/TSE_BSRNN_VIS
  save_results=True
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
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Start scoring ..."
  gpus="[0]"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  # Inferencing and scoring related
  use_pesq=true
  use_dnsmos=true
  dnsmos_use_gpu=true
  fs=16k
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
   real_data=/share/workspace/shared_datasets/speechdata/14_musan
   dest_dir=/share/workspace3/maduo/datasets/voxceleb2mix/data
   mkdir $dest_dir/musan
  find -L ${real_data} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >$dest_dir/musan/wav.scp
  # Convert all musan data to LMDB
  echo "conver musan data to LMDB ..."
  python tools/make_lmdb.py ${dest_dir}/musan/wav.scp ${dest_dir}/musan/lmdb
	
fi




if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Start training with musan..."
  data=/share/workspace3/maduo/datasets/voxceleb2mix/data
  data_type="raw" # shard/raw
  noise_type="clean"
  data=${data}/${noise_type}
  # Training related
  gpus="[0]"
  config=confs/tse_bsrnn_visual_with_musan.yaml
  exp_dir=/share/workspace3/maduo/exp/TSE_BSRNN_VIS_with_musan
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
data=/share/workspace3/maduo/datasets/voxceleb2mix/data/clean   # 按你机器改
exp_dir=/share/workspace3/maduo/exp/TSE_BSRNN_VIS_with_musan_debug
num_gpus=1
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_md.py \
    --config confs/tse_bsrnn_visual_with_musan.yaml \
    --data_type raw \
    --train_data $data/train/raw.list \
    --val_data $data/val/raw.list \
    --train_samples $data/train/samples.jsonl \
    --val_samples $data/val/samples.jsonl \
    --train_cues $data/train/cues.yaml \
    --val_cues $data/val/cues.yaml \
    --exp_dir $exp_dir \
    --gpus "0" \
    --num_epochs 150 \
    --optim_class AdamW \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --clip_grad 5.0 \
    --seed 42
fi


if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ]; then
data=/share/workspace3/maduo/datasets/voxceleb2mix/data/clean   
exp_dir=/share/workspace3/maduo/exp/online_avcrossnet_mamba_with_musan
num_gpus=1
export OMP_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train_md.py \
    --config confs/tse_bsrnn_visual_with_musan.yaml \
    --data_type raw \
    --train_data $data/train/raw.list \
    --val_data $data/val/raw.list \
    --train_samples $data/train/samples.jsonl \
    --val_samples $data/val/samples.jsonl \
    --train_cues $data/train/cues.yaml \
    --val_cues $data/val/cues.yaml \
    --exp_dir $exp_dir \
    --tse_model 'TSE_ONLINE_AVCROSSNET_MAMBA_VISUAL'\
    --loss_type 'OnlineAVCrossNetLoss'\
    --gpus "0" \
    --num_epochs 150 \
    --optim_class AdamW \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --clip_grad 5.0 \
    --seed 42
fi


if [ ${stage} -le 117 ] && [ ${stop_stage} -ge 117 ]; then
data_path_of_voxceleb2=/share/workspace/shared_datasets/speechdata/02_voxceleb2/01_voxceleb2_zk/mp4/train
lrs3_root=/share/workspace/shared_datasets/speechdata/11_lrs3/01_lrs3_ljj/mp4/trainval
exp_dir=/share/workspace3/maduo/exp/wesep_wenet/TSE_BSRNN_VISUAL_stage111_av_align_big_batch_lr_separator_causal_true
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
    --lrs3_root $lrs3_root \
    --chinese_lips_root "" \
    --train_speaker_fraction 0.8 \
    --output_dir $exp_dir \
    --tse_model 'TSE_BSRNN_VISUAL' \
    --loss_type 'SISDR_MRSTFT' \
    --visual_frontend "muse" \
    --separator_causal true \
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

