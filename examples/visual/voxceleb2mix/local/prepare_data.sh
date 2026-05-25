#!/bin/bash
# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#

stage=-1
stop_stage=-1

mix_data_path='./voxceleb2/mixture/'

data=data
noise_type=clean
num_spk=2

. tools/parse_options.sh || exit 1

real_data=$(realpath ${data})

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare the meta files for the datasets (JSONL)"

  for dataset in val test train; do
    echo "Preparing JSONL for" $dataset

    dataset_path=$mix_data_path/$dataset/mix
    out_dir="${real_data}/${noise_type}/${dataset}"
    mkdir -p "${out_dir}"

    python local/scan_voxceleb2mix.py \
      "${dataset_path}" \
      --outfile "${out_dir}/samples.jsonl"

    ln -sf samples.jsonl "${out_dir}/raw.list"

  done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Build fixed visual cues for datasets from samples.jsonl"

  for dset in train val test; do
    mix_index="${real_data}/${noise_type}/${dset}/samples.jsonl"
    out_dir="${real_data}/${noise_type}/${dset}/cues"
    mkdir -p "${out_dir}"

    # 1) Generate cues/visual.json
     if [ "$dset" == "test" ]; then
        mp4_split="test"
      else
        mp4_split="train"
      fi

      python local/build_visual_cues.py \
        --samples_jsonl "${mix_index}" \
        --mp4_root "${mix_data_path}/../mp4/${mp4_split}" \
        --outfile "${out_dir}/visual.json"

    # 2) Generate cues.yaml
cat > ${data}/${noise_type}/${dset}/cues.yaml << EOF
cues:
  visual:
    type: raw_mp4
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${data}/${noise_type}/${dset}/cues/visual.json
EOF
  done
fi


# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#   echo "Download the pre-trained speaker encoders (Resnet34 & Ecapa-TDNN512) from wespeaker..."
#   mkdir wespeaker_models
#   wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.zip
#   unzip voxceleb_resnet34.zip -d wespeaker_models
#   wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.zip
#   unzip voxceleb_ECAPA512.zip -d wespeaker_models
# fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  if [ ! -d "${real_data}/raw_data/musan" ]; then
    mkdir -p ${real_data}/raw_data/musan
    #
    echo "Downloading musan.tar.gz ..."
    echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."
    wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${real_data}/raw_data
    md5=$(md5sum ${real_data}/raw_data/musan.tar.gz | awk '{print $1}')
    [ $md5 != "0c472d4fc0c5141eca47ad1ffeb2a7df" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1

    echo "Decompress all archives ..."
    tar -xzvf ${real_data}/raw_data/musan.tar.gz -C ${real_data}/raw_data

    rm -rf ${real_data}/raw_data/musan.tar.gz
  fi

  echo "Prepare wav.scp for musan ..."
  mkdir -p ${real_data}/musan
  find -L ${real_data}/raw_data/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${real_data}/musan/wav.scp

  # Convert all musan data to LMDB
  echo "conver musan data to LMDB ..."
  python tools/make_lmdb.py ${real_data}/musan/wav.scp ${real_data}/musan/lmdb
fi
