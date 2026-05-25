# module av
module load cuda/12.8
module load cudnn/9.10.2-cuda12
#module load ffmpeg/latest
. "/share/software/anaconda3/2024.10/etc/profile.d/conda.sh"
conda activate  online_avcrossnet_mamba

#export PYTHONWARNINGS="ignore"

export PATH=$PWD:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../../:$PYTHONPATH
