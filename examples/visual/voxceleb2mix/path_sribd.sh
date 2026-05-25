. /maduo/package/cuda12.6_path.sh

. /maduo/miniconda3/etc/profile.d/conda.sh

# if you want to rename env name, you can follow the commands
# conda deactivate # 退出当前环境到base 环境
# conda rename -n old_name new_name
conda activate wesep_py310_cu126
export PATH=$PWD:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../../:$PYTHONPATH

