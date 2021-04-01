#!/bin/sh
### Modelarts Platform train command
export TF_CPP_MIN_LOG_LEVEL=2               ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0        ## Print log on terminal on(1), off(0)

code_dir=${1}
data_dir=${2}
result_dir=${3}
obs_url=${4}

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

#start exec
python3.7 ${code_dir}/train.py  \
	--iterations_per_loop=100 \
	--batch_size=256 \
	--data_dir=${data_dir} \
	--obs_dir=${obs_url} \
	--mode=train \
	--chip='npu' \
	--platform='modelarts' \
	--npu_profiling=True \
	--checkpoint_dir=${result_dir} \
	--max_train_steps=50 \
	--lr=0.015 \
	--log_dir=${result_dir}  2>&1 | tee ${result_dir}/${current_time}_npu.log

if [ $? -eq 0 ] ;
then
    echo "turing train success"
else
    echo "turing train fail"
fi

