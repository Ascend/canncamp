#!/bin/sh
EXEC_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "===>>>Python boot file dir: ${EXEC_DIR}"

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

#start exec
python3.7 ${EXEC_DIR}/train.py  \
	--iterations_per_loop=100 \
	--batch_size=256 \
	--data_dir=/root/imagenet2012 \
	--mode=train \
	--chip='gpu' \
	--platform='linux' \
	--checkpoint_dir=./model_1p/ \
	--max_train_steps=200 \
	--lr=0.015 \
	--log_dir=./model_1p  2>&1 | tee ${EXEC_DIR}/${current_time}_npu.log

if [ $? -eq 0 ] ;
then
    echo "turing train success"
else
    echo "turing train fail"
fi

