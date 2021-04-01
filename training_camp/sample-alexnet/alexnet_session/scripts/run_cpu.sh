#!/bin/bash
## for linux terminal
current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3.7 train.py \
        --dataset=/home/Flowers-Data-Set \
        --result=./log \
        --chip='cpu' \
        --num_classes=5 \
        --train_step=5 2>&1 | tee ${current_time}_train_gpu.log