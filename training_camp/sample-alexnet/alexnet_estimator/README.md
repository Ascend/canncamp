# Alexnet for Tensorflow 

This repository provides a script and recipe to train the AlexNet model .

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
  * [Optimizer](#Optimizer)
  * [Data augmentation](#Data-augmentation)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Key configuration changes](#Key-configuration-changes)
  * [Running the example](#Running-the-example)
    * [Training](#Training)
* [Advanced](#advanced)
  * [Command-line options](#Command-line-options)   

## Description

- AlexNet model from: `Alex Krizhevsky. "One weird trick for parallelizing convolutional neural networks". <https://arxiv.org/abs/1404.5997>.`
- reference implementation: <https://pytorch.org/docs/stable/_modules/torchvision/models/alexnet.html#alexnet>

## Requirements

- Tensorflow CPU/GPU/NPU environmemnt
- Download and preprocess ImageNet2012 dataset for training and evaluation. You can refer to the [link](https://github.com/tensorflow/models/tree/master/research/slim) to prepare the datasets from tensorflow offcial website. By the way, some imagenet2012 tfrecords have been preprocessed, you can download it directly by the [link](https://public-obs.obs.cn-north-4.myhuaweicloud.com/tfrecord_imagnet2012.rar).

## Default configuration

The following sections introduce the default configurations and hyperparameters for AlexNet model.

### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.015
- LR schedule: cosine_annealing
- Batch size : 256 for single NPU 
- Weight decay :  0.0001. 
- Label smoothing = 0.1
- We train for:
  - 150 epochs ->  60.1% top1 accuracy

### Data augmentation

This model uses the following data augmentation:

- For training:
  - RandomResizeCrop, scale=(0.08, 1.0), ratio=(0.75, 1.333)
  - RandomHorizontalFlip, prob=0.5
  - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- For inference:
  - Resize to (256, 256)
  - CenterCrop to (224, 224)
  - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

## Quick start guide

### Prepare the dataset

1. Please download the ImageNet2012 dataset by yourself. 
2. Please convert the dataset to tfrecord format file by yourself. Refer to the [link](https://github.com/tensorflow/models/tree/master/research/slim) to prepare the datasets from tensorflow offcial website. By the way, some imagenet2012 tfrecords have been preprocessed, you can download it directly by the [link](https://public-obs.obs.cn-north-4.myhuaweicloud.com/tfrecord_imagnet2012.rar).

### Running the example

#### Training
- NPU training
--Setting single NPU training parameters,examples are as follows.
Make sure that the "--data_dir" modify the path of the user generated tfrecord.
  ```
  python3.7 ${EXEC_DIR}/train.py  \
  --iterations_per_loop=100 \
  --batch_size=256 \
  --data_dir=/cache/imagenet2012 \
  --mode=train \
  --chip='npu' \
  --platform='modelarts' \
  --checkpoint_dir=./model_1p/ \
  --lr=0.015 \
  --log_dir=./model_1p 
  ```
- GPU training
--Setting single NPU training parameters,examples are as follows.
Make sure that the "--data_dir" modify the path of the user generated tfrecord. You can modify the parame `batch_size` smaller to adapte your hardware.
  ```
  python3.7 ${EXEC_DIR}/train.py  \
  --iterations_per_loop=10 \
  --batch_size=256 \
  --data_dir=/cache/imagenet2012 \
  --mode=train \
  --chip='gpu' \
  --platform='linux' \
  --checkpoint_dir=./model_1p/ \
  --lr=0.015 \
  --log_dir=./model_1p 
  ```

- CPU training
--Setting CPU training parameters,examples are as follows.
Make sure that the "--data_dir" modify the path of the user generated tfrecord. You can modify the parame `batch_size` smaller to adapte your hardware.
  ```
  ## For linux system
  python3.7 ${EXEC_DIR}/train.py  \
  --iterations_per_loop=10 \
  --batch_size=64 \
  --data_dir=/cache/imagenet2012 \
  --mode=train \
  --chip='cpu' \
  --platform='linux' \
  --checkpoint_dir=./model_1p/ \
  --lr=0.015 \
  --log_dir=./model_1p 

  ## For window system
  python.exe .\train.py  --iterations_per_loop=10 --batch_size=64 --data_dir=E:\Dataset\tfrecord_imagnet2012 --mode=train --chip='cpu' --platform='desktop' --checkpoint_dir=.\model_1p --max_train_steps=30 --lr=0.015 --log_dir=.\model_1p
  ```

## Advanced

### Command-line options

```
  --data_dir                        train data dir
  --num_classes                     num of classes in ImageNet（default:1000)
  --image_size                      image size of the dataset
  --batch_size                      mini-batch size (default: 128) per npu
  --chip                            which device your want to execute training
  --pretrained                      path of pretrained model
  --lr                              initial learning rate(default: 0.06)
  --max_epochs                      max number of epoch to train the model(default: 150)
  --warmup_epochs                   warmup epoch(when batchsize is large)
  --weight_decay                    weight decay (default: 1e-4)
  --momentum                        momentum(default: 0.9)
  --label_smoothing                 use label smooth in CE, (default 0.1)
  --save_summary_steps              logging interval(dafault:100)
  --log_dir                         path to save checkpoint and log
  --log_name                        name of log file
  --save_checkpoints_steps          the interval to save checkpoint
  --mode                            mode to run the program (train, evaluate)
  --checkpoint_dir                  path to checkpoint for evaluation
  --max_train_steps                 max number of training steps 
  --synthetic                       whether to use synthetic data or not
  --version                         weight initialization for model
  --do_checkpoint                   whether to save checkpoint or not 
  --rank_size                       local rank of distributed(default: 0)
  --group_size                      world size of distributed(default: 1)
  --max_train_steps                 number of training step , default : None, when set ,it will override the max_epoch
```
for a complete list of options, please refer to `train.py`



