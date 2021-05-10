"""
yexijoe; ZJUT, CETC36.
-*- coding:utf-8 -*-
"""

import pickle
import os
import numpy as np
import argparse
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import Normal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.nn.metrics import Accuracy
from mindspore import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.mindrecord import FileWriter
import resnet_mindspore_own as resnet_model


def load_dataset(data_path, batch_size=128, repeat_size=1, num_parallel_workers=3):
    """ load dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    data_set = ds.MindDataset(dataset_file=data_path)
    # buffer_size = 10000
    # data_set = data_set.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_size)
    # count = 0
    # for item in data_set.create_dict_iterator():
    #     count += 1
    # print("sample: {}".format(item))
    # print("Got {} samples".format(count))
    # print(data_set, "==================================================")

    return data_set


def train_net(network_model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    # load training dataset
    ds_train = load_dataset(data_path, args.batch_size, repeat_size)
    network_model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=sink_mode)


def test_net(network, network_model, data_path):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    file_list = os.listdir('./')
    for filename in file_list:
        de_path = os.path.join('./', filename)
        if de_path.endswith('.ckpt'):
            param_dict = load_checkpoint(de_path)
            # load parameter to the network
            load_param_into_net(network, param_dict)
            # load testing dataset
            ds_eval = load_dataset(data_path)
            acc = network_model.eval(ds_eval, dataset_sink_mode=False)
            print("============== Accuracy:{} ==============".format(acc), de_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore ResNet Example for Signal Classification')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--train_data_path', type=str, default="/home/ssd5/data/IQ_signal/RML2016.10a_train.mindrecord")
    parser.add_argument('--test_data_path', type=str, default="/home/ssd5/data/IQ_signal/RML2016.10a_test.mindrecord")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--device_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='device id used (Ascend)')
    parser.add_argument('--class_num', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_epoch', type=float, default=20)
    args = parser.parse_args()
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    dataset_sink_mode = not args.device_target == "CPU"
    # learning rate setting
    lr = args.lr
    momentum = args.momentum
    dataset_size = 1
    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_epoch = args.train_epoch
    # create the network
    print("************** Loading model **************")
    net = resnet_model.resnet50(args.class_num)
    print("************** Load successfully **************")
    # define the optimizer
    net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=int(110000 / args.batch_size), keep_checkpoint_max=10)
    # save the network model and parameters for subsequence fine-tuning
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    # group layers into an object with training and evaluation features
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    train_net(model, train_epoch, args.train_data_path, dataset_size, ckpoint, dataset_sink_mode)
    test_net(net, model, args.test_data_path)
