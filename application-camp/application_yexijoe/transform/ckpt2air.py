"""
yexijoe; ZJUT, CETC36.
-*- coding:utf-8 -*-
Process the initial IQ signal dataset, transform it to MindRecord which can be read by MindDataset.
"""

import numpy as np
import resnet_mindspore_own as resnet_model
from mindspore import Tensor, export, load_checkpoint, load_param_into_net

class_num = 11  # 类别数
resnet = resnet_model.resnet50(class_num)
# load the parameter into net
load_checkpoint("checkpoint_lenet-20_859.ckpt", net=resnet)
input = np.random.uniform(0.0, 1.0, size=[1, 1, 2, 128]).astype(np.float32)
export(resnet, Tensor(input), file_name='checkpoint_lenet-20_859', file_format='AIR')
