"""
yexijoe; ZJUT, CETC36.
-*- coding:utf-8 -*-
"""

import numpy as np
import pickle
import argparse
import csv
from atlas_utils.acl_resource import AclResource
from atlas_utils.acl_model import Model
import atlas_utils.constants as const
import atlas_utils.utils as utils


MODEL_PATH = "../model/resnet50_export.om"  # .om model path////////////////////////////
DATA_PATH = "../data/RML2016.10a_dict.pkl"  # dataset path////////////////////////////
modulation_type = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']  # 分别对应0—11


def load_data(data_path):
    """
    load and process data
    :param data_path: dataset's path
    :return: test dataset
    """
    Xd = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    X = np.vstack(X)
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    # train_2 = X[train_idx]  # (110000, 2, 128), float32
    test_2 = X[test_idx]  # (110000, 2, 128), float32
    # train_2 = np.expand_dims(train_2, 1)  # (110000, 1, 2, 128)
    test_2 = np.expand_dims(test_2, 1)  # (110000, 1, 2, 128)
    # train_label = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx))).astype(np.int32)  # (110000,), int32
    test_label_ = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx))).astype(np.int32)  # (110000,), int32
    print(f"测试集的形状为：{test_2.shape}, 测试集数据的类型为：{test_2.dtype}, "
          f"测试集类标签的形状为：{test_label_.shape}, 测试集类标签数据的形状为：{test_label_.dtype}")
    return test_2, test_label_


class Classify(object):
    """
    Class for portrait segmentation
    """
    def __init__(self, model_path):
        self._model_path = model_path
        self._model = None

    def init(self):
        """
        Initialize
        """
        # 加载模型
        self._model = Model(self._model_path)

        return const.SUCCESS

    @utils.display_time
    def inference(self, input_data):
        """
        model inference
        """
        return self._model.execute(input_data)


def main(arg):
    """
    main
    """
    # 初始化资源相关语句
    acl_resource = AclResource()
    acl_resource.init()
    # 初始化推理的类
    classify = Classify(MODEL_PATH)
    ret = classify.init()
    utils.check_ret("Classify init ", ret)
    # 得到测试样本及其类标签
    print(f"================正在加载测试集数据================")
    all_test_data, test_label = load_data(DATA_PATH)
    print(f"================测试集数据加载完毕================")
    # print(list(test_label).count(0))  # 类标签为0的测试集样本数量
    # 调制分类推理，依次预测每一个信号的调制类型
    print(f"=====================开始推理=====================")
    for now_num in range(arg.start_num, arg.end_num):
        initial_result = classify.inference(all_test_data[now_num, :, :, :])
        result = initial_result[0]  # 每一类的可能性大小(定性)
        # print(result)
        result = result.flatten()
        # x.argsort(),将x中的元素从小到大排列，返回其对应的索引
        pre_index = result.argsort()[-1]  # 可能性最大的类别的索引
        final_result = modulation_type[pre_index]  # 预测标签，即预测的测试样本的调制类型
        true_label = modulation_type[test_label[now_num]]
        print(f"================编号为{now_num}的信号的预测的调制类型为：{final_result}，实际的调制类型为：{true_label}================")
        with open("./result_modulation_type.csv", 'a', newline='') as t2:
            writer_train2 = csv.writer(t2)
            writer_train2.writerow([now_num, final_result, true_label])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A modified ResNet model for modulation classification of signal dataset')
    parser.add_argument('--start_num', type=int, default=5500, help='The start number of the signal to be inferred from.')
    parser.add_argument('--end_num', type=int, default=5520, help='The end number of the signal you want to infer.')
    args = parser.parse_args()
    # 保存测试结果的csv文件的第一行
    with open("./result_modulation_type.csv", 'a', newline='') as t:
        writer_train1 = csv.writer(t)
        writer_train1.writerow(["测试信号的编号", "预测的调制类型", "实际的调制类型"])
    main(args)
    