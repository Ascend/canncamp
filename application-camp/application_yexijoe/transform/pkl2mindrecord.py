"""
yexijoe; ZJUT, CETC36.
-*- coding:utf-8 -*-
Process the initial IQ signal dataset, transform it to MindRecord which can be read by MindDataset.
"""

import numpy as np
import pickle
import os
from tqdm import tqdm
from mindspore.mindrecord import FileWriter


# get train dataset and test dataset
dataset_path = "/home/huawei/data/IQ_signal/RML2016.10b.dat"
Xd = pickle.load(open(dataset_path, 'rb'), encoding='iso-8859-1')
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
train_2 = X[train_idx]  # (110000, 2, 128), float32
test_2 = X[test_idx]  # (110000, 2, 128), float32
train_label = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx))).astype(np.int32)  # (110000,), int32
test_label = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx))).astype(np.int32)  # (110000,), int32
print(train_2.shape, train_2.dtype, train_label.shape, train_label.dtype,
      test_2.shape, test_2.dtype, test_label.shape, test_label.dtype)

# get train MindRecord
MINDRECORD_FILE = "/home/huawei/data/IQ_signal/RML2016.10b_train.mindrecord"
if os.path.exists(MINDRECORD_FILE):
    os.remove(MINDRECORD_FILE)
    os.remove(MINDRECORD_FILE + ".db")
writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)
npy_schema = {"data": {"type": "float32",
                       "shape": [1, train_2.shape[1], train_2.shape[2]]},
              "label": {"type": "int32"}}
writer.add_schema(npy_schema, "it is a RML2016.10b IQ signal train dataset")
data = []
for i in tqdm(range(train_2.shape[0])):
    sample = {"data": train_2[i:i+1, :, :], "label": train_label[i]}
    data.append(sample)
    i += 1
    if i % 1000 == 0:
        writer.write_raw_data(data)
        data = []
if data:
    writer.write_raw_data(data)
writer.commit()

# get test MindRecord
MINDRECORD_FILE = "/home/huawei/data/IQ_signal/RML2016.10b_test.mindrecord"
if os.path.exists(MINDRECORD_FILE):
    os.remove(MINDRECORD_FILE)
    os.remove(MINDRECORD_FILE + ".db")
writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)
npy_schema = {"data": {"type": "float32",
                       "shape": [1, test_2.shape[1], test_2.shape[2]]},
              "label": {"type": "int32"}}
writer.add_schema(npy_schema, "it is a RML2016.10b IQ signal test dataset")
data = []
for i in tqdm(range(test_2.shape[0])):
    sample = {"data": test_2[i:i+1, :, :], "label": test_label[i]}
    data.append(sample)
    i += 1
    if i % 1000 == 0:
        writer.write_raw_data(data)
        data = []
if data:
    writer.write_raw_data(data)
writer.commit()
