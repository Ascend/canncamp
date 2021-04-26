#!/usr/bin/python3.6
"""
@Author: xiaxianyi<529379497@qq.com>
@Time: 2021/4/5 9:11
@File: test.py
Description:
"""
import tensorflow as tf
from Inference import inference
import tensorflow_core.examples.tutorials.mnist.input_data as input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    infer = inference()
    test_data = mnist.test
    num_examples = test_data._num_examples
    count = 0
    for i in range(num_examples):
        batch = mnist.train.next_batch(1)
        pred = infer.predict(batch[0])
        label = batch[1]
        if pred[0] == label[0]:
            count += 1
    print("test data accuracy: ", count / 10000)

