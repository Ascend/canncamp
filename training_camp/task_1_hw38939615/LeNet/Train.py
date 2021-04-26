import os
import tensorflow as tf
import tensorflow_core.examples.tutorials.mnist.input_data as input_data
import config as cfg
from lenet import Lenet
import moxing as mox
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

def main():
    # 将obs的训练数据拷贝到modelarts
    mox.file.copy_parallel(src_url="obs://canncamps-hw38939615/MNIST_data/", dst_url="MNIST_data")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
    sess = tf.Session(config=config)

    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER

    saver = tf.train.Saver()
    if os.path.exists(parameter_path):
        saver.restore(parameter_path)
    else:
        sess.run(tf.initialize_all_variables())

    for i in range(max_iter):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy, train_loss = sess.run([lenet.train_accuracy, lenet.loss], feed_dict={
                lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]
            })
            print("step %d, training accuracy %g, loss is %g" % (i, train_accuracy, train_loss))
        sess.run(lenet.train_op, feed_dict={lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]})
    save_path = saver.save(sess, parameter_path)
    print("save model in {}".format(save_path))
    # 将训练好的权重拷回到obs
    mox.file.copy_parallel(src_url="checkpoint/", dst_url="obs://canncamps-hw38939615/ckpt")


if __name__ == '__main__':
    main()
