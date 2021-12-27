import tensorflow as tf
from tensorflow.python.framework import graph_util
from npu_bridge.npu_init import *
import numpy as np
import matplotlib.pyplot as plt


# generate data and label
def generate_data():
    # generate data
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    x_data = x_data.astype("float32")

    # save data to x_data.bin file
    x_data.tofile("./data/x_data.bin")
    print(x_data.shape)

    # generate label, label = data * data + noise
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise
    y_data = y_data.astype("float32")

    # save label to y_data.bin file
    y_data.tofile("./data/y_data.bin")
    print(y_data.shape)
    return x_data, y_data


if __name__ == '__main__':
    x_data, y_data = generate_data()

    # define model's input
    x = tf.placeholder(tf.float32, [200, 1])
    y = tf.placeholder(tf.float32, [200, 1])

    # define model first layer
    weights_l1 = tf.Variable(tf.random.normal([1, 10]), dtype=tf.float32, name="weights_l1")
    bias_l1 = tf.Variable(tf.random.normal([1, 10]), dtype=tf.float32, name="bias_l1")
    matmul_l1 = tf.matmul(x, weights_l1) + bias_l1
    output_l1 = tf.nn.sigmoid(matmul_l1, name="output_l1_sigmoid")

    # define model second layer
    weights_l2 = tf.Variable(tf.random.normal([10, 1]), dtype=tf.float32, name="weights_l2")
    bias_l2 = tf.Variable(tf.random.normal([1, 1]), dtype=tf.float32, name="bias_l2")
    matmul_l2 = tf.matmul(output_l1, weights_l2) + bias_l2
    prediction = tf.nn.sigmoid(matmul_l2, name="prediction")

    # define loss function
    loss = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
    optimizer = tf.train.AdamOptimizer(0.1)
    train_step = optimizer.minimize(loss)

    # run session on ascend npu
    npu_session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False, )
    custom_op = npu_session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["mix_compile_mode"].b = True
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_exception_dump"].i = 1

    with tf.Session(config=npu_session_config) as sess:

        # session init
        sess.run(tf.global_variables_initializer())

        # train 2000 epoch
        for _ in range(2000):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})

        # save model
        tf.train.Saver().save(sess, "./models/model")

        # ckpt to pb
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=["prediction"])
        with tf.gfile.GFile("./models/model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        # plot prediction
        pred = sess.run(prediction, feed_dict={x: x_data})
        np.array(pred).astype("float32").tofile("./data/pred.bin")

        plt.figure()
        plt.scatter(x_data, y_data, alpha=0.4, c="orange", label="data")
        plt.plot(x_data, pred, "-+", alpha=0.4, c="darkgreen", label="pred")
        plt.legend()
        plt.savefig("result_plot.png")
