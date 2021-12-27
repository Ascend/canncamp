#  导入包
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
from npu_bridge.npu_init import *


# 引入minist数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 每个批次的大小
batch_size = 100
# 计算需要多少批次
n_batch = mnist.train.num_examples // batch_size

output_node_names = "Sigmoid"
output_graph = '/home/ma-user/AscendProjects/MyTraining/models/lenet.pb'

def config():
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,)
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True     # True表示在昇腾AI处理器上执行训练
    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    return session_config

#定义初始化函数
# 初始化权值
def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)  # 生成截断正态分布
    return tf.Variable(initial_value=initial)
# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial)
# 卷积层
def conv2d(x, W):
     '''
     :param x: input tensor of shape [batch,in_height_in_weight,in_channels]
     :param W: filter tensor of shape [filter_height,filter_width,in_channels,out_channels]
     :return:
     步长： strides[0] = strides[3] = 1 ，strides[1]代表x方向的步长，strides[2]代表x方向的步长
     padding : 两种填充方式，分别是SAME和VALID
     '''
     return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
# 池化层
def max_pool_2x2(x):
    # ksize [1,x,y,1] x,y 为池化窗口大小
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def main(unused_argv):
    # 定义placeholder
    # 定义placeholder
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])
    # 改变x的格式转为4D向量[batch,in_height_in_weight,in_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1是批次，暂时的-1，后面会变成100；1是指黑白图片（维度为1）
    """
    第一步 卷积
    与6个高为5，宽为5，深度为1的卷积核卷积，然后将卷积结果的每一个深度进行偏置，最后将结果经过激活函数处理，得到结果的尺寸高为28、宽为28、深度为6。其中卷积核都满足高斯随机分布的随机数生成，偏置设为常数
    """
    # 初始化第一个卷积层的权值和偏置
    W_conv1 = weight_variable([5, 5, 1, 6])  # 5*5采样窗口，32个卷积核从1个平面抽取特征
    b_conv1 = bias_variable([6])  # 每一个卷积核一个偏置值
    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用relu激活函数
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    """
    第二步 池化
    进行2×2的valid的最大池化，池化后的尺寸高为14、宽为14、深度为6。
    """
    h_pool1 = max_pool_2x2(h_conv1)  # 进行池化
    """
    第三步 卷积2
    先与16个高为5、宽为5、深度为6的卷积核SAME卷积，然后将卷积结果在每一个深度上加偏置，最后将加上偏置的结果经过激活函数。结果得到的是尺寸高为10、宽为10、深度为16。
    """
    # 初始化第二个卷积层的权值和偏置
    W_conv2 = weight_variable([5, 5, 6, 16])  # 5*5采样窗口，32个卷积核从32个平面抽取特征
    b_conv2 = bias_variable([16])  # 每一个卷积核一个偏置值
    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用relu激活函数
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    """
    第四步 池化2
    同第一次池化
    """
    h_pool2 = max_pool_2x2(h_conv2)  # 进行池化
    """
    第五步 全连接
    将第四步得到的结果拉伸为1个一维向量，其长度为7 ∗ 7 ∗ 16 = 784 7*7*16=7847∗7∗16=784,然后将这个向量经过一个全连接神经网络处理，该全连接网络有两个隐藏层，其中输出层有784个神经元，第一个隐藏层有120个，第二个有10个，然后输出层有10个（因为要处理的数据有10类别）
    """
    # 池化层的2D输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
    # 初始化第一个全连接层的权值
    W_fc1 = weight_variable([7 * 7 * 16, 120])  # 上一层有7*7*16个神经元，全连接层有120个神经元
    b_fc1 = bias_variable([120])  # 120 个节点
    # 求第一个全连接层的输出
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 初始化第二个全连接层
    W_fc2 = weight_variable([120, 84])  # 上一层有120个神经元
    b_fc2 = bias_variable([84])  # 84 个节点
    # 求第二个全连接层的输出
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # 初始化输出层
    W_fc3 = weight_variable([84, 10])  # 上一层有120个神经元
    b_fc3 = bias_variable([10])  # 84 个节点
    # 求第输出层的输出
    h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3)  # 输出层使用Sigmod作为激活函数输出
    # 梯度下降算法优化
    # 交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc3))
    # 使用AdamOptimizer进行优化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 结果存放在布尔型列表
    correct_prediction = tf.equal(tf.argmax(h_fc3, 1), tf.argmax(y, 1))  # argmax返回张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saveFile = "/home/ma-user/AscendProjects/MyTraining/models/lenet"
    saver = tf.train.Saver()
    # 创建Tensorflow会话
    with tf.Session(config=config()) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(5):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
                pass
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Iter" + str(epoch) + ',Testing Accuracy=' + str(acc))
        saver.save(sess, saveFile)
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == "__main__":
    tf.app.run()

