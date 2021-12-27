#  导入包
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
from npu_bridge.npu_init import *

output_graph = 'fc_mnist.pb'
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

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

def add_layer(input,input_size,output_size,activation_function=None):
    w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))
    Z = tf.add(tf.matmul(input, w), b)
    if activation_function == None:
        a = Z
    else:
        a = activation_function(Z)#a = f(z)
    return a

def predict(output,label):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def crossentropy(label,logit):
    return -tf.reduce_sum(tf.multiply(label,tf.log(logit)),reduction_indices=[1])

input_images = tf.placeholder(tf.float32,[None,784], name='input')
label = tf.placeholder(tf.float32,[None,10])

fc_layer = add_layer(input_images,784,16,tf.nn.leaky_relu)
output_layer = add_layer(fc_layer,16,10)
output = tf.nn.softmax(output_layer, name='output')

accuracy = predict(output,label)
loss = tf.reduce_mean(crossentropy(label, output))
op = tf.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(loss)

with tf.Session(config=config()) as sess:
    sess.run(tf.global_variables_initializer())
    for i in  range(10000):
        batch_image,batch_label = mnist.train.next_batch(128)
        sess.run(op, feed_dict={input_images: batch_image, label: batch_label})
        
        if i % 1000 == 0:
            train_acc = sess.run(accuracy, feed_dict={input_images: mnist.train.images, label: mnist.train.labels})
            test_acc, test_loss = sess.run([accuracy, loss], 
                                           feed_dict={input_images:mnist.test.images, label:mnist.test.labels})
            
#            import numpy as np
#            results = sess.run([output], feed_dict={input_images:mnist.test.images})
#            results = results[0]
#            labels = mnist.test.labels
#            total = len(results)
#            correct = 0.0
#            for j in range(len(results)):
#                if np.argmax(results[j])==np.argmax(labels[j]):
#                    correct+=1.0
#            print(f'test acc is {correct/total}')
            
            print(f"step {i},train_acc={train_acc} ,test_acc={test_acc},test_loss={test_loss}")
    
    output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess=sess,
        input_graph_def=sess.graph_def,  # 等于:sess.graph_def
        output_node_names=['output'])  # 如果有多个输出节点，以逗号隔开

    with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
    print(f'save {output_graph} finished')



