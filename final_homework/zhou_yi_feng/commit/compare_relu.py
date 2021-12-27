import logging            # Python标准库日志模块
import tensorflow as tf   # 导入TensorFlow开源库
from npu_bridge.estimator import npu_ops   # 导入TensorFlow开源库中的npu_ops模块
import numpy as np    # 导入Python的数学基础库
# np.allclose比较函数的相对公差参数
atol = 0.001
# np.allclose比较函数的绝对公差参数
rtol = 0.001


def config(excute_type):
    if excute_type == 'ai_core':
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = True   # 开启数据预处理下沉到Device侧执行
        custom_op.parameter_map["mix_compile_mode"].b = True
        custom_op.parameter_map["use_off_line"].b = True     # True表示在昇腾AI处理器上执行训练
        custom_op.parameter_map["enable_exception_dump"].i = 1

    elif excute_type == 'cpu':
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

    return session_config


def main(unused_argv):
    shape_params = (2, 2, 2)
    dtype_params = "float16"

    # 构造Relu算子的输入数据,shape为shape_params，范围在[-2,2]之间的随机数
    x_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)

    # 对Relu算子的输入数据进行占位
    x = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    
    # 计算算子输出
    out = tf.nn.leaky_relu(x)
    # 在Host侧CPU上运行单算子，得到期望运行结果
    with tf.compat.v1.Session(config=config('cpu')) as session:
        result_cpu = session.run(out, feed_dict={x: x_data})

    print('===========result_cpu===============')
    print(result_cpu)
    # 在昇腾AI处理器上运行单算子，得到实际运行结果
    with tf.compat.v1.Session(config=config('ai_core')) as session:
        result_ai_core = session.run(out, feed_dict={x: x_data})
    print('===========result_ai_core===============')
    print(result_ai_core)

    np.array(result_ai_core).astype(dtype_params)
    np.array(result_cpu).astype(dtype_params)
    print('====================================')
    # 通过np.allclose比较昇腾AI处理器上运行的实际结果和cpu上运行的期望结果，其中atol和rtol为np.allclose比较函数的相对公差参数和绝对公差参数，请见步骤3设置。
    cmp_result = np.allclose(result_ai_core, result_cpu, atol, rtol)
    print(cmp_result)
    print('===============verify success===============')


if __name__ == "__main__":
    tf.app.run()


