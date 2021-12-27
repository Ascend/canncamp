import tensorflow as tf
from npu_bridge.estimator import npu_ops
import numpy as np


def main():
    # generate twn input for sub ops
    x_data = np.random.uniform(-2, 2, size=(6, 6, 6)).astype("float32")
    y_data = np.random.uniform(-2, 2, size=(6, 6, 6)).astype("float32")

    # define two session input placeholder
    x = tf.compat.v1.placeholder("float32", shape=(6, 6, 6))
    y = tf.compat.v1.placeholder("float32", shape=(6, 6, 6))

    # use subtract to sub x and y
    out = tf.subtract(x, y)

    # run on cpu
    with tf.compat.v1.Session() as session:
        res_cpu = session.run(out, feed_dict={x: x_data, y: y_data})
    print('[INFO] cpu result:')
    print(res_cpu)

    # run on ascend
    npu_session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False, )
    custom_op = npu_session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["mix_compile_mode"].b = True
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_exception_dump"].i = 1

    with tf.compat.v1.Session(config=npu_session_config) as session:
        res_npu = session.run(out, feed_dict={x: x_data, y: y_data})
    print('[INFO] npu result:')
    print(res_npu)

    # check cpu result and ai_core result
    np.array(res_npu).astype("float32")
    np.array(res_cpu).astype("float32")

    if (res_cpu == res_npu).all():
        print("[INFO] Sub op correct!")
    else:
        print("[ERROR] Sub op incorrect!")


if __name__ == "__main__":
    main()
