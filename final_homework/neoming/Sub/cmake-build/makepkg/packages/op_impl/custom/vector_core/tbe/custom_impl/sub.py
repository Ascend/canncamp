import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check


@register_op_compute("Sub", op_mode="static", support_fusion=False)
def sub_compute(x, y, z, kernel_name="sub"):
    """
    :param x: input tensor
    :param y: input tensor
    :param z: output tensor
    :param kernel_name:  sub
    :return: x - y
    """
    print("[INFO] You are using custom_op Sub instead of tf.math.subtract")

    res = tbe.vsub(x, y)

    return res


# @para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
#                             para_check.KERNEL_NAME)
def sub(x, y, z, kernel_name="sub"):
    """
    :param x: input tensor
    :param y: input tensor
    :param z: output tensor
    :param kernel_name: sub
    :return: z = x - y
    """

    # check shape and type
    assert (x.get("shape") == y.get("shape"))
    assert (x.get("dtype") == y.get("dtype"))

    # create place holder
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    data_y = tvm.placeholder(y.get("shape"), dtype=y.get("dtype"), name="data_y")

    # compute
    res = sub_compute(data_x, data_y, z, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    tbe.build(schedule, config)
