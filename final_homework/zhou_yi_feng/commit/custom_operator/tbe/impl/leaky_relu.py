import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check


# pylint: disable=locally-disabled,unused-argument,invalid-name
@register_op_compute("LeakyRelu", op_mode="static", support_fusion=False)
def leaky_relu_compute(x, y,  kernel_name="leaky_relu"):
    print("=================执行自定义的leaky_relu算子============================")
    
    data_res = tbe.vlrelu(x, 0.2)

    return data_res

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def leaky_relu(x, y,  kernel_name="leaky_relu"):
    """leaky_relu op for input tensor

       f(x)= x(x>=0) or negative_slope*x(x<0) equal to
       f(x)=negative_slope*x

    Parameters
    ----------
    x : TVM tensor
        input tensor has shape and dtype attributes
    y : dict
        dict with keys(shape and dtype) of output

    kernel_name : str
        cce kernel name, default value is "leaky_relu"

    Returns
    ------
    None
    """

    # check input tensor shape
    shape = x.get("shape")
    dtype = x.get("dtype")

    # check input tensor data_type
    check_list = ["float16", "float32", "int32", "int8"]
    para_check.check_dtype(dtype.lower(), check_list, param_name="x")

    inp_dtype = dtype.lower()
    input_data_x = tvm.placeholder(shape, name="input_data_x", dtype=inp_dtype)

    with tvm.target.cce():

        res = leaky_relu_compute(input_data_x, y,  kernel_name)
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, res]}
    tbe.build(sch, config)