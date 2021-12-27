"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

sqrt
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


@fusion_manager.register("sqrt")
def sqrt_compute(input_x, output_y, kernel_name="sqrt"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sqrt"

    Returns
    -------
    output tensor
    """

    """
    TODO:
    Please refer to the TE DSL Manual, And code here with TE DSL.
    """
    print("=================当你看到这句话时，说明我这个自定义sqrt算子被执行了============================")

    dtype = input_x.dtype
    date = input_x
    if dtype == "float16":
        date = te.lang.cce.cast_to(input_x, "float32")
    log_val = te.lang.cce.vlog(date)
    const_val = tvm.const(0.5, "float32")
    mul_val = te.lang.cce.vmuls(log_val, const_val)
    res = te.lang.cce.vexp(mul_val)
    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")
    return res


@util.check_input_type(dict, dict, str)
def sqrt(input_x, output_y, kernel_name="sqrt"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sqrt"

    Returns
    -------
    None
    """

    """
    TODO:
    Please refer to the TE DSL Manual, And code here with TE DSL.
    """

    """
    TODO:
    operator check
    """

    """
    TODO:
    operator compute, invoke sqrt_compute
    """
    print("=================当你看到这句话时，说明我这个自定义sqrt算子被执行了============================")
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = sqrt_compute(data_input, output_y, kernel_name)

    """
    TODO:
    auto schedule
    """
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    """
    TODO:
    operator build
    """
    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)