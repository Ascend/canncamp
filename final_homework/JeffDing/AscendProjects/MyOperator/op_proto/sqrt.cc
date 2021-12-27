/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef GE_OP_SQRT_H
 #define GE_OP_SQRT_H

#include "graph/operator_reg.h"

namespace ge {

    REG_OP(sqrt)
    .INPUT(input_x, TensorType({DT_INT32, DT_FLOAT16})) /* "First operand." */
    .INPUT(input_y, TensorType({DT_INT32, DT_FLOAT16})) /* "Second operand." */
    .OUTPUT(output_z, TensorType({DT_INT32, DT_FLOAT16}))  /* "Result, has same element type as two inputs" */
    .OP_END_FACTORY_REG(sqrt)
} // namespace ge

#endif // GE_OP_SQRT_H