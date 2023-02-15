/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file nn_detect_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Obtains the ROI feature matrix from the feature map list. It is a customized fused operator for mmdetection. \n

*@par Inputs:
* Two inputs, including:
*@li features: A 5HD Tensor list of type float32 or float16.
*@li rois: ROI position. A 2D Tensor of float32 or float16 with shape (N, 5). "N" indicates the number of ROIs,
* the value "5" indicates the indexes of images where the ROIs are located, "x0", "y0", "x1", and "y1".

*@par Attributes:
*@li finest_scale: A optional attribute of type int, specifying the scale of calculate levels of "rois".
*@li roi_scale_factor: A optional attribute of type float32, specifying the rescaling of "rois" coordinates.
*@li spatial_scale: A optional attribute of type list float32, specifying the scaling ratio of "features"
* to the original image.
*@li pooled_height: A optional attribute of type int32, specifying the H dimension.
*@li pooled_width: A optional attribute of type int32, specifying the W dimension.
*@li sample_num: An optional attribute of type int32, specifying the horizontal and vertical sampling frequency
* of each output. If this attribute is set to "0", the sampling frequency is equal to the rounded up value of "rois",
* which is a floating point number. Defaults to "0".
*@li pool_mode: An optional attribute of type string to indicate pooling mode. Defaults to "avg" . \n
*@li aligned: An optional attribute of type bool, specifying the align to corner. Defaults to true . \n

*@par Outputs:
* output: Outputs the feature sample of each ROI position. The format is 5HD Tensor of type float32 or float16.
* The axis N is the number of input ROIs. Axes H, W, and C are consistent with the values of "pooled_height",
* "pooled_width", and "features", respectively.

*@par Third-party framework compatibility
*Compatible with mmdetection SingleRoIExtractor operator.
*/
REG_OP(RoiExtractor)
    .DYNAMIC_INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(finest_scale, Int, 56)
    .ATTR(roi_scale_factor, Float, 0)
    .ATTR(spatial_scale, ListFloat, {1.f / 4, 1.f / 8, 1.f / 16, 1.f / 32})
    .ATTR(pooled_height, Int, 7)
    .ATTR(pooled_width, Int, 7)
    .ATTR(sample_num, Int, 0)
    .ATTR(pool_mode, String, "avg")
    .ATTR(aligned, Bool, true)
    .OP_END_FACTORY_REG(RoiExtractor)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_
