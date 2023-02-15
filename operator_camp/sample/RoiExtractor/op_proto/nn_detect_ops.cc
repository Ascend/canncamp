/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file nn_detect_ops.cpp
 * \brief
 */
#include "inc/nn_detect_ops.h"
#include "inc/experiment_ops.h"
#include <cmath>
#include <string>
#include <vector>

#include "common/util/error_manager/error_manager.h"

#include "util/util.h"
#include "error_util.h"
#include "op_log.h"
#include "axis_util.h"
#include "register/infer_axis_slice_registry.h"

namespace ge {

// ----------------RoiExtractor-------------------
IMPLEMT_COMMON_INFERFUNC(RoiExtractorInferShape) {
  auto x0_desc = op.GetDynamicInputDesc("features", 0);
  auto input_dtype = x0_desc.GetDataType();
  auto x0_shape = x0_desc.GetShape();
  auto rois_shape = op.GetInputDescByName("rois").GetShape();

  int64_t pooled_height;
  int64_t pooled_width;
  if (op.GetAttr("pooled_height", pooled_height) == ge::GRAPH_FAILED) {
    OP_LOGI(
        TbeGetName(op).c_str(),
        "GetOpAttr pooled_height failed. Use default shape.");
    pooled_height = 7;
  }
  if (op.GetAttr("pooled_width", pooled_width) == ge::GRAPH_FAILED) {
    OP_LOGI(
        TbeGetName(op).c_str(),
        "GetOpAttr pooled_width failed. Use default shape.");
    pooled_width = 7;
  }

  std::vector<int64_t> dim_tmp;
  dim_tmp.push_back(rois_shape.GetDim(0));
  dim_tmp.push_back(x0_shape.GetDim(1));
  dim_tmp.push_back(pooled_height);
  dim_tmp.push_back(pooled_width);
  Shape valid_shape(dim_tmp);

  auto td = op.GetOutputDescByName("y");
  td.SetShape(valid_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(RoiExtractor, RoiExtractorVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(RoiExtractor, RoiExtractorInferShape);
VERIFY_FUNC_REG(RoiExtractor, RoiExtractorVerify);
// ----------------RoiExtractor-------------------

}  // namespace ge
