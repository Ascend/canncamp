/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "../onnx_common.h"

namespace domi {
static const uint32_t MIN_INPUT_NUM = 2;

Status ParseParamsRoiExtractor(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  uint32_t input_num = node->input_size();
  if (input_num < MIN_INPUT_NUM) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "input num must ge 2");
    return FAILED;
  }
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("features", input_num - 1, false);

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "finest_scale" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t finest_scale = attr.i();
      op_dest.SetAttr("finest_scale", finest_scale);
    }
    if (attr.name() == "roi_scale_factor" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      float roi_scale_factor = attr.f();
      op_dest.SetAttr("roi_scale_factor", roi_scale_factor);
    }
    if (attr.name() == "spatial_scale" && attr.type() == ge::onnx::AttributeProto::FLOATS) {
      std::vector<float> spatial_scale;
      for (auto s : attr.floats()) {
        spatial_scale.push_back(s);
      }
      op_dest.SetAttr("spatial_scale", spatial_scale);
    }
    if (attr.name() == "pooled_height" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t output_size = attr.i();
      op_dest.SetAttr("pooled_height", output_size);
    }
    if (attr.name() == "pooled_width" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t output_size = attr.i();
      op_dest.SetAttr("pooled_width", output_size);
    }
    if (attr.name() == "sample_num" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t sample_num = attr.i();
      op_dest.SetAttr("sample_num", sample_num);
    }
    if (attr.name() == "pool_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      std::string pool_mode = attr.s();
      op_dest.SetAttr("pool_mode", pool_mode);
    }
    if (attr.name() == "aligned" && attr.type() == ge::onnx::AttributeProto::INT) {
      bool aligned = attr.i();
      op_dest.SetAttr("aligned", aligned);
    }
  }

  return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("RoiExtractor")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::RoiExtractor",
                   "ai.onnx::9::RoiExtractor",
                   "ai.onnx::10::RoiExtractor",
                   "ai.onnx::11::RoiExtractor",
                   "ai.onnx::12::RoiExtractor",
                   "ai.onnx::13::RoiExtractor",
                   "ai.onnx::14::RoiExtractor",
                   "ai.onnx::15::RoiExtractor",
                   "ai.onnx::16::RoiExtractor"})
    .ParseParamsFn(ParseParamsRoiExtractor)
    .ImplyType(ImplyType::TVM);
}  // namespace domi