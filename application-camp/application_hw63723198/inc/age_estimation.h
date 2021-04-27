//
// Created by ascend on 12/14/20.
//

/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File sample_process.h
* Description: handle acl resource
*/
#pragma once
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"
#include "acl/acl.h"
#include "atlasutil/atlas_model.h"
#include "common.h"

using namespace std;

class AgeEstimation {
    public:
    AgeEstimation(aclrtRunMode runMode);
    ~AgeEstimation();

    AtlasError Init();
    void DestroyResource();
    
    AtlasError Process(cv::Mat& frame, DetectionResult& inputBox);
    
private:
    AtlasError create_input();
    AtlasError PreProcess(cv::Mat& frame);
    AtlasError Inference(vector<InferenceOutput>& inferOutputs);
    void PostProcess(cv::Mat image, vector<InferenceOutput>& modelOutput, DetectionResult& inputBox);
    
private:
    aclrtRunMode runMode_;
    uint32_t imageDataSize_; //Model input data size
    void*    imageDataBuf_;      //Model input data cache

    AtlasModel model_;
    
    bool isInited_;
    bool isReleased_;

};


