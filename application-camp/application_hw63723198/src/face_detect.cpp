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

* File sample_process.cpp
* Description: handle acl resource
*/
#include <iostream>

#include "acl/acl.h"
#include "atlasutil/atlas_model.h"
#include "face_detect.h"

using namespace std;

namespace {
uint32_t kModelWidth = 304;
uint32_t kModelHeight = 300;
const char* kModelPath = "../model/face_detection_bgr.om";
    
const static vector<string> ssdLabel = { "background", "face"};
const uint32_t kBBoxDataBufId = 1;
const uint32_t kBoxNumDataBufId = 0;
const uint32_t kItemSize = 8;
enum BBoxIndex { EMPTY = 0, LABEL,SCORE,TOPLEFTX,TOPLEFTY, 
                 BOTTOMRIGHTX, BOTTOMRIGHTY };  
}

FaceDetect::FaceDetect(aclrtRunMode runMode) : model_(kModelPath),isInited_(false), isReleased_(false){
    runMode_ = runMode;
    imageDataSize_ = RGBF32_CHAN_SIZE(kModelWidth, kModelHeight);
}

FaceDetect::~FaceDetect() {
    DestroyResource();
}

void FaceDetect::DestroyResource() {
    if (!isReleased_) {
        model_.DestroyResource();
        isReleased_ = true;
    }
}

AtlasError FaceDetect::create_input()
{
    //Request image data memory for input model
    aclError aclRet = aclrtMalloc(&imageDataBuf_, (size_t)imageDataSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("malloc device data buffer failed, aclRet is %d", aclRet);
        return ATLAS_ERROR;
    }

    //Use the applied memory to create the model and input dataset. After creation, only update the memory data for each frame of inference, instead of creating the input dataset every time
    AtlasError ret = model_.CreateInput(imageDataBuf_, imageDataSize_);
    if (ret != ATLAS_OK) {
        ATLAS_LOG_ERROR("Create mode input dataset failed");
        return ATLAS_ERROR;
    }
    ATLAS_LOG_INFO("Create model input dataset success.");

    return ATLAS_OK;
}

AtlasError FaceDetect::Init() {
    if (isInited_) {
        ATLAS_LOG_INFO("Face detection is initied already");
        return ATLAS_OK;
    }

    AtlasError atlRet = model_.Init();
    if (atlRet == ATLAS_ERROR) {
        ATLAS_LOG_ERROR("Model init failed, error %d", atlRet);
        return ATLAS_ERROR;
    }
    
    AtlasError ret = create_input();
    if (ret == ATLAS_ERROR) {
        ATLAS_LOG_ERROR("Create mode input dataset failed");
        return ATLAS_ERROR;
    }
    ATLAS_LOG_INFO("Create image info buf success");

    isInited_ = true;
    return ATLAS_OK;
}

AtlasError FaceDetect::PreProcess(cv::Mat& frame) {
    //Scale the frame image to the desired size of the model
    cv::Mat resizeMat;
    cv::resize(frame, resizeMat, cv::Size(kModelWidth, kModelHeight));
    if (resizeMat.empty()) {
        ATLAS_LOG_ERROR("Resize image failed");
        return ATLAS_ERROR;
    }

    //Copy the data into the cache of the input dataset
    aclrtMemcpyKind policy = (runMode_ == ACL_HOST)?ACL_MEMCPY_HOST_TO_DEVICE:ACL_MEMCPY_DEVICE_TO_DEVICE;
    aclError ret = aclrtMemcpy(imageDataBuf_, imageDataSize_, resizeMat.ptr<uint8_t>(), imageDataSize_, policy);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Copy resized image data to device failed.");
        return ATLAS_ERROR;
    }

    return ATLAS_OK;
}

AtlasError FaceDetect::Inference(vector<InferenceOutput>& inferOutputs) {
    AtlasError ret = model_.Execute(inferOutputs);
    if (ret != ATLAS_OK) {
        ATLAS_LOG_ERROR("Execute model inference failed\n");
    }

    return ret;
}

void FaceDetect::PostProcess(vector<DetectionResult>& detectResults, 
                             uint32_t imageWidth, uint32_t imageHeight,
                             vector<InferenceOutput>& modelOutput) {
    float* detectData = (float *)modelOutput[kBBoxDataBufId].data.get();
    uint32_t* boxNum = (uint32_t *)modelOutput[kBoxNumDataBufId].data.get();

    uint32_t totalBox = boxNum[0];

    for (uint32_t i = 0; i < totalBox; i++) {
        DetectionResult oneResult;
        Point point_lt, point_rb;
        uint32_t score = uint32_t(detectData[SCORE + i * kItemSize] * 100);
        if (score < 70) {
            break;
        }

        point_lt.x = detectData[TOPLEFTX + i * kItemSize] * imageWidth;
        point_lt.y = detectData[TOPLEFTY + i * kItemSize] * imageHeight;
        point_rb.x = detectData[BOTTOMRIGHTX + i * kItemSize] * imageWidth;
        point_rb.y = detectData[BOTTOMRIGHTY + i * kItemSize] * imageHeight;

        uint32_t objIndex = (uint32_t)detectData[LABEL + i * kItemSize];
        oneResult.lt = point_lt;
        oneResult.rb = point_rb;
        oneResult.result_text = ssdLabel[objIndex] + to_string(score) + "\%";
        detectResults.emplace_back(oneResult);
    }
}


AtlasError FaceDetect::Process(cv::Mat& frame, vector<DetectionResult>& detectResults) {
   
    AtlasError ret = PreProcess(frame);
    if (ret == ATLAS_ERROR) {
        ATLAS_LOG_ERROR("Preprocess image failed");
        return ATLAS_ERROR;
    }

    vector<InferenceOutput> inferOutputs;
    ret = Inference(inferOutputs);
    if (ret==ATLAS_ERROR) {
        ATLAS_LOG_ERROR("Inference image failed");
        return ATLAS_ERROR;        
    }

    PostProcess(detectResults, frame.cols, frame.rows, inferOutputs);

    return ATLAS_OK;
}


