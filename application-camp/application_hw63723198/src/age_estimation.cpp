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
#include "age_estimation.h"
using namespace std;

namespace {
    uint32_t kModelWidth = 227;
    uint32_t kModelHeight = 227;
    const char* kModelPath = "../model/inception_age.om";
    
    const static vector<string> ageLabel = { "(0, 2)","(4, 6)","(8, 12)","(15, 20)","(25, 32)","(38, 43)","(48, 53)","(60, 100)"};

    const uint32_t kBBoxDataBufId = 1;
    const uint32_t kBoxNumDataBufId = 0;
    uint32_t gSendNum = 0;
    const uint32_t kEachResultTensorNum = 136;
    // The center's size for the inference result
    const float kNormalizedCenterData = 0.5;
}

AgeEstimation::AgeEstimation(aclrtRunMode runMode) : model_(kModelPath),isInited_(false), isReleased_(false){
    runMode_ = runMode;
    imageDataSize_ = RGBU8_IMAGE_SIZE(kModelWidth, kModelHeight)*4;
}

AgeEstimation::~AgeEstimation() {
    DestroyResource();
}
void AgeEstimation::DestroyResource(){
    if (!isReleased_) {
        model_.DestroyResource();
        isReleased_ = true;
    }
}

AtlasError AgeEstimation::create_input()
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

AtlasError AgeEstimation::Init() {
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

AtlasError AgeEstimation::PreProcess(cv::Mat& frame) {
    //Scale the frame image to the desired size of the model
    cv::Mat resizeMat;
    cv::resize(frame, resizeMat, cv::Size(kModelWidth, kModelHeight));
    if (resizeMat.empty()) {
        ATLAS_LOG_ERROR("Resize image failed");
        return ATLAS_ERROR;
    }

    //Copy the data into the cache of the input dataset
    aclrtMemcpyKind policy = (runMode_ == ACL_HOST)?ACL_MEMCPY_HOST_TO_DEVICE:ACL_MEMCPY_DEVICE_TO_DEVICE;
    aclError ret = aclrtMemcpy(imageDataBuf_, imageDataSize_, resizeMat.ptr<uint32_t>(), imageDataSize_, policy);//uint8_t
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Copy resized image data to device failed.");
        return ATLAS_ERROR;
    }

    return ATLAS_OK;
}

AtlasError AgeEstimation::Inference(vector<InferenceOutput>& inferOutputs) {
    AtlasError ret = model_.Execute(inferOutputs);
    if (ret != ATLAS_OK) {
        ATLAS_LOG_ERROR("Execute model inference failed\n");
    }

    return ret;
}

void AgeEstimation::PostProcess(cv::Mat image, vector<InferenceOutput>& modelOutput, DetectionResult& inputBox) {
    float* inference_result1 = (float *)modelOutput[0].data.get();

    uint32_t max_confidence_index = 0;
    for (int j = 0; j < 8; j++) {
        if(inference_result1[j] > inference_result1[max_confidence_index]) {
            max_confidence_index = j;
        }

        //cout << j << "===========" << inference_result1[j]<< endl;
    }
    string age = ageLabel[max_confidence_index];
    inputBox.result_text = inputBox.result_text + " age:" + age;
}

AtlasError AgeEstimation::Process(cv::Mat& frame, DetectionResult& inputBox) {
    //Preprocess the picture: read the picture and zoom the picture to the size required by the model input
    AtlasError ret = PreProcess(frame);
    if (ret == ATLAS_ERROR) {
        ATLAS_LOG_ERROR("Preprocess image failed");
        return ATLAS_ERROR;
    }
  
    //Send the preprocessed pictures to the model for inference and get the inference results
    vector<InferenceOutput> inferOutputs;
    ret = Inference(inferOutputs);
    if (ret==ATLAS_ERROR) {
        ATLAS_LOG_ERROR("Inference image failed");
        return ATLAS_ERROR;        
    }
   
    PostProcess(frame, inferOutputs, inputBox);
    
    return ATLAS_OK;
}