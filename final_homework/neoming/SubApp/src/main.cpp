/**
* @file main.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <iostream>
#include "utils.h"
#include "acl/acl.h"

using namespace std;

bool g_isDevice = false;

void PrintResult(void * out_buffers,uint32_t out_tensor_size, std::string out_file){
    void* hostBuffer = nullptr;
    void* outData = nullptr;
    aclError ret = aclrtMallocHost(&hostBuffer, out_tensor_size);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to print result, malloc host failed");
    }
    ret = aclrtMemcpy(hostBuffer, out_tensor_size, out_buffers,out_tensor_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to print result, memcpy device to host failed, errorCode is %d", static_cast<int32_t>(ret));
        aclrtFreeHost(hostBuffer);
    }
    outData = reinterpret_cast<aclFloat16*>(hostBuffer);
    ofstream outstr(out_file, ios::out | ios::binary);
    outstr.write((char*)outData, out_tensor_size);
    outstr.close();
}

int main()
{
    // acl init
    const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init sucess");

    // set device
    int32_t deviceId_ = 0;
    ret = aclrtSetDevice(deviceId_);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrt set device failed");
        return FAILED;
    }
    INFO_LOG("acl set device sucess");

    // create context
    aclrtContext context_;
    ret = aclrtCreateContext(&context_, deviceId_);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("aclrt create context sucess");

    // create stream
    aclrtStream stream_;
    ret = aclrtCreateStream(&stream_);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("aclrt create stream sucess");

    // get run mode
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    INFO_LOG("aclrt get run mode sucess");

    g_isDevice = (runMode == ACL_DEVICE);

    uint32_t modelId_;
    size_t modelMemSize_;
    size_t modelWeightSize_;
    void* modelMemPtr_;
    void* modelWeightPtr_;
    bool loadFlag_;
    aclmdlDesc* modelDesc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;

    // define model path
    const char* modelPath = "../models/model.om";
    // get model mem size , weight size
    ret = aclmdlQuerySize(modelPath, &modelMemSize_, &modelWeightSize_);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclmdl query size failed");
        return FAILED;
    }
    INFO_LOG("aclmdl query model size sucess");

    // malloc model mem
    ret = aclrtMalloc(&modelMemPtr_, modelMemSize_, ACL_MEM_MALLOC_NORMAL_ONLY);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrt malloc modelMem failed");
        return FAILED;
    }
    INFO_LOG("aclrt malloc for model mem sucess");

    // malloc model weight
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_NORMAL_ONLY);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrt malloc modelWeight failed");
        return FAILED;
    }
    INFO_LOG("aclrt malloc for model weight sucess");

    // load model
    ret = aclmdlLoadFromFileWithMem(
        modelPath, &modelId_,
        modelMemPtr_, modelMemSize_,
        modelWeightPtr_, modelWeightSize_
    );
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclmdl load model from file with mem failed");
        return FAILED;
    }
    INFO_LOG("aclmdl load from file with mem  sucess");

    // get model desc
    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclmdl get model desc failed");
        return FAILED;
    }
    INFO_LOG("aclmdl create model desc sucess");

    // malloc input buffer
    size_t modelInputSize;
    void* modelInputBuffer;
    modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    ret = aclrtMalloc(&modelInputBuffer, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclmdl get malloc input buffer failed");
        return FAILED;
    }
    INFO_LOG("aclrt malloc for input sucess");

    // add input buffer to dataset buffer
    input_ = aclmdlCreateDataset();
    aclDataBuffer* inputData = aclCreateDataBuffer(modelInputBuffer, modelInputSize);
    ret = aclmdlAddDatasetBuffer(input_, inputData);
    if ( ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclmdl add input data to dataset buffer failed");
        return FAILED;
    }
    INFO_LOG("aclml add input buffer to dataset buffer sucess");

    // create output dataset
    output_ = aclmdlCreateDataset();
    size_t buffer_size = aclmdlGetOutputSizeByIndex(modelDesc_, 0);
    INFO_LOG("output buffer size is: %d", buffer_size);

    // malloc output buffer
    void *outputBuffer = nullptr;
    ret = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrt malloc output buffer failed");
        return FAILED;
    }
    INFO_LOG("aclrt malloc output buffer sucess");

    // add output buffer to output dataset
    aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
    ret = aclmdlAddDatasetBuffer(output_, outputData);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclmdl add dataset buffer failed");
        return FAILED;
    }
    INFO_LOG("aclmdl add output buffer to dataset sucess");

    // get input data
    string data_path = "../data/x_data.bin";
    void* inputBuffer;
    uint32_t inputBufferSize = 0;
    inputBuffer = Utils::ReadBinFile(data_path, inputBufferSize);

    if (!g_isDevice) {
        ret = aclrtMemcpy(
            modelInputBuffer, modelInputSize,
            inputBuffer, inputBufferSize,
            ACL_MEMCPY_HOST_TO_DEVICE
        );
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("aclrt memcpy from host to device failed");
            return FAILED;
        }
        INFO_LOG("aclrt memcpy from host to device sucess");
        (void)aclrtFreeHost(inputBuffer);
    } else {
        ret = aclrtMemcpy(
        modelInputBuffer, modelInputSize,
        inputBuffer, inputBufferSize,
        ACL_MEMCPY_DEVICE_TO_DEVICE
        );
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("aclrt memcpy from device to device failed");
            return FAILED;
        }
        INFO_LOG("aclrt memcpy from device to device sucess");
        (void)aclrtFree(inputBuffer);
    }

    // execute and get result
    ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclmdl execute model failed");
        return FAILED;
    }
    INFO_LOG("aclmdl execute sucess");
    PrintResult(outputBuffer, buffer_size, "../data/om_pred.bin");
    INFO_LOG("write res to %s sucess", "../data/om_pred.bin");

    // free input buffer
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(input_);
    input_ = nullptr;
    aclrtFree(modelInputBuffer);
    INFO_LOG("free input buffer success");

    // free output buffer
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
    INFO_LOG("free input buffer success");

    // free model resource
    ret = aclmdlUnload(modelId_);

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelMemPtr_ != nullptr) {
        (void)aclrtFree(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        (void)aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }
    INFO_LOG("free model resource success");

    // free stream context device
    ret = aclrtDestroyStream(stream_);
    if(ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrt destroy stream failed");
        return FAILED;
    }
    INFO_LOG("free model resource success");

    ret = aclrtDestroyContext(context_);
    if(ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrt destroy stream failed");
        return FAILED;
    }
    INFO_LOG("aclrt destroy stream success");

    ret = aclrtResetDevice(deviceId_);
    if(ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrt reset device failed");
        return FAILED;
    }
    INFO_LOG("aclrt reset device success");

    ret = aclFinalize();
    if(ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl finalize failed");
        return FAILED;
    }
    INFO_LOG("acl finalize success");
}
