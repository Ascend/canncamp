#pragma once
#include "acl/acl.h"
#include<string>
#include "colorization.h"
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"
#include <memory>

using namespace std;

// 资源初始化
void InitResource()
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE)
    {
        printf("Acl init failed\n");
        exit(1);
    }
    printf("Acl init success\n");
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE)
    {
        printf("Acl open device id failed\n");
        exit(1);
    }
    printf("Acl open device id success\n");
}

// 模型加载
void LoadModel(const char *modelPath)
{
    aclError ret = aclmdlLoadFromFile(modelPath, &modelId);
    if (ret != ACL_ERROR_NONE)
    {
        printf("Acl load model failed\n");
        exit(1);
    }
    printf("Acl load model success\n");
}

// 创建模型描述信息并根据modelId获取modelDesc
void CreateModelDesc(){
    // 创建模型描述
    modelDesc = aclmdlCreateDesc();
    // 根据modelId获取modelDesc
    aclError ret = aclmdlGetDesc(modelDesc, modelId);
    if (ret != ACL_ERROR_NONE)
    {
        printf("aclmdlGetDesc failed.\n");
        exit(1);
    }
    printf("CreateModelDesc success.\n");
}


// 创建模型的输入
void CreateModelInput()
{
    // 创建模型输入
    input = aclmdlCreateDataset();
    // 获取模型输入大小, 也就是opencv对图片处理之后的大小
    pictureDatasize = aclmdlGetInputSizeByIndex(modelDesc, 0);
    // 申请device侧内存，用来放input的
    aclError ret = aclrtMalloc(&pictureDeviceData, pictureDatasize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE)
    {
        printf("aclrtMalloc pictureDeviceData on device failed.\n");
        exit(1);
    }
    // 创建模型数据buffer
    inputBuffer = aclCreateDataBuffer(pictureDeviceData, pictureDatasize);
    // 添加databuffer
    ret = aclmdlAddDatasetBuffer(input, inputBuffer);
    if (ret != ACL_ERROR_NONE)
    {
        printf("CreateModelInput failed.\n");
        exit(1);
    }
    printf("CreateModelInput success.\n");
}

// 创建模型的输出
void CreateModelOutput()
{
    // 创建模型输出
    output = aclmdlCreateDataset();
    // 获取模型输出大小
    outputDataSize = aclmdlGetOutputSizeByIndex(modelDesc, 0);
    // 申请device侧模型输出的大小
    aclError ret = aclrtMalloc(&outputDeviceData, outputDataSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    // 创建databuffer
    outputBuffer = aclCreateDataBuffer(outputDeviceData, outputDataSize);
    // 添加databuffer
    ret = aclmdlAddDatasetBuffer(output, outputBuffer);
    if (ret != ACL_ERROR_NONE)
    {
        printf("CreateModelOutput failed.\n");
        exit(1);
    }
    printf("CreateModelOutput success.\n");
}

// 数据预处理
void DataProcess(const std::string &imageFile)
{
    // read image using OPENCV
    cv::Mat mat = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);
    //resize
    cv::Mat reiszeMat;
    cv::resize(mat, reiszeMat, cv::Size(kModelHeight, kModelHeight));
    // deal image
    reiszeMat.convertTo(reiszeMat, CV_32FC3);
    reiszeMat = 1.0 * reiszeMat / 255;
    cv::cvtColor(reiszeMat, reiszeMat, CV_BGR2Lab);

    // pull out L channel and subtract 50 for mean-centering
    std::vector<cv::Mat> channels;
    cv::split(reiszeMat, channels);
    cv::Mat reiszeMatL = channels[0] - 50;

    if (mat.empty())
    {
        printf("opencv image read failed!\n");
        exit(1);
    }

    aclError ret = aclrtMemcpy(pictureDeviceData, pictureDatasize, reiszeMatL.ptr<uint8_t>(), pictureDatasize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE)
    {
        printf("Copy resized image data to device failed.\n");
        exit(1);
    }
    printf("Copy resized image data to device success.\n");
}

// 模型推理
void Inference()
{
    aclError ret = aclmdlExecute(modelId, input, output);
    if (ret != ACL_ERROR_NONE)
    {
        printf("aclmdlExecute failed\n");
        exit(1);
    }
    printf("aclmdlExecute success\n");
}

// 保存最后结果
void SaveImage(const string& outImageFile, cv::Mat& image)
{
    cv::imwrite(outImageFile, image);
}

// 结果后处理
void ResultPostProcess()
{
    aclError ret = aclrtMallocHost(&outputHostData, outputDataSize);
    ret = aclrtMemcpy(outputHostData, outputDataSize, outputDeviceData, outputDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE)
    {
        printf("aclrtMemcpy data from device to host failed\n");
        exit(1);
    }
    printf("aclrtMemcpy data from device to host success\n");

    uint32_t size = static_cast<uint32_t>(outputDataSize) / sizeof(float);

    // get a channel and b channel result data
    cv::Mat mat_a(56, 56, CV_32FC1, const_cast<float *>((float *)outputHostData));
    cv::Mat mat_b(56, 56, CV_32FC1, const_cast<float *>((float *)outputHostData + size / 2));

    // pull out L channel in original image
    cv::Mat mat = cv::imread(inputImagePath, CV_LOAD_IMAGE_COLOR);
    mat.convertTo(mat, CV_32FC3);
    mat = 1.0 * mat / 255;
    cv::cvtColor(mat, mat, CV_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(mat, channels);

    // resize to match size of original image L
    int r = mat.rows;
    int c = mat.cols;
    cv::Mat mat_a_up(r, c, CV_32FC1);
    cv::Mat mat_b_up(r, c, CV_32FC1);
    cv::resize(mat_a, mat_a_up, cv::Size(c, r));
    cv::resize(mat_b, mat_b_up, cv::Size(c, r));

    // result Lab image
    cv::Mat newChannels[3] = {channels[0], mat_a_up, mat_b_up};
    cv::Mat resultImage;
    cv::merge(newChannels, 3, resultImage);

    //convert back to rgb
    cv::cvtColor(resultImage, resultImage, CV_Lab2BGR);
    resultImage = resultImage * 255;
    SaveImage(outputImagePath, resultImage);
    aclrtFreeHost(outputHostData);
}

// 回收malloc申请的资源
void DestoryMalloc()
{
    aclrtFree(pictureDeviceData);
    printf("pictureDeviceData free success\n");
    aclrtFree(outputDeviceData);
    printf("outputDeviceData free success\n");
}

// 卸载模型
void UnloadModel()
{
    aclmdlDestroyDesc(modelDesc);
    printf("modelDesc destroy success!\n");
    aclmdlUnload(modelId);
    printf("Model unload success!\n");
}

// 回收资源
void DestoryResource()
{
    aclError ret = aclrtResetDevice(deviceId);
    printf("acl reset device success!\n");
    ret = aclFinalize();
    printf("acl fainalize success!\n");
}

int main(int argc, char *argv[])
{
    //检查应用程序执行时的输入,程序执行要求输入图片目录参数
    if((argc < 3) || (argv[1] == nullptr) || argv[2] == nullptr){
        printf("Please input: ./main <input_image_path> <output_image_path. eg: ./main ../data/ansel_adams3.jpg ../output/ansel_adams3_output.jpg");
        exit(1);
    }
    inputImagePath = string(argv[1]);
    outputImagePath = string(argv[2]);
    cout << "input file: " << inputImagePath.c_str() << endl;
    cout << "output file: " << outputImagePath.c_str() << endl;
    InitResource();
    LoadModel("../model/colorization.om");
    CreateModelDesc();
    CreateModelInput();
    CreateModelOutput();
    DataProcess(inputImagePath);
    Inference();
    ResultPostProcess();
    DestoryMalloc();
    UnloadModel();
    DestoryResource();
    printf("colorization done!\n");
    return 0;
}
