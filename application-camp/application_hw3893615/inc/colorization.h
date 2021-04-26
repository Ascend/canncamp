#ifndef COLORIZATION_H
#define COLORIZATION_H
#include "acl/acl.h"
#include<string>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"

// 资源初始化
void InitResource();

// 数据预处理
void DataProcess(const std::string& imageFile);

void CreateModelInput();

void CreateModelOutput();

// 模型加载
void LoadModel(const char *modelPath);

// 创建模型描述信息
void CreateModelDesc();

// 模型推理
void Inference();

// 结果后处理
void ResultPostProcess();

// 保存最后结果
void SaveImage(const std::string& outImageFile, cv::Mat& image);

void DestoryMalloc();

// 卸载模型
void UnloadModel();

// 回收资源
void DestoryResource();

// 全局变量
uint32_t modelId;
aclmdlDesc *modelDesc;
aclmdlDataset *input;
aclmdlDataset *output;
aclDataBuffer *inputBuffer;
aclDataBuffer *outputBuffer;
int32_t deviceId = 0;

// 模型的weight和height
uint32_t kModelWidth = 224;
uint32_t kModelHeight = 224;

std::string inputImagePath;
std::string outputImagePath;

void * pictureHostData = nullptr;
void * pictureDeviceData = nullptr;
uint32_t pictureDatasize = 0;

void * outputHostData = nullptr;
void * outputDeviceData = nullptr;
size_t outputDataSize = 0;

#endif


