
#include <iostream>
#include "acl/acl.h"
#include "atlasutil/atlas_model.h"
#include "colorize_helper.h"

using namespace std;

ColorizeHelper::ColorizeHelper(const char* modelPath, uint32_t modelWidth, uint32_t modelHeight)
:deviceId_(0), inputBuf_(nullptr), modelWidth_(modelWidth), modelHeight_(modelHeight),
isInited_(false){
    modelPath_ = modelPath;
}

ColorizeHelper::~ColorizeHelper() {
    destroy_resource();
}

AtlasError ColorizeHelper::init() {
    if (isInited_) {
        ATLAS_LOG_INFO("Acl已完成初始化!");
        return ATLAS_OK;
    }

     // 初始化acl api
    AtlasError ret = aclInit(NULL);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Acl初始化失败.");
        return ATLAS_ERROR;
    }
    ATLAS_LOG_INFO("AclInit成功.");

    // 指定用于运算的Device，隐式创建Context和Stream
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Acl指定用于运算device[%d]失败", deviceId_);
        return ATLAS_ERROR;
    }
    ATLAS_LOG_INFO("指定用于运算device[%d]成功", deviceId_);

    ret = model_.Init(modelPath_);
    if (ret != ATLAS_OK) {
        ATLAS_LOG_ERROR("模型初始化失败.");
        return ATLAS_ERROR;
    }
    ATLAS_LOG_INFO("初始化model成功.");

    inputDataSize_ = model_.GetModelInputSize(0);

    // 申请buffer
    aclrtMalloc(&inputBuf_, inputDataSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (inputBuf_ == nullptr) {
        ATLAS_LOG_ERROR("Acl申请device内存失败.");
        return ATLAS_ERROR;
    }

    // 创建输入dataset
    ret = model_.CreateInput(inputBuf_, inputDataSize_);
    if (ret != ATLAS_OK) {
        ATLAS_LOG_ERROR("创建输入dataset失败.");
        return ATLAS_ERROR;
    }
    ATLAS_LOG_INFO("创建输入dataset成功.");

    isInited_ = true;
    return ATLAS_OK;
}

AtlasError ColorizeHelper::preprocess(const string& imageFile) {
    // 读取图片
    cv::Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);
    if (image.data == nullptr) {
        ATLAS_LOG_ERROR("读取图片失败.");
        return ATLAS_ERROR;
    }
    // 设置图片大小符合模型输入
    cv::Mat reiszeImage;
    cv::resize(image, reiszeImage, cv::Size(modelWidth_, modelHeight_));

    // 转换颜色模式、归一化
    reiszeImage.convertTo(reiszeImage, CV_32FC3);
    reiszeImage = 1.0 * reiszeImage / 255;
    cv::cvtColor(reiszeImage, reiszeImage, CV_BGR2Lab);

    // 分离图片通道
    std::vector<cv::Mat> channels;
    cv::split(reiszeImage, channels);
    if (channels.empty()) {
        return ATLAS_ERROR;
    }

    // L通道减去50，老谭也不知道为什么。。。 作用应该类似归一化
    cv::Mat reiszeMatL = channels[0] - 50;

    // 将L分量载入device内存
    aclError ret = aclrtMemcpy(inputBuf_, inputDataSize_,
                                reiszeMatL.ptr<uint8_t>(), inputDataSize_,
                                ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("载入device内存失败.");
        return ATLAS_ERROR;
    }

    return ATLAS_OK;
}

AtlasError ColorizeHelper::inference(std::vector<InferenceOutput>& inferOutputs) {
    AtlasError ret = model_.Execute(inferOutputs);
    if (ret != ATLAS_OK) {
        ATLAS_LOG_ERROR("执行模型推理失败.");
        return ATLAS_ERROR;
    }

    return ATLAS_OK;
}

AtlasError ColorizeHelper::postprocess(const string& imageFile, vector<InferenceOutput>& modelOutput)
{
    uint32_t dataSize = 0;
    void* data = modelOutput[0].data.get();
    if (data == nullptr)
    {
        return ATLAS_ERROR;
    }

    dataSize = modelOutput[0].size;
    // 计算数据长度
    uint32_t size = static_cast<uint32_t>(dataSize) / sizeof(float);
    // 将数据拆分为a通道和b通道
    cv::Mat mat_a(56, 56, CV_32FC1, const_cast<float*>((float*)data));
    cv::Mat mat_b(56, 56, CV_32FC1, const_cast<float*>((float*)data + size / 2));

    // 拿到原图像，将其L通道数据做预处理
    cv::Mat mat = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);
    if (mat.data == nullptr) {
        ATLAS_LOG_ERROR("读取图片失败.");
        return ATLAS_ERROR;
    }
    mat.convertTo(mat, CV_32FC3);
    mat = 1.0 * mat / 255;
    cv::cvtColor(mat, mat, CV_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(mat, channels);

    // 将推理出的数据尺寸转换为原图像大小
    int r = mat.rows;
    int c = mat.cols;
    cv::Mat mat_a_up(r, c, CV_32FC1);
    cv::Mat mat_b_up(r, c, CV_32FC1);
    cv::resize(mat_a, mat_a_up, cv::Size(c, r));
    cv::resize(mat_b, mat_b_up, cv::Size(c, r));

    // 合并原图像L通道和推理出的a、b通道
    cv::Mat newChannels[3] = { channels[0], mat_a_up, mat_b_up };
    cv::Mat resultImage;
    cv::merge(newChannels, 3, resultImage);

    // 转换图像颜色模式并保存
    cv::cvtColor(resultImage, resultImage, CV_Lab2BGR);
    resultImage = resultImage * 255;
    save_image(imageFile, resultImage);

    return ATLAS_OK;
}

void ColorizeHelper::save_image(const string& origImageFile, cv::Mat& image) {
    int pos = origImageFile.find_last_of("/");

    string filename(origImageFile.substr(pos + 1));
    stringstream sstream;
    sstream.str("");
    sstream << "./output/out_" << filename;

    string outputPath = sstream.str();
    cout << outputPath << endl;
    cv::imwrite(outputPath, image);
}

void ColorizeHelper::destroy_resource()
{
    model_.DestroyInput();
    model_.DestroyResource();

    aclrtFree(inputBuf_);
    inputBuf_ = nullptr;
    AtlasError ret;

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("释放device失败.");
    }
    ATLAS_LOG_INFO("成功释放device[%d].", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("释放acl失败.");
    }
    ATLAS_LOG_INFO("成功释放acl.");
}
