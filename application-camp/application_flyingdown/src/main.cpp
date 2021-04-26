#include "main.h"

using namespace std;

namespace
{
    uint32_t imageWidth = 224;
    uint32_t imageHeight = 224;
    const char *modelPath = "../model/colorization.om";
}

int main(int argc, char *argv[])
{
    // 检查参数输入是否正确
    if ((argc < 2) || (argv[1] == nullptr))
    {
        ATLAS_LOG_ERROR("Usage: ./main <image_dir>\n");
        return DO_FAIL;
    }

    // 创建辅助推理实例
    ColorizeHelper colorize(modelPath, imageWidth, imageHeight);

    // 初始化资源（acl资源、模型、输出空间创建）
    AtlasError ret = colorize.init();
    if (ret != ATLAS_OK)
    {
        ATLAS_LOG_ERROR("初始化资源失败.");
        return DO_FAIL;
    }

    // 获取带转换目录下的所有图片
    string inputImageDir = string(argv[1]);
    vector<string> images;
    GetAllFiles(inputImageDir, images);
    if (images.empty())
    {
        ATLAS_LOG_ERROR("path=%s 该目录下图片不存在.", inputImageDir.c_str());
        return DO_FAIL;
    }

    // 循环推理每一张图片
    for (string image : images)
    {
        // 图片预处理: 读取图片,调整图片大小
        ret = colorize.preprocess(image);
        if (ret != ATLAS_OK)
        {
            ATLAS_LOG_ERROR("%s 预处理失败.",
                            image.c_str());
            continue;
        }
        // 将预处理的图片送入模型推理,并获取推理结果
        std::vector<InferenceOutput> inferenceOutput;
        ret = colorize.inference(inferenceOutput);
        if (ret != ATLAS_OK)
        {
            ATLAS_LOG_ERROR("推理过程失败.");
            return DO_FAIL;
        }
        // 解析推理输出,并将推理得到的物体类别标记到图片上
        ret = colorize.postprocess(image, inferenceOutput);
        if (ret != ATLAS_OK)
        {
            ATLAS_LOG_ERROR("图片合成失败.");
            return DO_FAIL;
        }
    }

    ATLAS_LOG_INFO("执行成功.");
    return DO_SUCC;
}