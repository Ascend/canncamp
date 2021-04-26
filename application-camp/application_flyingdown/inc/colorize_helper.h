#ifndef _COLOR_HELPER_H_
#define _COLOR_HELPER_H_

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"
#include "acl/acl.h"
#include "atlasutil/atlas_model.h"
#include <memory>

class ColorizeHelper {
public:
    ColorizeHelper(const char* modelPath,
                    uint32_t imageWidth, uint32_t imageHight);
    ~ColorizeHelper();

    AtlasError init();
    AtlasError preprocess(const std::string& imageFile);
    AtlasError inference(std::vector<InferenceOutput>& inferOutputs);
    AtlasError postprocess(const std::string& imageFile, std::vector<InferenceOutput>& modelOutput);
private:
    void save_image(const std::string& origImageFile, cv::Mat& image);
    void destroy_resource();

private:
    int32_t deviceId_;
    AtlasModel model_;

    const char* modelPath_;
    uint32_t modelWidth_;
    uint32_t modelHeight_;
    uint32_t inputDataSize_;
    void*    inputBuf_;

    bool isInited_;
};

#endif