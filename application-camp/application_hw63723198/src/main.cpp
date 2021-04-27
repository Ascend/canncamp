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

* File main.cpp
*/

#include <iostream>
#include <stdlib.h>
#include <dirent.h>

#include "acl/acl.h"
#include "atlasutil/atlas_utils.h"
#include "atlasutil/atlas_error.h"
#include "atlasutil/acl_device.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"

#include "common.h"
#include "face_detect.h"
#include "age_estimation.h"


using namespace std;

void save_image(const string& origImageFile, cv::Mat& image) {
    int pos = origImageFile.find_last_of("/");

    string filename(origImageFile.substr(pos + 1));
    stringstream sstream;
    sstream.str("");
    sstream << "./output/out_" << filename;

    string outputPath = sstream.str();
    cout << outputPath << endl;
    cv::imwrite(outputPath, image);
}

int main(int argc, char *argv[]) {
    //检查应用程序执行时的输入,程序执行要求输入图片目录参数
    if((argc < 2) || (argv[1] == nullptr)){
        ATLAS_LOG_ERROR("Please input: ./main <image_dir>");
        return ATLAS_ERROR;
    }

    //init acl resource
    AclDevice aclDev;
    AtlasError ret = aclDev.Init();
    if (ret) {
        ATLAS_LOG_ERROR("Init resource failed, error %d", ret);
        return ATLAS_ERROR;
    }

    aclrtRunMode runMode = aclDev.GetRunMode();

    FaceDetect face_detect(runMode);
    //Initialize the acl resources, models and memory
    ret = face_detect.Init();
    if (ret == ATLAS_ERROR) {
        ATLAS_LOG_ERROR("FaceDetect Init resource failed\n");
        return ATLAS_ERROR;
    }

    AgeEstimation age_estimation(runMode);
    //Initialize the acl resources, models and memory
    ret = age_estimation.Init();
    if (ret == ATLAS_ERROR) {
        ATLAS_LOG_ERROR("FaceDetect Init resource failed\n");
        return ATLAS_ERROR;
    }

    //获取图片目录下所有的图片文件名
    string inputImageDir = string(argv[1]);
    vector<string> fileVec;
    GetAllFiles(inputImageDir, fileVec);
    if (fileVec.empty()) {
        ATLAS_LOG_ERROR("Failed to deal all empty path=%s.", inputImageDir.c_str());
        return 1;
    }

    //逐张图片推理
    for (string imageFile : fileVec) {

        // read image using OPENCV
        cv::Mat imageMat = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);

        vector<DetectionResult> face_detectResults;

        ret = face_detect.Process(imageMat, face_detectResults);
        if (ret == ATLAS_ERROR) {
            ATLAS_LOG_ERROR("face_detect process failed\n");
            return ATLAS_ERROR;
        }

        for ( auto onefaceResult : face_detectResults) {
            cv::Mat face_crop_mat = imageMat(cv::Rect(onefaceResult.lt.x, onefaceResult.lt.y,
                                    onefaceResult.rb.x - onefaceResult.lt.x , onefaceResult.rb.y - onefaceResult.lt.y));

            ret = age_estimation.Process(face_crop_mat, onefaceResult);
            if (ret == ATLAS_ERROR) {
                ATLAS_LOG_ERROR("age_estimation process failed\n");
                return ATLAS_ERROR;
            }

            ATLAS_LOG_INFO("==%d %d %d %d %s\n", onefaceResult.lt.x, onefaceResult.lt.y, onefaceResult.rb.x, onefaceResult.rb.y, onefaceResult.result_text.c_str());

            cv::Point p1, p2;
            p1.x = onefaceResult.lt.x;
            p1.y = onefaceResult.lt.y;
            p2.x = onefaceResult.rb.x;
            p2.y = onefaceResult.rb.y;
            cv::rectangle(imageMat, p1, p2, cv::Scalar(255, 0, 0), 2);
            cv::putText(imageMat, onefaceResult.result_text.c_str(), cv::Point(p1.x, p1.y - 11),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        save_image(imageFile, imageMat);
    }

    aclDev.Release();
    ATLAS_LOG_INFO("Execute success");
    return ATLAS_OK;
}
