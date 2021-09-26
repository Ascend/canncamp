#!/usr/bin/env python
# coding=utf-8
"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import cv2
import numpy as np
import json
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # 创建StreamManager资源
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 创建pipeline
    pipeline_path = b"test.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    img_path = "test.jpg"
    streamName = b"segmentation"
    inPluginId = 0
    dataInput = MxDataInput()
    with open(img_path, 'rb') as f:
        dataInput.data = f.read()
    # 输入jpg数据到pipeline
    ret = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if ret < 0:
        print("Failed to send data to stream")
        exit()

    # 从pipeline取结果
    keyVec = StringVector()
    keyVec.push_back(b"mxpi_imagedecoder0")
    keyVec.push_back(b"mxpi_semanticsegpostprocessor0")
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
    if infer_result[0].errorCode != 0:
        print("GetResult error=%d" %(infer_result[0].errorCode))
        exit()
    if infer_result[1].errorCode != 0:
        print("GetResult error=%d" %(infer_result[1].errorCode))
        exit()

    # 获取原图数据
    mxpiVisionList = MxpiDataType.MxpiVisionList()
    mxpiVisionList.ParseFromString(infer_result[0].messageBuf)
    vision_data = mxpiVisionList.visionVec[0].visionData.dataStr
    vision_info = mxpiVisionList.visionVec[0].visionInfo
    img = np.frombuffer(vision_data, np.uint8)
    img = img.reshape(vision_info.heightAligned, vision_info.widthAligned, 3)

    # 获取后处理结果
    mxpiImageMaskList = MxpiDataType.MxpiImageMaskList()
    mxpiImageMaskList.ParseFromString(infer_result[1].messageBuf)
    mask = np.frombuffer(mxpiImageMaskList.imageMaskVec[0].dataStr, np.uint8)
    mask = mask.reshape(mxpiImageMaskList.imageMaskVec[0].shape)

    # 分割结果映射到原图
    output_h = 513
    output_w = 513
    ratio = min(float(output_w) / float(vision_info.width), float(output_h) / float(vision_info.height))
    resize_w, resize_h = int(vision_info.width * ratio), int(vision_info.height * ratio)
    mask = mask[0:resize_h, 0:resize_w]
    mask = cv2.resize(mask, (vision_info.width, vision_info.height))
    
    # 可视化
    class_name = 21
    color_bin = int(255 / class_name)
    img = np.array(img)
    img[mask != 0] = 0.7 * mask[mask != 0, np.newaxis] * color_bin + 0.3* img[mask != 0]
    cv2.imwrite("./result.jpg", img)

    # 释放StreamManager资源
    streamManagerApi.DestroyAllStreams()