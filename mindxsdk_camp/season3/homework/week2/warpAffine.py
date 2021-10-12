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
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':

    # 创建stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 创建stream
    pipeline_path = b"test.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 输入图片
    img_path = "image/test1.jpg"
    streamName = b"detection"
    inPluginId = 0
    dataInput = MxDataInput()
    with open(img_path, 'rb') as f:
        dataInput.data = f.read()
    ret = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if ret < 0:
        print("Failed to send data to stream")
        exit()

    # 获取pipeline处理结果
    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer1")
    keyVec.push_back(b"mxpi_imagecrop0")
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode()))
        exit()

    # 解析关键点结果
    mxpiTensorPackageList = MxpiDataType.MxpiTensorPackageList()
    mxpiTensorPackageList.ParseFromString(infer_result[0].messageBuf)
    landmarks = np.frombuffer(mxpiTensorPackageList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    shape = mxpiTensorPackageList.tensorPackageVec[0].tensorVec[0].tensorShape
    print(shape)
    landmarks.resize(shape[1] // 2, 2)

    # 这里的图像数据格式为yuv，需要色域转换为bgr格式
    mxpiVisionList = MxpiDataType.MxpiVisionList()
    mxpiVisionList.ParseFromString(infer_result[1].messageBuf)
    visionData = mxpiVisionList.visionVec[0].visionData.dataStr
    visionInfo = mxpiVisionList.visionVec[0].visionInfo
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(visionData, dtype = np.uint8)
    img_yuv = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)

    # 坐标转换到112*112范围
    landmarks = landmarks * [112, 112]

    # 获取左右两个嘴角，两个眼的中心，鼻子的关键
    # 需要获取这5个点的位置索引
    # 请参考https://github.com/Hsintao/pfld_106_face_landmarks/blob/master/data/106%E4%BA%BA%E8%84%B8%E6%A0%87%E6%B3%A8%E8%A7%84%E5%88%99.jpg
    index = [0, 1, 2, 3, 4] 
    
    landmarks = landmarks[index]
    for ids in index:
        point = (landmarks.astype(np.int32)[ids][0], landmarks.astype(np.int32)[ids][1])
        cv2.circle(img, point, 1, (255, 0, 0), -1)
    cv2.imwrite("./result.jpg", img)

    # 人脸对齐代码补全
    # 需要自行实现
    cv2.imwrite("./affine.jpg", img)

    # 销毁stream manager资源
    streamManagerApi.DestroyAllStreams()
