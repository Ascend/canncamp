## AlexNet训练
### 前言
这个仓主要讲解如何将已经在昇腾Ascend910上迁移适配好的代码，在华为云的ModelArts平台或者依瞳Apulis平台上运行起来，并将训练得到的CheckPoint文件转成Protocol Buffers(简称PB)文件，不对AlexNet网络的结构进行详细描述。同时，该仓中的代码也保留了原CPU/GPU的代码，可以很明显的看出NPU适配处跟CPU/GPU的差异点。
### 介绍
1. ModelArts平台

    [Modelarts平台](https://support.huaweicloud.com/productdesc-modelarts/modelarts_01_0001.html)是华为云面向AI开发者的一站式开发平台，它能够支撑开发者从数据到AI应用的全流程开发过程。包含数据处理、模型训练、模型管理、模型部署等操作，并且提供AI Gallery功能，能够在市场内与其他开发者分享模型。

2. Apulis平台

    [Apulis平台](https://github.com/apulis/apulis_platform)是依瞳科技公司开发旨在为不同行业的用户提供基于深度学习的端到端解决方案，该平台融合了Tensorflow、PyTorch、MindSpore等开源深度学习框架，提供了模型训练、超参调优、集群状态监控等开发环境，方便AI开发者快速搭建人工智能开发环境，开展AI开发应用。

### 预置安装环境
1.  ModelArts平台，选择Ascend910硬件，镜像环境需要安装昇腾CANN3.x的软件包。 CANN训练迁移资料详见官方[链接](https://support.huaweicloud.com/tensorflowdevg-cann330alphaXtraining/atlasmprtg_13_0001.html)。
2.  Apulis平台，裸机基于Ascend910，且需要安装昇腾CANN3.x的软件包
3.  Gpu环境，如果需要使用GPU环境训练，需要实现安装好Nvidia-Driver环境, Cuda以及Cudnn包。
4.  Cpu环境，如果没有GPU和NPU环境，在笔记本上跑通该网络也是可以的，但需要先安装好python3.7.5和tensorflow1.15。

### 训练数据集
当前alexnet_session示例源码适配的数据集为Flower，下载[链接](https://public-obs.obs.cn-north-4.myhuaweicloud.com/Flowers-Data-Set.zip)，解压后修改代码路径即可。

### 使用说明
1. [在ModelArts平台上进行网络训练](https://gitee.com/ascend/canncamp/blob/master/training_camp/sample-alexnet/alexnet_session/doc/README_ModelArts.md)
2. [在Apulis平台上进行网络训练](https://gitee.com/ascend/canncamp/blob/master/training_camp/sample-alexnet/alexnet_session/doc/README_Apulis.md)
3. [在GPU平台上进行网络训练](https://gitee.com/ascend/canncamp/blob/master/training_camp/sample-alexnet/alexnet_session/doc/README_Gpu.md)
3. [在CPU平台上进行网络训练](https://gitee.com/ascend/canncamp/blob/master/training_camp/sample-alexnet/alexnet_session/doc/README_Cpu.md)
