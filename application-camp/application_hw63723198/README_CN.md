中文|[English](README.md)

**本样例为昇腾训练营作业，参考华为官方案例实现，非商业目的！**

**本样例适配3.0.0及以上版本，支持产品为Atlas200DK、Atlas300([ai1s](https://support.huaweicloud.com/productdesc-ecs/ecs_01_0047.html#ecs_01_0047__section78423209366))。**



## 人脸检测及年龄评估

功能：检测人脸，并识别检测到的人脸年龄属性。

样例输入：待推理的jpg图片。

样例输出：推理后的jpg图片。

### 前提条件

部署此Sample前，需要准备好以下环境：

- 请确认已按照[环境准备和依赖安装](https://gitee.com/ascend/samples/tree/master/cplusplus/environment)准备好环境。

- 已完成对应产品的开发环境和运行环境安装。

### 软件准备

1. 获取源码包。

   可以使用以下两种方式下载，请选择其中一种进行源码准备。

    - 命令行方式 或 压缩包方式下载源码包

2. 获取此应用中所需要的原始网络模型。

    参考下表获取此应用中所用到的原始网络模型及其对应的权重文件，并将其存放到开发环境普通用户下该样例的model文件夹中。
    
    |  **模型名称**  |  **模型说明**  |  **模型下载路径**  |
    |---|---|---|
    |  face_detection| 图片分类推理模型。是基于Caffe的resnet ssd模型。  |  请参考[https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/facedetection/ATC_resnet10-SSD_caffe_AE](https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/facedetection/ATC_resnet10-SSD_caffe_AE)目录中README.md下载原始模型章节下载模型和权重文件。 |
    |  inception_age| 年龄评估模型。  |  请参考[https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/inception_age](https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/inception_age)目录中README.md下载原始模型章节下载模型文件。 |


3. 将原始模型转换为Davinci模型。
    
   此案例中采用mindstudio转换模型。

   face_detection 模型转换配置如下：

![加载模型文件](https://images.gitee.com/uploads/images/2021/0423/183515_5ab8cd1a_5320986.png "屏幕截图.png")

![输入图片说明](https://images.gitee.com/uploads/images/2021/0424/015552_8cef924d_5320986.png "屏幕截图.png")

![输入图片说明](https://images.gitee.com/uploads/images/2021/0424/015638_6af17878_5320986.png "屏幕截图.png")

 
  inception_age 模型转换配置如下：

![输入图片说明](https://images.gitee.com/uploads/images/2021/0423/183721_b19322d6_5320986.png "屏幕截图.png")

![输入图片说明](https://images.gitee.com/uploads/images/2021/0423/183734_030ad6f5_5320986.png "屏幕截图.png")
   

   转换好的模型百度盘下载地址：链接: https://pan.baidu.com/s/1MedMZR5aOVOvPUZMfaGtnQ 提取码: 4cqi 

   将转换好的模型放在项目 model 目录下。


4. 获取样例需要的测试图片。

    执行以下命令，进入样例的data文件夹中，把测试图片放在该文件夹下。




### 样例部署
 
1. 开发环境命令行中设置编译依赖的环境变量。

   基于开发环境与运行环境CPU架构是否相同，请仔细看下面的步骤：

   - 当开发环境与运行环境CPU架构相同时，执行以下命令导入环境变量。

     **export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest**

     **export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub**

   - 当开发环境与运行环境CPU架构不同时，执行以下命令导入环境变量。例如开发环境为X86架构，运行环境为Arm架构，由于开发环境上同时部署了X86和Arm架构的开发套件，后续编译应用时需要调用Arm架构开发套件的ACLlib库，所以此处需要导入环境变量为Arm架构的ACLlib库路径。

     **export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest/arm64-linux**

     **export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub**

     ![](https://images.gitee.com/uploads/images/2020/1106/160652_6146f6a4_5395865.gif "icon-note.gif") **说明：**  
        > - 如果是3.0.0版本，此处 **DDK_PATH** 环境变量中的 **arm64-linux** 应修改为 **arm64-linux_gcc7.3.0**。
        > - 可以在命令行中执行 **uname -a**，查看开发环境和运行环境的cpu架构。如果回显为x86_64，则为x86架构。如果回显为arm64，则为Arm架构。

2. 用mindstudio打开项目。



3. 在mindstudio中进行编译。

    - 编译配置如下图所示。
      
     ![输入图片说明](https://images.gitee.com/uploads/images/2021/0423/183943_717d1a8d_5320986.png "屏幕截图.png")

     编译完成后，生成的可执行文件main在 **out** 目录下。



### 样例运行
      



    切换到目录 canncamp/application-camp/application_hw63723198/out 后，执行以下命令运行样例。


     mkdir output

     ./main ../data

### 查看结果

运行完成后，会在运行环境的命令行中打印出推理结果,并在application-camp/application_hw63723198/out/output目录下生成推理后的图片。

在 ai1s 上运行的过程如下图所示：

![输入图片说明](https://images.gitee.com/uploads/images/2021/0423/185951_0972fb09_5320986.png "屏幕截图.png")

生成的结果在output目录下：

![输入图片说明](https://images.gitee.com/uploads/images/2021/0423/190052_bddbf0ba_5320986.png "屏幕截图.png")