# 应用营大作业:黑白图像上色
README只提供在命令行方式下的运行方式，并为测试在Mindstudio下运行！
## 简介

功能：使用colorization模型对输入的黑白图片进行上色推理。    
样例输入：待推理的jpg图片。    
样例输出：推理后的jpg图片。

## 适配环境

弹性云服务ecs，A300(ai1s)，具体搭建过程请参考：

[应用营第一讲：开发&运行环境部署](https://support.huaweicloud.com/productdesc-ecs/ecs_01_0047.html#ecs_01_0047__section78423209366)

## 前提条件
参照[for_atlas300](https://gitee.com/ascend/samples/tree/master/cplusplus/environment)配置基础环境并安装ffmpeg和opencv，atlasutil可安装可不安装，本作业没有使用atlasutil接口，使用的是acl原生接口

## 程序准备

### 获取代码

    HwHiAiUser用户下
    
    cd $HOME
    
    git clone -b application_finalwork https://gitee.com/XxyMy/canncamp.git


### 代码结构

```
|--data: 需要处理得数据目录
   |--ansel_adams3.jpg  数据集
|--inc:  头文件目录
   |--colorization.h   头文件
|--src:  源文件目录 
   |--colorization.cpp  源代码  
   |--CMakeLists.txt    cmake 文件
|--model: 转好得模型文件目录
   |-- colorization.om
|--output: 转好后得图片路径
|--README.md
|--colorization.png  标准答案
```

没有提供最原始得caffe模型，直接提供好了转换得om模型

### 编译

    cd canncap/application-camp/application_hw3893615
    
    rm -rf build
    
    rm -rf output (存放输出结果的路径，先删掉后续再新建)

    mkdir -p build/intermediates/host
    
    mkdir output (如果没有的化创建一个，代码中没有自动新建目录得逻辑，就手动建了)

    cd build/intermediates/host
    
    cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    
    make

### 运行

    cd ../../../out/

    ./main ../data/ansel_adams3.jpg ../output/ansel_adams_output.jpg
    
## 结果

原图

![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/application_finalwork/application-camp/application_hw3893615/data/ansel_adams3.jpg)

上色后的图

![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/application_finalwork/application-camp/application_hw3893615/colorization.png)

咳咳~~，运行成功后自行看

整个流程在弹性云服务ai1s.large.4上亲测，可执行。。如果有疑问，欢迎骚扰 --~~--
