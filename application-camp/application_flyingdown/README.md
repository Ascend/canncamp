# 推理应用开发说明

安装驱动、`nnrt`及`toolkit`过程可以参考作业一，[开发环境配置](https://bbs.huaweicloud.com/forum/forum.php?mod=redirect&goto=findpost&ptid=113294&pid=882456&fromuid=296154)

安装环境依赖过程参考[`ffmpeg`及`opencv`安装](https://gitee.com/ascend/samples/blob/master/cplusplus/environment/opencv_install/README_300_CN.md)

安装`atlasutil`参考[`atlasutil`安装](https://gitee.com/ascend/samples/blob/master/cplusplus/environment/atlasutil_install/README_300_CN.md)


## 一、 转换离线模型

创建临时目录，下载模型及权重参数
```
wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization.prototxt

wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization.caffemodel
```

执行转换命令，并将结果拷贝至`~/application_flyingdown/model`

```
atc --input_shape="data_l:1,1,224,224" --weight="./colorization.caffemodel" --input_format=NCHW --output="colorization" --soc_version=Ascend310 --framework=0 --model="./colorization.prototxt"

cp colorization.om ~/application_flyingdown/model
```

## 二、 配置编译环境变量

打开`~/.bashrc`，将以下环境配置，并执行生效
```
export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest/x86_64-linux
export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub
```

## 三、 编译代码

进入`src`目录，尽量在该目录进行操作，否则，编译后的`out`目录位置会发生变化。创建`build`目录，进入`build`目录，并执行`cmake ../`命令，生成`Makefile`后，执行`make`编译
```
cd ~/application_flyingdown/src
mkdir build
cd build
cmake ../
make
```

## 四、 运行并产生结果

编译后，会在`~/application_flyingdown`路径下产生`out`目录，并生成可执行文件`main`，进入`~/application_flyingdown/out`，创建`output`目录，并执行命令`./main ../data/`，此时会在将`~/application_flyingdown/data`目录下的所有图片执行上色处理，并将结果存放于`~/application_flyingdown/out/output`下

```
cd ~/application_flyingdown/out
mkdir output
./main ../data/
```