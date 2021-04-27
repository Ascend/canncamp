
- 黑白上色应用源码结构如下所示：
- └─colorization
-     ├─bin         可执行代码
-     ├─etc         配置代码
-     ├─input       输入图片
-     ├─model       由ATC转换好的模型
-     ├─output      上色后的图片
-     └─src         源码及Makefile


所有操作均在Atlas 200DK开发板上完成。

（1）进入model目录，下载Caffe模型文件：
cd /home/HwHiAiUser/acl_project/colorization/model
wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization.prototxt
wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization.caffemodel

export install_path=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export PATH=${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:${install_path}/toolkit/bin:$PATH
export PYTHONPATH=${install_path}/toolkit/python/site-packages:${install_path}/atc/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/opp

![输入图片说明](https://images.gitee.com/uploads/images/2021/0421/012357_6e613b11_5630689.png "屏幕截图.png")

(2)转换Caffe模型为om格式
atc --input_shape="data_l:1,1,224,224" --weight="./colorization.caffemodel" --input_format=NCHW --output="colorization" --soc_version=Ascend310 --framework=0 --model="./colorization.prototxt"

![输入图片说明](https://images.gitee.com/uploads/images/2021/0421/012613_59ab0419_5630689.png "屏幕截图.png")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0421/012701_ae37499d_5630689.png "屏幕截图.png")

（3）切换到src目录，执行make clean和make，编译生成可执行文件 bin/main
![输入图片说明](https://images.gitee.com/uploads/images/2021/0421/012801_a0fba5fc_5630689.png "屏幕截图.png")

（4）在input目录下准备好一张灰度照片
![输入图片说明](https://images.gitee.com/uploads/images/2021/0421/012855_bb1c2010_5630689.png "屏幕截图.png")

（5）执行上色：../bin/main ../input/heben.jpg

![输入图片说明](https://images.gitee.com/uploads/images/2021/0421/013003_3762284b_5630689.png "屏幕截图.png")

（6）下载output目录下生成好的 out_heben.jpg
![输入图片说明](https://images.gitee.com/uploads/images/2021/0421/013100_e15cc39a_5630689.png "屏幕截图.png")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0421/013418_916919a8_5630689.png "屏幕截图.png")


