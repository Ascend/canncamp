# 【昇腾CANN训练营——应用开发营】大作业 yexijoe

## 课程作业回顾

- 应用营大作业内容：
参考社区样例https://ascend.huawei.com/zh/#/developer/mindx-sdk/cartoon/990674866img?fromPage=1
自己实现一个黑白图像上色应用，可以直接选用样例中的离线模型，编写推理应用代码即可。
注意点：
1. 不得与社区样例代码雷同或与其他参与者的代码雷同，一经发现，提交时间较晚的ID直接记0分
2. 使用C++或Python实现均可，但原则上C++实现难度大，得分会高一些
3. 不可执行的代码不给分
4. 作业最终得分会综合代码质量、README质量（README十分重要，如果按照README进行操作，过程不够丝滑，或者有步骤缺失，可能会扣分甚至没分的哦）、选用的编程语言
5. 如果实在不想实现黑白上色应用，可自行选择社区上当前没有的应用来开发一个，要求将模型训练代码和训练方法、模型转换方法、推理代码一并提交，对优秀的实现，会有一定加分。

## 个人作业说明

1. 目前gitee中Ascend / samples仓库中的例程大多都是用于处理图像视频等视觉领域的数据，而本次应用营大作业也是基于图像数据。因此我围绕课程作业的第5点展开实验，
从训练模型、转换模型到推理，自行开发应用。
2. 本次作业使用修改后的ResNet50模型对无线电信号数据集RML2016.10a进行调制样式分类。先在Atlas800-9000训练服务器上将.pkl格式的数据集划分为训练集和测试集，并
将其转换成MindRecord格式文件。然后在Atlas800-9000上基于1.1.1版本的MindSpore框架使用修改后的ResNet50模型对转换所得的数据集进行分类，训练得到.ckpt模型文件，
将.ckpt模型文件导出成AIR格式模型文件。在Atlas200DK开发环境（本地电脑的VM虚拟机上配置的Ubuntu18.04.4系统）中将AIR模型文件转成OM模型。最后基于Python编写推理
代码，将代码、原始数据集RML2016.10a_dict.pkl和.om格式模型上传到Atlas200DK开发板，进行推理。
3. 数据集说明：无线电信号数据集RML2016.10a包含8PSK、AM-DSB、AM-SSB、BPSK、CPFSK、GFSK、PAM4、QAM16、QAM64、QPSK和WBFM共11种调制类型的信号，每种调制类
型的信号数据都有20种信噪比，每个原始信号样本由长度都为128的I通道和Q通道数据构成，即每个信号样本的形状为2x128。设置随机种子值为2016，再随机划分得到训练集和
测试集。原始数据集RML2016.10a_dict.pkl的百度云链接为：https://pan.baidu.com/s/1-V7S0pGSEyPwp-HjR_oWBA ，提取码为：nfc9。
4. 涉及到的模型文件下载地址：https://pan.baidu.com/s/1gcwSrizQa7k7eddxB8rLmA ，提取码：fma3。

## requirements

Atlas200DK（配置好了运行环境）、带VM虚拟机的PC（配置好了开发环境）、MindSpore==1.1.1（本次作业基于Atlas800-9000）；Python=3.7.5、pickle、csv、numpy。

## 具体步骤
1. 数据集预处理及格式转换：编写代码将原始数据集RML2016.10a_dict.pkl以1:1的比例划分成训练集和测试集，再参照https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/dataset_conversion.html#mindrecord 中将非标准数据集转换MindRecord处的说明，生成训练集和测试集的MindRecord格式文件数据。执行：将数据集RML2016.10a_dict.pkl和数据转换代码pkl2mindrecord.py上传到Atlas800-9000训练服务器，通过"python3 pkl2mindrecord.py"运行，注意按自己的实际情况修改pkl2mindrecord.py中的路径。
2. 训练模型：简单修改ResNet50模型，输入(batch_size, 1, 2, 128)形状(相当于N,C,H,W)的数据进行训练，得到训练好的.ckpt模型文件。从https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/resnet/src 下载原始的ResNet50模型，重命名为resnet_mindspore_own.py并做相应修改，将
第一个卷积层的输入通道改成1，即代码中"ResNet"类里的"self.conv1 = _conv7x7(3, 64, stride=2)"改成"self.conv1 = _conv7x7(1, 64, stride=2)"，修改最大池化层的卷积核和步长，即"ResNet"类里的"self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")"改成"self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), pad_mode="same")"，删除"self.layer4(c4)"层，将"resnet50"函数中的"[256, 512, 1024, 2048]"改成"[256, 512, 2 * 1024, 2048]"，具体修改详见resnet_mindspore_own.py文件。从https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet 下载train.py，重命名为resnet_signal.py，做适当修改，具体修改详见resnet_signal.py。执行：将代码resnet_mindspore_own.py和resnet_signal.py上传到Atlas800-9000训练服务器，通过"python3 pkl2mindrecord.py"运行，注意按自己的实际情况修改resnet_signal.py中的路径，得到多个.ckpt模型文件，选择测试集精度最高的.ckpt文件，我的是checkpoint_lenet-20_859.ckpt，测试集精度为57.38%，目前无线电信号领域对该数据集的测试集分类精度最高可到62%，此次作业使用简单的修改后的ResNet50模型，得到的57.38%的测试集精度算中规中矩。
3. 转换模型（ckpt2air）：参照https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/use/save_model.html#air 编写代码，将训练得到的.ckpt模型文件转换成.air格式的模型文件，在Atlas800-9000上，使用ckpt2air.py将checkpoint_lenet-20_859.ckpt模型文件转换成checkpoint_lenet-20_859.air模型文件。
4. 转换模型（air2om）：参照https://www.mindspore.cn/tutorial/inference/zh-CN/r1.1/multi_platform_inference_ascend_310_air.html#air ，将.air格式模型文件转成.om格式模型。执行：将Atlas800-9000上的checkpoint_lenet-20_859.air模型文件下载到本地PC的VM虚拟机上，使用命令"/home/HwHiAiUser/Ascend/ascend-toolkit/latest/atc/bin/atc --framework=1 --model=./checkpoint_lenet-20_859.air --output=./resnet50_export --input_format=NCHW --soc_version=Ascend310"得到resnet50_export.om模型文件。
4. 推理：参照https://gitee.com/ascend/samples/tree/master/python/level2_simple_inference/1_classification/vgg16_cat_dog_picture 例程，编写基于Python的推理代码，执行推理。执行：本地PC通过网线连接Atlas200DK开发板，参照https://gitee.com/ascend/samples/tree/master/python/environment 准备好Atlas200DK的环境，将原始数据集RML2016.10a_dict.pkl、本地的resnet50_export.om模型文件、编写好的推理代码main.py和atlas_utils依赖文件夹及其文件上传到Atlas200DK，具体存放见infer文件夹说明，通过"python3 main.py --start_num=5500 --end_num=5550"开始推理，其中"--start_num"表示想要进行推理的起始测试集样本的编号，"--end_num"表示想要进行推理的最终测试集样本的编号，"python3 main.py --start_num=5500 --end_num=5550"表示对编号为5500到5550的测试集样本进行推理。推理结束会生成result_modulation_type.csv文件，保存了测试集样本的编号、预测的调制类型和实际的调制类型。
5. 结束：从训练模型到进行推理的过程中，涉及到的一些截图详见figures文件夹。

## 目录结构与说明

**./**  
├── figures：**作业过程中的部分截图存放位置**  
├── infer：**在Atlas200DK上基于Python使用.om模型进行推理**  
│   ├── data：**无线电信号原始数据集存放位置**  
│   │   └── RML2016.10a_dict.pkl  
│   ├── model：**转换所得的.om模型存放位置，.om模型从transform文件夹中复制**  
│   │   └── resnet50_export.om  
│   └── src：**运行推理的.py文件存放位置**  
│       ├── atlas_utils：**依赖文件**  
│       │   ├── lib  
│       │   │   ├── atlas200dk  
│       │   │   │   └── libatlasutil.so  
│       │   │   └── atlasutil_so.py  
│       │   ├── acl_image.py  
│       │   ├── acl_logger.py  
│       │   ├── acl_model.py  
│       │   ├── acl_resource.py  
│       │   ├── acl_venc.py  
│       │   ├── constants.py  
│       │   ├── resource_list.py  
│       │   └── utils.py  
│       ├── main.py：**执行推理的.py文件**  
│       └── result_modulation_type.csv  
├── train：**在Atlas800-9000上训练模型得到.ckpt模型文件**  
│   ├── checkpoint_lenet-20_859.ckpt：**训练得到的.ckpt模型文件。生成。**  
│   ├── resnet_mindspore_own.py：**修改后的ResNet50模型结构文件**  
│   └── resnet_signal.py：**执行训练的.py文件**  
└── transform：**将数据集转换成MindRecord格式文件；将.ckpt模型文件导出成AIR格式模型文件；将AIR模型文件转成OM模型**  
    ├── air2om命令.txt  
    ├── checkpoint_lenet-20_859.air：**.ckpt模型文件转换得到的.air模型文件。生成。**  
    ├── checkpoint_lenet-20_859.ckpt：**训练得到的.ckpt模型文件。从train文件夹中复制。**  
    ├── ckpt2air.py：**将.ckpt模型文件转换成.air模型文件的代码**  
    ├── pkl2mindrecord.py：**将.pkl数据集转换成.mindrecord数据集文件的代码**  
    └── resnet50_export.om：**.air模型文件转换得到的.om模型文件。生成。**  



