### 模型营第4周作业【共计50分，其中10分根据提交的PR和调试过程步骤由营长酌情给分】


使用Tensorflow1.15和Python3.7.5和[Kaggle官网CAT数据集](https://www.kaggle.com/crawford/cat-dataset)，将GitHub上开源的[BigGAN网络源码](https://github.com/taki0112/BigGAN-Tensorflow)迁移到Ascend910上。大作业只考虑img_size为128的情况，修改超惨batch_size的值，设置为32或64测试即可。

要求：
1. 在ModelArts平台上可以正常跑通，NPU利用率不为0，训练迭代次数不限。 【20分】

使用

epoch=5

iteration=5

batch_size=32

img_size=128

等参数进行训练作业。执行结果如下：

![ModelArts训练作业](https://bbs-img.huaweicloud.com/data/forums/attachment/forum/202104/18/090429p8jmgsqddiv9lzqu.png "ModelArts训练作业")

![NPU资源占用情况](https://bbs-img.huaweicloud.com/data/forums/attachment/forum/202104/18/090336bodjnut3wfgnwphl.png "NPU资源占用情况")


2. 使用Profiling工具分析网络性能。拿到profiling数据，并在Ai1S环境上使用toolkti工具包中的msprof工具进行算子耗时的简单分析。【10分】


![AICPU按照compute_times排序](https://bbs-img.huaweicloud.com/data/forums/attachment/forum/202104/18/171805h8k5mwlxohqpqmol.png "AICPU按照compute_times排序")

![op_summary按Task Duration排序](https://bbs-img.huaweicloud.com/data/forums/attachment/forum/202104/18/172013gg2lvrdldadolw5w.png "op_summary按Task Duration排序")

![op_statistic按照TotalTime排序](https://bbs-img.huaweicloud.com/data/forums/attachment/forum/202104/18/172138dy9j8vjdmuzqy3aa.png "op_statistic按照TotalTime排序")


3. 根据模型营所学的知识，尽可能的优化训练迭代耗时。 【10分】

优化训练时间，主要从 混合精度 的方式考虑，在main.py增加 混合精度的代码：
![混合精度](https://images.gitee.com/uploads/images/2021/0420/094501_3f8fd197_5630689.png "混合精度")

对比未实现混合精度前，5个epoch的训练情况如下：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0420/094615_211bf840_5630689.png "屏幕截图.png")

资源使用情况如下：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0420/094658_ad7981e0_5630689.png "屏幕截图.png")

耗时57分钟，5个epoch，其中每个epoch的每个iteration耗时如下：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0420/094755_47c3efe2_5630689.png "屏幕截图.png")

打开混合精度之后，跑10个epoch，训练情况如下：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0420/094849_03e54702_5630689.png "屏幕截图.png")

资源使用情况如下：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0420/094927_5ba1182c_5630689.png "屏幕截图.png")

耗时6分37秒，10个epoch，其中每个epoch的每个iteration耗时如下：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0420/095046_9517e1e7_5630689.png "屏幕截图.png")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0420/095100_718fc810_5630689.png "屏幕截图.png")

可见，Iteration时间从将近100-200秒，降到了0点几秒-20秒。性能大幅提升。


尽可能完善自己源码下的README，描述如何迁移的，做了什么修改/适配以及优化，最终需要提交完整代码PR，方便营长复现你的作业。


- 1.先仔细看钟老师的讲解视频 https://www.bilibili.com/video/BV1Fy4y14725
- 2.为了完成第一步作业，需首先下载代码和数据集。。
- 3.争取在本地跑通网络:使用PyCharm,选择tensorflow-gpu的conda环境，将数据集文件解压到 dataset目录，调试网络，保证main.py可以跑下去（笔者在这里报了OOM的错误），由于GPU资源所限，转入ModelArts调试处理。
- 4.修改代码（按照sess.run模式），使其满足在Ascend NPU运行的要求。删除dataset目录下的数据集。将数据集文件传到OBS，代码中增加OBS文件传输到ModelArts环境的部分。
- 5.使用PyCharm的ModelArts插件，在ModelArts上运行训练作业，并在ModelArts官网的控制台查看资源占用情况，确认训练作业是否用到了NPU。此时跑通后，第一步作业即已完成。
- 6.修改代码，增加Profilling采集部分。在代码中增加创建Profiling所需的目录，以及在训练结束后，将Profiling采集结果拷贝回OBS的过程。
- 7.使用PyCharm的ModelArts插件，在ModelArts上运行带Profiling采集的训练作业，并在ModelArts官网的控制台查看资源占用情况。等训练结束后，查看OBS的Profiling目录，并将该目录下载到本地备用。
- 8.等待小助手分配CANN 20.2alpha001-2021-3-1的共享镜像。接受该镜像并依此创建ECS AI1S云服务器。
- 9.将第7步下载的Profiling结果文件上传到云服务器。并编写分析脚本run_msprof.sh.执行该脚本，生成profiling分析结果目录（timeline和summary）。删除云服务器以避免额外扣费。下载带分析结果的profiling目录。
- 10.打开三个csv文件，并按照适当的字段排序，分析算子执行耗时。此时，第二步作业即已完成。
- 11.尝试使用比如zip压缩后上传OBS，在运行时解压；采用混合精度模式等方法，试图优化训练耗时。试图通过修改超参让训练收敛等等。。

详情可参见：https://bbs.huaweicloud.com/forum/thread-121645-1-1.html  


