### 模型营第4周作业【共计50分，其中10分根据提交的PR和调试过程步骤由营长酌情给分】


使用Tensorflow1.15和Python3.7.5和[Kaggle官网CAT数据集](https://www.kaggle.com/crawford/cat-dataset)，将GitHub上开源的[BigGAN网络源码](https://github.com/taki0112/BigGAN-Tensorflow)迁移到Ascend910上。要求：
1. 在ModelArts平台上可以正常跑通，NPU利用率不为0，训练迭代次数不限。 【20分】
2. 使用Profiling工具分析网络性能。拿到profiling数据，并在Ai1S环境上使用toolkti工具包中的msprof工具进行算子耗时的简单分析。【10分】
3. 根据模型营所学的知识，尽可能的优化训练迭代耗时。 【10分

尽可能完善自己源码下的README，描述如何迁移的，做了什么修改/适配以及优化，最终需要提交完整代码PR，方便营长复现你的作业。