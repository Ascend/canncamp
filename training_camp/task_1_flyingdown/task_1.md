### 模型营第2周作业【共计15分，其中3分根据提交的PR作业由营长酌情给分】

1. 如果需要导入NPU相关库，需要import什么包？ 【分值：2分】
	
    ```
    from npu_bridge.npu_init import *
    ```

2. 在**session run**模式下，原有网络的`tf.ConfigProto()`实现代码片段如下：【分值：2分】
    ```
    #创建session
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        sess.run(init)
    ```
	
    如果想迁移到CANN平台上运行，上述代码块上需要做什么修改适配？
    ```
    ### 适配修改点写在这里
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  
    
    with tf.Session(config=config) as sess:
        sess.run(init)
    ```
	
3. 在**Estimator**模式下，网络代码中的Runconfig配置运行参数代码片段如下：【分值：2分】
    ```
    config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir, 
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
    ```
    如果想迁移到CANN平台上运行，上述代码块上需要做什么修改适配？
    ```
    ### 适配修改点写在这里
    from npu_bridge.estimator.npu.npu_conifg import NPURunConfig
    config=tf.estimator.NPURunConfig(
            model_dir=FLAGS.model_dir, 
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
    ```
4. 使用Tensorflow1.15实现LeNet网络的minist手写数字识别。硬件平台不限，可以基于window或者linux系统，尽量给出整个过程的截图，并在最后给出Loss或者accuracy运行结果。 参考链接[Gitee](https://gitee.com/lai-pengfei/LeNet) ~ [Github](https://github.com/ganyc717/LeNet)。【分值：6分】
   
    由于自己的机子没有`gpu`，所以选择`cpu`进行试验。
    1. 创建虚拟环境，并检查`python`版本
    ![创建虚拟环境](https://images.gitee.com/uploads/images/2021/0401/131740_b99f7672_738550.png "创建虚拟环境.png")
    ![检查python版本](https://images.gitee.com/uploads/images/2021/0401/131944_641d1515_738550.png "检查python版本.png")
    
    2. 进入虚拟环境，并安装`tensorflow 1.15`，并检查是否可以正常导入
    ![安装tensorflow](https://images.gitee.com/uploads/images/2021/0401/133848_49310ace_738550.png "安装tensorflow.png")
    ![成功安装tensorflow](https://images.gitee.com/uploads/images/2021/0401/134052_8f0c3141_738550.png "成功安装tensorflow.png")
    ![检查tensorflow安装是否正确](https://images.gitee.com/uploads/images/2021/0401/134313_af92369e_738550.png "检查tensorflow安装是否正确.png")

    3. 按照作业要求，进入代码仓库下载代码
    ![下载LeNet代码](https://images.gitee.com/uploads/images/2021/0401/134918_a99dc6a5_738550.png "下载LeNet代码.png")

    直接进行训练会报错，原因是`1.15`版本中没有`tensorflow.examples.tutorials.mnist.input_data`
    ![直接训练报错](https://images.gitee.com/uploads/images/2021/0401/135200_38ee04ca_738550.png "直接训练报错.png")

    修改代码，从`tensorflow.contrib.learn.python.learn.datasets.mnist`中导入`read_data_sets`，同时修改`mnist`数据载入
    ![修改代码](https://images.gitee.com/uploads/images/2021/0401/140616_7349ac72_738550.png "修改代码.png")

    执行`python Train.py`，脚本成功运行！
    ![成功运行脚本](https://images.gitee.com/uploads/images/2021/0401/141225_7d94395b_738550.png "成功运行脚本.png")
    