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
    #创建session
	config = tf.ConfigProto()
	custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
	custom_op.name = "NpuOptimizer"
	config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
	sess = tf.Session(config=config)
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
    npu_config=NPURunConfig(
	  model_dir=FLAGS.model_dir,
	  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
	  session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False) 
	  )
    ```
4. 使用Tensorflow1.15实现LeNet网络的minist手写数字识别。硬件平台不限，可以基于window或者linux系统，尽量给出整个过程的截图，并在最后给出Loss或者accuracy运行结果。 参考链接[Gitee](https://gitee.com/lai-pengfei/LeNet) ~ [Github](https://github.com/ganyc717/LeNet)。【分值：6分】
    
    ## 实验环境
    windows10、Tensorflow-gpu-1.15、python3.7.5实现
    ## 实验步骤
    
    1.安装miniconda3（当然可以选择其他，个人习惯）<p>
    去[官网](https://docs.conda.io/en/latest/miniconda.html)下载minconda3，笔者选的是3.8<p>
    ![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/master/training_camp/task_1_hw38939615/miniconda3.png)
    然后双击安装，按照向导一路next<p>
    2.创建python3.7.5虚拟环境tensorflow-gpu-1.15.0<p>
    ```
    conda create -n tensorflow-gpu-1.15 python=3.7.5
    ```
    3.安装tensorflow-gpu-1.15.0<p>
    ```
    进入到刚装好的tensorflow虚拟环境
    conda activet tensorflow-gpu-1.15
    安装tensorflow-gpu-1.15包
    pip install tensorflow-gpu==1.15
    安装Pillow包（LeNet项目中有用到）
    pip install Pillow
    ```
    ![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/master/training_camp/task_1_hw38939615/tensorflow-gpu-1.15.0.png)
    4.clone LeNet代码<p>
    ```
    git clone https://gitee.com/lai-pengfei/LeNet
    ```
    5.修改代码中训练脚本<p>
    ![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/master/training_camp/task_1_hw38939615/modify_train_script.png)
    6.运行Train.py脚本<p>
    ![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/master/training_camp/task_1_hw38939615/run_train.png)
    7.训练集结果<p>
    ![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/master/training_camp/task_1_hw38939615/training_result.png)
    可以看到，在训练了5000次迭代之后，准确率达到0.98,同时训练集的loss也下降了0.197263，基本上已经收敛
    8.修改inference代码
    ![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/master/training_camp/task_1_hw38939615/inference.png)
    注意，要保持训练和推理的数据预处理是一致的，也即在训练脚本中没有做/255归一化，那么在测试集中也不能做/255归一化，否则预测结果就很差
    9.新建test.py脚本在测试集上验证模型
    ![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/master/training_camp/task_1_hw38939615/test.png)
    10.测试结果
    ![输入图片说明](https://gitee.com/XxyMy/canncamp/raw/master/training_camp/task_1_hw38939615/test_result.png)
    ###可以看到测试集上的accuracy是0.956，而训练集上的accuracy是0.98，说明训练过程出现了过拟合现象。。。。    
    
   

   