### 模型营第2周作业【共计15分，其中3分根据提交的PR作业由营长酌情给分】

1. 如果需要导入NPU相关库，需要import什么包？ 【分值：2分】
	
    ```
    xxx
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
    xxx
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
    xxx
    ```
4. 使用Tensorflow1.15实现LeNet网络的minist手写数字识别。硬件平台不限，可以基于window或者linux系统，尽量给出整个过程的截图，并在最后给出Loss或者accuracy运行结果。 参考链接[Gitee](https://gitee.com/lai-pengfei/LeNet) ~ [Github](https://github.com/ganyc717/LeNet)。【分值：6分】
    ```
    步骤1截图
    步骤2截图
    ...
    ```

   ![输入图片说明](https://images.gitee.com/uploads/images/2021/0331/130148_5de9d0b8_1482256.png "屏幕截图.png")
    