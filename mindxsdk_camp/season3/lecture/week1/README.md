## MindX SDK deeplabv3语义分割案例

### MindX SDK安装

```bash
cd /home/HwHiAiUser/MindX
mkdir MindXSDK
chmod +x ./Ascend-mindxsdk-mxvision_2.0.1_linux-x86_64.run
./Ascend-mindxsdk-mxvision_2.0.1_linux-x86_64.run --install --install-path=MindXSDK
```

### deeplabv3模型准备

- 模型下载

```bash
wget --no-check-certificate https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/modelCVersion/DeepLabv3%20for%20MindSpore/zh/1.4/m/DeeplabV3_for_MindSpore_1.4_model.zip
```

- 解压到`model`目录

```bash
unzip DeeplabV3_for_MindSpore_1.4_model.zip -d ./model/
```

### 运行前配置

- 修改`test.pipeline`配置

`${Mx_SDK_HOME}`改成对应的SDK安装路径

```bash
"mxpi_semanticsegpostprocessor0": {
"props": {
    "dataSource": "mxpi_tensorinfer0",
    "postProcessConfigPath": "model/deeplabv3.cfg",
    "labelPath": "model/deeplabv3.names",
    "postProcessLibPath":"${Mx_SDK_HOME}/lib/modelpostprocessors/libdeeplabv3post.so"
},
"factory": "mxpi_semanticsegpostprocessor",
"next": "mxpi_dataserialize0"
}
```

- 配置环境变量

```bash
export MX_SDK_HOME=SDK安装目录
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
export PYTHONPATH=${MX_SDK_HOME}/python
```

### 程序运行

```
python3 main.py
```

