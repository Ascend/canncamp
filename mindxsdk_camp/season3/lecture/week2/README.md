## 人脸关键点检测

### MindStudio (window)

#### 安装指导

https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-3MindStuido%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md

#### 依赖安装下载

- JDK 11

https://repo.huaweicloud.com/java/jdk/11.0.1+13/jdk-11.0.1_windows-x64_bin.exe

- Python3.7.5

https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe

- MinGW

https://nchc.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/8.1.0/threads-posix/seh/x86_64-8.1.0-release-posix-seh-rt_v6-rev0.7z

- CMake

https://github-releases.githubusercontent.com/537699/bb003702-495e-44d7-a5c0-fab03631ad08?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210926%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210926T130723Z&X-Amz-Expires=300&X-Amz-Signature=5df77f3e710c0f34cd41d6ed39a044b3f192147cd3845e42bd59ad95918db31c&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=537699&response-content-disposition=attachment%3B%20filename%3Dcmake-3.20.6-windows-x86_64.msi&response-content-type=application%2Foctet-stream

### 程序执行

**修改test.pipeline的后处理库路径**

```bash
"mxpi_objectpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "yolov4/yolov4.cfg",
                "labelPath": "yolov4/coco.names",
                "postProcessLibPath": "${MX_SDK_HOME}/lib/modelpostprocessors/libyolov3postprocess.so"
            },
            "factory": "mxpi_objectpostprocessor",
            "next": "mxpi_distributor0"
        }
```

修改`${MX_SDK_HOME}`为具体路径

**MindStudio run配置**

- 配置执行环境变量

```bash
MX_SDK_HOME=安装的SDK的路径
LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
PYTHONPATH=${MX_SDK_HOME}/python
```

**注意：**由于`MindStudio`无法自动替换`${MX_SDK_HOME}`，需要将所有`${MX_SDK_HOME}`替换为具体的路径

- 执行脚本

点击`run`按钮执行

