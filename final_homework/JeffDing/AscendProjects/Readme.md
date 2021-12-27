## 自定义算子复现  (华为云ID：JeffDing）

### 安装环境

```bash
cd ~/work
pip3.7.5 install tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl --user
./Ascend-cann-tfplugin_5.0.2.alpha003_linux-aarch64.run --install
tar zxvf AscendProjects.tar.gz
cp -r AscendProjects/ ../
cp .bashrc ../
cd ..
source .bashrc
```

自定义算子的编写在Mindstudio 上进行，本环境上有安装好的Mindstudio。  
在home 目录下执行以下命令： 
```bash 
cd Mindstudio/bin  
./Mindstudio.sh
```
在MindStudio中打开MyOperation文件夹，点击菜单栏Build->edit build configuration。菜单栏中TargetOS选择LINUX,目标架构选择AARCH64

随后点击Build

出现编译完成就说明编译成功

然后在菜单栏选择Ascend-Deploy，随后选择locally，点击Deploy部署自定义算子