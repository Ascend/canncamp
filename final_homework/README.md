账号：epsilon-code


一、按照教程部署环境

二、自定义算子rsqrt

解压缩 tar -zxvf AscendProjects.tar.gz

启动mindstudio

打开算子工程：open-MyOperator

配置python之后在rsqrt.py页面点击右键选择Run

部署：

菜单Build- edit build configuration-选择local Build

菜单Ascend-Deploy-Deploy Locally

二、在tf代码中调用

打开训练工程：open-MyTraing

先执行train_relu.py，在编辑界面右键，Run train_rsqrt.py 通过自定义算子中插入的打印语句
验证自定义算子有被调用：
