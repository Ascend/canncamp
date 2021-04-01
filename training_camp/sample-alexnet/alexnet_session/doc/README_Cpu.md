本仓的代码仍然可以在CPU上执行，只需要你在执行脚本中指定参数chip为cpu即可。

- 如果你使用window系统，对应着修改下dataset数据集路径，然后执行下面命令：
    ```
    python.exe .\train.py --dataset=E:\Dataset\Flowers-Data-Set --result=.\log --chip='cpu' --num_classes=5 --train_step=6
    ```
- 如果你在linux下，直接在terminal上执行命令 `sh scripts/run_cpu.sh`
    ```
    --data_url=/home/Flowers-Data-Set \          ## dataset path
    --chip='cpu' \                               ## set chip 
    --train_url=./log \                          ## log save path
    --num_classes=5 \                            ## number of classes
    --train_step=6                               ## Total train steps
    ```