# Operator Campus

### 介绍
昇腾CANN训练营(第三期)——算子开发营。算子开发营作业共有两道题目，请大家将代码提交到canncamp仓

    
首先，将[canncamp仓](https://gitee.com/ascend/canncamp)fork到个人分支，基于个人分支在`operator_camp`文件夹下新增自己的工程文件夹。文件夹命名规则为`华为云账号`；比如：`fulltower`。

其次，上传源码。将自己的源码添加到新建的文件夹中，删除冗余的代码/文件，完善readme，并更新到自己的个人分支。

最后，提交PR。在提交PR时，算子营的标题请标注为【昇腾CANN训练营第三期——算子训练营】华为云账号。比如：
```
【昇腾CANN训练营第三期——算子训练营】fulltower
```

### 课程作业

##### 题目一：使用TIK实现element-wise的加法算子

其中：
输入和输出数据类型都是float16类型，tensor大小为（16,16,16,16,16）。

提交的文件名命名为 **eltwise.py** 。

其逻辑表示为：

```
for (int i0 = 0; i < 16; i++)
    for (int i1 = 0; i < 16; i++)
        for (int i2 = 0; i < 16; i++)
            for (int i3 = 0; i < 16; i++)
                for (int i4 = 0; i < 16; i++)
                    C[i0, i1, i2, i3, i4] = A[i0, i1, i2, i3, i4] + B[i0, i1, i2, i3, i4]
```

##### 题目二：使用TIK实现softmax算子
其中：
需要支持数据类型为float，同时支持任意维度的shape，即算子需要支持泛化能力。

提交的文件名命名为 **softmax.py** 。



