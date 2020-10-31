论文结构

- 摘要
- 介绍此论文的工作以及文章结构
- 此问题的文献综述
- 详细介绍自己的设计，网络架构，算法
- 描述应用此架构或算法的实验
- 总结及讨论

# Background

构建小而高效的神经网络的方法一般是：

- 压缩与训练模型
- 直接训练得到一个小而高效的模型

其他模型只关注模型足够小，不太考虑速度问题，而mobilenet**专注于优化延迟**问题。

一种使用depthwise separable convolutions获得小模型；

其他不同的方法：

- 收缩、因式分解、压缩预训练模型(shrinking, factorizing or compressing pretrained networks.
  Compression)
- 蒸馏 distillation
- low bit networks





# MobileNets

## Motivation

构建一个高效的网络架构，以及两个超参数，可以使网络更小、延迟更低、更加容易应用在受资源限制的移动以及嵌入式视觉应用中。



# methods

## 概述

**depthwise  Separable Convolution**：它的核心思想是将一个完整的卷积运算分解为两步进行，分别为Depthwise Convolution与Pointwise Convolution  

- Depthwise Convolution： 
- Pointwise Convolution   

利用上述的两个方法降低了参数数量。首先每一层用不同的卷积核只在自己平面上做卷积，这样产生了相同深度的feature-map，但是每一channel是独立的没有充分利用空间信息。再使用1 * 1 * depth卷积核，融合深度信息，有几个卷积核就产生几个feature-map。

------

MobileNet的架构即不断堆叠 depthwise  Separable Convolution，最后使用global avg-pooling代替flatten展平向量连接全连接层。

再使用两个超参数给mobilenet做瘦身，以适应不同的应用场景：

- Width Multiplier：减少input和output的feature-map的数量，p*，减少参数两和计算量
- Resolution Multiplier：减少feature-map的size，减少了计算量，不改变参数量

## Depthwise Separable Convolution



![image-20201006152314580](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006152314580.png)

M表示channe数量，$$D_k$$表示卷积核size，N表示输出的channel数量。

分两步进行

- 首先使用M个$$D_k * D_k * 1$$的的卷积核，分别对每一channel进行卷积
- 再使用N个$$1 * 1 * M$$的卷积核融合空间信息。

目的是降低参数数量以及运算量。

### 参数量分析：

1. 正常卷积

$$D_k * D_k * M * N$$

2. Depthwise Separable Convolution

$$D_k * D_k * 1 * M + 1 * 1 * M * N $$

化简后即证明 $$\frac{1}{N} + \frac{1}{D_k*D_k} < 1$$，因为N表示输出的feature_map数量，一般较大，$$D_k$$一般为3，所以参数量减少。

### 计算量分析

$$D_F$$表示输出的feature_map的height或width

1. 正常卷积：

$$D_k * D_k * D_F * D_F * M * N $$

2. Depthwise Separable Convolution

$$D_k * D_k * 1 * D_F* D_F * M + 1 * 1 * D_F* D_F * M * N $$

证明与上面一致。

### 参数分布

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006160957142.png" alt="image-20201006160957142" style="zoom:80%;" />

集中再1 * 1 卷积容易优化。

## 架构



<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006155313322.png" alt="image-20201006155313322" style="zoom:80%;" />

使用右边的结构代替传统的卷积操作。

mobileNet的整体架构如下：

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006155429791.png" alt="image-20201006155429791" style="zoom:80%;" />

# Experiments

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006160045702.png" alt="image-20201006160045702" style="zoom:80%;" />



<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006160121129.png" alt="image-20201006160121129" style="zoom:80%;" />

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006160457625.png" alt="image-20201006160457625" style="zoom:80%;" />

2. 脸部特征分类

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006160653024.png" alt="image-20201006160653024" style="zoom:80%;" />

3. 目标检测

   <img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006160736867.png" alt="image-20201006160736867" style="zoom:80%;" />

4. 人脸嵌入

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201006160843478.png" alt="image-20201006160843478" style="zoom:80%;" />

