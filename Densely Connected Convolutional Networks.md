# Densely Connected Convolutional Networks



## 1.motivation

**是为了解决什么问题**

随着一系列CNN模型的出现，其深度越来越深，由此出现了**新的问题**：

- 1.图像信息流传递过程中出现信息丢失
- 2.梯度回传过程中出现梯度消失现象

第一点可能会造成模型的训练误差变大；

第二点，由于深度很深，当进行链式法则计算梯度时，如果出现了过多的小于0的数相乘，则会造成最终结果趋近于0，所以造成前面的参数不能有效的更新。使得模型的训练时间增大。



![](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20200928235238935.png)

## 2.How

是怎么解决问题的，算法描述

鉴于之前一些列模型的出现，ResNets, Highway Networks,  Stochastic depth, FractalNets, 他们都有一个共同点：构造了连接从前面层到后面层的一个短连接（skip connections, shortcuts），此论文则总结出了一个简单的连接模式：为了确保最大信息流的传播，此模型把所有层都和其他的所有层进行了连接。

与ResNet的区别：

- 连接数量
  - ResNet: 两个、层或三层加一个shortcuts
  - DeseNet: 每一个输出都添加到后面所有层
- feature-map的连接方式(*为什么这种连接方式更有利于信息的传播*)
  - ResNet: add 
  
  - DenseNet: Channel维度上的融合
  
  - **原因探索**：x1 x2当前层的两个channels,x3 x4 前一层的两个channels
  
  - add的方式进行卷积：w1 * (x1+x2) + w2 * (x3+x4) ，对应channel共享了参数
  
  - channel维度连接方式进行卷积：w1 * x1+w2* x2+w3 * x3 +w4 * x4 每一层使用不同参数训练
  
    > 这样为什么效果好？？？？？

其相较其他算法的优势：

* 有更少的参数，不需要学习冗余的feature -maps（可以随机放弃一些层）
* 更加容易训练，因为最终的Loss Function得到的梯度可以直接传递到每一层。

### 2.1算法的架构

![image-20200929001238491](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20200929001238491.png)

1. Dense Bolck
2. Translation

每一个bolck中的size是相同的便于连接，不同block由于size不同所以靠translation中的pooling来减少size；每层使用相同的卷积核数目，K，称其为growth_rate

naive：

* 在每个DenseBlock内部进行BN -> ReLU -> 3*3Conv
* Translation: BN -> 1*1Conv -> avg-Pooling

问题1：在DenseBlock中的最后几层，有过多的feature-map输入

解决方法：DenseNet-B, 在进行3* 3Conv之前先进行一次1* 1的Conv，BN -> ReLU -> 1* 1Conv -> BN -> ReLU -> 3*3Conv

问题2：每个DenseBlock的输出为K，模型过大

解决方法：DenseNet-C在每个Translation层中使用1* 1Conv时，卷积核数目 = input<_feature_map_nums * $$\theta$$ , $$\theta$$ < 1时，实现压缩效果。

*减少深度，减少了参数数量，但是精度能够保证吗？？*

实验使用模型为DenseNet-BC，

### 2.2算法的实现思路

### 2.3 算法的实现及实现细节

模型训练的技巧：

1. 使用SGD优化器，batch-size=64,epoch=40~300
2. learning_rate,初始为0.1，在训练完50%和75% 数据量后分别divide10
3. weight-decay 设为 $$10^{-4}$$ 
4. momentum = 0.9 

## 3.Why

算法为什么work,为什么起作用

我认为此模型work的原因：

* 高度重用了图像信息，弥补了图像信息在深度网络中传递过程中不可逆的消息丢失（且已经证明[^1]，如果在恒等映射中添加激活、1* 1Conv和dropout时，效果会下降，而单纯使用激活或1* 1Conv时短层效果提升，深层依旧效果变差，是不是可以说明图像信息流确实在深层的传播中丢失了很多信息。）

  [^1]: identity mapping in deep resnet

  

## 4.该论文对我的启发

此模型的得出是基于之前一些列的论文，加几条短连接后有很好的效果，即可以实现梯度的成功回传，又可以保证当此卷积层丢失过多图像信息时，将前几层的feature-map拿过来弥补。

基于以上信息，该论文就大胆尝试加入更多的短连接。

------

想要降低channel的维度，使用1* 1进行卷积；

想要降低feature_map的size，使用 **pooling** 或直接**卷积**。



使用dropout，卷积或造成图像信息的丢失。

## 5.该论文方法的不足



