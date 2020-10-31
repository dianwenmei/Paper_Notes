# Inception

1* 1Conv的作用：

* 降维或升维
* 跨通道信息交融
* 减少参数量
* 增加模型深度



<<Going deeper wit convolutions>>

演变：V1(googLeNet) -> BN-Inception V2 -> V3 -> V4 -> Inception ResNet -> Xception

通过精心设计的一个模块，实现了增加深度与宽度的同时，大大减少参数，增加计算速度，减少计算预算。

核心：设计Inception模块，使用不同大小的卷积核收集特征信息，然后通过channel融合在一起（使得feature-map的size相同）

启发文献：《Network in Network》（1*1Conv降维，采用global avg pooling取代全连接层中的展开）、《》（用稀疏网络可以取代密集网络）

基于神经研究，不同神经元识别不同的特征

## 思想来源：

传统的提升网络效果的方法：

* 增加深度
* 增加卷积核数量，每层的参数数量

问题：参数过多、算力过大，过拟合

本论文的方法： **使用稀疏连接取代密集连接**



特征信息多尺度的并行处理

辅助分类器：相当于添加了正则化、给前面层注入梯度 



解决问题:

* 图像识别
* 目标检测

## Motivation

传统的提升网络效果的方法：

* 增加深度
* 增加卷积核数量，每层的参数数量

问题1：

1. 大量参数，易过拟合，尤其数据量小的时候
2. 增加计算量，如果训练后权重是0，则算力被浪费

------

解决以上问题的方法： **用稀疏连接取代密集连接**

```markdown
理解：
密集连接：上一层的输出，经过*多个相同尺寸的卷积核* 过程后形成下一个输出，数据密集
稀疏连接：上一层的输出，经过*多个不同的卷积核* 过程形成下一层的输出，数据更稀疏
```

考虑改进网络的结构，使得更好的利用稀疏性



## Architecture

### naive Inception构建过程

低层获取小范围的局部信息，高层获取大范围的信息，所以设计卷积核时使得一开始卷积核小，后面使用大的卷积核。为了使得尺寸相当，使用了padding对齐方法。1*1Conv、3 * 3Conv padding=1、5 * 5Conv padding= 2

```
n - m + 2x   = n  - 1
2x = m - 1
```

由此设计出Inception模块，Inception模块是一个结合体，结合了前一层由不同尺寸的卷积核（1 * 1、3 * 3、5 * 5）卷积后的feature_map，同时池化对卷积很重要，所以在每一层中也加入了池化层（pooling）。naive版本如下所示。

![image-20201005142353744](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201005142353744.png)

模块内部使用多尺度的卷积核收集图像信息， 网络不断堆叠此模块，且网络越深模块中 3 * 3、5 * 5Conv的比例越高(即其数量越来越多，使得组成后下一层输入的占比变多) ，因为越深层次偏向于学习更高抽象的特征（因为通过卷积与池化，使得feature-map中的每个值都是一个更大的感受野的映射，所以越到后面每个值融合了之前的一部分信息，只有扩大尺寸才能识别到范围更大的特征）

#### 问题1：

* leading to a computational blow up within a few stages 计算爆炸问题

**embeddings**:  even low dimensional embeddings might contain a lot of information about a relatively large image patch.

**改进**：受其启发，使用1 * 1Conv进行降维，又由于此操作是密集的，所以只在需要时使用，即在进行3 * 3、5 * 5Conv前使用1 * 1 降维，且使用ReLU增加非线性。

由于初始图片尺寸过大，Inception使用的卷积核尺寸太小，易造成经过几层后feature_map的尺寸依然很大，造成内存问题，所以初始先使用传统卷积方法降低feature_map的尺寸，降低内存使用，后面再开始使用Inception模块。

改进版如下所示。

![image-20201005152218673](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201005152218673.png)

#### 问题2：

网络太深，使得梯度回传出现问题

**解决方法**：在中间层加入分类器，将其loss加入到总loss中，且占0.3，使得加速梯度

回传，且加入正则化。

GoogLeNet的总体架构参数设计：

特点：

1. 首先使用传统卷积网络
2. 后面使用inception模块
3. 中间额外添加分类器，将其
4. 最后使用global-avg-pooling，用平均值代表整张feature-map，省去了大量的参数计算（传统方法，将7 * 7 * 1024展成向量，再进行线性计算，参数为7 * 7 * 1024 * 1000 + 1000；现在只有1 * 1 * 1024 * 1000  + 1000个参数）

![image-20201005155234094](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201005155234094.png)



![image-20201005183145631](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201005183145631.png)

辅助分类器的设计：

	* 5 * 5， stride = 3
	* 1 * 1, 128个
	* 1024全连接层
	* dropout 0.7
	* 全连接层 1000

# 训练的技巧

使用了数据并行，异步的SGD，图片裁剪等方法

预测技巧：

使用7个模型对同一张图片的不同裁剪取平均值作为最终的预测结果。

图片裁剪方法：

- 按照短边裁剪到四个scale，256，288，320，352，
- 裁剪左，中 右（上中下）三张图片
- 裁剪四个角，中间以及原图到224 * 224，6张图片
- 最后做水平翻转，所以一张图片被裁减成4 * 3 * 6 * 2 = 144





# work的原因



**我的理解**：使用不同尺寸的卷积核并行处理，把multi-scale filter生成的不同感受野的特征融合到一起，有利于识别不同尺度的对象

**疑问**：

Inceptiont系列始终坚持不用identity shortcut，但在性能上却能跟用了identity shortcut的网络打得难解难分。更奇怪的是，Inception v4这篇paper中验证了增加identity shortcut对于Inception精度的提升非常有限，仅仅是加速了训练而已。为什么对其它结构可以立竿见影涨点的identity shortcut，对于Inception来说却略显无力？



inception work的真正原因是什么？

同一层数据，使用不同的尺寸的卷积核进行卷积，且使用1 * 1Conv，充分利用了feature-map的信息。



优势：减少计算量，但是收到内存的限制。