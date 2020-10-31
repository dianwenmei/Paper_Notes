# Background

构建越深越大的卷积神经网络是视觉分类的最重要解决方法，并且效果最好的CNNs通常拥有上百层、上千层channels,因此需要大量的计算力。而此

# Motivation

致力于构建一个可以在有限的计算资源中获得最好的准确率，例如在智能手机、机器人等平台中。

**使用的主要方法**：

- pointwise group convolutions
- channels shuffle

相较于MobileNet的优势是：可以使用更多的Channels，以获得更高的准确率



# Approach

**MobileNet的问题**：

使用了 depthwise separable convolutions方法，但是其忽略了1 * 1 Conv的复杂性，因其占用了大量了计算量。所以为了减少1 * 1Conv的计算量，只能通过减少Channeld的数量，这就造成了准确率的降低。

如何解决此问题：既要保证channel的数量，也要保证更少的计算量

## 什么是pointwise group convolutions

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201007001828969.png" alt="image-20201007001828969" style="zoom:80%;" />

分组卷积即将feature-map分组，每组就可以使用更浅的卷积核对每组单独进行卷积，造成的结果是每次卷积只利用了一部分的空间信息，没有全部利用到。如果堆叠group convolution 则会造成每组每次都是只包含之前输入的一部分channel信息。但是这样可以**减少参数量**

所以本论文使用了pointwise group convolution方法，在进行1 * 1Conv融合channel时，使用分组卷积，降低了计算量但是维持了维度，**问题**是：会造成某些组只与input的某些层相关，所以再使用shuffle channel方法解决。

------

因此根据以上的想法，构造了shuffle unit，如下所示。借助resident block，通过对其改进实现。

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201007151341135.png" alt="image-20201007151341135" style="zoom:80%;" />

## 什么是shuffle channel

shuffle channel就是为了解决分组卷积后每组只是一部分channel的信息融合。它通过在每组内进行分更多的子分组，然后分别从每组中挑选一个子组重新组合成一个新的分组，这样channel的分布就更加均匀，到后面在进行分组卷积时，便能更好的信息流传递。

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201007164438528.png" alt="image-20201007164438528" style="zoom:80%;" />

# Architecture

**细节1**：

ShuffleNet的架构是通过堆叠shuffle unit构建出来的，其中每个stage中的第一个unit，stride设为2，实现**下采样**，后面的units的stride设为1，保持相同的feature-map size,且池化过程也通过添加padding保持size。

**细节2**：

在每个unit中的bottleneck中保持resident的设置，将输入的feature-map数量reduce为此unit输出channel数量的 1/4。

**细节3**：

保持每个stage的输出是前一层stage输出的feature-map数量的2倍。

**细节4**：

通过增加更多的分组，实现了输出更多的channel数量，保持更多的信息流的传播，同时参数量不会增加太多（实现了增加channel数量，同时保持了较少的参数量）。

**细节5**：

最后输入到全连接层时，使用global-pooling将数据展成向量。

架构表如下：

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201007165720577.png" alt="image-20201007165720577" style="zoom:80%;" />

 

