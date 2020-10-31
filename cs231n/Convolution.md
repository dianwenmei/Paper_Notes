# lecture_7
## Convolution

1. pointwise convolution的意义是：
   - 降维
   - 整合更多信息
2. 卷积核的size与stride应该要与输入的feature-map的size相匹配，最好不要出现不能整除的现象，因为这样会造成信息的不对称。（==可能会造成更多的计算法复杂？？？==）
3. CNN向后的发展有用 **Conv** 代替 **pooling**的趋势，原因分析：
   - pooling的作用，降低feature-map的size，去掉一些无用信息（背景信息？）
   - 使用稍大的stride的Conv可以实现降低size的功能，同时Conv并没有直接丢弃图像信息，而是将其融合，尽可能的保留了图像的所有信息，但是==提高了计算量，增加了更多参数，而pooling不会增加新的参数和计算量==
4. CNN的架构一般是：Conv - > ReLu ->**** -> pooling ->  **** -> FC -> ReLU -> softmax，但是最近也有了新的架构形式，
   - ResNet：没有使用pooling，增加了shortcut
   - GoogLeNet：每层中使用了多个尺度的卷积核，充分利用了图像数据，reuse

## Activations

<img src="C:\Users\mdw\AppData\Roaming\Typora\typora-user-images\image-20201009205919682.jpg" alt="image-20201009205919682" style="zoom:50%;" />

1. 添加activation function得原因：**增加非线性，否则不管使用多少层依然相当于一层的线性函数**
2.  1和3都是将数据压缩到[0, 1]，且1可能造成的问题：
   - 饱和函数，容易造成梯度为0，导致向后传递梯度时，传来的梯度太小，不断乘W得到下一层梯度时全都为0，
   - 数据不是压缩在以零为中心。
   - e()计算复杂度过高
3. 3ReLU可能会造成的问题：
   * 当初始W， b设为0时，则导致W*X + b = 0, 使用ReLU激活后依然为0，则后面所有输出全为0，当梯度反向传播时，由于所有层的x都为0，则每层的W的梯度为 $$\frac{dL}{dz} * \frac{dz}{dW} = \frac{dL}{dz} * X$$，X为0，所以每层的W都不会更新，造成无法学习。
4. 所以出现了4对3的改进。
5. ==为什么通过每层使用ReLU可以有很好的效果，ReLU的根据作用是什么？？？==

## Data Preprocessing 

1. ==为什么更喜欢数据以0为中心分布？？？==：
   - 数据更集中，更加有助于找到模式
   - 数据不会一直为positive或negative，导致一层这种的所有参数要么全正要么全负。

<img src="C:\Users\mdw\AppData\Roaming\Typora\typora-user-images\image-20201009213121865.png" alt="image-20201009213121865" style="zoom:50%;" />

## Weight Initialization

1.  First idea: Small random numbers （g==aussian with zero mean and 1e-2 standard deviation），小网络可以，如果是深层网络会遇到什么问题？？？==：
   - 每层的W梯度为：**后面传来的梯度值 * X **，而下一层的梯度值为：**后面传来的梯度值 * W **，每一层的梯度为$$\frac{dL}{dZ} * W_i^n$$，所以当W以0为中心、且网络过深时，每一层的梯度值会趋于0，导致每层的W的梯度趋于0，造成无法学习。
   - W趋于0，后面深层的的activation function的输出也会趋于0，$$W_{i+1}tanh(W_i*x + b) + b$$，一直相乘。
2.  ==“Xavier” Initialization== ：
3. ==Kaiming / MSRA Initialization==

<img src="C:\Users\mdw\AppData\Roaming\Typora\typora-user-images\image-20201009214801634.png" alt="image-20201009214801634" style="zoom:50%;" />

## Batch Normalization

1. 使用方法$$\frac{x^k - E[x^k]}{\sqrt{Var[x^k]}}$$，计算出每个维度的均值与方差，对所有数据进行归一化处理，使得每个维度的数据都是服从**标准正态E(0, 1)分布**的。

   因为在激活时，最后做到更加随机分布，所以需要在激活函数之前使用BN，防止激活后所有都是正或负的情况。FC -> BN -> act -> FC -> BN -> act...

2. BN的优势：
   - 网络更容易训练，==why==，数据始终是正态分布？ ？



------

1. 在训练数据之前，可以先使用小数据集、去掉正则项，查看loss的输出能否逐渐变小到0，检查完毕后，再使用所有数据集。
2. 损失值变化很小，表示learning_rate设置的过小
3. loss输出变为NAN或越来越大，表示设置的learning_rate过大。
4. 超参数的学习，使用交叉验证集来确定

------



