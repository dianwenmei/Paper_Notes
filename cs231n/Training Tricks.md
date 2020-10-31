# Training Tricks

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201013001246834.png" alt="image-20201013001246834" style="zoom:50%;" />

## Weight的初始化

1. Weight初始化
   - 太小：激活值为0
   - 太大：激活值过大，爆炸

使用Batch Normalization的原因是，将数据分布在正态分布上，这样使得可以使的权重也在0 附近，不会造成W很大，降低学习难度。

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201013104104792.png" alt="image-20201013104104792" style="zoom:50%;" />

2. train上的准确率还在提升，而验证集开始下降，说明模型过拟合了需要加入正则化。 

## 优化器策略

SGD的问题：

* 如果一个梯度很大，一个梯度很小，横坐标移动一点，纵坐标移动很多，则会造成走很多弯路 **之** 形，==那么如何避免==？
* 落在鞍点处或梯度很小处
* minibatch，如果有噪音则会影响结果：意味着X是会变化的，如果X不变则一直改变W就可以找到最小值，而X不断变化则W也会不断变化即使上一次已经到了一个局部最优解

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201013110238024.png" alt="image-20201013110238024" style="zoom:50%;" />

### momentum + SGD

带动量的SGD，momentum：更新梯度时不是直接学习率 乘 梯度，而是学习率 乘 动量，而动量是一个累计值Vx = p * Vx + dx。可以解决以上问题，但是==为什么==？，会加速梯度较小的方向，但是也会加速提速更大的地方；虽然可以在梯度为0的极小值地方继续前进，但是到了最小值怎么办。

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201013122029609.png" alt="image-20201013122029609" style="zoom:80%;" />

**理解**：足够多次后，由于 乘 p，且p<1，所以刚开始的梯度乘以趋于零的数，则表示越到后期，前期的梯度的影响就越小。如果最小值是一个周围梯度比较缓的情况下，带动量的SGD会在周围迭代很多次后梯度值都很小，以至于最后收敛在此地。如果坡度很陡，则可能会因为梯度的权重和过大而跳过最小点。所以，==最终问题点在于：什么样的点是好的点，梯度更缓的还是更陡的==

==因为坡度更缓的点说明泛化能力更强，所以找的是更缓的点==。

==带动量的SGD特性：可以跨过陡峭的极值点。==

### AdaGrad

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201013125939335.png" alt="image-20201013125939335" style="zoom:100%;" />

累加每次 dx * dx的值，然后根据**之前累加的所有梯度的平方和**对当前的梯度进行放缩。 这样就可以解决在某一方向上梯度值特别大，某一方向上梯度特别小，造成 **之** 形形况。==梯度大则除以大的数，当前梯度将会缩小，梯度小则会除以一个较小值，如果小于1，则会加速在此方向的梯度。==

 ==adagrad的问题==：由于grad_squard是一直递增的，所以会导致步长越来越小，如果是凸函数还好，如果是非凸函数则会无法继续更新。

一般不使用adagrid，因为其到后期步长会不断变小，训练变慢。改进版本：

### RMSProp

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201013221214230.png" alt="image-20201013221214230" style="zoom:50%;" />

decay_rate: 小于1，使得之前累加的梯度的平方和进行衰减，并且对当前的梯度也进行一个衰减，这样当进行很多迭代后，总的grad_squared将会减少，就不会出现步长减慢的现象了。 

------

带动量的优势：

​	累计了之前的所有梯度，所以其更新速度更快，且可以跨越陡峭的极值点，在坡的极值点收敛。

Adagrid和RMSProp的优势：

​	加速梯度较小的方向，减缓梯度较大的方向。

结合两者的优点：Adam

### Adam

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201014001419944.png" alt="image-20201014001419944" style="zoom:80%;" />

==beta1 = 0.9, beta2 = 0.999, learning_rate = 1e-3, 4e-4==

### 总结

带动量的SGD的学习衰减很常见，但是Adam很少有学习率衰减

Adam可以在任何情况表现良好以恒定的learning_rate;

SGD + Momentum 可能会超过Adam，但是需要调整learning_rate;

如果可以承受所有数据，可以使用L-BFGS。 

## Learning rate schedules

什么样的learning_rate是好的：

- 初始较大
- 逐渐减小
- 迭代次数越多应该越小

==策略==：

1. step: 每隔多次epoch，learning_rate除以10.

   ![image-20201017100159195](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201017100159195.png)

2. Cosine： ![image-20201017100227970](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201017100227970.png)

3. Linear：![image-20201017100251722](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201017100251722.png)

4. inverse:![image-20201017100358824](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201017100358824.png)

5. Linear Warmup: 在初始阶段learning_rate迅速升高，然后再慢慢减小。

==检验法则==：如果Batch变大，则learning_rate可以增大（If you increase the batch size by N, also scale the initial learning rate by N）



### 总结

1. Adam很好的默认选择，即是使用固定的学习率表现得也很好。
2. SGD+Momentum: 效果可能比Adam更好，但是需要调整学习率
3. 如果使用full batch，可以使用==L-BFGS==

## 提升测试的结果

提升test_data的结果：

==使用模型集成==：

- 单独训练多个模型
- 预测test_data时，取各模型的平均值。

==如何提升单个模型==：

### Regularization: 加到loss函数中

- L1
- L2
- (L1 + L2) 

### dropout

随机将一部分的激活函数值置为0，那么输入到下层激活函数时就有一部分的值是0。

一般是在全连接层使用。

解释：

![image-20201014125108362](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201014125108362.png)

随机去掉一些特征，再进行分类，限制了不同特征间的依赖。有效的防止了过拟合。

在训练时，激活函数有P概率被激活，而在测试时所有的激活函数都被激活，所以为了保证输出值的期望相等，在测试时每个激活函数要乘P的概率。We must scale the activations so that for each neuron: output at test time = expected output at training time



## 数据增强

数据增强的方法:

1. 水平翻转

2. 随机裁剪和缩放

   Training: 

   	1. sample random crops / scales
   ResNet:

   1. Pick random L in range [256, 480]
   2. Resize training image, short side

   Testing: average a fixed set of crops
   ResNet:
   1. Resize image at 5 scales: {224, 256, 384, 480, 640}
   2. For each size, use 10 224 x 224 crops: 4 corners + center, + flips

3. Color Jitter：随机调整对比度和亮度

根据自己的问题使用不同的方法：translation - rotation - stretching shearing, - lens distortions

### 总结

A common pattern： dropout、BatchNormalization、data Augmentation、

DropConnect、

Fractional Max Pooling

Stochastic Depth：添加短连接

Cutout / Random Crop： 随机选择图片中的一块区域像素点置为0

Mixup：随机选择两张图片进行融合

## Choosing Hyperparameters

Step 1: Check initial loss

Step 2: Overfit a small sample

Step 3: Find LR that makes loss go down：Good learning rates to try: 1e-1, 1e-2, 1e-3, 1e-4

Step 4: Coarse grid, train for ~1-5 epochs

Step 5: Refine grid, train longer：train them for longer (~10-20 epochs) without learning rate decay

Step 6: ==Look at loss curves==：use a **plot** and also plot **moving average** to see trends better

	1. Accuracy still going up, youneed to train longer
 	2. **Huge train / val gap** means **overfitting! **Increase regularization, get more data
 	3. No gap between train / val means **underfitting**: train longer, use a bigger model

## 迁移学习

### motivation 

使得仅使用小数据集就可以训练得到一个卷积神经网络，在此之前训练一个卷积网络需要大量的数据。

### Methods 

所有的卷积层中的参数不变，重新训练最后的全连接层。
 即前面的卷积层只是完成了特征提取的功能，分类任务还是在最后的全连接层。

|          | 与之前的相似数据 | 与之前的不相似 |
| -------- | ---------------- | -------------- |
| 少量数据 | 只训练全连接层   | trouble        |
| 大量数据 | 可以微调其他层   | 需要调整很多层 |

​					 