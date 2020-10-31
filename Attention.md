**其他使用神经网络的方法与问题**

The models proposed recently for neural machine translation often belong to a family of ==encoder–decoders== and encode a source sentence into a fixed-length vector from which a decoder generates a translation.

In this paper, we conjecture that ==the use of a fixed-length vector== is a ==bottleneck== in improving the performance of this basic encoder–decoder architecture（即之前的方式是 将源语言编码成C，然后利用C生成目标语言的所有单词）

we propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly.

In order to address this issue, we introduce an extension to the encoder–decoder model which learns to ==align and translate== jointly.

**改进的方法**

Each time the proposed model generates a word in a translation, it (soft-)searches for a set of positions in a source sentence where the most relevant information is concentrated. The model then predicts a target word based on the context vectors associated with these source positions and all the previous generated target words.（利用已经翻译好的目标语言单词 来学习到 **下一个目标单词** 与 **源语言中各单词** 的相关程度，进而生成更好的Ci，从而更好的预测下一个单词）



**架构的底层框架-RNN**

an ==encodet== reads the input sentence, a sequence of vectors x = (x1, · · · , xTx ), into a vector c.ｈ和ｑ是非线性函数，且ｃ一般取htx

![image-20201023111831386](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201023111831386.png)

The ==decoder== is often trained to predict the next word yt0 given the context vector c and all the previously predicted words {y1, · · · , yt0−1}. In other words, the decoder defines a probability over the translation y by decomposing the joint probability into the ordered conditionals:

![image-20201023121615494](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201023121615494.png)

![屏幕截图 2020-10-23 122208](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\屏幕截图 2020-10-23 122208.png)

**LEARNING TO ALIGN AND TRANSLATE**

<img src="C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201029094550625.png" alt="image-20201029094550625" style="zoom:50%;" />

The new architecture consists of a bidirectional RNN as an encoder (Sec. 3.2) and a decoder that emulates searching through a source sentence during decoding a translation (Sec. 3.1).

It should be noted that unlike the existing encoder–decoder approach (see Eq. (2)), here the probability is conditioned on a distinct context vector ci for each target word yi.（对于预测的每一个yi，都重新计算一个上下文向量c，并未每一个中间层学习到一个权重）

The context vector ci depends on a sequence of annotations (h1, · · · , hTx ) to which an encoder maps the input sentence. ==Each annotation hi contains information about the whole input sequence with a strong focus on the parts surrounding the i-th word of the input sequence.==（每个hi都包含了所有的源输入，且重点关注其附近的单词，所以存在遗忘问题，使用**GRU**解决）



==attention中的核心思想==：在预测下一个单词时，使用一个带权重的向量Ci，同时把$$s_{i-1}$$作为输入，得到yi。最主要的区别是 Ci是一个带权重的向量和，因此可以 寻找到下一个预测单词的源单词，即对齐。==最重点是如何学习到 权重==（==每个源单词 与 已预测完的单词信息$$s_{i-1}$$ 之间的权重 作为每个源单词与下一个预测单词之间的权重==）

i like to eat apple.

我喜欢吃苹果

We parametrize the alignment model ==$$a$$== as a feedforward neural network which is jointly trained with all the other components of the proposed system.

We can understand the approach of taking a weighted sum of all the annotations as computing an expected annotation,where the expectation is over possible alignments. Let αij be a probability that the target word yi is aligned to, or translated from, a source word xj. Then, the i-th context vector ci is the expected annotation over all the annotations with probabilities αij. 

**使用双向的RNN**

A BiRNN consists of forward and backward RNN’s. The forward RNN !f reads the input sequence as it is ordered (from x1 to xTx ) and calculates a sequence of forward hidden states ( !h 1, ··· , !h Tx ). The backward RNN f reads the sequence in the reverse order (from xTx to x1), resulting in a sequence of backward hidden states ( h 1, ··· , h Tx ).

 In this way, the annotation hj contains the summaries of both the preceding words and the following words. Due to the tendency of RNNs to better represent recent inputs, the annotation hj will be focused on the words around xj . This sequence of annotations is used by the decoder and the alignment model later to compute the context vector（encoder模型训练出hj之后，就不用encoder了，只是用decoder每次根据$$s_{i-1}$$ ,与hj生成hj的权重，然后求hj的权重和，得到每次预测yi用到的ci）



**词向量**：把一个维数为所有词的数量的高维空间嵌入到一个维数低的多的连续向量空间中，每个单词或词组被映射到实数域上的向量

使用one-hot编码表示词，每一维表示一个词，如果有3000个词，则有3000维的向量，而且每个位置上只有一个位置是1，其余都是0。由于维度过高，容易造成**维度灾难**，所以在深度学习中一般使用词向量的表示形式。（维度灾难指特征过多导致容易过拟合，尤其是应用小数据样本时，如果样本数量足够多，则更不容易出现维度灾难）





**模型的选择**



1. RNN中的激活函数选择：GHU：

For the activation function f of an RNN, we use the gated hidden unit recently proposed by Cho et al. The gated hidden unit is an alternative to the conventional simple units such as an element-wise tanh. This gated unit is similar to a long short-term memory (LSTM) unit. ==sharing with it the ability to better model and learn long-term dependencies==

 This is made possible by having computation paths in the unfolded RNN for which the product of derivatives is close to 1. These paths allow gradients to flow backward easily without suffering too much from the vanishing effect

encoder:

![image-20201029122505210](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201029122505210.png)

decoder:

![image-20201029122531017](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201029122531017.png)

都是使用了GRU，目的都是捕捉长距离的依赖关系

其中对于权重的学习即对齐模型 $$a$$,![image-20201029122713321](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\image-20201029122713321.png)

The update gates ==zi== allow each hidden unit to maintain its previous activation, and the reset gates ==ri== control how much and what information from the previous state should be reset. ($$\sigma$$ 是sigmoid激活函数，取值0-1，所以$$r_i$$是减弱之前单词state的信息权重，$$z_i$$是为了保持之前的单词信息的激活，即实现长距离的直接依赖)

![GRU](C:\Users\DianwenMei\AppData\Roaming\Typora\typora-user-images\GRU.jpg)

GRU的计算图