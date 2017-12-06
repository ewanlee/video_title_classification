This project is developed under python2, Ubuntu Server 16.04 and I use one
Nvidia GPX 1080 Ti graphic card.

Notice that the word2vec model files are not uploaded to this repository because of
Github upload size limitation.

You can find the model files at [here](https://pan.baidu.com/s/1o89R8Oa).

I have used three models:
- TextCNN
- TextRNN
- TextRCNN
- Hierarchical Attention

train\_\*.py are model training files and model\_name.py are model define files
You can view more details on comments in source code.

For now, I get the best multi-class classification accuracy by use TextCNN (0.7). The
hyperparameters have written in code.

--------------------------------------------------

本项目解决的是视频短标题的多分类问题，目前涉及到33个类，所采用的算法包括TextCNN，TextRNN，TextRCNN以及HAN。目前效果最好的是TextCNN算法。

项目流程大体框架如下：



![1](http://o7ie0tcjk.bkt.clouddn.com/github-video-title-classify/1.png)



# 数据预处理

数据预处理部分主要涉及到的文件有：

- `ordered_set.py`
- `preprocess.py`

大致流程如下：

![2](http://o7ie0tcjk.bkt.clouddn.com/github-video-title-classify/2.png)



## 数据加载

初始的文件包括三个：

- `all_video_info.txt` 该文件是后两个数据的合并，作为数据预处理算法输入
- `all_video_info_month_day.txt`（这里的month和day由具体数值替换）这类文件包含多个，**只使用最新的**，是正式的标题数据， 包括已标记的以及未标记的
- `add_signed_video_info.txt` 该文件是从其他数据库中选取的经人工标注的数据，只含有已标记的标题

所有文件的格式都是一样的，每一行代表一个样本，分为四列，中间用制表符间隔。

![3](http://o7ie0tcjk.bkt.clouddn.com/github-video-title-classify/3.png)

其中第一列代表视频URL；第二列为该视频类别是否经过算法修改，最开始全都为0；第三列为视频标签；第四列为视频标题。

视频标签的映射表如下：

![4](http://o7ie0tcjk.bkt.clouddn.com/github-video-title-classify/4.png)

在数据加载部分，我们将数据分为有标记数据以及无标记数据，有标记数据将用来训练以及测试分类器，然后用训练好的分类器预测无标记数据的标签。

分类的依据首先是根据视频标签是否为0，如果为0，代表视频是未标记的。其次，已标记的数据中有些类别是会对算法造成干扰，这里我们也将其去掉。

具体代码参照`preprocess.py`文件中的`load_data`方法。



## 去除特殊符号

由于视频标题中存在一些表情等特殊符号，在这个阶段将其去掉。

具体代码参照`preprocess.py`文件中的`remove_emoji`方法。



## 分词

本项目采用结巴分词作为分词器。

具体代码参照`preprocess.py`文件中的`cut`方法。



## 去停止词

本项目采用了`data/stopword.dic`文件中的停止词表，值得注意的是，句子去停止词前后去停止词后，单词的相对顺序保持不变。这里我们采用了有序集合（具体实现在`ordered_set.py`文件中）实现。

经过这一步之后，句子中重复的非停止词将只会取一次。但是由于视频标题较短，出现重复词的概率非常小，因此不会有太大影响。

具体代码参照`preprocess.py`文件中的`remove_stop_words`方法。



## 建立词典

将所有视频标题经过分词后的单词汇总起来建立一个词典，供后续句子建模使用。

具体代码参照`preprocess.py`文件中的`vocab_build`方法。



## 句子建模

将分词后的视频标题中的每个词替换为其在词典中的序号，这样每个标题将会转换为由一串数组构成的向量。

具体代码参照`preprocess.py`文件中的`word2index`方法。



# 训练

之前提到过，本文一共运用了四种深度学习模型，采用tensorflow框架，训练过程中涉及到的文件分为两类：

- 模型文件， 包括`textcnn.py`, `textrnn.py`, `textrcnn.py`以及`han.py`
- 训练文件，包括`train_cnn.py`, `train_rnn.py`, `train_rcnn.py`以及`train_han.py`

模型文件定义了具体的模型，本篇文档将不会具体地讲解实现代码，只会从理论层面介绍模型。训练文件包含了算法的训练过程，由于不同算法的训练流程一致，这里单挑TextCNN讲解。

下面开始介绍模型，如果只关注实现可以跳过到训练部分。



## 模型

### 词向量

分布式表示（Distributed Representation）是Hinton 在1986年提出的，基本思想是将每个词表达成 n 维稠密、连续的实数向量，与之相对的one-hot encoding向量空间只有一个维度是1，其余都是0。分布式表示最大的优点是具备非常powerful的特征表达能力，比如 n 维向量每维 k 个值，可以表征 $k^n$个概念。事实上，不管是神经网络的隐层，还是多个潜在变量的概率主题模型，都是应用分布式表示。下图是03年Bengio在 [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 的网络结构：

![img](https://pic3.zhimg.com/50/v2-dc007baa415cf1674df6d323419cc2de_hd.jpg)

这篇文章提出的神经网络语言模型（NNLM，Neural Probabilistic Language Model）采用的是文本分布式表示，即每个词表示为稠密的实数向量。NNLM模型的目标是构建语言模型：

![img](https://pic1.zhimg.com/50/v2-855f785d33895960712509982199c4b4_hd.jpg)

词的分布式表示即词向量（word embedding）是训练语言模型的一个附加产物，即图中的Matrix C。

尽管Hinton 86年就提出了词的分布式表示，Bengio 03年便提出了NNLM，词向量真正火起来是google Mikolov 13年发表的两篇word2vec的文章 [Efficient Estimation of Word Representations in Vector Space](http://ttic.uchicago.edu/%7Ehaotang/speech/1301.3781.pdf)和[Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)，更重要的是发布了简单好用的**word2vec工具包**，在语义维度上得到了很好的验证，极大的推进了文本分析的进程。下图是文中提出的CBOW 和 Skip-Gram两个模型的结构，基本类似于NNLM，不同的是模型去掉了非线性隐层，预测目标不同，CBOW是上下文词预测当前词，Skip-Gram则相反。

![img](https://pic2.zhimg.com/50/v2-04bfc01157c1c3ae1480299947315251_hd.jpg)

除此之外，提出了Hierarchical Softmax 和 Negative Sample两个方法，很好的解决了计算有效性，事实上这两个方法都没有严格的理论证明，有些trick之处，非常的实用主义。详细的过程不再阐述了，有兴趣深入理解word2vec的，推荐读读这篇很不错的paper: [word2vec Parameter Learning Explained](http://www-personal.umich.edu/%7Eronxin/pdf/w2vexp.pdf)。额外多提一点，实际上word2vec学习的向量和真正语义还有差距，更多学到的是具备相似上下文的词，比如“good” “bad”相似度也很高，反而是文本分类任务输入有监督的语义能够学到更好的语义表示。

至此，文本的表示通过词向量的表示方式，把文本数据从高纬度高稀疏的神经网络难处理的方式，变成了类似图像、语音的的连续稠密数据。深度学习算法本身有很强的数据迁移性，很多之前在图像领域很适用的深度学习算法比如CNN等也可以很好的迁移到文本领域了，



### 深度学习文本分类模型



#### TextCNN

本篇文章的题图选用的就是14年这篇文章提出的TextCNN的结构（见下图）。卷积神经网络[CNN Convolutional Neural Network](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)最初在图像领域取得了巨大成功，CNN原理就不讲了，核心点在于可以**捕捉局部相关性**，具体到文本分类任务中可以利用CNN来提取句子中类似 n-gram 的关键信息。

![img](https://pic1.zhimg.com/50/v2-ab904178abf9241329e3e2d0fa7c0584_hd.jpg)



TextCNN的详细过程原理图见下：

![img](https://pic3.zhimg.com/50/v2-bb10ad5bbdc5294d3041662f887e60a6_hd.jpg)

**TextCNN详细过程**：第一层是图中最左边的7乘5的句子矩阵，每行是词向量，维度=5，这个可以类比为图像中的原始像素点了。然后经过有 filter_size=(2,3,4) 的一维卷积层，每个filter_size 有两个输出 channel。第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示了，最后接一层全连接的 softmax 层，输出每个类别的概率。

**特征**：这里的特征就是词向量，有静态（static）和非静态（non-static）方式。static方式采用比如word2vec预训练的词向量，训练过程不更新词向量，实质上属于迁移学习了，特别是数据量比较小的情况下，采用静态的词向量往往效果不错。non-static则是在训练过程中更新词向量。推荐的方式是 non-static 中的 fine-tunning方式，它是以预训练（pre-train）的word2vec向量初始化词向量，训练过程中调整词向量，能加速收敛，当然如果有充足的训练数据和资源，直接随机初始化词向量效果也是可以的。

**通道（Channels）**：图像中可以利用 (R, G, B) 作为不同channel，而文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。

**一维卷积（conv-1d）**：图像是二维数据，经过词向量表达的文本为一维数据，因此在TextCNN卷积用的是一维卷积。一维卷积带来的问题是需要设计通过不同 filter_size 的 filter 获取不同宽度的视野。

**Pooling层**：利用CNN解决文本分类问题的文章还是很多的，比如这篇 [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/pdf/1404.2188.pdf) 最有意思的输入是在 pooling 改成 (dynamic) k-max pooling ，pooling阶段保留 k 个最大的信息，保留了全局的序列信息。比如在情感分析场景，举个例子：

```
            “ 我觉得这个地方景色还不错，但是人也实在太多了 ”
```

虽然前半部分体现情感是正向的，全局文本表达的是偏负面的情感，利用 k-max pooling能够很好捕捉这类信息。



#### TextRNN

尽管TextCNN能够在很多任务里面能有不错的表现，但CNN有个最大问题是固定 filter_size 的视野，一方面无法建模更长的序列信息，另一方面 filter_size 的超参调节也很繁琐。CNN本质是做文本的特征表达工作，而自然语言处理中更常用的是递归神经网络（RNN, Recurrent Neural Network），能够更好的表达上下文信息。具体在文本分类任务中，Bi-directional RNN（实际使用的是双向LSTM）从某种意义上可以理解为可以捕获变长且双向的的 "n-gram" 信息。

RNN算是在自然语言处理领域非常一个标配网络了，在序列标注/命名体识别/seq2seq模型等很多场景都有应用，[Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)文中介绍了RNN用于分类问题的设计，下图LSTM用于网络结构原理示意图，示例中的是利用最后一个词的结果直接接全连接层softmax输出了。

![img](https://pic3.zhimg.com/50/v2-92e49aef6626add56e85c2ee1b36e9aa_hd.jpg)



#### TextRCNN (TextCNN + TextRNN)

我们参考的是中科院15年发表在AAAI上的这篇文章 Recurrent Convolutional Neural Networks for Text Classification 的结构：

![img](https://pic3.zhimg.com/50/v2-263209ce34c0941fece21de00065aa92_hd.jpg)

利用前向和后向RNN得到每个词的前向和后向上下文的表示：

![img](https://pic1.zhimg.com/50/v2-d97b136cbb9cd98354521a827e0fd8b4_hd.jpg)

这样词的表示就变成词向量和前向后向上下文向量concat起来的形式了，即：

![img](https://pic4.zhimg.com/50/v2-16378ac29633452e7093288fd98d3f73_hd.jpg)

最后再接跟TextCNN相同卷积层，pooling层即可，唯一不同的是卷积层 filter_size = 1就可以了，不再需要更大 filter_size 获得更大视野，这里词的表示也可以只用双向RNN输出。



#### HAN (TextRNN + Attention)

CNN和RNN用在文本分类任务中尽管效果显著，但都有一个不足的地方就是不够直观，可解释性不好，特别是在分析badcase时候感受尤其深刻。而注意力（Attention）机制是自然语言处理领域一个常用的建模长时间记忆机制，能够很直观的给出每个词对结果的贡献，基本成了Seq2Seq模型的标配了。

**Attention机制介绍**：

详细介绍Attention恐怕需要一小篇文章的篇幅，感兴趣的可参考14年这篇paper [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473v7.pdf)。

以机器翻译为例简单介绍下，下图中$x_t$是源语言的一个词，$y_t$是目标语言的一个词，机器翻译的任务就是给定源序列得到目标序列。翻译$y_t$的过程产生取决于上一个词 $y_{t-1}$ 和源语言的词的表示 $h_{j}$($x_{j}$) 的 bi-RNN 模型的表示），而每个词所占的权重是不一样的。比如源语言是中文 “我 / 是 / 中国人” 目标语言 “i / am / Chinese”，翻译出“Chinese”时候显然取决于“中国人”，而与“我 / 是”基本无关。下图公式, $\alpha _{ij}$则是翻译英文第$i$个词时，中文第$j$个词的贡献，也就是注意力。显然在翻译“Chinese”时，“中国人”的注意力值非常大。

![img](https://pic3.zhimg.com/50/v2-de9146388978dfe7ef467993b9cf12ae_hd.jpg)

![img](https://pic1.zhimg.com/50/v2-0ebc7b64a7d34a908b8d82d87c92f6b8_hd.jpg)

Attention的核心point是在翻译每个目标词（或 预测商品标题文本所属类别）所用的上下文是不同的，这样的考虑显然是更合理的。

**TextRNN + Attention 模型**：

我们参考了这篇文章 [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/%7Ediyiy/docs/naacl16.pdf)，下图是模型的网络结构图，它一方面用层次化的结构保留了文档的结构，另一方面在word-level和sentence-level。标题场景只需要 word-level 这一层的 Attention 即可。

![img](https://pic3.zhimg.com/50/v2-4ff2c8099ccf0b2d8eb963a0ac248296_hd.jpg)

加入Attention之后最大的好处自然是能够直观的解释各个句子和词对分类类别的重要性。



## 训练

现在来详细讲解训练过程，涉及到的文件`train_cnn.py`, `utils.py`, `textcnn.py`

注意到`train_cnn.py`文件最后：

```python
if  __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    tf.app.run()
```

其中第一行是指定只用一个GPU。第二行是tensorflow的一个运行框架，`run`会运行文件内的`main`方法，并且传入文件最开始设定的参数：

```python
# configuration
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("num_classes", 33, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer(
  "batch_size", 64, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer(
  "decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float(
  "decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string(
  "ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer(
  "sentence_len", 15, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 64, "embedding size")
tf.app.flags.DEFINE_boolean(
  "is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer(
  "num_epochs", 30, "number of epochs to run.")
tf.app.flags.DEFINE_integer(
  "validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean(
  "use_embedding", True, "whether to use embedding or not.")
tf.app.flags.DEFINE_integer(
  "num_filters", 256, "number of filters")
tf.app.flags.DEFINE_boolean(
  "multi_label_flag", False, "use multi label or single label.")
tf.app.flags.DEFINE_boolean(
  "just_train", False, "whether use all data to train or not.")
```

第一个参数代表参数名（调用这个参数的方法：`FLAGS.name`），第二个参数是默认值，第三个参数是描述。值得说明的是这里有一个`just_train`参数，它代表是否将测试集放入训练集一起训练，一般在用模型最终确定之后。

所以运行`python train_cnn.py`就是启动训练过程，同时可以传入参数，方法为`python train_cnn.py --name value`, 这里的name就是文件定义的参数名，value就是你要设定的值。如果不传入参数，则参数为默认值。



下面我们来看一下`main`函数，流程如下：

![5](http://o7ie0tcjk.bkt.clouddn.com/github-video-title-classify/5.png)



### 数据加载

这个过程主要是调用`train_test_loader`方法切分训练集与测试集。

```python
X_train, X_val, y_train, y_val, n_classes = 
	train_test_loader(FLAGS.just_train)
```



### 词典加载

加载数据预处理过程中建立的词典。目的是用来从预训练的词向量词典中拿出对应的词向量。

```python
with open('data/vocab.dic', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab) + 1
print('size of vocabulary: {}'.format(vocab_size))
```

这里将词典的长度加一是为了给一个特殊词“空”加入位置，“空”的作用是填充短标题，让所有标题长度一样。



### Padding

这个阶段就是将所有标题长度变成一致，短了就填充，长了就截断。标题长度是一个参数，可以设置。

```python
# padding sentences
    X_train = pad_sequences(X_train, maxlen=FLAGS.sentence_len, 
    	value=float(vocab_size - 1))
    if not FLAGS.just_train:
        X_val = pad_sequences(
         	X_val, maxlen=FLAGS.sentence_len, value=float(vocab_size - 1))
```



### 模型实例化

```python
textcnn = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.num_classes,
                FLAGS.learning_rate, FLAGS.batch_size,
                FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len,
                vocab_size, FLAGS.embed_size, FLAGS.is_training, 
                multi_label_flag=False)
```

如果有之前训练到一半的模型，那我们就加载那个模型的参数，继续训练，否则进行参数初始化

```python
# Initialize save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + 'checkpoint'):
            print('restoring variables from checkpoint')
            saver.restore(
            	sess, 
            	tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:
                assign_pretrained_word_embedding(
                	sess, vocab, vocab_size, textcnn)
```



### 模型训练

模型训练过程中包括两个循环，第一个是大循环，表示遍历所有训练数据多少遍。第二个是mini-batch循环，小循环走过一遍代表遍历了所有训练数据一遍。

```python
for epoch in range(curr_epoch, total_epochs):
            loss, acc, counter = .0, .0, 0
            for start, end in zip(
                    range(0, number_of_training_data, batch_size),
                    range(batch_size, number_of_training_data, 
                    batch_size)):
```

下面就是将训练数据喂到模型中:

```python
feed_dict = {textcnn.input_x: X_train[start:end], 		
			textcnn.dropout_keep_prob: 0.5}
```

第二个参数是模型相关的dropout参数，用于减少过拟合，范围是(0, 1]，基本不用改变。

```python
curr_loss, curr_acc, _ = sess.run(
                        [textcnn.loss_val, textcnn.accuracy, 
                        textcnn.train_op], feed_dict)
```

这一步就是得到这一小部分训练数据对应的准确率以及loss。



然后每经过`validate_every`个大循环的训练，在测试集上看看模型性能。如果性能比上一次更好，就保存模型，否则就退出，因为算法开始发散了。

模型训练完毕检查性能之后，如果模型可行，下一步就将所有数据用于训练，也即运行以下命令`python train_cnn.py --just_train True`。这个过程会迭代固定的20个大循环。训练完毕之后，下面的预测过程将使用这个模型。



# 预测

预测涉及到的文件`predict_cnn.py`以及`utils.py`

预测的流程和训练差不多，只不过不再进行多次对数据集的遍历，只进行对未标记数据进行一次遍历，拿到结果之后，由于算法输出的结果是[0, 32]这样一个序号，我们需要转化为中文标签。

具体参照代码，不再赘述。


# 引用

【1】https://zhuanlan.zhihu.com/p/25928551

【2】https://github.com/brightmart/text_classification
