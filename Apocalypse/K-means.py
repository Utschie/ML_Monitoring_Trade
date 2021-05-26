#把每一帧拉直成向量再标准化后用聚类方法把无限种帧的情况转化成有限个类(用1，2，3表示)，因为doc2vec需要有限个数的词汇表，这样就把每场比赛都用有限多个类代号组成的长串来表示
#如果不搞词汇表的话那就是相当于直接拉直，每一种情况都是一个unique词汇
#然后用gensim的Doc2vec（或者GloVe或者sentence2vec）得到每一场比赛的向量+xgboost或随机森林或fasttext或深度森林模型对序列进行操作————20210517
#本程序暂时是使用K均值聚类，对训练集进行聚类，然后输出一个包含所有训练集比赛的符合gensim的doc2vec输入规则的文件————20210517
#之后可以尝试子空间分割聚类，毕竟拉直后的帧的向量有至少4000多个指标，维度很高而且很稀疏————20210517
#可以把数值的数据用谱聚类给聚类成有限个类名，然后文字型的当做单独的词汇也写入单个样本的这个Taged_document里,然后再进入doc2vec————20210517
#用普通kmeans聚类很有可能内存会爆掉，所以暂时使用sklearn的minibatchkmeans
#在训练及59367个文件里，共有121554754次，即1.2亿次转移,近算上有数据的数据行的话，有22.5亿行————20210524
#这里隐藏了一个可能的问题，就是赔率的数据是偏态分布，它不可能小于1,但是上不封顶，不知道对标准化会不会有影响————20210525
#赔率的分布更接近卡方分布，及左边有界，然后隆起，然后迅速下降至0,此时用平方根法可以把分布转化成正态分布（不过这里可能设计到一个正负号的问题）————20210525
#随机抽100场比赛的所有行，标准化后进行主成分分析，方差解释度为[0.4339, 0.3027, 0.1411 , 0.0585, 0.0310,0.0147 , 0.0102, 0.0051, 0.0023]
#如果随机抽10000场比赛，每场比赛只提供最后两帧的话，标准化后进行主成分分析，方差解释度为[0.368, 0.2548, 0.1777, 0.0715, 0.0537,0.0277, 0.0220, 0.0179, 0.0059]
#如果所有列取对数，然后用PCA包自带的标准化库标准化后做主成分分析，方差解释度为[0.44,0.21,0.18,0.076,0.04...]
#如果只有赔率列取对数，则方差解释度为[0.42971035, 0.20809988, 0.16917282, 0.08865169, 0.05207994,0.04530344, 0.00458814, 0.00159367, 0.00079466]
#赔率列取了倒数方差解释度也差不多，取了平方根也差不多，就是基本上都得取前4个才能保存足够总方差解释度————20210526
#如果先用数字的均值和标准差把每一帧标准化，然后补0,拉直成6010维的向量，然后再进行pca的话，那么方差解释度为（未完成）
#如果先把每一帧拉直，然后通过拉直后的均值和标准差（包含0求出的那个均值和方差），那么方差解释度为（未完成）
#如果前面两个都不太行，实在不行那就给每个公司都分配一个权重，然后取加权平均值，然后再取前4-5个主成分
#如果实在都不行的话，那就只能各行求简单平均了
#使用pca或者svd降维，应该对全样本做一个pca或者svd而不是每个都做一次，然后保存得到的特征向量矩阵，每次通过这个固定的矩阵进行数据变换————20210527
#或者使用因子分析————20210527