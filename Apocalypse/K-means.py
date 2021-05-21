#把每一帧拉直成向量再标准化后用聚类方法把无限种帧的情况转化成有限个类(用1，2，3表示)，因为doc2vec需要有限个数的词汇表，这样就把每场比赛都用有限多个类代号组成的长串来表示
#如果不搞词汇表的话那就是相当于直接拉直，每一种情况都是一个unique词汇
#然后用gensim的Doc2vec（或者GloVe或者sentence2vec）得到每一场比赛的向量+xgboost或随机森林或fasttext或深度森林模型对序列进行操作————20210517
#本程序暂时是使用K均值聚类，对训练集进行聚类，然后输出一个包含所有训练集比赛的符合gensim的doc2vec输入规则的文件————20210517
#之后可以尝试子空间分割聚类，毕竟拉直后的帧的向量有至少4000多个指标，维度很高而且很稀疏————20210517
#可以把数值的数据用谱聚类给聚类成有限个类名，然后文字型的当做单独的词汇也写入单个样本的这个Taged_document里,然后再进入doc2vec————20210517
#用普通kmeans聚类很有可能内存会爆掉，所以暂时使用sklearn的minibatchkmeans