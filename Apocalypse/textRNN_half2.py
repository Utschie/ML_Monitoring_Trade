#本程序是采用平衡数据集的用tsvd的模型，看看会不会比非平衡数据集好————20210126
#把优化器的eps跳到1e-16,并把初始学习率调低两个数量级，看看loss会不会继续持续下降
'''
经过初始学习率分别为0.01,0.001,0.0001,0.00001的比对后，发现0.001甚至0.01是最好的学习率，loss最低降到1.1左右
而0.0001和0.00001下降非常慢，甚至在初始学习率为0.00001时loss到了5之后下降就非常慢了
而当学习率为0.01时loss很快就下降到了1.1,而且非常稳定，甚至比0.001还稳
'''
#这次试一下如果不用正交初始化而是用默认初始化的话，loss会是个什么结果
#使用默认初始化的效果比正交初始化的结果要略好，甚至大冷的平均收益率超过1（约1.002这样）。
#虽然只跑了可能3个epoch，但是总准确率提高到了42%，所以还是默认初始化比较好————20210202
#如果按照这个测试结果，那么爆大冷，小冷和正常的预测准确率分别为28.14%，29.01%和64.62%————20210203
#按照各种预测结果的最后一帧的最大值购买，那么平均收益分别为12.89%,3.87%和10.52%————20210203
#如果分别按每次命中收益算，即相当于加权，那么算出的平均收益分别为11%，3.6%和7%————20210203
#但实际上就算只卖大冷或者只买小冷，正常，只要都取最后一帧的最大值平均收益也有14%，1%和6.6%————20210203
#如果把模型预测的概率当做投资比例来做投资，那么72532场比赛的平均收益是6.6%，但标准差高达22%————20210203
#加入分布来自均值为6.6%，标准差为22%，那么样本均值的标准差=总标准差/sqrt(n),也就意味着至少要抽12个，才能是1倍标准差内(即以68%的概率)为正收益
#2倍标准差才能以95%的概率>1,此时需要抽45个，才能有95%的概率>1————20210203
#如果不用模型预测的概率，而是用最后一帧的平均概率来投资，然后也用最大那一组赔率，它的均值和标准差会是多少呢？————20210203
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
import pandas as pd
import numpy as np
import csv
import random
import re
from sklearn.decomposition import TruncatedSVD
from torch.nn.utils.rnn import pad_sequence#用来填充序列
import time
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter
from pywick.optimizers.nadam import Nadam#使用pywick包里的nadam优化器
from torch.optim import lr_scheduler
from torch.optim import Adam
with open('/home/jsy/data/cidlist_complete.csv') as f:
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表
cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
class BisaiDataset(Dataset):#数据预处理器
    def __init__(self,filepath):#filepath是个列表
        with open(filepath,'r') as f:#读取filepath文件并做成filelist列表
            self.filelist = []
            for line in f:
                self.filelist.append(line.strip('\n'))
        self.lablelist = pd.read_csv('/home/jsy/data/lablelist.csv',index_col = 0)#比赛id及其对应赛果的列表
        self.lables = {'win':0,'draw':1,'lose':2}#分类问题要从0开始编号,而且要对应好了表中的顺序编，
    
    def __getitem__(self, index):
        #todo
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        #这里需要注意的是，第一步：read one data，是一个dat
        data_path = self.filelist[index]
        bisai_id = int(re.findall(r'/(\d*?).csv',data_path)[0])
        # 2. Preprocess the data (e.g. torchvision.Transform).
        data = self.csv2frame(data_path)
        # 3. Return a data pair (e.g. image and label).
        lable = self.lablelist.loc[bisai_id].result
        lable = self.lables[lable]
        return data,lable      
       
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.filelist)

    def csv2frame(self,filepath):#给出单场比赛的csv文件路径，并转化成帧列表和对应变帧时间列表，以及比赛结果
        data = pd.read_csv(filepath)#读取文件
        data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
        frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
        framelist =list()#framelist为一个空列表,长度与frametimelist相同,一定要规定好具体形状和float类型，否则dataloader无法读取
        '''
        此处两个循环算法太慢，用pandas更慢，完全抛弃pandas后，数据处理速度从109秒降到了10秒，降到10秒后cpu利用率20%，再往上提也提不上去了，可能需要C++或C来写了
        '''
        new_data = np.array(data)
        lables = new_data[:,0]
        if len(frametimelist)>250:
            frametimelist = [frametimelist[0]]+random.sample(list(frametimelist)[1:-1],248)+[frametimelist[-1]]#如果长度大于500,保留头尾，并在中间随机抽取498个，共计500个
            frametimelist.sort(reverse=True)#并降序排列
        for i in frametimelist:
            state = new_data[lables==i]#从第一次变盘开始得到当次转移
            #state = np.array(state)#不必转成numpy多维数组，因为已经是了
            state = np.delete(state,(0,1), axis=-1)#去掉frametime和cid
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            framelist.append(state)
        frametimelist = np.array(frametimelist)
        vectensor = self.mrx2vec(framelist)
        len_frame = vectensor.shape[0]
        if len_frame<250:
            vectensor = np.concatenate((np.zeros((250-len_frame,10),dtype=np.float64),vectensor),axis=0)#如果不足500，则在前面用0填充
        vectensor = torch.from_numpy(vectensor)
        return vectensor#传出一个帧列表,也可以把frametimelist一并传出来，此处暂不考虑位置参数的问题
    
    def tsvd(self,frame):
        tsvd = TruncatedSVD(1)
        if frame.shape[0] != 1:
            newframe = tsvd.fit_transform(np.transpose(frame))#降维成（1,10）的矩阵
        else:
            return frame.reshape((10,1))#第一行需要reshape一下
        return newframe
    
    def mrx2vec(self,flist):#把截断奇异值的方法把矩阵变成向量(matrix2vec/img2vec)，传入：len(frametimelist)*(306*10),传出：len(frametimelist)*10
        vectensor = np.array(list(map(self.tsvd,flist))).squeeze(2)
        #veclist = veclist.transpose()
        #vectensor = torch.from_numpy(veclist)#转成张量
        return vectensor#传出一个形状为(1,序列长度,10)的张量，因为后面传入模型之前，还需要做一下pad_sequence(0维是batch_size维)

 


class Lstm(nn.Module):#在模型建立之处就把它默认初始化
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(input_size=10, 
                                hidden_size=250,#选择对帧进行保留首尾的均匀截断采样
                                num_layers=1,#暂时就只有一层
                                bidirectional=True)
        #nn.init.orthogonal_(self.encoder.weight_ih_l0)
        #nn.init.orthogonal_(self.encoder.weight_hh_l0)
        #nn.init.constant_(self.encoder.bias_ih_l0,0.0)
        #nn.init.constant_(self.encoder.bias_hh_l0,0.0)
        self.decoder = nn.Sequential(
            nn.Linear(1000, 250),#把LSTM的输出
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(250, 3)
        )
        #nn.init.normal_(self.decoder[0].weight,mean=0.0)
        #nn.init.constant_(self.decoder[0].bias, 0.0)
        #nn.init.normal_(self.decoder[3].weight,mean=0.0)
        #nn.init.constant_(self.decoder[3].bias, 0.0)

    def forward(self,inputs):
        output, _= self.encoder(inputs.permute(1,0,2))#inputs需要转置一下再输入lstm层，因为pytorch要求第0维为长度，第二维才是batch_size
        encoding = torch.cat((output[0], output[-1]), -1)#双向的lstm，就把两个都放进去
        return self.decoder(encoding)#把最后一个时间步的输出输入MLP


def get_parameter_number(model):#参数统计
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



if __name__ == "__main__":
    train_writer = SummaryWriter('/home/jsy/log3/train')#自动建立
    test_writer = SummaryWriter('/home/jsy/log3/test')#自动建立
    checkpoint_path = '/home/jsy/log3/checkpoint.pth'#ckpoint文件夹需要提前建立
    train_path = '/home/jsy/balanced_train_path.txt'
    test_path = '/home/jsy/balanced_test_path.txt'
    dataset = BisaiDataset(train_path)#训练集
    test_set = BisaiDataset(test_path)#验证集
    print('数据集读取完成')
    loader = DataLoaderX(dataset, 128 ,shuffle=True,num_workers=4,pin_memory=True)#num_workers>0情况下无法在交互模式下运行
    test_loader = DataLoaderX(test_set, 128, shuffle=True,num_workers=4,pin_memory=True)#验证dataloader
    print('dataloader准备完成')
    net = Lstm().double().cuda()#双精度
    print('网络构建完成')
    stat1 = get_parameter_number(net)
    print(str(stat1))
    lr, num_epochs = 0.001, 2000
    optimizer= Adam(net.parameters(), lr=lr,eps=1e-16)
    start_epoch = 1#如果没有checkpoint则初始epoch为1
    gesamt_counter = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        gesamt_counter = checkpoint['gesamt_counter']
    loss = nn.CrossEntropyLoss()
    scheuler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.25)
    for epoch in range(start_epoch, num_epochs + 1):
        l_list = list()
        epoch_start = time.time()#记录整个epoch除验证外所用的时间
        net.train()
        counter = 0
        start = time.time()
        train_output = torch.zeros((1,3))
        train_y = torch.zeros((1)).long()#得是long类型
        #if epoch == 6:
        #    loader = DataLoaderX(dataset,64,shuffle=True,num_workers=4,pin_memory=True)#在经过10个epoch后，batch改成64
        #    optimizer= Nadam(net.parameters(), lr=lr/pow(4,2))
        for x, y in iter(loader):
            #但是还需要使填充后的那些0不参与计算，所以可能需要制作掩码矩阵
            #或者需要时序全局最大池化层来消除填充的后果
            x = x.double().cuda()
            y = y.long().cuda()
            output = net(x)#x要求是一个固定shape的第0维是batch_size的张量，所以需要批量填充
            l = loss(output, y)
            optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
            end = time.time()
            train_period = end-start
            counter+=1
            gesamt_counter+=1
            print('第'+str(epoch)+'个epoch已学习'+str(counter)+'个batch,'+'用时'+str(train_period)+'秒')
            l_list.append(l.item())
            train_writer.add_scalar('step_loss',l.item(),gesamt_counter)#随着每一步学习的loss下降图
            #print('loss: %f' % (l.item()))
            start = time.time()
            train_output = torch.cat((train_output,output.cpu()),0)#把这一个batch的输出连起来
            train_y = torch.cat((train_y,y.cpu()),0)#把这一个batch的lable连起来
        epoch_end = time.time()
        print('epoch %d, loss: %f' % (epoch,np.mean(l_list)))
        print('第'+str(epoch)+'个epoch训练用时'+str(int(epoch_end-epoch_start))+'秒')
        prediction = torch.argmax(train_output, 1)#找出每场比赛预测输出的最大值的坐标
        correct = (prediction == train_y).sum().float()#找出预测正确的总个数
        accuracy = correct/len(train_y)#计算Top-1正确率,总共就三分类，就不看top-2的了
        train_writer.add_scalar('Top-1 Accuracy',accuracy,epoch)#写入文件
        #下面是一个epoch结束的验证部分
        #if epoch>=20:
        print('开始验证......')
        test_start = time.time()
        net.eval()
        torch.cuda.empty_cache()#释放一下显存
        with torch.no_grad():#这样在验证时显存才不会爆
            test_output = torch.zeros((1,3))
            test_y = torch.zeros((1)).long()#得是long类型
            test_counter = 0
            for x,y in iter(test_loader):
                x = x.double().cuda()
                output = net(x).cpu()#把输出转到内存
                test_output = torch.cat((test_output,output),0)#把这一个batch的输出连起来
                test_y = torch.cat((test_y,y),0)#把这一个batch的lable连起来
                #torch.cuda.empty_cache()
                test_counter+=1
                print('验证集已完成'+str(test_counter)+'个batch')
            #验证输出和验证lable的第一个元素都是0，鉴于到时候占比很小就不删除了
            print('计算结果......')
            l_test = loss(test_output,test_y)#用整个验证集的输出和lable算一个总平均loss(nn.CrossEntropyLoss默认就是求平均值)
            test_writer.add_scalar('epoch_loss',l_test.item(),epoch)#每一个epoch算一次验证loss
            train_writer.add_scalar('epoch_loss',np.mean(l_list),epoch)#每一个epoch把训练loss也加上
            prediction = torch.argmax(test_output, 1)#找出每场比赛预测输出的最大值的坐标
            correct = (prediction == test_y).sum().float()#找出预测正确的总个数
            accuracy = correct/len(test_y)#计算Top-1正确率,总共就三分类，就不看top-2的了
            test_writer.add_scalar('Top-1 Accuracy',accuracy,epoch)#写入文件
        test_end = time.time()
        print('验证已完成，用时'+str(int(test_end-test_start))+'秒')
        print('验证完成，开始保存......')
        #下面是模型保存部分
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'gesamt_counter':gesamt_counter
                }
        torch.save(checkpoint, checkpoint_path)#保存checkpoint到路径
        torch.cuda.empty_cache()#释放一下显存
        scheuler.step()
        print('保存完毕')
                
