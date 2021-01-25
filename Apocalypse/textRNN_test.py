#专门用来对textRNN_half来做的测试程序
#由于训练集的比例是1：1：1,测试集则采用非平衡数据集查看最终效果
#不需要跑循环只需要测试一次就好
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
with open('/home/jsy/data/cidlist_complete.csv') as f:
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表
cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
class BisaiDataset(Dataset):#数据预处理器
    def __init__(self,filepath):#filepath是个列表
        self.lablelist = pd.read_csv('/home/jsy/data/lablelist.csv',index_col = 0)#比赛id及其对应赛果的列表
        self.filelist0 = list()
        for x in filepath:#把filepath这个列表中的所有目录下的路径组合成一个列表
            self.filelist0+=[i+'/'+k for i,j,k in os.walk(x) for k in k]#得到所有csv文件的路径列表
        #self.filelist0 = [i+'/'+k for i,j,k in os.walk(filepath) for k in k]#得到所有csv文件的路径列表
        self.filelist = [data_path for data_path in self.filelist0 if int(re.findall(r'/(\d*?).csv',data_path)[0]) in  self.lablelist.index]#只保留有赛果的文件路径
        self.lables = {'win':0,'draw':1,'lose':2}#分类问题要从0开始编号,而且要对应好了表中的顺序编
        self.all_mean = np.array([0.92236896, 2.56799437, 3.64873151, 3.94965805])#全样本均值
        self.all_std = np.array([0.03154385, 1.76454825, 1.0027637 , 3.01005411])#全样本标准差
    
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
        if len(frametimelist)>250:#如果总帧数大于250,则随机挑选250，若不足250,则在后面处理vectensor时补0
            frametimelist = [frametimelist[0]]+random.sample(list(frametimelist)[1:-1],248)+[frametimelist[-1]]#如果长度大于500,保留头尾，并在中间随机抽取498个，共计500个
            frametimelist.sort(reverse=True)#并降序排列
        for i in frametimelist:
            state = new_data[lables==i]#从第一次变盘开始得到当次转移
            #state = np.array(state)#不必转成numpy多维数组，因为已经是了
            state = np.delete(state,(0,1), axis=-1)#去掉frametime和cid
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            max = (state.max(axis=0)[0:4]-self.all_mean)/self.all_std#标准化
            mean = (state.mean(axis=0)[0:4]-self.all_mean)/self.all_std
            min = (state.min(axis=0)[0:4]-self.all_mean)/self.all_std
            max_min = np.concatenate((max,mean,min),axis=0)
            framelist.append(max_min)
        frametimelist = np.array(frametimelist)
        vectensor = np.array(framelist)
        len_frame = vectensor.shape[0]
        if len_frame<250:
            vectensor = np.concatenate((np.zeros((250-len_frame,12),dtype=np.float64),vectensor),axis=0)#如果不足500，则在前面用0填充
        vectensor = torch.from_numpy(vectensor)
        return vectensor#传出一个帧列表,也可以把frametimelist一并传出来，此处暂不考虑位置参数的问题
    


class Lstm(nn.Module):#在模型建立之处就把它默认初始化
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(input_size=12, 
                                hidden_size=250,#选择对帧进行保留首尾的均匀截断采样
                                num_layers=1,#暂时就只有一层
                                bidirectional=True)
        nn.init.orthogonal_(self.encoder.weight_ih_l0)
        nn.init.orthogonal_(self.encoder.weight_hh_l0)
        nn.init.constant_(self.encoder.bias_ih_l0,0.0)
        nn.init.constant_(self.encoder.bias_hh_l0,0.0)
        self.decoder = nn.Sequential(
            nn.Linear(1000, 250),#把LSTM的输出
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(250, 3)
        )
        nn.init.normal_(self.decoder[0].weight,mean=0.0)
        nn.init.constant_(self.decoder[0].bias, 0.0)
        nn.init.normal_(self.decoder[3].weight,mean=0.0)
        nn.init.constant_(self.decoder[3].bias, 0.0)

    def forward(self,inputs):
        output, _= self.encoder(inputs.permute(1,0,2))#inputs需要转置一下再输入lstm层，因为pytorch要求第0维为长度，第二维才是batch_size
        encoding = torch.cat((output[0], output[-1]), -1)#双向的lstm，就把两个都放进去
        return self.decoder(encoding)#把最后一个时间步的输出输入MLP


def get_parameter_number(model):#参数统计
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



if __name__ == "__main__":
    checkpoint_path = '/home/jsy/log2/checkpoints/checkpoint.pth'#ckpoint文件夹需要提前建立
    test_path1 = ['/home/jsy/train_20160701-20170531','/media/jsy/Samsung/train_20180211-20190224','/media/jsy/WD/train_20170601-20180210']
    test_path2 = ['/media/jsy/Samsung/test2','/home/jsy/test4']
    test_set1 = BisaiDataset(test_path1)#验证集
    test_set2 = BisaiDataset(test_path2)
    print('数据集读取完成')
    test_loader1 = DataLoaderX(test_set1, 64 ,shuffle=True,num_workers=16,pin_memory=True)#num_workers>0情况下无法在交互模式下运行
    test_loader2 = DataLoaderX(test_set2, 64, shuffle=True,num_workers=16,pin_memory=True)#验证dataloader
    print('dataloader准备完成')
    net = Lstm().double().cuda()#双精度
    print('网络构建完成')
    stat1 = get_parameter_number(net)
    print(str(stat1))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    loss = nn.CrossEntropyLoss()
    print('开始验证......')
    net.eval()
    torch.cuda.empty_cache()#释放一下显存
    with torch.no_grad():#这样在验证时显存才不会爆
        test_output = torch.zeros((1,3))
        test_y = torch.zeros((1)).long()#得是long类型
        test_counter = 0
        #先测试test_loader1
        start = time.time()
        for x,y in iter(test_loader1):
            x = x.double().cuda()
            output = net(x).cpu()#把输出转到内存
            test_output = torch.cat((test_output,output),0)#把这一个batch的输出连起来
            test_y = torch.cat((test_y,y),0)#把这一个batch的lable连起来
            #torch.cuda.empty_cache()
            test_counter+=1
            print('test_set1已完成'+str(test_counter)+'个batch')
        #验证输出和验证lable的第一个元素都是0，鉴于到时候占比很小就不删除了
        print('计算结果......')
        l_test = loss(test_output,test_y)#用整个验证集的输出和lable算一个总平均loss(nn.CrossEntropyLoss默认就是求平均值)
        prediction = torch.argmax(test_output, 1)#找出每场比赛预测输出的最大值的坐标
        correct = (prediction == test_y).sum().float()#找出预测正确的总个数
        accuracy1 = correct/len(test_y)#计算Top-1正确率,总共就三分类，就不看top-2的了
        end = time.time()
        print('test_set1测试完成,用时'+str(end-start)+'秒,准确率为'+str(accuracy1))
        #再测试test_loader2
        start = time.time()
        for x,y in iter(test_loader2):
            x = x.double().cuda()
            output = net(x).cpu()#把输出转到内存
            test_output = torch.cat((test_output,output),0)#把这一个batch的输出连起来
            test_y = torch.cat((test_y,y),0)#把这一个batch的lable连起来
            #torch.cuda.empty_cache()
            test_counter+=1
            print('test_set2已完成'+str(test_counter)+'个batch')
        #验证输出和验证lable的第一个元素都是0，鉴于到时候占比很小就不删除了
        print('计算结果......')
        l_test = loss(test_output,test_y)#用整个验证集的输出和lable算一个总平均loss(nn.CrossEntropyLoss默认就是求平均值)
        prediction = torch.argmax(test_output, 1)#找出每场比赛预测输出的最大值的坐标
        correct = (prediction == test_y).sum().float()#找出预测正确的总个数
        accuracy2 = correct/len(test_y)#计算Top-1正确率,总共就三分类，就不看top-2的了
        end = time.time()
        print('test_set2测试完成,用时'+str(end-start)+'秒,准确率为'+str(accuracy2))
    print('test_set1的Top-1精确度为'+str(accuracy1))
    print('test_set2的Top-1精确度为'+str(accuracy2))
            
    
                

        

