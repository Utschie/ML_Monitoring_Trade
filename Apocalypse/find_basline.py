#本程序是根据验证集来找到一个基准线的，就只看终赔平均赔率哪种结果最高就那么预测，以此预测为baseline
import pandas as pd
import os
import numpy as np
import re
from tqdm import tqdm
filepath = 'C:\\data\\test'
lablelist = pd.read_csv('C:\\data\\lablelist.csv',index_col = 0)#比赛id及其对应赛果的列表
filelist0 = [i+'\\'+k for i,j,k in os.walk(filepath) for k in k]#得到所有csv文件的路径列表
filelist = [data_path for data_path in filelist0 if int(re.findall(r'\\(\d*?).csv',data_path)[0]) in lablelist.index]#只保留有赛果的文件路径
lables = {'win':0,'draw':1,'lose':2}#分类问题要从0开始编号，否则出错
resultlist = list()
peilvs = list()
try:
    with tqdm(filelist,ascii=True) as t:#在tqdm中显示加入ascii=True,才能在cmd下载一行输出进度条
        for data_path in t:
            bisai_id = int(re.findall(r'\\(\d*?).csv',data_path)[0])
            data = pd.read_csv(data_path)#读取文件
            data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
            frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
            new_data = np.array(data)
            timepoint = new_data[:,0]
            state = new_data[timepoint==0]
            mean = np.mean(state, axis=0)
            peilv = mean[3:6]#第3,4,5列用3:6来表示
            peilvs.append(peilv)#把赔率
            result = lablelist.loc[bisai_id].result
            result = lables[result]
            resultlist.append(result)
except KeyboardInterrupt:
    t.close()
    raise
t.close()
predictions = np.argmin(peilvs, 1)#谁赔率低谁赢
correct = (predictions == resultlist).sum()
baseline = correct/len(resultlist)
print(str(baseline))
'''
for data_path in tqdm(filelist):
    bisai_id = int(re.findall(r'\\(\d*?).csv',data_path)[0])
    data = pd.read_csv(data_path)#读取文件
    data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
    frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
    new_data = np.array(data)
    timepoint = new_data[:,0]
    state = new_data[timepoint==0]
    mean = np.mean(state, axis=0)
    peilv = mean[3:6]#第3,4,5列用3:6来表示
    peilvs.append(peilv)#把赔率
    result = lablelist.loc[bisai_id].result
    result = lables[result]
    resultlist.append(result)
    

'''