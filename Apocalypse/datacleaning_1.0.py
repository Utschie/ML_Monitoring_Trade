#本程序是数据清洗程序用与Apocalypse 1.0 系列的开发
#每个urlnum代表一场单独的比赛，同样的urlnum下，不同的cid是同一个公司对该厂比赛的不同赔率，再同一个cid下不同的restime是不同的剩余时间
#用sql很麻烦，改用pandas做成数据框，然后直接洗试试
#现在是把字典列表逐个插入dataframe，写进一天的文件要好多分钟————20200618
#一个76.5M的原文件写入dataframe要5472秒，压缩成csv变成28.6M，读取只需1秒，所以要先把所有的文件转成csv文件————20200618
#按上面那个速度400G数据全转完需要8000多个小时，需要用并行的方式插入数据框————20200618（已解决）
#找到问题了，其实数据转换很快，但是插入pandas的dataframe的过程特别慢，那就直接写入csv吧
#直接写入csv用时18秒，非常棒
import re
import datetime
import pandas as pd
import time
import csv
#从文本文件读入记录，由于当时写入文件的时候没分行也没分分隔符，所以整个文件只有一行
f=open('D:\\data\\20141130-20160630\\2014-11-30.txt','r')
line=f.readline()
f.close()#要关闭数据集
#把这一行数据用re转成列表
datalist=re.findall('{.*?}',line)
keys=['date', 'urlnum', 'league', 'cid', 'zhudui', 'kedui', 'companyname', 'resttime', 'fanhuanlv', 
'peilv_host', 'peilv_fair', 'peilv_guest', 
'gailv_host', 'gailv_fair', 'gailv_guest', 
'kailizhishu_host', 'kailizhishu_fair', 'kailizhishu_guest']
df=pd.DataFrame(columns=keys)#先列好字段
#把列表中各个元素转成字典,并且把peilv，gailv和kailizhishu分拆成三列，否则无法正确读入pandas
def str2dict(str):#讲datalist中的单个元素转换插入dataframe的函数
    dic=eval(str)
    del dic['timestamp']
    dic['peilv_host']=dic['peilv'][0]
    dic['peilv_fair']=dic['peilv'][1]
    dic['peilv_guest']=dic['peilv'][2]
    del dic['peilv']
    dic['gailv_host']=dic['gailv'][0]
    dic['gailv_fair']=dic['gailv'][1]
    dic['gailv_guest']=dic['gailv'][2]
    del dic['gailv']
    dic['kailizhishu_host']=dic['kailizhishu'][0]
    dic['kailizhishu_fair']=dic['kailizhishu'][1]
    dic['kailizhishu_guest']=dic['kailizhishu'][2]
    del dic['kailizhishu']
    return dic

def dict2DataFrame(dic):
    global df
    df=df.append(dic,ignore_index=True)

start=time.time()
dictlist=list(map(str2dict,datalist))#把datalist变成了字典形式的dictlist，即每个元素都是字典
with open('D:\\data_csv\\2014-11-30.csv','w',newline='') as f:
    w=csv.DictWriter(f,keys)
    w.writeheader()
    w.writerows(dictlist)

df=pd.read_csv('D:\\data_csv\\2014-11-30.csv',encoding='GBK')#再讲csv读入dataframe，注意编码
#接下来进行数据重组处理




end=time.time()
print('耗时'+str(end-start)+'秒')







