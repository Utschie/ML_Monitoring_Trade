#本程序是数据清洗程序用与Apocalypse 1.0 系列的开发
#每个urlnum代表一场单独的比赛，同样的urlnum下，不同的cid是同一个公司对该厂比赛的不同赔率，再同一个cid下不同的restime是不同的剩余时间
#用sql很麻烦，改用pandas做成数据框，然后直接洗试试
#现在是把字典列表逐个插入dataframe，写进一天的文件要好多分钟————20200618
import re
import datetime
import pandas as pd
import ast
#从文本文件读入记录，由于当时写入文件的时候没分行也没分分隔符，所以整个文件只有一行
f=open('D:\\data\\20141130-20160630\\2014-11-30.txt','r')
line=f.readline()
#把这一行数据用re转成列表
datalist=re.findall('{.*?}',line)
#把列表中各个元素转成字典,并且把peilv，gailv和kailizhishu分拆成三列，否则无法正确读入pandas
for i in range(0,len(datalist)-1):
    datalist[i]=eval(datalist[i])#先转成字典
    del datalist[i]['timestamp']#去掉时间戳
    #键值分拆
    datalist[i]['peilv_host']=datalist[i]['peilv'][0]
    datalist[i]['peilv_fair']=datalist[i]['peilv'][1]
    datalist[i]['peilv_guest']=datalist[i]['peilv'][2]
    del datalist[i]['peilv']
    datalist[i]['gailv_host']=datalist[i]['gailv'][0]
    datalist[i]['gailv_fair']=datalist[i]['gailv'][1]
    datalist[i]['gailv_guest']=datalist[i]['gailv'][2]
    del datalist[i]['gailv']
    datalist[i]['kailizhishu_host']=datalist[i]['kailizhishu'][0]
    datalist[i]['kailizhishu_fair']=datalist[i]['kailizhishu'][1]
    datalist[i]['kailizhishu_guest']=datalist[i]['kailizhishu'][2]
    del datalist[i]['kailizhishu']
    #将字典读入数据框
    if i==0:
        df=pd.DataFrame(datalist[i],index=[0])
    else:
        df=df.append(datalist[i],ignore_index=True)
    print(str(i))#显示添加进度





#把列表中各个字典去除时间戳后逐个插入mysql






