#寻找cid
#经过修改后，用来提取20141130-20190224之间全部数据中的cid————20201128
import pandas as pd
import os
cidlist=list()
filelist = os.listdir('F:\\cleaned_data_20141130-20160630')#读出洗好的数据的文件夹名
for i in filelist:
    bisailist = os.listdir('F:\\cleaned_data_20141130-20160630\\'+i)
    for j in bisailist:
        filepath = 'F:\\cleaned_data_20141130-20160630\\'+i+'\\'+j
        data = pd.read_csv(filepath)
        cid = data.cid.drop_duplicates().values.tolist()
        cidlist = list(set(cidlist+cid))

filelist = os.listdir('H:\\cleaned_data_new_20160701-20190224')
for i in filelist:
    bisailist = os.listdir('H:\\cleaned_data_20160701-20190224\\'+i)
    for j in bisailist:
        filepath = 'H:\\cleaned_data_20160701-20190224\\'+i+'\\'+j
        data = pd.read_csv(filepath)
        cid = data.cid.drop_duplicates().values.tolist()
        cidlist = list(set(cidlist+cid))

filepath = 'H:\\cidlist_new.csv'
ser=pd.Series(cidlist)#转成series
ser.to_csv(filepath)#写入csv