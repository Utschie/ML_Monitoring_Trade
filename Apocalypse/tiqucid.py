#寻找cid
#经过修改后，用来提取20141130-20190224之间全部数据中的cid————20201128
#不过这个程序比较慢，要遍历10万多个文件
import pandas as pd
import os
cidlist=list()
filenum = 0#已遍历文件数
filelist = os.listdir('F:\\cleaned_data_new_20141130-20160630')#读出洗好的数据的文件夹名
for i in filelist:
    bisailist = os.listdir('F:\\cleaned_data_new_20141130-20160630\\'+i)
    for j in bisailist:
        filepath = 'F:\\cleaned_data_new_20141130-20160630\\'+i+'\\'+j
        data = pd.read_csv(filepath)
        cid = data.cid.drop_duplicates().values.tolist()
        cidlist = list(set(cidlist+cid))
        filenum+=1
        print('已遍历'+str(filenum)+'个文件')

filelist = os.listdir('H:\\cleaned_data_new_20160701-20190224')
for i in filelist:
    bisailist = os.listdir('H:\\cleaned_data_new_20160701-20190224\\'+i)
    for j in bisailist:
        filepath = 'H:\\cleaned_data_new_20160701-20190224\\'+i+'\\'+j
        data = pd.read_csv(filepath)
        cid = data.cid.drop_duplicates().values.tolist()
        cidlist = list(set(cidlist+cid))
        filenum+=1
        print('已遍历'+str(filenum)+'个文件')

filepath = 'H:\\cidlist_complete.csv'
ser=pd.Series(cidlist)#转成series
ser.to_csv(filepath)#写入csv