#寻找cid
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

