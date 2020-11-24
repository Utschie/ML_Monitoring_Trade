#本文件是datacleaning_new_all_to_csv后，把所有比赛的csv文件转化成按帧排布的矩阵列表的csv形式
#本文件可直接用numpy读取成矩阵列表,以及对应的frametime作为位置编码
#最终还是要搞成直接读取一帧帧的形式，以及相应的位置信息（frametime），这里的位置信息可能会在可能的帧填充时用到的————20201112
import numpy as np
import pandas as pd
import csv
with open('D:\\data\\cidlist.csv') as f:
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表
cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
def csv2dflist(filepath,filename):#把csv转成dflist然后写入npy文件,filepath是读入文件，filename是输出文件
    data = pd.read_csv(filepath)#读取文件
    data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
    frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
    framelist = list()#framelist为一个空列表
    for i in frametimelist:#其中frametimelist里的数据是整型
        state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
        state = np.array(state)#转成numpy多维数组
        #在填充成矩阵之前需要知道所有数据中到底有多少个cid
        statematrix=np.zeros((410,12))#
        for j in state:
            cid = j[1]#得到浮点数类型的cid
            index = cidlist.index(cid)
            statematrix[index] = j#把对应矩阵那一行给它
        statematrix=np.delete(statematrix,(0,1), axis=-1)#去掉frametime和cid列
        framelist.append(statematrix)
    framelist = np.array(framelist)#转成numpy数组
    frametimelist = np.array(frametimelist)
    np.savez(filename,framelist=framelist,frametimelist=frametimelist)#framelist和frametimelist分别是自定义的key，将来读取用这两个key来引用
    