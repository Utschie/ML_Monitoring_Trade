def proc(datelist):
    for i in datelist:
        start=time.time()
        outputpath1='F:\\cleaned_data_new\\'+i#为这一天建立一个文件夹
        outputpath2='F:\\cleaned_data_new_dflist\\'+i
        os.makedirs(outputpath1)#建立保存csv的文件夹
        os.makedirs(outputpath2)#建立保存npz的文件夹
        df=txt2csv(i)#将txt文件导出csv后读入dataframe
        urlnumlist=list(df['urlnum'].value_counts().index)#获得当天比赛列表
        bisailist=list(map(bisaiquery(df),urlnumlist))#获得由各个比赛的dataframe组成的表
        coprocess(bisailist)#协程并发分别写入文件
        end=time.time()
        outputstr='日期：'+i+'已清洗完成\n'+'耗时：'+str(end-start)+'秒\n'
        print(outputstr)
