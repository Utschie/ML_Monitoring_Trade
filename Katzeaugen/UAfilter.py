#为了过滤那些已经失效的UA
import requests
import urllib
import re
UAcontent = urllib.request.urlopen('file:///D:/data/useragentswitcher.xml').read()
UAcontent = str(UAcontent)
UAname = re.findall('(useragent=")(.*?)(")',UAcontent)
UAlist = list()
for z in range(0,int(len(UAname))):
    UAlist.append(UAname[z][1])

UAlist = UAlist[0:586]#这样就得到了一个拥有586个UA的UA池
UAlist.append('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36')#再加一个
r = requests.Session()
for i in range(0,len(UAlist)-1):
    header = {}
    header['User-Agent'] = UAlist[i]
    header['Host'] = 'www.okooo.com'#必须加上这个才能进入足球日历
    header['Upgrade-Insecure-Requests'] = '1'#这个也得加上
    ceshi = r.get('http://www.okooo.com/jingcai/',headers = header,verify=False,allow_redirects=False,timeout = 31)#从首页开启会话
    if (ceshi.status_code != 200):
        UAlist.remove(header['User-Agent'])
        print('第'+ str(i)+'个UA无效')
    else:
        print('第'+ str(i)+'个UA有效')
        with open('D:\\data\\UAlist.txt','a') as f:
            f.write(header['User-Agent'])
            f.write('\n')


