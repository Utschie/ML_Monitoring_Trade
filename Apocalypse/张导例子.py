#一些杂七杂八的库，平时都直接复制使用的，懒得删了，连接操作的话只会用到某几个
import numpy as np
import pandas as pd
import datetime,time
import os,sys
import paramiko
import pymysql
import math
from pyecharts.charts import Line,Page
from pyecharts import options as opts
from sshtunnel import SSHTunnelForwarder
from  multiprocessing import Process,Pool,Manager
# 导入相关库-email
from email.mime.multipart import MIMEMultipart  # 构建邮件头信息，包括发件人，接收人，标题等
from email.mime.text import MIMEText  # 构建邮件正文，可以是text，也可以是HTML
from email.mime.application import MIMEApplication  # 构建邮件附件，理论上，只要是文件即可，一般是图片，Excel表格，word文件等
from email.header import Header  # 专门构建邮件标题的，这样做，可以支持标题中文
import smtplib

#在服务器上直接运行python脚本的命令
#sudo /home/zhangjl2/anaconda3/bin/python3 /home/zhangjl2/python/高配冗余指标.py

#定一个类，后续引用，里面的date日期不用管
class High_Target():
	def __init__(self,startDate,endDate):
		self.start_date = startDate
		self.end_date = endDate
		self.host = "192.168.29.24"
		self.username = "zhangjl2"
		self.password = "zhangjl2@rpp999"
		self.port = 57891
		self.ssh_host = "10.8.0.35"
		self.ssh_port = 57891
		self.ssh_username = "zhangjl2"
		self.ssh_password = "Gd9113210"
		self.db_host = "10.16.158.73"
		self.db_port = 63751
		self.db_username = "rpp-read"
		self.db_password = "rpp-read@rpp999"
		self.db_name = "ROAS_DB"
		self.db_name1 = "ROAS_SRC_DB"
		self.db2_host = "10.8.18.33"
		self.db2_port = 3306
		self.db2_username = "root"
		self.db2_password = "oa_Adm@1843"
		self.db2_name = "OA_db"


	#连接24服务器
	def Connect_server(self):
		self.ssh = paramiko.SSHClient()
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		try:
			self.ssh.connect(self.host,self.port,self.username,self.password)
			print("24连接成功")
			#return self.ssh
		except:
			print("24连接失败")
			#return None
		return None


	#连接数据库，三个
	def Connect_db(self):
		self.db=pymysql.connect(host=self.db_host,port=self.db_port,user=self.db_username,passwd=self.db_password,db=self.db_name)
		self.db2=pymysql.connect(host=self.db2_host,port=self.db2_port,user=self.db2_username,passwd=self.db2_password,db=self.db2_name)
		self.db3=pymysql.connect(host=self.db_host,port=self.db_port,user=self.db_username,passwd=self.db_password,db=self.db_name1)


	#关闭数据库，也是三个
	def Close_db(self):
		self.db.close()
		self.db2.close()
		self.db3.close()


	#定义一个执行sql，并返回输出结果的函数
	def Exec_sql_commond(self,sqlCmd,db):
		try:
			data = pd.read_sql(sqlCmd,db)
			return data
		except Exception as e:
			print("数据库获取报错")
			raise


	#定义一个执行shell命令，并返回输出的函数
	def Exec_command(self,cmd):
		stdin,stdout,stderr = self.ssh.exec_command(cmd)
		results = stdout.read().decode()
		return results


	#连接数据库例子
	def sql_example(self):
		cmd = "SELECT date(cust_peak_time) '日期',isp '运营商',cache,appService_type '应用服务类型',node '节点中文',cust_peak_time '全平台峰值时刻',original_capacity '最大能力(不扣冗余故障)(Mkbs)',cust_peak_bandwidth '设备已用带宽(Mkbs)' FROM OA_db.pd_cache_Band_custPeakDay1 WHERE isp in ('电信','网通','移动') and DATE(cust_peak_time)<='%s' and DATE(cust_peak_time)>='%s' and pop_enableState='解冻' and pop_openState='开通'"%(self.day,one_month_day)#你要执行的sql语句
		print(cmd)
		df = self.Exec_sql_commond(cmd,self.db2)#这里，要连接哪个库，就对应改哪个
		return df


	#连接服务器，读取落地文件的例子
	def shell_example(self):
		cmd = "ls /home/zhangjl2/high_deploy/high_fault|grep high_fault_detail"
		#在服务器上执行shell命令，也可以不用函数，改用os.popen()
		df = self.Exec_command(cmd)
		#将结果切割成矩阵后生成dataframe，我个人都是用数据框，几乎不用字典。。。。
		df = pd.DataFrame(bs.Split_text_to_array(df.split('\n')[:-1],'|'),columns=["dns_map", "time", "bw"])
		return df


work=High_Target(start_date,end_date)#类
work.Connect_server()#连接服务器
work.Connect_db()#连接数据库
work.Get_all_time_result()#类下你需要执行的函数
work.Close_db()#关闭数据库