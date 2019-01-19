#一次请求只能return一组数据，所以不能简单通过一次请求进行全面同步和上传。也就是说，本地也要写一个同步函数，不断的提出请求知道同步结束为止
from flask import Flask
from flask import request
from flask import make_response
app = Flask(__name__)
app.debug = True


@app.route('/')
def hello_world():
    return 'OK!', 200

@app.route('/sync',methods=['GET','POST'])#当url地址为这个的时候，开始同步函数，令本地机器和云服务器的mysql同步
def sync():
    if request.method == 'Post':
        return '上传文件请post"/upload"'
    else:
        resp = make_response('那么开始吧',200)
        return resp
        #然后就是具体同步本地mysql和云服务器mysql的方法

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'GET':
        return '请求方法错误',400
    else:
        #此处应该是具体从远程机器上传监控数据的方法
        return 'XXX上传成功'
