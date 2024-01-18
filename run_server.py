from flask import Flask
import subprocess

app = Flask(__name__)

# 启动后台服务
label_studio_process = subprocess.Popen(["label-studio", "start", "--port", "8081"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
label_studio_ml_process = subprocess.Popen(["label-studio-ml", "start", "ml_backend", "--port", "8082"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
pinfer_process = subprocess.Popen(["pinfer", "--backend-port", "8585", "api:service"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 定义路由
@app.route('/')
def hello():
    return 'Hello, World!'

# 运行 Flask 应用
if __name__ == '__main__':
    app.run()

