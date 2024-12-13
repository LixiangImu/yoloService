FROM python:3.9

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt
COPY requirements.txt .

# 安装Python依赖
RUN pip install -r requirements.txt

# 明确复制配置文件和应用文件
COPY config.py /app/
COPY app_Service.py /app/
COPY best.pt /app/

# 创建必要的目录
RUN mkdir -p static/uploads static/anomalies

# 安装Python依赖
RUN pip install -r requirements.txt

# 暴露端口
EXPOSE 5000

# 启动服务
CMD ["python", "app_Service.py"]