class Config:
    # 服务器配置
    SERVER_HOST = 'localhost'
    SERVER_PORT = 5000
    SERVER_URL = f'http://{SERVER_HOST}:{SERVER_PORT}'
    
    # API端点
    API_ENDPOINTS = {
        'health_check': '/',
        'video_check': '/check_video'
    }
    
    # 超时设置
    CONNECTION_TIMEOUT = 30  # 连接超时时间（秒）
    REQUEST_TIMEOUT = 180    # 请求超时时间（秒）
    
    # 文件大小限制
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 最大允许的文件大小（50MB）
    
    # 支持的视频格式
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov']
    
    # 服务器端文件夹配置
    UPLOAD_FOLDER = 'static/uploads'
    ANOMALY_FOLDER = 'static/anomalies'
    
    # 模型配置
    TRUCK_CONFIDENCE_THRESHOLD = 0.3
    PERSON_CONFIDENCE_THRESHOLD = 0.3
    
    # 视频处理配置
    FRAME_SKIP = 5               # 每隔5帧处理一次
    MIN_FRAME_INTERVAL = 10      # 保存异常帧的最小间隔
    MAX_ANOMALY_FRAMES = 2       # 最多保存的异常帧数量
    
    # 客户端配置
    LOCAL_ANOMALY_FOLDER = 'anomalies'  # 本地保存异常图片的文件夹
    
    # 清理配置
    MAX_STORAGE_DAYS = 7    # 文件保存天数
    MAX_STORAGE_SIZE = 1024 * 1024 * 1024  # 最大存储空间(1GB)