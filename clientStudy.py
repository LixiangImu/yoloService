import requests
import logging
import os
from datetime import datetime
from config import Config
import cv2
import time
import base64
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s)'
)
logger = logging.getLogger(__name__)

class VideoSafetyClient:
    def __init__(self):
        self._int_config()
        self._setup_session()
        self._init_storage()

    def _init_config(self):
        """初始化配置"""
        self.config = Config()
        self.api_endpoints = {
            'health_check': '/api/v1/health',
            'video_check': '/api/v1/video/analyze'
        }
    
    def _setup_session(self):
        """设置会话"""
        self.session = requests.Session()
        self.session.mount('http://',requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))

    def _init_storage(self):
        """初始化存储"""
        self.local_anomaly_folder = os.path.join(os.getcwd(),'anomalies')
        os.makedirs(self.local_anomaly_folder,exist_ok=True)
        logger.info(f"异常图片将保存至: {self.local_anomaly_folder}")

    def check_server_health(self):
        """检查服务器健康状况"""
        try:
            response = self.session.get(
                f"{self.config.SERVER_URL}/api/v1/health",
                timeout=self.config.CONNECTION_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                return True,data.get('message','服务器正常')

            return False, f"服务器相应错误: {response.status_code}"
        
        except Exception as e:
            return False, f"连接失败: {str(e)}"

    def check_video_safety(self,video_path):
        """检查视频安全性"""
        try:
            start_time = time.time()

            # 预检查
            if not os.path.exists(video_path):
                return self._format_result({
                    'safe':False,
                    'message':'视频文件不存在',
                    'process_time':0
                })
            
            #打开视频文件
            with open(video_path,'rb') as video_file:
                logger.info(f"正在发送视频:{video_path}")

                #发送请求
                response = self.session.post(
                    f"{self.config.SERVER_URL}/api/v1/video/analyze",
                    files={'video':video_file},
                    timeout=(self.config.CONNECTION_TIMEOUT,self.config.REQUEST_TIMEOUT)
                )

                if response.status_code == 200:
                    data = response.json()

                    # 如果检测到异常，保存图片
                    local_images = []
                    if not data['safe'] and 'anomaly_images' in data:
                        #保存每张异常图片
                        for img_data in data['anomaly_images']:
                            # 解码base64图片数据
                            img_bytes = base64.b64decode(img_data['image_data'])

                            # 构建保存路径
                            filename = img_data['filename']   
                            save_path = os.path.join(self.local_anomaly_folder,filename)

                            # 保存图片
                            with open(save_path,'wb') as f:
                                f.write(img_bytes)

                            local_images.append(save_path)
                            logger.info(f"已保存异常图片: {save_path}")

                return self._format_result({
                    'safe': data['safe'],
                    'message': data['message'],
                    'local_images': local_images,
                    'process_time': time.time() - start_time
                })
            
            return self._format_result({
                'safe': False,
                'message':f"服务器响应错误: {response.status_code}",
                'process_time': time.time() - start_time
            })

        except Exception as e:
             logger.error(f"请求失败: {str(e)}")
             return self._format_result({
                 'safe':False,
                 'messaage': f"请求失败: {str(e)}",
                 'process_time': time.time() - start_time
             })    
        
    def _format_result(self,result):
        """格式化检测结果"""
        return {
            'safe':result.get('safe',False),
            'message':result.get('message','未知错误'),
            'local_images':result.get('local_images',[]),
            'process_time':result.get('process_time',0),
            'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
def print_result(result):
    """打印检测结果"""
    print("\n" + "="*50)
    print("视频安全检测结果：")
    print("="*50)

    # 添加默认值，防止键不存在
    timestamp = result.get('timestamp',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    process_time = result.get('process_time',0)
    safe = result.get('safe',False)
    message = result.get('message','未知错误')
    locals_images = result.get('local_images',[])

    print(f"检测时间: {timestamp}")
    print(f"处理用时: {process_time:.2f}秒")
    print(f"安全状态: {'安全' if safe else '不安全'}")
    print(f"详细信息: {message}")

    if locals_images:
        print("\n异常图片已保存至:")
        for img_path in locals_images:
            print(f"- {img_path}")
        
    print("="*50 + "\n")

def main():
    """主函数"""
    try:
        # 使用默认配置创建客户端实例
        client = VideoSafetyClient()

        #检查服务器状态
        logger.info("正在检查服务器状态...")
        is_health, health_message = client.check_server_health()
        if not is_health:
            logger.error(f"服务器状态异常: {health_message}")
            return
        logger.info(f"服务器状态: {health_message}")

        # 获取视频路径
        video_path = input("请输入视频文件路径: ").strip('"')
        if not video_path:
            logger.error("未输入视频路径")
            return
        
        #检测视频
        logger.info("开始检测视频...")
        result = client.check_video_safety(video_path)

        # 打印结果
        print_result(result)

    except KeyboardInterrupt:
        print("\n程序已终止")
    except Exception as e:
        logger.error(f"程序执行错误: {str(e)}")
    finally:
        input("按回车键退出程序...")

if __name__ == "__main__":
    main()