from flask import Flask,jsonify,request,url_for
from flask_restful import Api,Resource
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import logging
from config import Config
import base64

# 配置日志
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s -%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__) # 整个应用的载体
api = Api(app) # 创建api
config = Config()

# Flask配置
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# 确保必要的文件存在
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.ANOMALY_FOLDER, exist_ok=True)

class VideoProcessor:
    def __init__(self):
        self.model = None
    
    def init_model(self):
        """初始化YOLO模型"""
        try:
            self.model = YOLO('best.pt')
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败：{str(e)}")
            raise

    def process_frame(self,frame):
        """处理单帧图像，返回是否危险和检测到的框"""
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        results = self.model(frame)
        boxs = results[0].boxs

        truck_boxs = []
        person_boxs = []
        dangerous_pairs = []

        for box in boxs:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 1 and conf > config.TRUCK_CONFIDENCE_THRESHOLD:
                truck_boxs.append(box.xyxy[0].cpu().numpy())
            elif cls == 0 and conf > config.PERSON_CONFIDENCE_THRESHOLD:
                person_boxs.append(box.xyxy[0].cpu().numpy())

        is_dangerous = False
        for person_box in person_boxs:
            for trucl_box in truck_boxs:
                if self.check_person_in_danger_zone(person_box, trucl_box):
                    is_dangerous = True
                    dangerous_pairs.append((person_box, trucl_box))
                
        return is_dangerous, dangerous_pairs

    def check_person_in_danger_zone(self, person_box, truck_box):
         """检查一个人是否在危险区域内"""
         person_bottom = person_box[3]      
         truck_top = truck_box[1]
         truck_bottom = truck_box[3]

         # 计算卡车下1/3区域的起始y坐标
         truck_height = truck_bottom - truck_top
         truck_danger_zone = truck_bottom - (truck_height / 3)

         # 判断是否处于危险区域
         in_lower_third = (person_bottom >= truck_danger_zone and 
                           person_bottom <= truck_bottom)
         below_truck = person_bottom > truck_bottom

         return in_lower_third or below_truck       
    
    def process_video_stream(self,video_stream):
        """处理视频流，返回是否危险和检测到的框"""
        try:
            # 将视频流转换为numpy数组
            video_arry = np.frombuffer(video_stream.read(),np.uint8)

            # 创建临时文件名
            temp_path = os.path.join(
                config.UPLOAD_FOLDER,
                f'temp_{datetime.now().strftime("%Y%m%d%H%M%S")}.mp4'
            )

            # 保存视频流到临时文件
            with open(temp_path,'wb') as f:
                f.write(video_arry)

            # 使用OpenCV读取视频
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频流")    
            frame_count = 0
            anomaly_frames = [] # 存储异常帧的列表
            last_saved_frame = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1

                # 每隔一定帧数进行检测
                if frame_count % config.FRAME_SKIP !=0:
                    continue

                # 检测当前帧
                is_dangerous, dangerous_pairs = self.process_frame(frame)

                # 如果检测到危险情况
                if is_dangerous:
                    # 确保两张图片间隔足够远
                    if not anomaly_frames or (frame_count - last_saved_frame > config.MIN_FRAME_INTERVAL):
                        anomaly_info = self._save_anomaly_frame(frame, frame_count,cap,dangerous_pairs)
                        last_saved_frame = frame_count

                        # 如果已经保存了足够的异常图片，提前结束检测
                        if len(anomaly_frames) >= config.MAX_ANOMALY_FRAMES:
                            break

            # 格式化返回结果
            if anomaly_frames:
                return self._format_safe_result(anomaly_frames)
            else:
                return self._format_safe_result()

        except Exception as e:  
            logger.error(f"视频流处理失败：{str(e)}")
            raise
        finally:
            if 'cap' in locals():
                cap.release()
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _save_anomaly_frame(self,frame,frame_count,cap,dangerous_pairs):
        """保存异常帧"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        frame_time = frame_count / cap.get(cv2.CAP_PROP_FPS) # 计算帧时间
        filename = f'anomaly_{timestamp}_frame_{frame_count}.jpg' # 生成文件名

        # 在图片上绘制检测框和警告信息
        frame_with_boxs = frame.coyp()
        for person_box, truck_box in dangerous_pairs:
            # 绘制人员边界框（红色）
            cv2.rectangle(frame_with_boxs,
                          (int(person_box[0]),int(person_box[1])),
                          (int(person_box[2]),int(person_box[3])),
                          (0,0,255),2)
            
            # 绘制卡车边界框（蓝色）
            cv2.rectangle(frame_with_boxs,
                          (int(truck_box[0]),int(truck_box[1]))
                          (int(truck_box[2],int(truck_box[3])))
                          (255,0,0),2)
            
        # 添加警告文字
        cv2.putText(frame_with_boxs,
                    f"Danger! Frame:{frame_count}, Time: {int(frame_time)} s",
                     (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        # 保存图片
        filepath = os.path.join(config.ANOMALY_FOLDER,filename)
        cv2.imwrite(filepath,frame_with_boxs) 

        return {
            'frame': frame_with_boxs,
            'frame_number': frame_count,
            'time': frame_time,
            'filename': filename
        }
    
    def _format_anomaly_result(self, anomaly_frames):
         """格式化异常检测结果"""
         anomaly_images = [] # 存储异常图片的base64编码
         anomaly_details = []

         for frame_info in anomaly_frames:
            # 将图片编码保存为Numpy数字
            _,img_encoded = cv2.imencode('.jpg', frame_info['frame'])
            # 将编码后的图片转换为base64字符串
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

            anomaly_images.append({
                'image_data': img_base64,
                'filename': frame_info['filename'],
                'frame': frame_info['frame_number'],
                'time': frame_info['time']
            })

            anomaly_details.append(
                f"异常{len(anomaly_details)+1}:"
                f"帧{frame_info['framg_number']},"
                f"时间{int(frame_info['time'])}秒"
            )
        
         return {
            'status': 'success',
            'data': {
                'safe':False,
                'message':f"检测到{len(anomaly_frames)}处异常行为: {';'.join(anomaly_details)}",
                'anomaly_images': anomaly_images
            }
        }

    def _format_safe_result(self):
        """格式化安全检测结果"""
        return {
            'status': 'success',
            'data': {
                'safe': True,
                'message': "未检测到异常行为"
            }
        }

# 创建视频处理器实例
video_processor = VideoProcessor()

class HealthCheck(Resource):
    """健康检查接口"""
    def get(self):
        """处理健康检查请求"""
        return _create_response(
            message = "服务器运行正常"
        )

class VideoAnalysis(Resource):
    """视频分析接口"""
    def post(self):
        """处理视频分析请求""" 
        try:
            if 'video'not in request.files:
                return _create_response(
                    success = False,
                    message = "没有文件上传",
                    status_code = 400
                )

            video_file = request.files['video']
            if video_file.filename == '':
                return _create_response(
                    success=False,
                    message='没有选择文件',
                    status_code=400
                )
            
            # 检查文件类型
            if not video_file.filename.lower().endswith(tuple(config.SUPPORTED_VIDEO_FORMATS)):
                return _create_response(
                    success=False,
                    message='不支持的文件格式',
                    status_code=400
                )
            
            # 处理视频流
            result = video_processor.process_video_stream(video_file)
            if result['status'] == 'success':
                return result['data']
            else:
                return {
                    'safe': False,
                    'message': result.get('message','处理失败')
                }

        except Exception as e:
             logger.error(f"视频分析失败：{str(e)}")
             return {
                 'safe': False,
                 'message': f'视频分析失败:{str(e)}'
             }
def _create_response(massage, data = None, success = True, status_code = 200):
    """创建统一的响应格式"""
    response = {
        'status': 'success' if success else 'error',
        'massage': massage
    }

    if data is not None:
        response['data'] = data

    return response, status_code

# 注册API路由
api.add_resource(HealthCheck, '/api/v1/health',endpoint='healthcheck')
api.add_resource(VideoAnalysis, '/api/v1/video/analysis',endpoint='videoanalysis')

if __name__ == '__main__':
    logger.info("正在启动服务...")
    video_processor.init_model()
    app.run(host='0,0,0,0',port=config.SERVER_PORT,debug=False)