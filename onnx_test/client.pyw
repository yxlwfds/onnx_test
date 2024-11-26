import base64
import io

from utils import *
import datetime
import json
import multiprocessing
import threading
import time
from multiprocessing import freeze_support
from concurrent.futures import ThreadPoolExecutor
import shutil

import cv2
import numpy as np
from humanfriendly.testing import retry

import config
import dectonnx

import onnxruntime as rt

setup_logging()
executor = ThreadPoolExecutor(max_workers=10)


class Service:
    """
    Yolo服务
    """

    def __init__(self):
        self.queue_send = multiprocessing.Queue(40)  # 发送数据队列qg

    def restart_yolo_service(self, d):
        """
        重启YOLO服务
        Args:
            d: 摄像头配置
        Returns:
        """
        if ping_camera(d.camera):
            if not yolo_by_thread:
                try:
                    logging.info(f"重启YOLO服务...{d.id}")
                    pid = r.get(f'yolo_{d.id}')
                    kill_pid(pid)
                except Exception as e:
                    logging.exception(e)
                yolo_process = multiprocessing.Process(target=self.yolo_service, args=(d,))
                yolo_process.start()
                r.set(f'yolo_{d.id}', yolo_process.pid)
            else:
                try:
                    logging.info(f"重启YOLO服务...{d.id}")
                    pid = r.get(f'yolo_{d.id}')
                    kill_pid(pid, False)
                except Exception as e:
                    logging.exception(e)
                yolo_thread = threading.Thread(target=self.yolo_service, args=(d,))
                yolo_thread.start()
                r.set(f'yolo_{d.id}', yolo_thread.ident)
        else:
            logging.info(f"{d.id}摄像头不可用,暂不重启YOLO服务...")

    def restart_push_service(self, d):
        """
        重启推流服务
        Args:
            d:  摄像头配置
        Returns:
        """
        # if ping_camera(d.camera):
        #     # 小于10说明推流服务属性激活状态,不需要重启
        #     if not r.get(f'{d.id}_push_active') or (
        #             current_time() - float(r.get(f'{d.id}_push_active')) > 4):
        #         try:
        #             logging.info(f"重启推流服务...{d.id}")
        #             kill_python(f'python_{d.id}.exe')
        #             kill_python(f'pythonw_{d.id}.exe')
        #             time.sleep(6)
        #         except Exception as e:
        #             logging.exception(e)
        #         # self.frame_push(d)
        #         yolo_thread = threading.Thread(target=frame_push, args=(d,))
        #         yolo_thread.start()
        # else:
        #     logging.info(f"{d.id}摄像头不可用,暂不重启推流服务...")
        ...

    def yolo_service(self, d: dict):
        """
        YOLO服务
            1. 连接摄像头
            2. 加载模型
            3. yolo检测
            4. 推送数据到队列,等待其它进程处理
        Args:
            d: config dict 某个摄像头配置
        Returns:

        """
        # q = getattr(self, f'video_queues_{d.id}')  # 视频图片帧队列
        logging.info(f'{d.id} service starting...')

        filename = os.path.splitext(os.path.basename(f'{config.path}models/{d.model}'))[0]
        cap, model = connect_camera(d, d.camera, d.id)
        if cap and cap.isOpened():
            dectonnx.readClass()
            model = rt.InferenceSession(f'{config.path}models/{d.model}', None)
            model.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # 如果不存在image_base/摄像头id目录，则创建
        if not os.path.exists(f"{image_base}/{d.id}"):
            os.makedirs(f"{image_base}/{d.id}")
        save = False
        set_cap = True
        try:
            s_i = 0
            retry = 0
            last_heartbeat = time.time()
            while 1:
                if not cap or not cap.isOpened():
                    r.hset(f'stream_info_{d.id}', 'status', 'offline')
                    if cap:
                        cap.release()
                    cap, pipe = connect_camera(d, d.camera, d.id, retry)
                    retry += 1
                    time.sleep(20)
                else:
                    ret, raw_frame = cap.read()

                    if ret:
                        if set_cap:
                            height, width = raw_frame.shape[:2]
                            logging.info(f"{d.id} infos::{fps}, {width}, {height}")
                            r.hset(f'stream_info_{d.id}',
                                   mapping={'fps': fps, 'width': width, 'height': height, 'status': 'online'})
                            set_cap = False
                        now = current_time()
                        if now - last_heartbeat >= 1:
                            r.set(f'{d.id}_report', current_time())
                            last_heartbeat = now
                        s_i = s_i + 1
                        out_img, num_box = dectonnx.detectBox(model, raw_frame, filename)  # 检测
                        # if len(num_box) > 0:
                            # if save:
                            #     # 保存图片
                            #     # 不再直接保存图片，而是将图片base64后推送到 Redis Stream
                            #     out_img_base64 = base64.b64encode(out_img.tobytes()).decode('utf-8')
                            #     r.lpush(f"frame_base64", out_img_base64)
                            #     # 限制队列长度为10
                            #     r.ltrim(f"frame_base64", 0, 100)
                            #     save = False
                            # self.process_box(d, num_box, out_img)
                        # else:
                        #     if s_i == 1:  # 当检测框为空时，每隔5帧保存一张图片
                        #         save = True
                        r.xadd(f"frame_{d.id}", {'frame': raw_frame.tobytes(), 'timestamp': current_time()}, maxlen=20)
                        if s_i > 5:
                            s_i = 0
        except KeyboardInterrupt:
            logging.info("Stopping yolo process...")
        except Exception as e:
            logging.exception(e)
            raise e
        finally:
            if cap is not None:
                cap.release()
            del model
            r.delete(f'{d.id}_starting')

    def service_manager(self):
        """
        连接与推流服务管理器
            通过redis记录摄像头状态，检查摄像头连接状态，推流状态来关闭或重启服务
        Returns:
        """

        logging.debug('service_manager starting...')
        global first
        cap_time = 0  # 摄像头重启时间
        push_time = 0  # 推流重启时间
        cap_start = False  # 摄像头重启标志
        push_start = False  # 推流重启标志
        while 1:
            try:
                for d in config.stream_list:
                    if first:
                        if r.exists(f"frame_{d.id}"):
                            r.delete(f"frame_{d.id}")

                        r.delete(f'{d.id}_starting')
                        s = self.restart_yolo_service(d)
                        # time.sleep(3)
                        # s = self.restart_push_service(d)
                    else:
                        cap_start, push_start = self.check_cap(cap_start, d, push_start)
                        # push_start = self.check_push_stream(d, push_start, push_time)
                first = False
                time.sleep(10)
            except Exception as e:
                logging.exception(e)
            if cap_start:
                cap_time = current_time()
                cap_start = False
            if push_start:
                push_time = current_time()
                push_start = False

    def check_push_stream(self, d, push_start, push_time):
        """
        检查推流服务
            1. 检查推流服务是否超时
            2. 检查摄像头是否有推流
            3. 决定是否重启推流服务
        Args:
            d:  摄像头配置
            push_start:  推流重启标志
            push_time:  推流重启时间
        Returns: push_start
        """
        if not r.get(f'{d.id}_push_report') or (
                current_time() - float(r.get(f'{d.id}_push_report')) > 31):
            if not r.get(f'{d.id}_report') or (current_time() - float(r.get(f'{d.id}_report')) < 200):
                # if not r.get(f'{d.id}_push_active') or (
                #         current_time() - float(r.get(f'{d.id}_push_active'))) > 30:
                logging.info(f"推流超时,准备重启推流服务...{d.id}")
                if current_time() - push_time > 30:
                    self.restart_push_service(d)
                    push_start = True
            else:
                logging.info(f"{d.id}摄像头无流,暂不重启推流服务...")
        return push_start

    def check_cap(self, cap_start, d, push_start):
        """
        检查摄像头连接
            1. 检查摄像头是否超时
            2. 决定是否重启摄像头服务
            3. 重启摄像头服务后，重启推流服务
        Args:
            cap_start:  摄像头重启标志
            d:  摄像头配置
            push_start:  推流重启标志
        Returns: cap_start, push_start
        """
        if not r.get(f'{d.id}_report') or (
                current_time() - float(r.get(f'{d.id}_report')) > 30):
            if not r.get(f'{d.id}_starting'):
                logging.info(f"摄像头连接超时,准备重启服务...{d.id}")
                self.restart_yolo_service(d)
                cap_start = True
                time.sleep(15)
                # self.restart_push_service(d)
                # push_start = True
        return cap_start, push_start

    def process_box(self, d: dict, num_box, out_img):
        """
        处理检测框
        Args:
            d: 摄像头配置
            num_box: 检测框
        Returns:
        """
        data_box = json.dumps(num_box.tolist())
        np_img = np.array(out_img)
        _, buffer = cv2.imencode('.png', np_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 90])
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        # 保存img_base64到文件
        # with open(f"{image_base}/{d.id}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png", "wb") as f:
        #     f.write(base64.b64decode(img_base64))
        data_out = {'id': f'{d.id}', 'res': data_box, 'image': img_base64}
        if not self.queue_send.full():
            self.queue_send.put(data_out)

    def box_service(self):
        """
        检测框推数据到box_data_queue redis服务
        Returns:
        """
        logging.debug('box_service started')
        try:
            while 1:
                if not self.queue_send.empty():
                    data_out = self.queue_send.get()
                    r.lpush(f'box_data_queue', str(data_out))
                    r.ltrim(f'box_data_queue', 0, 1000)
                time.sleep(0.00001)
        except KeyboardInterrupt:
            logging.info("Stopping box service...")

    def start(self):
        """
        启动服务
        Returns:
        """
        try:
            monitor_yolo = multiprocessing.Process(target=self.service_manager)
            monitor_yolo.start()
            box_process = multiprocessing.Process(target=self.box_service)
            box_process.daemon = True
            box_process.start()
            monitor_yolo.join()
            box_process.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, initiating shutdown...")
        finally:
            cleanup()


if __name__ == '__main__':
    freeze_support()
    rstp_service()
    s = Service()
    s.start()
