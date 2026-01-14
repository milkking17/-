# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import json
import numpy as np
import cv2
import tempfile  # 用于创建临时文件

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast

import paddle
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import Trainer, Trainer_ARSL  # 只用原生Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_gcu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')

# OCR相关导入
from paddleocr import PaddleOCR

# ======================== 全局配置（可根据需求调整） ========================
CAMERA_ID = 0  # 电脑内置摄像头默认ID=0
CAMERA_WIDTH = 640  # 摄像头分辨率（降低分辨率提升速度）
CAMERA_HEIGHT = 480
SEND_INTERVAL = 0.1  # 发送间隔（秒）
DRAW_THRESHOLD = 0.3  # 降低阈值，更容易检测到目标
SAVE_CROP_IMG = False  # 实时场景默认不保存裁剪图，提升速度
TEMP_IMG_PATH = "temp_frame.jpg"  # 临时保存摄像头帧的文件
# 全局OCR实例（只初始化一次，提升速度）
OCR_INSTANCE = None
# ===========================================================================

def parse_args():
    # 仅使用PaddleDetection内置ArgsParser（自带-c/--config、-o/--opt参数）
    parser = ArgsParser()
    # 只添加自定义参数，不添加--config/--weights（用内置的-c/-o）
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing temporary files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=DRAW_THRESHOLD,
        help="Threshold to reserve the result.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    # OCR相关参数
    parser.add_argument(
        "--enable_ocr",
        action='store_true',
        default=True,
        help="Whether to enable OCR text recognition for road signs.")
    parser.add_argument(
        "--ocr_lang",
        type=str,
        default='en',
        help="OCR language, 'en' for English, 'ch' for Chinese.")
    parser.add_argument(
        "--label_list_path",
        type=str,
        default="dataset/roadsign_voc/label_list.txt",
        help="Path to label list file for category name mapping.")
    args = parser.parse_args()
    return args

# ======================== 红绿灯颜色识别函数（优化版） ========================
def detect_traffic_light_color(traffic_light_img):
    """优化版：减少计算量，提升实时性"""
    # 快速预处理
    blurred = cv2.GaussianBlur(traffic_light_img, (3, 3), 0)  # 缩小卷积核
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 提取中心ROI（缩小计算区域）
    h, w = hsv.shape[:2]
    roi_x1, roi_y1 = int(w * 0.2), int(h * 0.2)
    roi_x2, roi_y2 = int(w * 0.8), int(h * 0.8)
    hsv_roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2] if h > 20 and w > 20 else hsv

    # 颜色阈值（精简版）
    mask_red = cv2.inRange(hsv_roi, np.array([0, 100, 80]), np.array([10, 255, 255])) + \
               cv2.inRange(hsv_roi, np.array([160, 100, 80]), np.array([179, 255, 255]))
    mask_yellow = cv2.inRange(hsv_roi, np.array([20, 100, 80]), np.array([35, 255, 255]))
    mask_green = cv2.inRange(hsv_roi, np.array([40, 80, 80]), np.array([80, 255, 255]))

    # 计算有效像素面积（转成Python原生int）
    area_red = int(cv2.countNonZero(mask_red))
    area_yellow = int(cv2.countNonZero(mask_yellow))
    area_green = int(cv2.countNonZero(mask_green))

    # 判断结果（过滤小噪声）
    color_areas = {'red': area_red, 'yellow': area_yellow, 'green': area_green}
    max_color = max(color_areas, key=color_areas.get)
    max_area = color_areas[max_color]
    
    return max_color if max_area > 50 else 'unknown', color_areas

# ======================== 裁剪图片函数（轻量化版） ========================
def crop_image_by_bbox(img, bbox, save_path=None):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    # 边界保护
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop_img = img[y1:y2, x1:x2]
    
    # 仅当开启保存且裁剪图有效时保存
    if SAVE_CROP_IMG and save_path and crop_img.size > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, crop_img)
    return crop_img

# ======================== 识别信息处理函数（返回结构化数据） ========================
def process_detection_result(img, result, FLAGS, label_list):
    """
    处理单帧检测结果，返回结构化的识别信息（仅包含有效结果）
    关键修复：1. 移除OCR的show_log参数 2. 全局OCR实例避免重复初始化 3. 异常处理防止崩溃
    """
    global OCR_INSTANCE
    # 初始化全局OCR实例（只初始化一次，提升速度+避免重复报错）
    if FLAGS.enable_ocr and OCR_INSTANCE is None:
        try:
            # 关键修复：移除show_log=False，适配新版本PaddleOCR
            OCR_INSTANCE = PaddleOCR(lang=FLAGS.ocr_lang, use_angle_cls=True)
        except Exception as e:
            logger.warning(f"OCR初始化失败：{e}，将关闭OCR功能")
            FLAGS.enable_ocr = False

    valid_info = {
        "timestamp": float(time.time()),  # 转成Python float
        "road_signs": [],  # 路牌信息
        "traffic_lights": []  # 红绿灯信息
    }

    # 固定类别ID
    ROAD_SIGN_ID = 0    # 路牌
    TRAFFIC_LIGHT_ID = 2 # 红绿灯

    if 'bbox' in result and len(result['bbox']) > 0:
        for bbox_idx, bbox_info in enumerate(result['bbox']):
            cls_id, score, x1, y1, x2, y2 = bbox_info
            cls_id = int(cls_id)  # 转Python int
            score = float(score)  # 转Python float
            if score < FLAGS.draw_threshold:
                continue  # 跳过低置信度结果
            
            cls_name = label_list[cls_id] if (label_list and cls_id < len(label_list)) else f"class_{cls_id}"
            # 所有坐标转Python float
            bbox_coords = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            }

            # 处理红绿灯
            if cls_id == TRAFFIC_LIGHT_ID:
                crop_img = crop_image_by_bbox(img, [x1, y1, x2, y2])
                if crop_img.size > 0:
                    light_color, color_areas = detect_traffic_light_color(crop_img)
                    traffic_light_info = {
                        "class_name": cls_name,
                        "confidence": round(score, 2),  # 已转Python float
                        "bbox": bbox_coords,
                        "color": light_color,
                        "color_areas": color_areas  # 已转Python int
                    }
                    valid_info["traffic_lights"].append(traffic_light_info)
            
            # 处理路牌（OCR）
            elif cls_id == ROAD_SIGN_ID and FLAGS.enable_ocr:
                crop_img = crop_image_by_bbox(img, [x1, y1, x2, y2])
                ocr_text = ""
                if crop_img.size > 0:
                    try:
                        # 异常处理：避免单帧OCR识别失败导致程序崩溃
                        ocr_result = OCR_INSTANCE.ocr(crop_img)
                        if ocr_result and ocr_result[0]:
                            ocr_text = ' '.join([line[1][0] for line in ocr_result[0]])
                    except Exception as e:
                        logger.warning(f"单帧OCR识别失败：{e}")
                        ocr_text = "识别失败"
                
                road_sign_info = {
                    "class_name": cls_name,
                    "confidence": round(score, 2),  # 已转Python float
                    "bbox": bbox_coords,
                    "ocr_text": ocr_text
                }
                valid_info["road_signs"].append(road_sign_info)

    # 仅保留有有效识别结果的数据
    if not valid_info["road_signs"] and not valid_info["traffic_lights"]:
        return None
    return valid_info

# ======================== 信息发送函数（可替换为串口/网络发送） ========================
def send_recognition_info(info):
    """
    识别信息发送接口（核心函数）
    可根据需求修改：比如发送到串口、网络、MQTT等
    """
    if info is None:
        return  # 无有效信息，不发送
    
    # 格式化发送内容（JSON格式，便于解析）
    send_data = json.dumps(info, ensure_ascii=False)
    # ========== 这里替换为你的实际发送逻辑 ==========
    print(f"[发送信息] {send_data}")
    # 示例：串口发送（需安装pyserial）
    # import serial
    # ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
    # ser.write((send_data + '\n').encode('utf-8'))
    # ==============================================

# ======================== 实时检测主函数 ========================
def run_realtime_detection(FLAGS, cfg):
    # 检查cfg中是否有weights参数（通过-o传递的）
    if not hasattr(cfg, 'weights') or not cfg.weights:
        logger.error("未指定权重文件！请通过 -o weights=xxx.pdparams 传递权重路径")
        return

    # 初始化Trainer（原生支持，无接口兼容问题）
    if cfg.get('ssod_method', None) == 'ARSL':
        trainer = Trainer_ARSL(cfg, mode='test')
        trainer.load_weights(cfg.weights, ARSL_eval=True)
    else:
        trainer = Trainer(cfg, mode='test')
        trainer.load_weights(cfg.weights)

    # 加载类别列表（添加UTF-8编码）
    label_list = []
    if os.path.exists(FLAGS.label_list_path):
        with open(FLAGS.label_list_path, 'r', encoding='utf-8') as f:
            label_list = [line.strip() for line in f.readlines() if line.strip()]
    else:
        logger.warning(f"标签文件不存在：{FLAGS.label_list_path}，将使用默认类别名")

    # 摄像头ID重试逻辑
    cap = None
    for cam_id in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            CAMERA_ID = cam_id
            logger.info(f"成功打开摄像头（ID={CAMERA_ID}）")
            break
    if not cap or not cap.isOpened():
        logger.error("无法打开任何摄像头，请检查连接或是否被其他程序占用！")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 初始化时间控制变量
    last_send_time = time.time()
    frame_count = 0
    fps_start_time = time.time()

    logger.info("开始实时检测（按q键退出）...")
    logger.info(f"摄像头分辨率：{CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    logger.info(f"发送间隔：{SEND_INTERVAL}秒")
    logger.info(f"检测置信度阈值：{FLAGS.draw_threshold}")

    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                logger.warning("读取摄像头帧失败，跳过...")
                time.sleep(0.01)
                continue

            # 帧计数与帧率计算
            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count / (time.time() - fps_start_time)
                logger.info(f"实时帧率：{fps:.1f} FPS")
                frame_count = 0
                fps_start_time = time.time()

            # 核心逻辑：将摄像头帧临时保存为图片文件（Trainer只认文件路径）
            cv2.imwrite(TEMP_IMG_PATH, frame)
            # 构造文件路径列表，传给trainer.predict（适配Trainer的输入要求）
            image_paths = [TEMP_IMG_PATH]
            
            # 执行检测（关闭可视化和保存，提升速度）
            detection_result = trainer.predict(
                image_paths,  # 传入文件路径而非numpy数组
                draw_threshold=FLAGS.draw_threshold,
                output_dir=FLAGS.output_dir,
                save_results=False,
                visualize=False,
                do_eval=False)[0]

            # 处理检测结果
            valid_info = process_detection_result(frame, detection_result, FLAGS, label_list)

            # ========== 绘制检测框和标签 ==========
            if valid_info is not None:
                # 绘制红绿灯检测框（红色）
                for light in valid_info["traffic_lights"]:
                    bbox = light["bbox"]
                    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                    # 画框：红色框，线宽2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # 写标签：颜色+置信度
                    label = f"{light['color']} {light['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 绘制路牌检测框（蓝色）
                for sign in valid_info["road_signs"]:
                    bbox = sign["bbox"]
                    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                    # 画框：蓝色框，线宽2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # 写标签：OCR文本+置信度
                    label = f"{sign['ocr_text']} {sign['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 控制发送频率
            current_time = time.time()
            if current_time - last_send_time >= SEND_INTERVAL:
                send_recognition_info(valid_info)
                last_send_time = current_time

            # 显示画面+退出逻辑
            cv2.imshow('Real-time Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("用户按下q键，退出检测...")
                break

    except KeyboardInterrupt:
        logger.info("检测被中断，退出...")
    finally:
        # 释放资源+删除临时文件
        cap.release()
        cv2.destroyAllWindows()
        if os.path.exists(TEMP_IMG_PATH):
            os.remove(TEMP_IMG_PATH)  # 清理临时图片
        logger.info("摄像头资源已释放，临时文件已清理，检测结束")

def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # 设置设备
    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    if cfg.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
        logger.warning("未启用GPU，将使用CPU推理（速度较慢）")

    # 模型压缩相关
    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    # 检查配置
    check_config(cfg)
    check_version()

    # 启动实时检测
    run_realtime_detection(FLAGS, cfg)

if __name__ == '__main__':
    main()