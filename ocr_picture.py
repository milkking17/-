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
import numpy as np  # 新增：用于数组计算

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
from ppdet.engine import Trainer, Trainer_ARSL
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_gcu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')

# ======================== 新增：OCR相关导入 ========================
import cv2
from paddleocr import PaddleOCR
# ==================================================================

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_list",
        type=str,
        default=None,
        help="The file path containing path of image to be infered. Valid only when --infer_dir is given."
    )
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--save_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for saving.")
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
    parser.add_argument(
        "--do_eval",
        type=ast.literal_eval,
        default=False,
        help="Whether to eval after infer.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether to save inference results to output_dir.")
    parser.add_argument(
        "--slice_infer",
        action='store_true',
        help="Whether to slice the image and merge the inference results for small object detection."
    )
    parser.add_argument(
        '--slice_size',
        nargs='+',
        type=int,
        default=[640, 640],
        help="Height of the sliced image.")
    parser.add_argument(
        "--overlap_ratio",
        nargs='+',
        type=float,
        default=[0.25, 0.25],
        help="Overlap height ratio of the sliced image.")
    parser.add_argument(
        "--combine_method",
        type=str,
        default='nms',
        help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.6,
        help="Combine method matching threshold.")
    parser.add_argument(
        "--match_metric",
        type=str,
        default='ios',
        help="Combine method matching metric, choose in ['iou', 'ios'].")
    parser.add_argument(
        "--visualize",
        type=ast.literal_eval,
        default=True,
        help="Whether to save visualize results to output_dir.")
    parser.add_argument(
        "--rtn_im_file",
        type=bool,
        default=False,
        help="Whether to return image file path in Dataloader.")
    # ======================== 新增：OCR相关参数 ========================
    parser.add_argument(
        "--enable_ocr",
        action='store_true',
        default=False,
        help="Whether to enable OCR text recognition for detected bboxes.")
    parser.add_argument(
        "--ocr_lang",
        type=str,
        default='en',
        help="OCR language, 'en' for English, 'ch' for Chinese.")
    parser.add_argument(
        "--crop_dir",
        type=str,
        default="output/crops",
        help="Directory for storing cropped images by detected bboxes.")
    parser.add_argument(
        "--label_list_path",
        type=str,
        default="dataset/roadsign_voc/label_list.txt",
        help="Path to label list file for category name mapping.")
    # ==================================================================
    args = parser.parse_args()
    return args

# ======================== 新增：红绿灯颜色识别函数 ========================
def detect_traffic_light_color(traffic_light_img):
    """
    识别红绿灯裁剪图的颜色（红/黄/绿）
    :param traffic_light_img: 裁剪后的红绿灯图像（cv2格式）
    :return: 识别结果（red/yellow/green/unknown）、各颜色像素面积
    """
    # 步骤1：预处理（降噪+聚焦灯体区域）
    # 1.1 高斯模糊降噪
    blurred = cv2.GaussianBlur(traffic_light_img, (5, 5), 0)
    # 1.2 转换到HSV空间
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 1.3 提取中心ROI（排除边框，取70%区域）
    h, w = hsv.shape[:2]
    roi_x1 = int(w * 0.15)
    roi_y1 = int(h * 0.15)
    roi_x2 = int(w * 0.85)
    roi_y2 = int(h * 0.85)
    hsv_roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]

    # 步骤2：定义HSV颜色阈值（可根据实际场景微调）
    # 红色（两个区间，因为Hue在0-180中红色跨0和179）
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv_roi, lower_red1, upper_red1) + cv2.inRange(hsv_roi, lower_red2, upper_red2)
    
    # 黄色
    lower_yellow = np.array([20, 120, 70])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    
    # 绿色
    lower_green = np.array([40, 120, 70])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)

    # 步骤3：计算各颜色掩码的有效像素面积
    area_red = cv2.countNonZero(mask_red)
    area_yellow = cv2.countNonZero(mask_yellow)
    area_green = cv2.countNonZero(mask_green)

    # 步骤4：判断最大面积的颜色
    color_areas = {
        'red': area_red,
        'yellow': area_yellow,
        'green': area_green
    }
    max_color = max(color_areas, key=color_areas.get)
    # 过滤小面积噪声（如果最大面积<100，判定为未知）
    if color_areas[max_color] < 100:
        max_color = 'unknown'

    return max_color, color_areas

# ======================== 裁剪图片函数 ========================
def crop_image_by_bbox(img, bbox, save_path=None):
    """
    Crop image by detection bbox, avoid out of bounds
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    crop_img = img[y1:y2, x1:x2]
    if save_path and crop_img.size > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, crop_img)
    return crop_img
# ==================================================================

def get_test_images(infer_dir, infer_img, infer_list=None):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    if infer_list:
        assert os.path.isfile(
            infer_list), f"infer_list {infer_list} is not a valid file path."
        with open(infer_list, 'r') as f:
            lines = f.readlines()
        for line in lines:
            images.update([os.path.join(infer_dir, line.strip())])
    else:
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images

# ======================== OCR处理函数（新增红绿灯颜色识别） ========================
def process_ocr(FLAGS, images, detection_results):
    """
    核心逻辑：
    - 路牌（ID=0）：裁剪+OCR
    - 红绿灯（ID=2）：裁剪+保存+颜色识别
    """
    # Initialize PaddleOCR
    ocr = PaddleOCR(lang=FLAGS.ocr_lang, use_angle_cls=True)
    
    # Load label list
    label_list = []
    if os.path.exists(FLAGS.label_list_path):
        with open(FLAGS.label_list_path, 'r') as f:
            label_list = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"加载类别列表：{label_list} | 路牌ID=0，红绿灯ID=2")
    else:
        logger.warning(f"Label list file {FLAGS.label_list_path} not found，仅显示类别ID")

    # 固定类别ID
    ROAD_SIGN_ID = 0    # 路牌：第一个类别
    TRAFFIC_LIGHT_ID = 2 # 红绿灯：第三个类别

    # Process each image
    for img_path, res in zip(images, detection_results):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to read image {img_path}, skip OCR.")
            continue
        
        img_name = os.path.basename(img_path)[:-4]
        if 'bbox' in res and len(res['bbox']) > 0:
            for bbox_idx, bbox_info in enumerate(res['bbox']):
                cls_id, score, x1, y1, x2, y2 = bbox_info
                cls_id = int(cls_id)
                if score < FLAGS.draw_threshold:
                    continue
                
                cls_name = label_list[cls_id] if (label_list and cls_id < len(label_list)) else f"class_{cls_id}"
                
                # ======================== 红绿灯（ID=2）：裁剪+保存+颜色识别 ========================
                if cls_id == TRAFFIC_LIGHT_ID:
                    crop_path = os.path.join(FLAGS.crop_dir, f"{img_name}_trafficlight_{bbox_idx}.png")
                    crop_img = crop_image_by_bbox(img, [x1, y1, x2, y2], crop_path)
                    # 颜色识别
                    if crop_img.size > 0:
                        light_color, color_areas = detect_traffic_light_color(crop_img)
                        # 日志输出
                        logger.info(f"\n===== 检测到红绿灯 (Bbox {bbox_idx+1}) =====")
                        logger.info(f"类别：{cls_name} (ID={cls_id}), 置信度：{score:.2f}")
                        logger.info(f"坐标：x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                        logger.info(f"裁剪图保存至：{crop_path}")
                        logger.info(f"红绿灯颜色识别结果：{light_color.upper()}")
                        logger.info(f"各颜色像素面积：红={color_areas['red']}, 黄={color_areas['yellow']}, 绿={color_areas['green']}")
                    else:
                        logger.warning(f"红绿灯裁剪图为空，跳过颜色识别")
                    logger.info("-" * 50)
                
                # ======================== 路牌（ID=0）：裁剪+OCR ========================
                elif cls_id == ROAD_SIGN_ID:
                    crop_img = crop_image_by_bbox(img, [x1, y1, x2, y2])
                    ocr_text = ""
                    if crop_img.size > 0:
                        ocr_result = ocr.ocr(crop_img)
                        if ocr_result and ocr_result[0]:
                            ocr_text = ' '.join([line[1][0] for line in ocr_result[0]])
                    # 日志输出
                    logger.info(f"\n===== 检测到路牌 (Bbox {bbox_idx+1}) =====")
                    logger.info(f"类别：{cls_name} (ID={cls_id}), 置信度：{score:.2f}")
                    logger.info(f"坐标：x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                    logger.info(f"OCR识别结果：{ocr_text}")
                    logger.info("-" * 50)
                
                # 其他类别：无操作
                else:
                    continue

# ==================================================================

def run(FLAGS, cfg):
    if FLAGS.rtn_im_file:
        cfg['TestReader']['sample_transforms'][0]['Decode'][
            'rtn_im_file'] = FLAGS.rtn_im_file
    ssod_method = cfg.get('ssod_method', None)
    if ssod_method == 'ARSL':
        trainer = Trainer_ARSL(cfg, mode='test')
        trainer.load_weights(cfg.weights, ARSL_eval=True)
    else:
        trainer = Trainer(cfg, mode='test')
        trainer.load_weights(cfg.weights)
    
    if FLAGS.do_eval:
        dataset = create('TestDataset')()
        images = dataset.get_images()
    else:
        images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img, FLAGS.infer_list)

    detection_results = None
    if FLAGS.slice_infer:
        trainer.slice_predict(
            images,
            slice_size=FLAGS.slice_size,
            overlap_ratio=FLAGS.overlap_ratio,
            combine_method=FLAGS.combine_method,
            match_threshold=FLAGS.match_threshold,
            match_metric=FLAGS.match_metric,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize)
    else:
        detection_results = trainer.predict(
            images,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize,
            save_threshold=FLAGS.save_threshold,
            do_eval=FLAGS.do_eval)
    
    if FLAGS.enable_ocr and detection_results is not None:
        process_ocr(FLAGS, images, detection_results)

def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    if 'use_npu' not in cfg:
        cfg.use_npu = False
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False
    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False
    if 'use_gcu' not in cfg:
        cfg.use_gcu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    elif cfg.use_mlu:
        place = paddle.set_device('mlu')
    elif cfg.use_gcu:
        place = paddle.set_device('gcu')
    else:
        place = paddle.set_device('cpu')

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)
    check_gcu(cfg.use_gcu)
    check_version()
    run(FLAGS, cfg)


if __name__ == '__main__':
    main()