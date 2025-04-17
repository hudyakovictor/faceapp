"""
Основной скрипт для обработки лиц (аналог demo2.py)
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import face_alignment
import mediapipe
from skimage import io
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render, render_color
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose, calc_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import _parse_param, similar_transform, str2bool

def main():
    parser = argparse.ArgumentParser(description='Face analysis system')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input image path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory or format')
    args = parser.parse_args()
    
    # Инициализация TDDFA и FAN точно так же, как в demo2.py
    tddfa = TDDFA()
    face_alignment = FaceAlignment()
    
    # Загрузка изображения
    img = cv2.imread(args.file)
    if img is None:
        print(f'Cannot read image: {args.file}')
        return
    
    # Обработка с помощью TDDFA
    param_lst, roi_box_lst = tddfa.detect_and_reconstruct(img)
    
    # Получение landmarks от TDDFA
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    
    # Получение углов головы
    pose = calc_pose(ver_lst[0])
    
    # Обработка с помощью FAN
    fan_landmarks = face_alignment.get_landmarks(img)
    
    # Вывод результатов
    print(f"Processing completed for {args.file}")
    print(f"Output format: {args.output}")
    print(f"Pose: {pose}")
    
    # Визуализация результатов (если нужно)
    if args.output:
        # Рисуем landmarks на изображении
        img_drawn = draw_landmarks(img, ver_lst, show_flag=False)
        
        # Сохраняем результат
        cv2.imwrite(f"{args.output}_result.jpg", img_drawn)
        print(f"Result saved to {args.output}_result.jpg")

if __name__ == "__main__":
    main()
