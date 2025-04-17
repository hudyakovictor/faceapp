# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
import time
import json

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark


def param2alpha_shp(param):
    # предполагается param — [R, offset, alpha_shp, alpha_exp]
    param = param.squeeze()
    if param.ndim == 1:
        param = param[np.newaxis, :]
    alpha_shp = param[:, 12:52]  # параметры формы (40 значений по умолчанию)
    return alpha_shp[0]

def calculate_face_metrics(ver, ref_vector=None):
    metrics = {}

    # Симметрия
    metrics["eye_symmetry_x"] = np.abs(ver[36][0] - ver[45][0])
    metrics["eye_symmetry_y"] = np.abs(ver[37][1] - ver[46][1])

    # Геометрия глаз
    metrics["interocular_distance"] = np.linalg.norm(ver[36] - ver[45])
    metrics["eye_width_left"] = np.linalg.norm(ver[42] - ver[45])
    metrics["eye_width_right"] = np.linalg.norm(ver[36] - ver[39])
    metrics["eye_height_left"] = np.linalg.norm(ver[43] - ver[47])
    metrics["eye_height_right"] = np.linalg.norm(ver[37] - ver[41])

    # Ширина лица по скулам
    metrics["cheekbone_width"] = np.linalg.norm(ver[2] - ver[14])

    # Длина носа
    metrics["nose_length"] = np.linalg.norm(ver[27] - ver[30])

    # Высота челюсти
    metrics["jaw_height"] = np.linalg.norm(ver[5] - ver[33])

    # Длина лица
    metrics["face_length"] = np.linalg.norm(ver[8] - ver[30])

    # Shape Error
    metrics["shape_error"] = int(np.linalg.norm(param2alpha_shp(ver) - ref_vector)) if ref_vector is not None else 0

    # Стандартное отклонение shape_vector
    metrics["shape_std"] = int(np.std(param2alpha_shp(ver)))

    # Delta от эталонного shape_vector
    metrics["shape_delta"] = int(np.linalg.norm(param2alpha_shp(ver) - ref_vector)) if ref_vector is not None else 0

    # Объём треугольника "лоб-нос-подбородок"
    metrics["triangle_volume"] = 0.5 * np.abs(np.linalg.det(np.vstack((ver[0], ver[27], ver[8]))))

    # Угол между скулами и подбородком (V-образность)
    metrics["cheek_angle"] = np.degrees(np.arccos(np.dot(ver[2] - ver[14], ver[8] - ver[2]) / (np.linalg.norm(ver[2] - ver[14]) * np.linalg.norm(ver[8] - ver[2]))))

    # Соотношение расстояния глаз - нос - рот
    metrics["eye_nose_mouth_ratio"] = metrics["interocular_distance"] / metrics["nose_length"]

    return metrics


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    reader = imageio.get_reader("<video0>")

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    if 'metrics_list' not in globals():
        metrics_list = []

    # run
    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)
            if len(boxes) == 0:
                print("❌ Лицо не обнаружено на первом кадре.")
                break
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                if len(boxes) == 0:
                    print("❌ Лицо потеряно при отслеживании")
                    continue
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # Получаем метрики
        metrics = calculate_face_metrics(ver, ref_vector)

        # Сохраняем метрики в JSON
        metrics_list.append(metrics)

        # Выводим метрики в консоль или лог
        print(metrics)

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)

            img_draw = frame_bgr.copy()

            img_draw = cv_draw_landmark(img_draw, ver_ave, size=3)

            shape_vector = param2alpha_shp(param_lst[0])
            if 'last_check' not in globals():
                last_check = time.time()
                shape_snapshot = shape_vector
            else:
                if time.time() - last_check > 20:
                    shape_snapshot = shape_vector.copy()
                    last_check = time.time()

            shape_error = int(np.linalg.norm(shape_snapshot))
            shape_max = int(np.max(np.abs(shape_snapshot)))
            shape_mean = int(np.mean(np.abs(shape_snapshot)))
            shape_std = int(np.std(shape_snapshot))

            x0, y0, x1, y1 = map(int, roi_box)
            color = (0, 255, 0) if shape_error < 50000 else (0, 255, 255) if shape_error < 100000 else (0, 0, 255)
            debug_text = [
                f'shape_error: {shape_error}',
                f'max: {shape_max}',
                f'mean: {shape_mean}',
                f'std: {shape_std}',
            ]
            if delta is not None:
                debug_text.insert(0, f'delta: {delta}')

            for idx, line in enumerate(debug_text):
                cv2.putText(img_draw, line, (x0, max(y0 - 10 - idx*20, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.rectangle(img_draw, (x0, y0), (x1, y1), color, 2)

            cv2.imshow('image', img_draw)

            k = cv2.waitKey(20)
            if (k & 0xff == ord('q')):
                break
            if (k & 0xff == ord('s')):
                ref_vector = shape_vector.copy()
                print("✅ Эталон сохранён.")

            queue_ver.popleft()
            queue_frame.popleft()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
