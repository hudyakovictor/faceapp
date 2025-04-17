import sys
import argparse
import cv2
import mediapipe as mp
import yaml
import json
import numpy as np
from scipy.spatial.transform import Rotation
from collections import OrderedDict
from utils.pose import calc_pose
from utils.pose import P2sRt
import mediapipe as mp
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool
import shutil
import insightface
from insightface.app import FaceAnalysis
import face_alignment
from skimage import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # отключает логи TensorFlow
os.environ['KMP_WARNINGS'] = '0'          # отключает предупреждения OpenMP
os.environ['OMP_NUM_THREADS'] = '1'       # ограничивает потоки (если нужно)
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)  # подавляет TF-логи
import warnings
warnings.filterwarnings('ignore')  # подавляет Python warnings
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextlib.contextmanager
def suppress_stderr():
    """
    Контекстный менеджер для подавления вывода в stderr
    """
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr






 



#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------




# Функция для добавления метрик в зависимости от ракурса
def add_metrics(pose_type, param, landmarks, mp_landmarks, delta_pose, img=None, roi_box=None):
    if landmarks is None or len(landmarks) == 0:
        print("⚠️ landmarks пустой — пропускаем расчёт метрик")
        return {}
    metrics = {}
    # Утилитарная функция для вычисления угла между тремя точками
    def calculate_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Избегаем ошибок округления
        return float(np.degrees(np.arccos(cos_theta)))
    # Утилитарная функция для вычисления симметрии между двумя массивами точек
    def symmetry_score(points1, points2):
        if len(points1) != len(points2):
            return 0.0
        diffs = np.linalg.norm(points1 - points2, axis=1)
        return float(np.mean(diffs))



    # Метрики для фронтального ракурса (старые на основе 3DDFA)
    if pose_type == "frontal":
        try:
            metrics["face_asymmetry"] = float(np.abs(np.linalg.norm(landmarks[0] - landmarks[8]) - np.linalg.norm(landmarks[16] - landmarks[8])))
            metrics["eye_angle"] = float(np.degrees(np.arctan2(landmarks[45][1] - landmarks[36][1], landmarks[45][0] - landmarks[36][0])))
            left_eye_center = np.mean(landmarks[36:42], axis=0)
            right_eye_center = np.mean(landmarks[42:48], axis=0)
            metrics["eye_distance"] = float(np.linalg.norm(left_eye_center - right_eye_center))
            metrics["eye_socket_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["eye_asymmetry"] = float(np.abs(np.linalg.norm(landmarks[36] - landmarks[39]) - np.linalg.norm(landmarks[42] - landmarks[45])))
            metrics["face_width"] = float(np.linalg.norm(landmarks[0] - landmarks[16]))
            metrics["nose_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["cheek_width"] = float(np.linalg.norm(landmarks[2] - landmarks[14]))
            metrics["mouth_width"] = float(np.linalg.norm(landmarks[48] - landmarks[54]))
            metrics["nose_width"] = float(np.linalg.norm(landmarks[31] - landmarks[35]))
            metrics["brow_height"] = float(np.linalg.norm(landmarks[19] - landmarks[36]))
            metrics["brow_angle"] = float(np.degrees(np.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            metrics["chin_height"] = float(np.linalg.norm(landmarks[8] - landmarks[57]))
            metrics["jaw_asymmetry"] = float(np.abs(np.linalg.norm(landmarks[4] - landmarks[8]) - np.linalg.norm(landmarks[12] - landmarks[8])))
            metrics["forehead_width"] = float(np.linalg.norm(landmarks[0] - landmarks[16]))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых фронтальных метрик: {e}")



        # Новые метрики FAN для фронтального ракурса
        try:
            metrics["fn_inter_pupil_distance"] = float(np.linalg.norm(landmarks[42] - landmarks[39]))
            metrics["fn_left_eye_openness"] = float(np.linalg.norm(landmarks[37] - landmarks[41]))
            metrics["fn_right_eye_openness"] = float(np.linalg.norm(landmarks[44] - landmarks[48]))
            metrics["fn_eye_symmetry_score"] = symmetry_score(landmarks[36:42], landmarks[42:48])
            metrics["fn_left_eye_corner_angle"] = calculate_angle(landmarks[36], landmarks[39], landmarks[37])
            metrics["fn_right_eye_corner_angle"] = calculate_angle(landmarks[42], landmarks[45], landmarks[44])
            left_eye_bulge = np.mean([np.linalg.norm(landmarks[i] - landmarks[39]) for i in [37, 38, 40, 41]])
            right_eye_bulge = np.mean([np.linalg.norm(landmarks[i] - landmarks[45]) for i in [43, 44, 46, 47]])
            metrics["fn_eye_bulge_score"] = float(left_eye_bulge + right_eye_bulge) / 2
            metrics["fn_left_eyebrow_height"] = float(np.linalg.norm(landmarks[38] - landmarks[20]))
            metrics["fn_right_eyebrow_height"] = float(np.linalg.norm(landmarks[43] - landmarks[25]))
            metrics["fn_eyebrow_symmetry_score"] = symmetry_score(landmarks[17:22], landmarks[22:27])
            metrics["fn_left_eyebrow_slope"] = float(np.degrees(np.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            metrics["fn_right_eyebrow_slope"] = float(np.degrees(np.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_nose_length"] = float(np.linalg.norm(landmarks[27] - landmarks[30]))
            metrics["fn_nose_width"] = float(np.linalg.norm(landmarks[31] - landmarks[35]))
            metrics["fn_nostril_asymmetry_score"] = float(np.abs(landmarks[31][1] - landmarks[35][1]))
            metrics["fn_nose_angle"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_nose_tip_alignment"] = float(np.abs(landmarks[32][0] - landmarks[34][0]))
            metrics["fn_mouth_width"] = float(np.linalg.norm(landmarks[48] - landmarks[54]))
            metrics["fn_mouth_height"] = float(np.linalg.norm(landmarks[51] - landmarks[57]))
            metrics["fn_lip_thickness_upper"] = float(np.linalg.norm(landmarks[50] - landmarks[52]))
            metrics["fn_lip_thickness_lower"] = float(np.linalg.norm(landmarks[58] - landmarks[56]))
            metrics["fn_mouth_corner_angle"] = calculate_angle(landmarks[48], landmarks[51], landmarks[54])
            metrics["fn_mouth_asymmetry_score"] = float(np.abs(landmarks[48][1] - landmarks[54][1]))
            metrics["fn_jaw_width"] = float(np.linalg.norm(landmarks[5] - landmarks[13]))
            metrics["fn_chin_height"] = float(np.linalg.norm(landmarks[57] - landmarks[8]))
            # Глобальная симметрия лица
            left_half = landmarks[1:17]
            right_half = landmarks[16:0:-1]
            metrics["fn_face_symmetry_score"] = symmetry_score(left_half, right_half)
            # Золотое сечение (примерное)
            eye_to_nose = np.linalg.norm(landmarks[36] - landmarks[30])
            nose_to_lip = np.linalg.norm(landmarks[30] - landmarks[51])
            lip_to_chin = np.linalg.norm(landmarks[51] - landmarks[8])
            metrics["fn_golden_ratio_score"] = float(abs(1.618 - (eye_to_nose + nose_to_lip) / lip_to_chin))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых фронтальных метрик FAN: {e}")



    # Метрики для левого профиля (старые на основе 3DDFA)
    elif pose_type == "profile_left":
        try:
            metrics["left_chin_depth"] = float(np.linalg.norm(landmarks[4] - landmarks[6]))
            metrics["left_nose_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["left_nose_width"] = float(np.linalg.norm(landmarks[31] - landmarks[30]))
            metrics["left_chin_height"] = float(np.linalg.norm(landmarks[8] - landmarks[57]))
            metrics["left_cheek_width"] = float(np.linalg.norm(landmarks[2] - landmarks[4]))
            metrics["left_jaw_width"] = float(np.linalg.norm(landmarks[0] - landmarks[4]))
            metrics["left_eye_distance"] = float(np.linalg.norm(np.mean(landmarks[36:42], axis=0) - np.mean(landmarks[42:48], axis=0)))
            metrics["left_brow_height"] = float(np.linalg.norm(landmarks[19] - landmarks[36]))
            metrics["left_eye_socket_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["left_nose_angle"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["left_jaw_angle"] = float(np.degrees(np.arctan2(landmarks[8][1] - landmarks[4][1], landmarks[8][0] - landmarks[4][0])))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых метрик profile_left: {e}")



        # Новые метрики FAN для левого профиля
        try:
            metrics["fn_left_eye_openness"] = float(np.linalg.norm(landmarks[37] - landmarks[41]))
            metrics["fn_left_eye_corner_angle"] = calculate_angle(landmarks[36], landmarks[39], landmarks[37])
            left_eye_bulge = np.mean([np.linalg.norm(landmarks[i] - landmarks[39]) for i in [37, 38, 40, 41]])
            metrics["fn_left_eye_bulge_score"] = float(left_eye_bulge)
            metrics["fn_left_eyebrow_height"] = float(np.linalg.norm(landmarks[38] - landmarks[20]))
            metrics["fn_left_eyebrow_slope"] = float(np.degrees(np.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            metrics["fn_nose_projection_left"] = float(np.linalg.norm(landmarks[30] - landmarks[27]))
            metrics["fn_nostril_visibility_left"] = float(np.abs(landmarks[31][2] - landmarks[30][2]))
            metrics["fn_nose_angle_left"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_mouth_corner_angle_left"] = calculate_angle(landmarks[48], landmarks[51], landmarks[54])
            metrics["fn_left_cheek_offset"] = float(np.linalg.norm(landmarks[3] - landmarks[30]))
            metrics["fn_jaw_angle_left"] = calculate_angle(landmarks[3], landmarks[5], landmarks[6])
            jaw_curve_left = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 5)])
            metrics["fn_jaw_curve_left"] = float(jaw_curve_left)
            metrics["fn_chin_offset_left"] = float(np.abs(landmarks[8][0] - landmarks[30][0]))
            metrics["fn_mouth_asymmetry_left"] = float(np.abs(landmarks[48][1] - landmarks[51][1]))
            contour_density_left = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 7)])
            metrics["fn_face_contour_visibility_left"] = float(contour_density_left)
            metrics["fn_left_nose_projection"] = float(np.linalg.norm(landmarks[27] - landmarks[30]))
            metrics["fn_left_forehead_nose_angle"] = calculate_angle(landmarks[19], landmarks[27], landmarks[30])
            lip_profile_left = np.mean([np.linalg.norm(landmarks[i] - landmarks[51]) for i in [48, 54, 57]])
            metrics["fn_left_lip_profile"] = float(lip_profile_left)
            metrics["fn_left_mouth_corner_angle"] = calculate_angle(landmarks[48], landmarks[51], landmarks[54])
            metrics["fn_left_jaw_angle"] = calculate_angle(landmarks[3], landmarks[5], landmarks[6])
            metrics["fn_left_jaw_curve"] = float(jaw_curve_left)
            metrics["fn_left_chin_shape"] = float(np.linalg.norm(landmarks[8] - landmarks[57]))
            cheek_curve_left = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in [3, 4, 5]])
            metrics["fn_left_cheek_curve"] = float(cheek_curve_left)
            eye_slit_left = np.linalg.norm(landmarks[39] - landmarks[41])
            metrics["fn_left_eye_profile_slit"] = float(eye_slit_left)
            metrics["fn_left_eyebrow_profile_angle"] = float(np.degrees(np.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            metrics["fn_left_nostril_contour_visibility"] = float(np.abs(landmarks[31][2] - landmarks[30][2]))
            face_curve_left = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 7)])
            metrics["fn_left_face_curve_profile"] = float(face_curve_left)
            metrics["fn_left_ear_to_nose_ratio"] = float(np.linalg.norm(landmarks[0] - landmarks[30]))
            metrics["fn_left_face_silhouette_density"] = float(contour_density_left)
            metrics["fn_left_face_angle_score"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[0][1], landmarks[30][0] - landmarks[0][0])))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых метрик FAN для profile_left: {e}")



    # Метрики для правого профиля (старые на основе 3DDFA)
    elif pose_type == "profile_right":
        try:
            metrics["right_chin_depth"] = float(np.linalg.norm(landmarks[10] - landmarks[12]))
            metrics["right_chin_height"] = float(np.linalg.norm(landmarks[8] - landmarks[57]))
            metrics["right_nose_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["right_nose_width"] = float(np.linalg.norm(landmarks[35] - landmarks[30]))
            metrics["right_nose_angle"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["right_cheek_width"] = float(np.linalg.norm(landmarks[12] - landmarks[14]))
            metrics["right_jaw_width"] = float(np.linalg.norm(landmarks[12] - landmarks[16]))
            metrics["right_eye_distance"] = float(np.linalg.norm(np.mean(landmarks[36:42], axis=0) - np.mean(landmarks[42:48], axis=0)))
            metrics["right_jaw_angle"] = float(np.degrees(np.arctan2(landmarks[8][1] - landmarks[12][1], landmarks[8][0] - landmarks[12][0])))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых метрик profile_right: {e}")



        # Новые метрики FAN для правого профиля
        try:
            metrics["fn_right_eye_openness"] = float(np.linalg.norm(landmarks[44] - landmarks[48]))
            metrics["fn_right_eye_corner_angle"] = calculate_angle(landmarks[42], landmarks[45], landmarks[44])
            right_eye_bulge = np.mean([np.linalg.norm(landmarks[i] - landmarks[45]) for i in [43, 44, 46, 47]])
            metrics["fn_right_eye_bulge_score"] = float(right_eye_bulge)
            metrics["fn_right_eyebrow_height"] = float(np.linalg.norm(landmarks[43] - landmarks[25]))
            metrics["fn_right_eyebrow_slope"] = float(np.degrees(np.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_nose_projection_right"] = float(np.linalg.norm(landmarks[30] - landmarks[27]))
            metrics["fn_nostril_visibility_right"] = float(np.abs(landmarks[35][2] - landmarks[30][2]))
            metrics["fn_nose_angle_right"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_mouth_corner_angle_right"] = calculate_angle(landmarks[51], landmarks[54], landmarks[48])
            metrics["fn_right_cheek_offset"] = float(np.linalg.norm(landmarks[13] - landmarks[30]))
            metrics["fn_jaw_angle_right"] = calculate_angle(landmarks[13], landmarks[15], landmarks[16])
            jaw_curve_right = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(10, 15)])
            metrics["fn_jaw_curve_right"] = float(jaw_curve_right)
            metrics["fn_chin_offset_right"] = float(np.abs(landmarks[8][0] - landmarks[30][0]))
            metrics["fn_mouth_asymmetry_right"] = float(np.abs(landmarks[54][1] - landmarks[51][1]))
            contour_density_right = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(10, 16)])
            metrics["fn_face_contour_visibility_right"] = float(contour_density_right)
            metrics["fn_right_nose_projection"] = float(np.linalg.norm(landmarks[27] - landmarks[30]))
            metrics["fn_right_forehead_nose_angle"] = calculate_angle(landmarks[24], landmarks[27], landmarks[30])
            lip_profile_right = np.mean([np.linalg.norm(landmarks[i] - landmarks[51]) for i in [48, 54, 57]])
            metrics["fn_right_lip_profile"] = float(lip_profile_right)
            metrics["fn_right_mouth_corner_angle"] = calculate_angle(landmarks[54], landmarks[51], landmarks[48])
            metrics["fn_right_jaw_angle"] = calculate_angle(landmarks[13], landmarks[15], landmarks[16])
            metrics["fn_right_jaw_curve"] = float(jaw_curve_right)
            metrics["fn_right_chin_shape"] = float(np.linalg.norm(landmarks[8] - landmarks[57]))
            cheek_curve_right = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in [13, 14, 15]])
            metrics["fn_right_cheek_curve"] = float(cheek_curve_right)
            eye_slit_right = np.linalg.norm(landmarks[45] - landmarks[47])
            metrics["fn_right_eye_profile_slit"] = float(eye_slit_right)
            metrics["fn_right_eyebrow_profile_angle"] = float(np.degrees(np.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_right_nostril_contour_visibility"] = float(np.abs(landmarks[35][2] - landmarks[30][2]))
            face_curve_right = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(8, 16)])
            metrics["fn_right_face_curve_profile"] = float(face_curve_right)
            metrics["fn_right_ear_to_nose_ratio"] = float(np.linalg.norm(landmarks[16] - landmarks[30]))
            metrics["fn_right_face_silhouette_density"] = float(contour_density_right)
            metrics["fn_right_face_angle_score"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[16][1], landmarks[30][0] - landmarks[16][0])))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых метрик FAN для profile_right: {e}")



    # Метрики для полулевого ракурса (старые на основе 3DDFA)
    elif pose_type == "semi_left":
        try:
            metrics["semi_left_nose_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["semi_left_nose_angle"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["semi_left_skull_width"] = float(np.linalg.norm(landmarks[0] - landmarks[4]))
            metrics["semi_left_cheek_width"] = float(np.linalg.norm(landmarks[2] - landmarks[4]))
            metrics["semi_left_jaw_width"] = float(np.linalg.norm(landmarks[0] - landmarks[4]))
            metrics["semi_left_jaw_angle"] = float(np.degrees(np.arctan2(landmarks[8][1] - landmarks[4][1], landmarks[8][0] - landmarks[4][0])))
            metrics["semi_left_chin_width"] = float(np.linalg.norm(landmarks[4] - landmarks[8]))
            metrics["semi_left_eye_socket_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых метрик semi_left: {e}")



        # Новые метрики FAN для полулевого ракурса
        try:
            metrics["fn_left_eye_openness"] = float(np.linalg.norm(landmarks[37] - landmarks[41]))
            metrics["fn_left_eye_corner_angle"] = calculate_angle(landmarks[36], landmarks[39], landmarks[37])
            left_eye_bulge = np.mean([np.linalg.norm(landmarks[i] - landmarks[39]) for i in [37, 38, 40, 41]])
            metrics["fn_left_eye_bulge_score"] = float(left_eye_bulge)
            metrics["fn_left_eyebrow_height"] = float(np.linalg.norm(landmarks[38] - landmarks[20]))
            metrics["fn_left_eyebrow_slope"] = float(np.degrees(np.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            metrics["fn_nose_projection_left"] = float(np.linalg.norm(landmarks[30] - landmarks[27]))
            metrics["fn_nostril_visibility_left"] = float(np.abs(landmarks[31][2] - landmarks[30][2]))
            metrics["fn_nose_angle_left"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_mouth_corner_angle_left"] = calculate_angle(landmarks[48], landmarks[51], landmarks[54])
            metrics["fn_left_cheek_offset"] = float(np.linalg.norm(landmarks[3] - landmarks[30]))
            metrics["fn_jaw_angle_left"] = calculate_angle(landmarks[3], landmarks[5], landmarks[6])
            jaw_curve_left = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 5)])
            metrics["fn_jaw_curve_left"] = float(jaw_curve_left)
            metrics["fn_chin_offset_left"] = float(np.abs(landmarks[8][0] - landmarks[30][0]))
            metrics["fn_mouth_asymmetry_left"] = float(np.abs(landmarks[48][1] - landmarks[51][1]))
            contour_density_left = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 7)])
            metrics["fn_face_contour_visibility_left"] = float(contour_density_left)
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых метрик FAN для semi_left: {e}")



    # Метрики для полуправого ракурса (старые на основе 3DDFA)
    elif pose_type == "semi_right":
        try:
            metrics["semi_right_nose_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["semi_right_nose_angle"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], 
                                                                        landmarks[30][0] - landmarks[27][0])))
            metrics["semi_right_skull_width"] = float(np.linalg.norm(landmarks[12] - landmarks[16]))
            metrics["semi_right_cheek_width"] = float(np.linalg.norm(landmarks[12] - landmarks[14]))
            metrics["semi_right_jaw_width"] = float(np.linalg.norm(landmarks[12] - landmarks[16]))
            metrics["semi_right_jaw_angle"] = float(np.degrees(np.arctan2(landmarks[8][1] - landmarks[12][1], 
                                                                         landmarks[8][0] - landmarks[12][0])))
            metrics["semi_right_chin_width"] = float(np.linalg.norm(landmarks[8] - landmarks[12]))
            metrics["semi_right_eye_socket_depth"] = float(np.linalg.norm(landmarks[27] - landmarks[33]))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых метрик semi_right: {e}")



        # Новые метрики FAN для полуправого ракурса
        try:
            metrics["fn_right_eye_openness"] = float(np.linalg.norm(landmarks[44] - landmarks[48]))
            metrics["fn_right_eye_corner_angle"] = calculate_angle(landmarks[42], landmarks[45], landmarks[44])
            right_eye_bulge = np.mean([np.linalg.norm(landmarks[i] - landmarks[45]) for i in [43, 44, 46, 47]])
            metrics["fn_right_eye_bulge_score"] = float(right_eye_bulge)
            metrics["fn_right_eyebrow_height"] = float(np.linalg.norm(landmarks[43] - landmarks[25]))
            metrics["fn_right_eyebrow_slope"] = float(np.degrees(np.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_nose_projection_right"] = float(np.linalg.norm(landmarks[30] - landmarks[27]))
            metrics["fn_nostril_visibility_right"] = float(np.abs(landmarks[35][2] - landmarks[30][2]))
            metrics["fn_nose_angle_right"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_mouth_corner_angle_right"] = calculate_angle(landmarks[51], landmarks[54], landmarks[48])
            metrics["fn_right_cheek_offset"] = float(np.linalg.norm(landmarks[13] - landmarks[30]))
            metrics["fn_jaw_angle_right"] = calculate_angle(landmarks[13], landmarks[15], landmarks[16])
            jaw_curve_right = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(10, 15)])
            metrics["fn_jaw_curve_right"] = float(jaw_curve_right)
            metrics["fn_chin_offset_right"] = float(np.abs(landmarks[8][0] - landmarks[30][0]))
            metrics["fn_mouth_asymmetry_right"] = float(np.abs(landmarks[54][1] - landmarks[51][1]))
            contour_density_right = np.mean([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(10, 16)])
            metrics["fn_face_contour_visibility_right"] = float(contour_density_right)
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых метрик FAN для semi_right: {e}")




    # Новые метрики на основе MediaPipe (если mp_landmarks доступен)
    if mp_landmarks is not None and len(mp_landmarks) >= 68:
        if pose_type == "frontal":
            try:
                left_eye_center = np.mean(mp_landmarks[36:42], axis=0)
                right_eye_center = np.mean(mp_landmarks[42:48], axis=0)
                metrics["MP_eye_distance"] = float(np.linalg.norm(left_eye_center - right_eye_center))
                eye_vector = right_eye_center - left_eye_center
                metrics["MP_eye_angle"] = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
                left_eye_width = np.linalg.norm(mp_landmarks[36] - mp_landmarks[39])
                right_eye_width = np.linalg.norm(mp_landmarks[42] - mp_landmarks[45])
                metrics["MP_eye_asymmetry"] = float(np.abs(left_eye_width - right_eye_width))
                metrics["MP_eye_socket_depth"] = float(np.abs(mp_landmarks[36][2] - mp_landmarks[27][2]))
                metrics["MP_face_width"] = float(np.linalg.norm(mp_landmarks[0] - mp_landmarks[16]))
                metrics["MP_brow_height"] = float(np.linalg.norm(mp_landmarks[19] - mp_landmarks[36]))
                brow_vector = mp_landmarks[21] - mp_landmarks[17]
                metrics["MP_brow_angle"] = float(np.degrees(np.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_forehead_width"] = float(np.linalg.norm(mp_landmarks[0] - mp_landmarks[16]))
                metrics["MP_nose_width"] = float(np.linalg.norm(mp_landmarks[31] - mp_landmarks[35]))
                metrics["MP_nose_depth"] = float(np.abs(mp_landmarks[30][2] - mp_landmarks[27][2]))
                metrics["MP_cheek_width"] = float(np.linalg.norm(mp_landmarks[2] - mp_landmarks[14]))
                metrics["MP_mouth_width"] = float(np.linalg.norm(mp_landmarks[48] - mp_landmarks[54]))
                metrics["MP_chin_height"] = float(np.linalg.norm(mp_landmarks[8] - mp_landmarks[57]))
                metrics["MP_jaw_asymmetry"] = float(np.abs(np.linalg.norm(mp_landmarks[4] - mp_landmarks[8]) - np.linalg.norm(mp_landmarks[12] - mp_landmarks[8])))
                metrics["MP_face_asymmetry"] = float(np.abs(np.linalg.norm(mp_landmarks[0] - mp_landmarks[8]) - np.linalg.norm(mp_landmarks[16] - mp_landmarks[8])))
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для frontal: {e}")



        elif pose_type == "semi_left":
            try:
                metrics["MP_left_forehead_width"] = float(np.linalg.norm(mp_landmarks[0] - mp_landmarks[4]))
                metrics["MP_left_nose_width"] = float(np.linalg.norm(mp_landmarks[31] - mp_landmarks[30]))
                metrics["MP_left_nose_depth"] = float(np.abs(mp_landmarks[30][2] - mp_landmarks[27][2]))
                jaw_vector = mp_landmarks[8] - mp_landmarks[4]
                metrics["MP_left_jaw_angle"] = float(np.degrees(np.arctan2(jaw_vector[1], jaw_vector[0])))
                metrics["MP_left_jaw_width"] = float(np.linalg.norm(mp_landmarks[0] - mp_landmarks[4]))
                metrics["MP_left_cheek_width"] = float(np.linalg.norm(mp_landmarks[2] - mp_landmarks[4]))
                brow_vector = mp_landmarks[21] - mp_landmarks[17]
                metrics["MP_left_brow_angle"] = float(np.degrees(np.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_left_face_width"] = float(np.linalg.norm(mp_landmarks[0] - mp_landmarks[8]))
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для semi_left: {e}")



        elif pose_type == "semi_right":
            try:
                metrics["MP_right_forehead_width"] = float(np.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_nose_width"] = float(np.linalg.norm(mp_landmarks[30] - mp_landmarks[35]))
                metrics["MP_right_nose_depth"] = float(np.abs(mp_landmarks[30][2] - mp_landmarks[27][2]))
                jaw_vector = mp_landmarks[8] - mp_landmarks[12]
                metrics["MP_right_jaw_angle"] = float(np.degrees(np.arctan2(jaw_vector[1], jaw_vector[0])))
                metrics["MP_right_jaw_width"] = float(np.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_cheek_width"] = float(np.linalg.norm(mp_landmarks[12] - mp_landmarks[14]))
                brow_vector = mp_landmarks[26] - mp_landmarks[22]
                metrics["MP_right_brow_angle"] = float(np.degrees(np.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_right_face_width"] = float(np.linalg.norm(mp_landmarks[8] - mp_landmarks[16]))
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для semi_right: {e}")

        elif pose_type == "profile_left":
            try:
                metrics["MP_left_forehead_width"] = float(np.linalg.norm(mp_landmarks[0] - mp_landmarks[4]))
                metrics["MP_left_nose_width"] = float(np.linalg.norm(mp_landmarks[31] - mp_landmarks[30]))
                metrics["MP_left_nose_depth"] = float(np.abs(mp_landmarks[30][2] - mp_landmarks[27][2]))
                jaw_vector = mp_landmarks[8] - mp_landmarks[4]
                metrics["MP_left_jaw_angle"] = float(np.degrees(np.arctan2(jaw_vector[1], jaw_vector[0])))
                metrics["MP_left_jaw_width"] = float(np.linalg.norm(mp_landmarks[0] - mp_landmarks[4]))
                metrics["MP_left_cheek_width"] = float(np.linalg.norm(mp_landmarks[2] - mp_landmarks[4]))
                brow_vector = mp_landmarks[21] - mp_landmarks[17]
                metrics["MP_left_brow_angle"] = float(np.degrees(np.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_left_face_width"] = float(np.linalg.norm(mp_landmarks[0] - mp_landmarks[8]))
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для profile_left: {e}")



        elif pose_type == "profile_right":
            try:
                metrics["MP_right_forehead_width"] = float(np.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_nose_width"] = float(np.linalg.norm(mp_landmarks[30] - mp_landmarks[35]))
                metrics["MP_right_nose_depth"] = float(np.abs(mp_landmarks[30][2] - mp_landmarks[27][2]))
                jaw_vector = mp_landmarks[8] - mp_landmarks[12]
                metrics["MP_right_jaw_angle"] = float(np.degrees(np.arctan2(jaw_vector[1], jaw_vector[0])))
                metrics["MP_right_jaw_width"] = float(np.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_cheek_width"] = float(np.linalg.norm(mp_landmarks[12] - mp_landmarks[14]))
                brow_vector = mp_landmarks[26] - mp_landmarks[22]
                metrics["MP_right_brow_angle"] = float(np.degrees(np.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_right_face_width"] = float(np.linalg.norm(mp_landmarks[8] - mp_landmarks[16]))
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для profile_right: {e}")




    # Преобразование numpy-типов в обычные Python-типы
    mimetic_keys = [k for k in metrics if 'mouth' in k or 'lip' in k or 'eyebrow' in k]
    for k in mimetic_keys:
        metrics[k] = round(metrics[k] * 0.5, 5)

    for k, v in metrics.items():
        if isinstance(v, np.generic):
            metrics[k] = v.item()
    
    
    
    
    import cv2
    
    def calculate_texture_metrics(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gabor_kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_filtered = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
        gabor_energy = float(np.mean(np.abs(gabor_filtered)))
    
        contrast = float(gray.std())
        texture_variance = float(np.var(gray))
        highlight_skewness = float(np.mean((gray - gray.mean())**3)) / (gray.std()**3 + 1e-6)
    
        return gabor_energy, contrast, texture_variance, highlight_skewness
    
    
    
    
    try:
        # Ensure face_crop is defined by extracting the face region
        if roi_box is not None:
            x_min, y_min, x_max, y_max = [int(i) for i in roi_box]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img.shape[1], x_max)
            y_max = min(img.shape[0], y_max)
            face_crop = img[y_min:y_max, x_min:x_max]
        else:
            raise ValueError("ROI box list is empty, cannot extract face region.")

        gabor_mean, contrast_score, texture_var, highlight_skew = calculate_texture_metrics(face_crop)
    except Exception as e:
        print(f"⚠️ Ошибка при расчёте текстурных метрик: {e}")
        gabor_mean = contrast_score = texture_var = highlight_skew = 0.0
        texture_anomaly = False
    
    metrics["texture_gabor_energy_mean"] = round(gabor_mean, 4)
    metrics["texture_contrast_ratio_score"] = round(contrast_score, 4)
    metrics["texture_skin_texture_variance"] = round(texture_var, 4)
    metrics["texture_highlight_skewness"] = round(highlight_skew, 4)

    #Простейшая эвристика: если энергия Габора слишком низкая, текстура слабая; если разброс слишком мал — аномалия
    texture_anomaly = False
    if gabor_mean < 5.0 or texture_var < 200.0 or highlight_skew > 1.5:
        texture_anomaly = True
    metrics["texture_anomaly_flag"] = texture_anomaly
    # -------------------- МЕТРИКИ ПО КАРТЕ ГЛУБИНЫ --------------------
    try:
        from utils.depth import depth_map
        depth_data = depth_map(img, ver_lst[i], tddfa.tri, return_array=True)
        depth_std = float(np.std(depth_data))
        depth_range = float(np.max(depth_data) - np.min(depth_data))
        metrics["depth_std"] = round(depth_std, 4)
        metrics["depth_range"] = round(depth_range, 4)
        metrics["depth_flatness_score"] = round(1.0 / (depth_std + 1e-6), 4)
    except Exception as e:
        print(f"⚠️ Ошибка при извлечении depth-карты: {e}")

    # -------------------- МЕТРИКИ ПО КАРТЕ НОРМАЛЕЙ --------------------
    try:
        from ut.normal import normal_map
        normal_img = normal_map(img, ver_lst[i], tddfa.tri, return_array=True)
        avg_normal = np.mean(normal_img, axis=(0, 1))
        normal_magnitude = float(np.linalg.norm(avg_normal))
        metrics["normal_avg_magnitude"] = round(normal_magnitude, 4)
        metrics["normal_direction_bias"] = round(np.mean(np.abs(avg_normal - [0.5, 0.5, 1.0])), 4)
    except Exception as e:
        print(f"⚠️ Ошибка при извлечении normal-карты: {e}")

    metrics = adapt_mediapipe_metrics_by_pose(metrics, pose_type, delta_pose)
    metrics = adapt_fan_metrics_by_pose(metrics, pose_type, delta_pose)
    metrics = adapt_3ddfa_metrics_by_pose(metrics, pose_type, delta_pose)
    return metrics




#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------










#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------




# Функция для вычисления изменений в блоках метрик
def calculate_block_deltas(current_metrics, previous_metrics):
    """
    Вычисляет изменения между текущими и предыдущими метриками для каждого блока.
    """
    deltas = {}
    for key in current_metrics.keys():
        if key in previous_metrics:
            deltas[key] = round(current_metrics[key] - previous_metrics[key], 5)
        else:
            deltas[key] = None
    return deltas











def adapt_mediapipe_metrics_by_pose(metrics, pose_type, delta_pose):
    """
    Адаптация метрик MediaPipe в зависимости от силы виртуального доворота.
    Чем сильнее поворот, тем больше понижаем нестабильные метрики.
    """
    def penalty(delta):
        norm = sum([abs(d) for d in delta])  # суммарный "поворот"
        if norm > 60:
            return 0.1
        elif norm > 30:
            return 0.3
        elif norm > 15:
            return 0.6
        return 1.0

    for key in list(metrics.keys()):
        if not key.startswith("MP_"):
            continue
        weight = penalty(delta_pose)
        if weight < 1.0:
            metrics[key] = round(metrics[key] * weight, 5)
    return metrics


def adapt_fan_metrics_by_pose(metrics, pose_type, delta_pose):
    """
    Адаптация метрик FAN в зависимости от виртуального доворота.
    По аналогии с MediaPipe — глазные и мимические метрики ослабляются при сильных поворотах.
    """

    for key in list(metrics.keys()):
        if not key.startswith("fn_"):
            continue
        weight = penalty(delta_pose)
        if weight < 1.0:
            metrics[key] = round(metrics[key] * weight, 5)
    return metrics

def adapt_3ddfa_metrics_by_pose(metrics, pose_type, delta_pose):
    """
    Адаптация метрик 3DDFA в зависимости от отклонения от целевого ракурса.
    """
    def penalty(delta):
        norm = sum([abs(d) for d in delta])
        if norm > 60:
            return 0.1
        elif norm > 30:
            return 0.3
        elif norm > 15:
            return 0.6
        return 1.0

    for key in list(metrics.keys()):
        if not key.startswith("MP_") and not key.startswith("fn_"):
            weight = penalty(delta_pose)
            if weight < 1.0:
                metrics[key] = round(metrics[key] * weight, 5)
    return metrics













#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------




# Основная функция обработки изображения
def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Инициализация FaceBoxes и TDDFA
    if args.onnx:
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

        # Только внутри suppress — то, что реально шумит
        with suppress_stdout(), suppress_stderr():
            insight_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            insight_app.prepare(ctx_id=0)

    # Остальные print будут работать

    # Инициализация MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Загрузка изображения в BGR
    img = cv2.imread(args.img_fp)

    # Если передана папка, обработать все изображения
    if os.path.isdir(args.img_fp):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        img_files = [os.path.join(args.img_fp, f) for f in os.listdir(args.img_fp) if f.lower().endswith(exts)]
        for img_fp in sorted(img_files):
            args.img_fp = img_fp
            main(args)
        return

    # Обнаружение лиц
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'Лицо не обнаружено, завершение')
        sys.exit(-1)
    print(f'Обнаружено {n} лиц')

    # Выбираем только самое большое лицо (по площади)
    def box_area(box):
        x_min, y_min, x_max, y_max = box[:4]
        return (x_max - x_min) * (y_max - y_min)

    largest_box = max(boxes, key=box_area)
    boxes = [largest_box]

    # Получение параметров 3DMM и ROI боксов
    param_lst, roi_box_lst = tddfa(img, boxes)

    # Визуализация и сериализация
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix
    original_copy_path = wfp.replace(f"_{args.opt}{new_suffix}", f"_original{old_suffix}")
    shutil.copy2(args.img_fp, original_copy_path)
    print(f"📷 Скопировано оригинальное фото: {original_copy_path}")




    # Реконструкция вершин
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)


    # Инициализация предыдущих метрик
    previous_metrics = {}

    # Обработка параметров и сохранение результатов
    for i, param in enumerate(param_lst):
        param = np.array(param)
        if param.ndim == 1:
            param = param[np.newaxis, :]
        _, pose = calc_pose(param[0])
        yaw, pitch, roll = pose
        P = param[0][:12].reshape(3, -1)
        s, R, t3d = P2sRt(P)

        shape_vector = np.array(param[0][:62])

        # Определение типа ракурса
        if yaw < -60:
            pose_type = "profile_left"
        elif yaw < -15:
            pose_type = "semi_left"
        elif yaw <= 15:
            pose_type = "frontal"
        elif yaw <= 60:
            pose_type = "semi_right"
        else:
            pose_type = "profile_right"

        # Целевой yaw по ракурсу
        target_yaws = {
            "frontal": 0,
            "semi_left": -30,
            "semi_right": 30,
            "profile_left": -90,
            "profile_right": 90
        }
        target_yaw = target_yaws.get(pose_type, 0)
        
        # Выравнивание landmarks по среднему ракурсу
        mean_pose = {
            'frontal': (0, 0, 0),
            'semi_left': (-30, 0, 0),
            'semi_right': (30, 0, 0),
            'profile_left': (-90, 0, 0),
            'profile_right': (90, 0, 0)
        }
        target_pose = mean_pose.get(pose_type, (0, 0, 0))

        # MediaPipe обработка — перенесено сюда, чтобы yaw/pitch/roll уже были определены
        with suppress_stderr():
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mp_landmarks = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = img.shape
            mp_raw_landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks.landmark])  # 468 landmarks
            mp_landmarks = mp_raw_landmarks  # больше не нормализуем

        # Расчёт ошибок формы
        shape_error_raw = float(np.linalg.norm(shape_vector))
        pose_penalty = {
            "frontal": 1.0,
            "semi_left": 0.8,
            "semi_right": 0.8,
            "profile_left": 0.6,
            "profile_right": 0.6
        }.get(pose_type, 1.0)
        shape_error = shape_error_raw * pose_penalty
        anomaly_score = shape_error * pose_penalty
        MAX_REF = 500000  # Опорный максимум для калибровки
        anomaly_index = min((anomaly_score / MAX_REF) * 100, 100)







        # Сбор метрик (старые и новые)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        landmarks = ver_lst[i]
        x_min, y_min, x_max, y_max = [int(i) for i in roi_box_lst[i]]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)
        face_crop = img[y_min:y_max, x_min:x_max]
        faces = insight_app.get(face_crop)
        embedding_vector = faces[0].embedding.tolist() if faces else None
        # Выравнивание landmarks по среднему ракурсу
        mean_pose = {
            'frontal': (0, 0, 0),
            'semi_left': (-30, 0, 0),
            'semi_right': (30, 0, 0),
            'profile_left': (-90, 0, 0),
            'profile_right': (90, 0, 0)
        }
        



        
        if landmarks is not None:
            if landmarks.shape[0] == 3 and landmarks.shape[1] > 3:
                landmarks = landmarks.T
            elif landmarks.ndim == 2 and landmarks.shape[1] != 3:
                print(f"⚠️ Некорректная форма landmarks: {landmarks.shape}")
                landmarks = landmarks.reshape(-1, 3)

        

            
        if landmarks is None or len(landmarks) == 0:
            print(f"⚠️ landmarks пустой для файла {args.img_fp}")
            additional_metrics = {}
        else:
            # Вычисление delta_pose для адаптации метрик
            mean_pose = {
                'frontal': (0, 0, 0),
                'semi_left': (-30, 0, 0),
                'semi_right': (30, 0, 0),
                'profile_left': (-90, 0, 0),
                'profile_right': (90, 0, 0)
            }
            target_pose = mean_pose.get(pose_type, (0, 0, 0))
            _, (yaw, pitch, roll) = calc_pose(param[0])
            delta_pose = (
                yaw - target_pose[0],
                pitch - target_pose[1],
                roll - target_pose[2]
            )

            # Добавляем метрики с учётом отклонения delta_pose
            additional_metrics = add_metrics(
                pose_type=pose_type,
                param=param,
                landmarks=fan_landmarks,
                mp_landmarks=mp_landmarks,
                delta_pose=delta_pose,
                img=img,
                roi_box=roi_box_lst[i]
            )
        
        
        
        
        
        # Генерация изображения с FaceMesh и затемнённым фоном
        mesh_output_path = os.path.splitext(wfp)[0] + '_mesh.png'
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mesh_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        with suppress_stderr():
            results = mesh_face_mesh.process(rgb_img)
        if results.multi_face_landmarks:
            mesh_img = (img * 0.2).astype("uint8")
            for face_landmarks in results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=mesh_img,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec
                )
            cv2.imwrite(mesh_output_path, mesh_img)
            print(f"📸 Сохранён FaceMesh: {mesh_output_path}")
        mesh_face_mesh.close()

        # 🎯 Отладочная визуализация: FAN landmarks + куб ориентации
        debug_img = img.copy()
        
        # Рисуем FAN-точки как сетку (соединённые линии)
        fan_indices = [
            list(range(0, 17)),     # контур лица
            list(range(17, 22)),    # левая бровь
            list(range(22, 27)),    # правая бровь
            list(range(27, 31)),    # нос (вверх)
            list(range(31, 36)),    # нос (низ)
            [30, 36], [30, 45],     # нос → глаза
            list(range(36, 42)) + [36],  # левый глаз
            list(range(42, 48)) + [42],  # правый глаз
            list(range(48, 60)) + [48],  # внешний контур губ
            list(range(60, 68)) + [60],  # внутренний контур губ
        ]
        h, w = debug_img.shape[:2]
        for group in fan_indices:
            for j in range(len(group) - 1):
                pt1 = tuple(landmarks[group[j]].astype(int)[:2])       # <<< используй landmarks (до нормализации)
                pt2 = tuple(landmarks[group[j+1]].astype(int)[:2])     # <<< тоже
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(debug_img, pt1, pt2, (0, 255, 0), 1)
        
        # Куб для визуализации ориентации до и после нормализации
        from utils.pose import viz_pose
        from scipy.spatial.transform import Rotation as R

        # Куб ДО нормализации — стандартная функция viz_pose из 3DDFA_V2
        viz_pose(debug_img, [param[0]], [landmarks], wfp=None)

        # Куб ПОСЛЕ нормализации — вручную через draw_pose_cube
        center_3d = t3d  # центр головы по модели
        raw_rot = R.from_euler('xyz', [pitch, yaw, roll], degrees=True)
        norm_rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
        draw_pose_cube(debug_img, center_3d, raw_rot, color=(255, 0, 0))  # до нормализации — красный
        draw_pose_cube(debug_img, center_3d, norm_rot, color=(0, 255, 255))  # после нормализации — жёлтый

        debug_path = os.path.splitext(wfp)[0] + '_debug.png'
        cv2.imwrite(debug_path, debug_img)
        print(f"🧪 Сохранено отладочное изображение: {debug_path}")

        # Summary block — анализ комплексных отклонений
        summary_status = "normal 🟢"
        summary_description = "No significant anomalies detected."



        # Исправляем отступы
        if shape_error > 400000 or anomaly_index > 80:
            critical_keys = ["eye_distance", "face_asymmetry", "jaw_asymmetry", 
                            "chin_height", "fn_face_symmetry_score"]
            triggered = [k for k in critical_keys 
                        if k in additional_metrics and additional_metrics[k] > 1.5]
            
            if len(triggered) >= 2:
                summary_status = "atypical facial structure 🟡"
                summary_description = ("Multiple key facial metrics deviate from average: " + 
                                     ", ".join(triggered) + 
                                     ". Possible mask or artificial modification.")
            elif shape_error > 600000:
                summary_status = "strong geometric anomaly 🔴"
                summary_description = "Shape error exceeds threshold. Potential structural inconsistency."





        # Формирование итогового JSON с требуемым порядком полей
        result = OrderedDict([
            ("filename", os.path.basename(args.img_fp)),
            ("anomaly_status", "normal" if shape_error < 500000 else "anomalous"),
            ("summary", {
                "status": f"{summary_status} {'🟢' if 'normal' in summary_status else ('🟡' if 'atypical' in summary_status else '🔴')}",
                "description": summary_description,
                "pose_correction": {
                    "delta_yaw": round(delta_pose[0], 2),
                    "delta_pitch": round(delta_pose[1], 2),
                    "delta_roll": round(delta_pose[2], 2)
                },
                "suppressed_blocks": []
            }),
            ("pose_type", pose_type),
            ("metrics", result_metrics),
            ("shape_error", round(shape_error, 5)),
            ("pose", {
                "yaw": round(float(yaw), 2),
                "pitch": round(float(pitch), 2),
                "roll": round(float(roll), 2)
            }),
            ("scale", round(float(s), 4)),
            ("head_center", [round(float(i), 2) for i in t3d]),
            ("shape_error_raw", round(shape_error_raw, 5)),
            ("embedding_vector", embedding_vector),
            ("shape_vector", [round(float(x), 5) for x in param[0][:62]]),
            ("alpha_shp", [round(float(x), 6) for x in param[0][12:52]]),
            ("roi_box", [round(float(x), 2) for x in roi_box_lst[i]])
        ])




        # Сохранение в JSON
        out_path = os.path.splitext(wfp)[0] + '.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"🧾 Сохранён JSON: {out_path}")





    # Визуализация в зависимости от опции
    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '2d_dense':
        img = (img * 0.5).astype(np.uint8)
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp, color=(0, 255, 255), size=4)
    elif args.opt == '3d':
        img = (img * 0.5).astype(np.uint8)
        render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'depth':
        depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'pncc':
        pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'uv_tex':
        uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'pose':
        viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'ply':
        ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    elif args.opt == 'obj':
        ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    else:
        raise ValueError(f'Неизвестная опция {args.opt}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Демо обработки неподвижного изображения с помощью 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='режим gpu или cpu')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='показывать ли результат визуализации')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)





def aggregate_all_metrics_by_pose(metrics):
    """
    Агрегирует метрики из разных источников (FAN, 3DDFA, MediaPipe) по текущему ракурсу.
    Выделяет ключевые блоки и формирует удобную структуру для анализа.
    """
    aggregated = {
        "symmetry": {},
        "eyes": {},
        "mouth": {},
        "nose": {},
        "brows": {},
        "chin_jaw": {},
        "texture": {},
        "geometry": {},
        "mediapipe": {},
        "other": {}
    }

    for key, value in metrics.items():
        key_lower = key.lower()
        if any(k in key_lower for k in ["symmetry", "golden"]):
            aggregated["symmetry"][key] = value
        elif "eye" in key_lower:
            aggregated["eyes"][key] = value
        elif "mouth" in key_lower or "lip" in key_lower:
            aggregated["mouth"][key] = value
        elif "nose" in key_lower:
            aggregated["nose"][key] = value
        elif "brow" in key_lower or "eyebrow" in key_lower:
            aggregated["brows"][key] = value
        elif "jaw" in key_lower or "chin" in key_lower:
            aggregated["chin_jaw"][key] = value
        elif "texture" in key_lower:
            aggregated["texture"][key] = value
        elif key_lower.startswith("mp_"):
            aggregated["mediapipe"][key] = value
        elif "depth" in key_lower or "angle" in key_lower or "projection" in key_lower:
            aggregated["geometry"][key] = value
        else:
            aggregated["other"][key] = value

    return aggregated

def process_single_image(args):
    print(f"📥 Обработка файла: {args.img_fp}")
    # сюда позже будет перенесена вся логика из main