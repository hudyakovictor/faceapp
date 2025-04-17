# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
from utils.pose import calc_pose
from utils.pose import P2sRt

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

def calculate_block_deltas(current_metrics, previous_metrics):
    """
    Вычисляет изменения между текущими и предыдущими метриками для каждого блока.
    Для каждого ключа из current_metrics, если он есть в previous_metrics,
    возвращает разницу (текущая – предыдущая), округленную до 5 знаков после запятой;
    если нет – возвращает None.
    """
    deltas = {}
    for key in current_metrics.keys():
        if key in previous_metrics and previous_metrics[key] is not None:
            try:
                deltas[key] = round(current_metrics[key] - previous_metrics[key], 5)
            except Exception:
                deltas[key] = None
        else:
            deltas[key] = None
    return deltas

def calculate_block_scores(metrics):
    """
    Группирует метрики по блокам и вычисляет среднее значение для каждого блока.
    Блоки: глаза, нос, рот, симметрия, череп.
    """
    blocks = {
        "eyes": [],
        "nose": [],
        "mouth": [],
        "symmetry": [],
        "skull": []
    }
    for key, value in metrics.items():
        if "eye" in key:
            blocks["eyes"].append(value)
        elif "nose" in key:
            blocks["nose"].append(value)
        elif "mouth" in key or "lip" in key:
            blocks["mouth"].append(value)
        elif "asymmetry" in key or "symmetry" in key:
            blocks["symmetry"].append(value)
        elif "face" in key or "jaw" in key or "chin" in key:
            blocks["skull"].append(value)
    block_scores = {}
    for block, values in blocks.items():
        if values:
            block_scores[block] = round(sum(values) / len(values), 5)
        else:
            block_scores[block] = None
    return block_scores
import mediapipe as mp  # Импорт MediaPipe для новых метрик

# Функция для добавления метрик в зависимости от ракурса
def add_metrics(pose_type, param, landmarks, mp_landmarks):
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
            metrics["semi_right_nose_angle"] = float(np.degrees(np.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["semi_right_skull_width"] = float(np.linalg.norm(landmarks[12] - landmarks[16]))
            metrics["semi_right_cheek_width"] = float(np.linalg.norm(landmarks[12] - landmarks[14]))
            metrics["semi_right_jaw_width"] = float(np.linalg.norm(landmarks[12] - landmarks[16]))
            metrics["semi_right_jaw_angle"] = float(np.degrees(np.arctan2(landmarks[8][1] - landmarks[12][1], landmarks[8][0] - landmarks[12][0])))
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

    # 🔍 Адаптация метрик в зависимости от поворота головы
    stable_metrics_by_pose = {
        "frontal": list(metrics.keys()),  # фронтальные — все метрики сохраняются
        "semi_left": [k for k in metrics if 'eye' in k or 'nose' in k or 'jaw' in k],
        "semi_right": [k for k in metrics if 'eye' in k or 'nose' in k or 'jaw' in k],
        "profile_left": [k for k in metrics if 'jaw' in k or 'chin' in k or 'nose' in k],
        "profile_right": [k for k in metrics if 'jaw' in k or 'chin' in k or 'nose' in k],
    }

    stable_keys = stable_metrics_by_pose.get(pose_type, [])
    for k in list(metrics.keys()):
        if k not in stable_keys:
            metrics[k] = round(metrics[k] * 0.1, 5)  # понижаем вес нестабильных

    # Комментарий: приоритет групп метрик (например, глаза — высокий, губы — средний, ambient — низкий)
    return metrics

# Нормализация landmarks для заданного yaw
def normalize_landmarks(param, target_yaw):
    """
    Нормализует landmarks в пределах текущего ракурса, выравнивая по yaw, pitch и roll.

    :param param: массив параметров [1, 62]
    :param target_yaw: желаемый yaw в градусах (например, 0, -30, +90 и т.д.)
    :return: нормализованный param
    """
    P = param[0][:12].reshape(3, -1)
    s, R_mat, t3d = P2sRt(P)

    # Определяем текущие углы
    _, (yaw, pitch, roll) = calc_pose(param[0])
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Текущая ориентация головы
    rot_current = R.from_euler('xyz', [pitch_rad, yaw_rad, roll_rad])
    # Целевая ориентация
    rot_target = R.from_euler('xyz', [0, np.radians(target_yaw), 0])
    # Матрица, которая "повернёт" текущую позу к целевой
    delta_rot = rot_current.inv() * rot_target

    # Применим поворот к landmark-координатам
    landmarks = param[0][62:].reshape(-1, 3) if len(param[0]) > 62 else np.zeros((68, 3))
    normalized = delta_rot.apply(landmarks)

    param[0][62:] = normalized.reshape(-1) if len(param[0]) > 62 else param[0]
    return param

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
        # Инициализация InsightFace
        insight_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        insight_app.prepare(ctx_id=0)

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

    # Обработка через MediaPipe
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mp_68_landmarks = None
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = img.shape
        mp_landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks.landmark])  # 468 landmarks
        # Примерное преобразование 468 точек в 68 (нужна точная карта соответствия)
        mp_to_68_indices = [
            33, 263, 1, 61, 291, 199,  # нижняя челюсть
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,  # левая бровь
            334, 296, 336, 285, 295, 282, 283, 276,  # правая бровь
            4,  # кончик носа
            45, 55, 65, 52,  # переносица
            133, 173, 157, 158, 159, 160, 161, 246,  # левый глаз
            362, 398, 384, 385, 386, 387, 388, 466,  # правый глаз
            70, 63, 105, 66, 107,  # верхняя губа
            336, 296, 334, 293, 300,  # нижняя губа
            33, 7, 163, 144, 145, 153, 154, 155, 133  # контур лица (частично)
        ]
        if len(mp_to_68_indices) == 68:
            mp_68_landmarks = mp_landmarks[mp_to_68_indices]
        else:
            print("⚠️ Ошибка в длине массива mp_to_68_indices, используем только 3DDFA landmarks")

    # Обработка параметров и сохранение результатов
    for i, param in enumerate(param_lst):
        param = np.array(param)
        if param.ndim == 1:
            param = param[np.newaxis, :]
        _, pose = calc_pose(param[0])
        yaw, pitch, roll = pose
        P = param[0][:12].reshape(3, -1)
        s, R_mat, t3d = P2sRt(P)

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

        # Нормализация landmarks (старый метод отключён)
        # param = normalize_landmarks(param, target_yaw)

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
        landmarks = ver_lst[i]
        faces = insight_app.get(img)
        embedding_vector = faces[0].embedding.tolist() if faces else None
        # Выравнивание landmarks по среднему ракурсу
        mean_pose = {'frontal': (0, 0, 0), 'semi_left': (-30, 0, 0), 'semi_right': (30, 0, 0),
                     'profile_left': (-90, 0, 0), 'profile_right': (90, 0, 0)}
        target_pose = mean_pose.get(pose_type, (0, 0, 0))
        delta_rot = R.from_euler('xyz', [np.radians(pitch), np.radians(yaw), np.radians(roll)]).inv() * R.from_euler('xyz', [np.radians(target_pose[1]), np.radians(target_pose[0]), np.radians(target_pose[2])])
        
        if landmarks is not None:
            print(f"landmarks shape before reshape: {landmarks.shape}")
            if landmarks.ndim == 1:
                landmarks = landmarks.reshape(-1, 3)
            elif landmarks.shape[0] == 3 and landmarks.shape[1] > 3:
                landmarks = landmarks.T
            landmarks = delta_rot.apply(landmarks)
        
        if mp_68_landmarks is not None:
            print(f"mp_68_landmarks shape before reshape: {mp_68_landmarks.shape}")
            if mp_68_landmarks.ndim == 1:
                mp_68_landmarks = mp_68_landmarks.reshape(-1, 3)
            elif mp_68_landmarks.shape[0] == 3 and mp_68_landmarks.shape[1] > 3:
                mp_68_landmarks = mp_68_landmarks.T
            mp_68_landmarks = delta_rot.apply(mp_68_landmarks)

        if landmarks is None or len(landmarks) == 0:
            print(f"⚠️ landmarks пустой для файла {args.img_fp}")
            additional_metrics = {}
        else:
            additional_metrics = add_metrics(pose_type, param, landmarks, mp_68_landmarks)

        # Summary block — анализ комплексных отклонений (вычисляем до формирования JSON)
        summary_status = "normal"
        summary_description = "No significant anomalies detected."
        if shape_error > 400000 or anomaly_index > 80:
            critical_keys = ["eye_distance", "face_asymmetry", "jaw_asymmetry", "chin_height", "fn_face_symmetry_score"]
            triggered = [k for k in critical_keys if k in additional_metrics and additional_metrics[k] > 1.5]
            if len(triggered) >= 2:
                summary_status = "atypical facial structure"
                summary_description = "Multiple key facial metrics deviate from average: " + ", ".join(triggered) + ". Possible mask or artificial modification."
            elif shape_error > 600000:
                summary_status = "strong geometric anomaly"
                summary_description = "Shape error exceeds threshold. Potential structural inconsistency."

        # Формирование итогового JSON с требуемым порядком полей
        previous_metrics = {}
        result = OrderedDict([
            ("filename", os.path.basename(args.img_fp)),
            ("anomaly_status", "normal" if shape_error < 500000 else "anomalous"),
            ("summary", {
                "status": summary_status,
                "description": summary_description
            }),
            ("pose_type", pose_type),
            ("metrics", additional_metrics),
            ("shape_error", round(shape_error, 5)),
            ("pose", {
                "yaw": round(float(yaw), 2),
                "pitch": round(float(pitch), 2),
                "roll": round(float(roll), 2)
            }),
            ("confidence_level", (
                "high" if abs(yaw) <= 20 else
                "medium" if abs(yaw) <= 45 else
                "low"
            )),
            ("scale", round(float(s), 4)),
            ("head_center", [round(float(i), 2) for i in t3d]),
            ("shape_error_raw", round(shape_error_raw, 5)),
            ("embedding_vector", embedding_vector),
            ("shape_vector", [round(float(x), 5) for x in param[0][:62]]),
            ("alpha_shp", [round(float(x), 6) for x in param[0][12:52]]),
            ("roi_box", [round(float(x), 2) for x in roi_box_lst[i]]),
            ("block_scores", calculate_block_scores(additional_metrics)),
            ("block_change_score", calculate_block_deltas(additional_metrics, previous_metrics)),
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