# analysis.py

import numpy
import os
import sys
from logger import logger, log_exception, log_function_call, LogEmoji, log_metrics, log_face
from core import compute_anomaly_score_v2
from utils.tddfa_util import _parse_param
from utils.pose import calc_pose  # Для estimate_pose_3ddfa

# Контекстный менеджер для подавления вывода
class SuppressOutput:
    def __enter__(self):
        # Открываем null-устройство
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        
        # Сохраняем оригинальные файловые дескрипторы
        self.stdout_fd = 1  # stdout file descriptor
        self.stderr_fd = 2  # stderr file descriptor
        self.stdout_copy = os.dup(self.stdout_fd)
        self.stderr_copy = os.dup(self.stderr_fd)
        
        # Перенаправляем stdout и stderr в null
        os.dup2(self.null_fd, self.stdout_fd)
        os.dup2(self.null_fd, self.stderr_fd)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Восстанавливаем оригинальные файловые дескрипторы
        os.dup2(self.stdout_copy, self.stdout_fd)
        os.dup2(self.stderr_copy, self.stderr_fd)
        
        # Закрываем копии и null-устройство
        os.close(self.stdout_copy)
        os.close(self.stderr_copy)
        os.close(self.null_fd)

@log_function_call
def run_anomaly_analysis_and_append(landmarks_data, img, ver_lst, tddfa_tri):
    """
    Анализ аномалий и расчет метрик с добавлением данных о глубине и нормалях.
    """
    logger.debug(f"{LogEmoji.PROCESSING} Начало анализа аномалий")
    
    if landmarks_data.get('3ddfa') and landmarks_data.get('fan'):
        try:
            # Получаем точки от всех детекторов
            pts_3ddfa = numpy.array(landmarks_data['3ddfa'][0])[:68]
            pts_fan = numpy.array(landmarks_data['fan'][0])[:68]
            
            # Получаем точки MediaPipe (если есть)
            mp_pts = numpy.array(landmarks_data['mediapipe'][0])[:68] if landmarks_data.get('mediapipe') else pts_fan
            
            param = landmarks_data.get('3ddfa_param')
            
            # Получаем углы поворота головы
            yaw, pitch, roll = estimate_pose_3ddfa(param)
            
            # Определяем ракурс
            pose_type = classify_pose_type(yaw)
            landmarks_data['pose_type'] = pose_type
            landmarks_data['pose_angles'] = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll)
            }
            
            # Извлекаем карту глубины и нормалей
            depth_buffer = extract_depth_map(img, ver_lst, tddfa_tri, show_flag=False, with_bg_flag=False)
            normal_map = extract_normal_map(img, ver_lst, tddfa_tri, show_flag=False, with_bg_flag=False)
            
            # Подготавливаем данные для core.py
            depth_data = {str(i): float(depth_buffer[y, x]) for i, (x, y) in enumerate(pts_3ddfa)}
            normal_data = {str(i): list(normal_map[y, x]) for i, (x, y) in enumerate(pts_3ddfa)}
            
            # Вызов основной функции расчёта аномалий
            anomaly = compute_anomaly_score_v2(
                fan_pts=pts_fan,
                ddfa_pts=pts_3ddfa,
                mp_pts=mp_pts,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                depth_data=depth_data,
                normal_data=normal_data
            )
            
            landmarks_data['anomaly'] = float(anomaly['A_face'])
            landmarks_data['anomaly_details'] = anomaly
            
        except Exception as e:
            log_exception(e, "Ошибка в run_anomaly_analysis_and_append")
            landmarks_data['anomaly'] = 1.0
            
    return landmarks_data

# Вспомогательные функции

@log_function_call
def estimate_pose_3ddfa(param):
    """Возвращает углы поворота головы (yaw, pitch, roll)."""
    try:
        if param is None:
            return 0.0, 0.0, 0.0
        
        # Подавляем вывод при вызове calc_pose
        with SuppressOutput():
            pose_values = calc_pose(param)
            
        return float(pose_values[0]), float(pose_values[1]), float(pose_values[2])
    except Exception as e:
        log_exception(e, "Ошибка при расчете углов поворота")
        return 0.0, 0.0, 0.0

@log_function_call
def classify_pose_type(yaw):
    """Классифицирует ракурс лица."""
    if abs(yaw) <= 15:
        return 'frontal'
    elif yaw < -30:
        return 'profile_left'
    elif yaw > 30:
        return 'profile_right'
    elif yaw < 0:
        return 'semi_left'
    else:
        return 'semi_right'

@log_function_call
def run_anomaly_analysis_and_append(landmarks_data, img, ver_lst, tddfa_tri):
    """
    Анализ аномалий и расчет метрик с добавлением данных о глубине и нормалях.
    """
    logger.debug(f"{LogEmoji.PROCESSING} Начало анализа аномалий")

    if landmarks_data.get('3ddfa') and landmarks_data.get('fan'):
        try:
            # Получаем точки от всех детекторов
            pts_3ddfa = numpy.array(landmarks_data['3ddfa'][0])[:68]
            pts_fan = numpy.array(landmarks_data['fan'][0])[:68]
            
            # Получаем точки MediaPipe (если есть)
            mp_pts = numpy.array(landmarks_data['mediapipe'][0])[:68] if landmarks_data.get('mediapipe') else pts_fan

            param = landmarks_data.get('3ddfa_param')
            
            # Получаем углы поворота головы
            yaw, pitch, roll = estimate_pose_3ddfa(param)
            
            # Определяем ракурс
            pose_type = classify_pose_type(yaw)
            landmarks_data['pose_type'] = pose_type
            landmarks_data['pose_angles'] = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll)
            }
            
            # Извлекаем карту глубины и нормалей
            depth_buffer = extract_depth_map(img, ver_lst, tddfa_tri, show_flag=False, with_bg_flag=False)
            normal_map = extract_normal_map(img, ver_lst, tddfa_tri, show_flag=False, with_bg_flag=False)
            
            # Подготавливаем данные для core.py
            depth_data = {str(i): float(depth_buffer[y, x]) for i, (x, y) in enumerate(pts_3ddfa)}
            normal_data = {str(i): list(normal_map[y, x]) for i, (x, y) in enumerate(pts_3ddfa)}
            
            # Вызов основной функции расчёта аномалий
            anomaly = compute_anomaly_score_v2(
                fan_pts=pts_fan,
                ddfa_pts=pts_3ddfa,
                mp_pts=mp_pts,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                depth_data=depth_data,
                normal_data=normal_data
            )
            
            landmarks_data['anomaly'] = float(anomaly['A_face'])
            landmarks_data['anomaly_details'] = anomaly

        except Exception as e:
            log_exception(e, "Ошибка в run_anomaly_analysis_and_append")
            landmarks_data['anomaly'] = 1.0

    return landmarks_data

# Вспомогательные функции
@log_function_call
def estimate_pose_3ddfa(param):
    """Возвращает углы поворота головы (yaw, pitch, roll)."""
    try:
        if param is None:
            return 0.0, 0.0, 0.0
        pose_values = calc_pose(param)
        return float(pose_values[0]), float(pose_values[1]), float(pose_values[2])
    except Exception as e:
        log_exception(e, "Ошибка при расчете углов поворота")
        return 0.0, 0.0, 0.0

@log_function_call
def classify_pose_type(yaw):
    """Классифицирует ракурс лица."""
    if abs(yaw) <= 15:
        return 'frontal'
    elif yaw < -30:
        return 'profile_left'
    elif yaw > 30:
        return 'profile_right'
    elif yaw < 0:
        return 'semi_left'
    else:
        return 'semi_right'

# Заглушки для функций, которые должны быть перенесены из demo2.py
def extract_depth_map(img, ver_lst, tri, **kwargs):
    """Заглушка для примера (реализация должна быть перенесена из demo2.py)."""
    h, w = img.shape[:2]
    return numpy.zeros((h, w), dtype=numpy.float32)

def extract_normal_map(img, ver_lst, tri, **kwargs):
    """Заглушка для примера (реализация должна быть перенесена из demo2.py)."""
    h, w = img.shape[:2]
    return numpy.zeros((h, w, 3), dtype=numpy.float32)