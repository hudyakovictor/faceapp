"""
Модуль для расчета confidence (достоверности) метрик
"""

import numpy as np
from app.config import YAW_THRESHOLDS, PITCH_THRESHOLDS, ROLL_THRESHOLDS

def calculate_confidence(angles, metric_name, pose_info):
    """
    Рассчитывает confidence (достоверность) для метрики на основе углов головы
    
    Args:
        angles (list): Список углов [yaw, pitch, roll] в градусах
        metric_name (str): Название метрики
        pose_info (dict): Информация о ракурсе
        
    Returns:
        float: Значение confidence от 0 до 1
    """
    yaw, pitch, roll = angles
    main_pose = pose_info['main']
    
    # Базовый confidence для метрики
    base_confidence = get_base_confidence(metric_name, main_pose)
    
    # Модификаторы confidence на основе углов
    yaw_modifier = calculate_yaw_modifier(yaw, metric_name)
    pitch_modifier = calculate_pitch_modifier(pitch, metric_name)
    roll_modifier = calculate_roll_modifier(roll, metric_name)
    
    # Итоговый confidence
    confidence = base_confidence * yaw_modifier * pitch_modifier * roll_modifier
    
    # Ограничиваем значение от 0 до 1
    return max(0.0, min(1.0, confidence))

def get_base_confidence(metric_name, pose):
    """
    Возвращает базовый confidence для метрики в зависимости от ракурса
    
    Args:
        metric_name (str): Название метрики
        pose (str): Ракурс
        
    Returns:
        float: Базовый confidence
    """
    # Словарь базовых confidence для каждой метрики и ракурса
    base_confidences = {
        'frontal': {
            'eye_distance': 1.0,
            'eye_ratio': 1.0,
            'nose_width': 1.0,
            'mouth_width': 1.0,
            'face_width': 1.0,
            'face_height': 0.9,
            'jaw_width': 0.9,
            'symmetry': 1.0,
            'eyebrow_angle': 1.0,
            'eye_aspect_ratio': 1.0,
            'mouth_aspect_ratio': 1.0,
            'face_oval_ratio': 0.9,
            'nose_tip_position': 0.9
        },
        'half_profile_left': {
            'visible_eye_size': 0.9,
            'nose_projection': 0.9,
            'cheek_contour': 0.8,
            'jaw_angle': 0.8,
            'face_depth': 0.9,
            'visible_eye_ratio': 0.9,
            'cheek_contour': 0.8,
            'jaw_projection': 0.8
        },
        'half_profile_right': {
            'visible_eye_size': 0.9,
            'nose_projection': 0.9,
            'cheek_contour': 0.8,
            'jaw_angle': 0.8,
            'face_depth': 0.9,
            'visible_eye_ratio': 0.9,
            'cheek_contour': 0.8,
            'jaw_projection': 0.8
        },
        'profile_left': {
            'nose_projection': 1.0,
            'face_depth': 1.0,
            'jaw_line': 0.9,
            'forehead_curve': 0.8,
            'forehead_slope': 0.8,
            'chin_projection': 0.9
        },
        'profile_right': {
            'nose_projection': 1.0,
            'face_depth': 1.0,
            'jaw_line': 0.9,
            'forehead_curve': 0.8,
            'forehead_slope': 0.8,
            'chin_projection': 0.9
        }
    }
    
    # Если метрика определена для данного ракурса
    if pose in base_confidences and metric_name in base_confidences[pose]:
        return base_confidences[pose][metric_name]
    
    # Если метрика не определена для данного ракурса
    return 0.5  # Средний confidence по умолчанию

def calculate_yaw_modifier(yaw, metric_name):
    """
    Рассчитывает модификатор confidence на основе угла yaw
    
    Args:
        yaw (float): Угол yaw в градусах
        metric_name (str): Название метрики
        
    Returns:
        float: Модификатор confidence
    """
    # Словарь оптимальных диапазонов yaw для каждой метрики
    optimal_ranges = {
        'eye_distance': (-10, 10),
        'eye_ratio': (-10, 10),
        'nose_width': (-15, 15),
        'mouth_width': (-15, 15),
        'face_width': (-15, 15),
        'face_height': (-20, 20),
        'jaw_width': (-20, 20),
        'symmetry': (-5, 5),
        'visible_eye_size': (-45, -15) if 'left' in metric_name else (15, 45),
        'nose_projection': (-90, -30) if 'left' in metric_name else (30, 90),
        'cheek_contour': (-60, -20) if 'left' in metric_name else (20, 60),
        'jaw_angle': (-60, -20) if 'left' in metric_name else (20, 60),
        'face_depth': (-90, -30) if 'left' in metric_name else (30, 90),
        'jaw_line': (-90, -60) if 'left' in metric_name else (60, 90),
        'forehead_curve': (-90, -60) if 'left' in metric_name else (60, 90)
    }
    
    # Если метрика определена
    if metric_name in optimal_ranges:
        min_angle, max_angle = optimal_ranges[metric_name]
        
        # Если угол в оптимальном диапазоне
        if min_angle <= yaw <= max_angle:
            return 1.0
        
        # Если угол близок к оптимальному диапазону
        if min_angle - 15 <= yaw < min_angle or max_angle < yaw <= max_angle + 15:
            # Линейное уменьшение confidence
            if yaw < min_angle:
                return 1.0 - (min_angle - yaw) / 15
            else:
                return 1.0 - (yaw - max_angle) / 15
        
        # Если угол далек от оптимального диапазона
        return 0.5
    
    # Если метрика не определена
    return 1.0  # Не влияет на confidence

def calculate_pitch_modifier(pitch, metric_name):
    """
    Рассчитывает модификатор confidence на основе угла pitch
    
    Args:
        pitch (float): Угол pitch в градусах
        metric_name (str): Название метрики
        
    Returns:
        float: Модификатор confidence
    """
    # Словарь оптимальных диапазонов pitch для каждой метрики
    optimal_ranges = {
        'eye_distance': (-15, 15),
        'eye_ratio': (-15, 15),
        'nose_width': (-20, 20),
        'mouth_width': (-20, 20),
        'face_width': (-20, 20),
        'face_height': (-15, 15),
        'jaw_width': (-15, 15),
        'symmetry': (-15, 15),
        'forehead_curve': (-30, 0),
        'chin_projection': (0, 30)
    }
    
    # Если метрика определена
    if metric_name in optimal_ranges:
        min_angle, max_angle = optimal_ranges[metric_name]
        
        # Если угол в оптимальном диапазоне
        if min_angle <= pitch <= max_angle:
            return 1.0
        
        # Если угол близок к оптимальному диапазону
        if min_angle - 15 <= pitch < min_angle or max_angle < pitch <= max_angle + 15:
            # Линейное уменьшение confidence
            if pitch < min_angle:
                return 1.0 - (min_angle - pitch) / 15
            else:
                return 1.0 - (pitch - max_angle) / 15
        
        # Если угол далек от оптимального диапазона
        return 0.5
    
    # Если метрика не определена
    return 1.0  # Не влияет на confidence

def calculate_roll_modifier(roll, metric_name):
    """
    Рассчитывает модификатор confidence на основе угла roll
    
    Args:
        roll (float): Угол roll в градусах
        metric_name (str): Название метрики
        
    Returns:
        float: Модификатор confidence
    """
    # Словарь оптимальных диапазонов roll для каждой метрики
    optimal_ranges = {
        'eye_distance': (-10, 10),
        'eye_ratio': (-10, 10),
        'nose_width': (-15, 15),
        'mouth_width': (-15, 15),
        'face_width': (-15, 15),
        'face_height': (-15, 15),
        'jaw_width': (-15, 15),
        'symmetry': (-5, 5)
    }
    
    # Если метрика определена
    if metric_name in optimal_ranges:
        min_angle, max_angle = optimal_ranges[metric_name]
        
        # Если угол в оптимальном диапазоне
        if min_angle <= roll <= max_angle:
            return 1.0
        
        # Если угол близок к оптимальному диапазону
        if min_angle - 15 <= roll < min_angle or max_angle < roll <= max_angle + 15:
            # Линейное уменьшение confidence
            if roll < min_angle:
                return 1.0 - (min_angle - roll) / 15
            else:
                return 1.0 - (roll - max_angle) / 15
        
        # Если угол далек от оптимального диапазона
        return 0.5
    
    # Если метрика не определена
    return 1.0  # Не влияет на confidence
