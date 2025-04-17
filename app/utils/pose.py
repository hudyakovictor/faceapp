"""
Модуль для определения ракурса по углам поворота головы
"""

import numpy as np
from app.config import ACTIVE_METRICS, MP_ACTIVE_METRICS

# Пороговые значения для определения ракурса по углу yaw (в градусах)
YAW_THRESHOLDS = {
    'frontal': (-15, 15),
    'half_profile_left': (-45, -15),
    'half_profile_right': (15, 45),
    'profile_left': (-90, -45),
    'profile_right': (45, 90)
}

def determine_pose(angles):
    """
    Определяет ракурс головы на основе углов yaw, pitch, roll
    
    Args:
        angles (list): Список углов [yaw, pitch, roll] в градусах
        
    Returns:
        dict: Словарь с информацией о ракурсе
    """
    yaw, pitch, roll = angles
    
    # Определение ракурса по yaw (поворот влево-вправо)
    pose = determine_yaw_pose(yaw)
    
    return {
        'pose': pose,
        'angles': {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }
    }

def determine_yaw_pose(yaw):
    """
    Определяет ракурс по углу yaw
    
    Args:
        yaw (float): Угол yaw в градусах
        
    Returns:
        str: Название ракурса
    """
    for pose, (min_angle, max_angle) in YAW_THRESHOLDS.items():
        if min_angle <= yaw < max_angle:
            return pose
    
    # Если угол выходит за пределы определенных диапазонов
    if yaw < -90:
        return 'profile_left'
    else:
        return 'profile_right'

def get_active_metrics(pose_info):
    """
    Возвращает список активных метрик для данного ракурса
    
    Args:
        pose_info (dict): Информация о ракурсе
        
    Returns:
        list: Список активных метрик
    """
    pose = pose_info['pose']
    
    if pose in ACTIVE_METRICS:
        return ACTIVE_METRICS[pose]
    else:
        # Если ракурс не определен, возвращаем метрики для фронтального ракурса
        return ACTIVE_METRICS['frontal']

def get_mp_active_metrics(pose_info):
    """
    Возвращает список активных метрик MediaPipe для данного ракурса
    
    Args:
        pose_info (dict): Информация о ракурсе
        
    Returns:
        list: Список активных метрик MediaPipe
    """
    pose = pose_info['pose']
    
    if pose in MP_ACTIVE_METRICS:
        return MP_ACTIVE_METRICS[pose]
    else:
        # Если ракурс не определен, возвращаем метрики для фронтального ракурса
        return MP_ACTIVE_METRICS['frontal']

def is_frontal(pose_info):
    """
    Проверяет, является ли ракурс фронтальным
    
    Args:
        pose_info (dict): Информация о ракурсе
        
    Returns:
        bool: True, если ракурс фронтальный, иначе False
    """
    return pose_info['pose'] == 'frontal'

def is_profile(pose_info):
    """
    Проверяет, является ли ракурс профильным
    
    Args:
        pose_info (dict): Информация о ракурсе
        
    Returns:
        bool: True, если ракурс профильный, иначе False
    """
    return pose_info['pose'] in ['profile_left', 'profile_right']

def is_half_profile(pose_info):
    """
    Проверяет, является ли ракурс полупрофильным
    
    Args:
        pose_info (dict): Информация о ракурсе
        
    Returns:
        bool: True, если ракурс полупрофильный, иначе False
    """
    return pose_info['pose'] in ['half_profile_left', 'half_profile_right']

def is_left_side(pose_info):
    """
    Проверяет, является ли ракурс левосторонним
    
    Args:
        pose_info (dict): Информация о ракурсе
        
    Returns:
        bool: True, если ракурс левосторонний, иначе False
    """
    return pose_info['pose'] in ['profile_left', 'half_profile_left']

def is_right_side(pose_info):
    """
    Проверяет, является ли ракурс правосторонним
    
    Args:
        pose_info (dict): Информация о ракурсе
        
    Returns:
        bool: True, если ракурс правосторонний, иначе False
    """
    return pose_info['pose'] in ['profile_right', 'half_profile_right']

def get_pose_confidence(pose_info):
    """
    Возвращает уверенность в определении ракурса
    
    Args:
        pose_info (dict): Информация о ракурсе
        
    Returns:
        float: Значение уверенности от 0 до 1
    """
    yaw = pose_info['angles']['yaw']
    pose = pose_info['pose']
    
    # Получаем границы для данного ракурса
    if pose in YAW_THRESHOLDS:
        min_angle, max_angle = YAW_THRESHOLDS[pose]
        
        # Вычисляем расстояние до границы ракурса
        if min_angle <= yaw <= max_angle:
            # Если угол в центре диапазона, уверенность максимальна
            center = (min_angle + max_angle) / 2
            distance_to_center = abs(yaw - center)
            range_half_width = (max_angle - min_angle) / 2
            
            # Линейное уменьшение уверенности от центра к границе
            confidence = 1.0 - (distance_to_center / range_half_width) * 0.5
            return confidence
        else:
            # Если угол вне диапазона, уверенность минимальна
            return 0.5
    else:
        # Если ракурс не определен, уверенность средняя
        return 0.5
