"""
Модуль для расчета отклонений между landmarks от разных библиотек
"""

import numpy as np
from app.config import LANDMARKS_INDICES

def calculate_deviations(landmarks_3ddfa, landmarks_fan):
    """
    Рассчитывает отклонения между landmarks от 3DDFA и FAN
    
    Args:
        landmarks_3ddfa (numpy.ndarray): Массив landmarks от 3DDFA размером (68, 3)
        landmarks_fan (numpy.ndarray): Массив landmarks от FAN размером (68, 2)
        
    Returns:
        dict: Словарь с отклонениями
    """
    # Преобразуем landmarks_fan в формат (68, 3), добавляя нулевую координату Z
    landmarks_fan_3d = np.zeros_like(landmarks_3ddfa)
    landmarks_fan_3d[:, :2] = landmarks_fan
    
    # Вычисляем общее отклонение
    total_deviation = np.mean(np.linalg.norm(landmarks_3ddfa[:, :2] - landmarks_fan[:, :2], axis=1))
    
    # Вычисляем отклонения по группам точек
    deviations = {
        'total': float(total_deviation)
    }
    
    # Вычисляем отклонения для каждой группы точек
    for group_name, indices in LANDMARKS_INDICES.items():
        group_deviation = np.mean(np.linalg.norm(
            landmarks_3ddfa[indices, :2] - landmarks_fan[indices, :2], axis=1
        ))
        deviations[group_name] = float(group_deviation)
    
    # Вычисляем отклонения симметрии
    symmetric_pairs = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (48, 54), (49, 53), (50, 52), (59, 55), (58, 56), (57, 57)
    ]
    
    symmetry_deviations_3ddfa = []
    symmetry_deviations_fan = []
    
    for left_idx, right_idx in symmetric_pairs:
        # Отклонения для 3DDFA
        left_point_3ddfa = landmarks_3ddfa[left_idx, :2]
        right_point_3ddfa = landmarks_3ddfa[right_idx, :2]
        symmetry_deviation_3ddfa = np.linalg.norm(left_point_3ddfa - right_point_3ddfa)
        symmetry_deviations_3ddfa.append(symmetry_deviation_3ddfa)
        
        # Отклонения для FAN
        left_point_fan = landmarks_fan[left_idx]
        right_point_fan = landmarks_fan[right_idx]
        symmetry_deviation_fan = np.linalg.norm(left_point_fan - right_point_fan)
        symmetry_deviations_fan.append(symmetry_deviation_fan)
    
    # Вычисляем разницу в симметрии между 3DDFA и FAN
    symmetry_diff = np.mean(np.abs(np.array(symmetry_deviations_3ddfa) - np.array(symmetry_deviations_fan)))
    deviations['symmetry_diff'] = float(symmetry_diff)
    
    # Вычисляем отклонения по осям
    deviations['x_axis'] = float(np.mean(np.abs(landmarks_3ddfa[:, 0] - landmarks_fan[:, 0])))
    deviations['y_axis'] = float(np.mean(np.abs(landmarks_3ddfa[:, 1] - landmarks_fan[:, 1])))
    
    # Вычисляем максимальное отклонение
    max_deviation = float(np.max(np.linalg.norm(landmarks_3ddfa[:, :2] - landmarks_fan[:, :2], axis=1)))
    deviations['max'] = max_deviation
    
    # Вычисляем процент точек с большим отклонением
    threshold = 0.05  # Пороговое значение для большого отклонения
    large_deviations = np.linalg.norm(landmarks_3ddfa[:, :2] - landmarks_fan[:, :2], axis=1) > threshold
    percent_large_deviations = float(np.mean(large_deviations) * 100)
    deviations['percent_large'] = percent_large_deviations
    
    # Вычисляем отклонения по ключевым областям лица
    key_regions = {
        'eyes': list(range(36, 48)),  # Оба глаза
        'nose': list(range(27, 36)),  # Нос
        'mouth': list(range(48, 68)),  # Рот
        'eyebrows': list(range(17, 27)),  # Брови
        'jaw': list(range(0, 17))  # Челюсть
    }
    
    for region_name, indices in key_regions.items():
        region_deviation = np.mean(np.linalg.norm(
            landmarks_3ddfa[indices, :2] - landmarks_fan[indices, :2], axis=1
        ))
        deviations[f'{region_name}_deviation'] = float(region_deviation)
    
    # Вычисляем стандартное отклонение для всех точек
    std_deviation = float(np.std(np.linalg.norm(landmarks_3ddfa[:, :2] - landmarks_fan[:, :2], axis=1)))
    deviations['std'] = std_deviation
    
    # Вычисляем нормализованные отклонения (относительно размера лица)
    face_size = np.max(landmarks_3ddfa[:, :2], axis=0) - np.min(landmarks_3ddfa[:, :2], axis=0)
    face_diagonal = np.linalg.norm(face_size)
    
    if face_diagonal > 0:
        normalized_deviation = total_deviation / face_diagonal
        deviations['normalized'] = float(normalized_deviation)
        
        # Нормализованные отклонения по ключевым областям
        for region_name, indices in key_regions.items():
            region_deviation = np.mean(np.linalg.norm(
                landmarks_3ddfa[indices, :2] - landmarks_fan[indices, :2], axis=1
            ))
            deviations[f'{region_name}_normalized'] = float(region_deviation / face_diagonal)
    
    # Вычисляем отклонения в пропорциях лица
    # Пропорции глаз
    eye_width_3ddfa = np.linalg.norm(landmarks_3ddfa[36, :2] - landmarks_3ddfa[39, :2])
    eye_width_fan = np.linalg.norm(landmarks_fan[36] - landmarks_fan[39])
    eye_width_diff = abs(eye_width_3ddfa - eye_width_fan) / max(eye_width_3ddfa, eye_width_fan)
    deviations['eye_width_proportion'] = float(eye_width_diff)
    
    # Пропорции рта
    mouth_width_3ddfa = np.linalg.norm(landmarks_3ddfa[48, :2] - landmarks_3ddfa[54, :2])
    mouth_width_fan = np.linalg.norm(landmarks_fan[48] - landmarks_fan[54])
    mouth_width_diff = abs(mouth_width_3ddfa - mouth_width_fan) / max(mouth_width_3ddfa, mouth_width_fan)
    deviations['mouth_width_proportion'] = float(mouth_width_diff)
    
    # Пропорции носа
    nose_width_3ddfa = np.linalg.norm(landmarks_3ddfa[31, :2] - landmarks_3ddfa[35, :2])
    nose_width_fan = np.linalg.norm(landmarks_fan[31] - landmarks_fan[35])
    nose_width_diff = abs(nose_width_3ddfa - nose_width_fan) / max(nose_width_3ddfa, nose_width_fan)
    deviations['nose_width_proportion'] = float(nose_width_diff)
    
    # Вычисляем отклонения в углах между ключевыми точками
    # Угол между глазами и носом
    eye_nose_angle_3ddfa = calculate_angle(
        landmarks_3ddfa[39, :2], landmarks_3ddfa[42, :2], landmarks_3ddfa[33, :2]
    )
    eye_nose_angle_fan = calculate_angle(
        landmarks_fan[39], landmarks_fan[42], landmarks_fan[33]
    )
    eye_nose_angle_diff = abs(eye_nose_angle_3ddfa - eye_nose_angle_fan)
    deviations['eye_nose_angle'] = float(eye_nose_angle_diff)
    
    # Угол челюсти
    jaw_angle_3ddfa = calculate_angle(
        landmarks_3ddfa[0, :2], landmarks_3ddfa[8, :2], landmarks_3ddfa[16, :2]
    )
    jaw_angle_fan = calculate_angle(
        landmarks_fan[0], landmarks_fan[8], landmarks_fan[16]
    )
    jaw_angle_diff = abs(jaw_angle_3ddfa - jaw_angle_fan)
    deviations['jaw_angle'] = float(jaw_angle_diff)
    
    # Вычисляем общую аномальность отклонений
    # Комбинируем различные факторы для определения аномальности
    anomaly_score = (
        deviations['normalized'] * 2.0 +
        deviations['symmetry_diff'] * 1.5 +
        deviations['percent_large'] / 100.0 +
        deviations['eye_width_proportion'] * 1.2 +
        deviations['mouth_width_proportion'] * 1.2 +
        deviations['nose_width_proportion'] * 1.2 +
        deviations['eye_nose_angle'] / 180.0 +
        deviations['jaw_angle'] / 180.0
    ) / 8.0
    
    deviations['anomaly_score'] = float(anomaly_score)
    
    return deviations

def calculate_angle(p1, p2, p3):
    """
    Вычисляет угол между тремя точками (в градусах)
    
    Args:
        p1 (numpy.ndarray): Первая точка
        p2 (numpy.ndarray): Вторая точка (вершина угла)
        p3 (numpy.ndarray): Третья точка
        
    Returns:
        float: Угол в градусах
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Нормализуем векторы
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Вычисляем угол
    dot_product = np.dot(v1, v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def analyze_deviations(deviations, threshold=0.1):
    """
    Анализирует отклонения и определяет, являются ли они аномальными
    
    Args:
        deviations (dict): Словарь с отклонениями
        threshold (float): Пороговое значение для аномалий
        
    Returns:
        dict: Результаты анализа
    """
    analysis = {
        'is_anomalous': False,
        'anomaly_reasons': [],
        'anomaly_score': deviations.get('anomaly_score', 0.0)
    }
    
    # Проверяем различные критерии аномальности
    if deviations.get('normalized', 0.0) > threshold:
        analysis['is_anomalous'] = True
        analysis['anomaly_reasons'].append('high_normalized_deviation')
    
    if deviations.get('symmetry_diff', 0.0) > threshold * 1.5:
        analysis['is_anomalous'] = True
        analysis['anomaly_reasons'].append('high_symmetry_difference')
    
    if deviations.get('percent_large', 0.0) > 20.0:  # Если более 20% точек имеют большие отклонения
        analysis['is_anomalous'] = True
        analysis['anomaly_reasons'].append('many_large_deviations')
    
    if deviations.get('eye_width_proportion', 0.0) > threshold * 1.2:
        analysis['is_anomalous'] = True
        analysis['anomaly_reasons'].append('eye_proportion_mismatch')
    
    if deviations.get('mouth_width_proportion', 0.0) > threshold * 1.2:
        analysis['is_anomalous'] = True
        analysis['anomaly_reasons'].append('mouth_proportion_mismatch')
    
    if deviations.get('nose_width_proportion', 0.0) > threshold * 1.2:
        analysis['is_anomalous'] = True
        analysis['anomaly_reasons'].append('nose_proportion_mismatch')
    
    if deviations.get('eye_nose_angle', 0.0) > 10.0:  # Если угол отличается более чем на 10 градусов
        analysis['is_anomalous'] = True
        analysis['anomaly_reasons'].append('facial_angle_mismatch')
    
    if deviations.get('anomaly_score', 0.0) > threshold:
        analysis['is_anomalous'] = True
        analysis['anomaly_reasons'].append('high_overall_anomaly_score')
    
    # Определяем наиболее проблемные области
    region_deviations = {
        k: v for k, v in deviations.items() 
        if k.endswith('_deviation') and not k.startswith('symmetry')
    }
    
    if region_deviations:
        max_region = max(region_deviations.items(), key=lambda x: x[1])
        analysis['most_problematic_region'] = max_region[0].replace('_deviation', '')
        
        # Если отклонение в проблемной области значительно выше среднего
        if max_region[1] > deviations.get('total', 0.0) * 1.5:
            analysis['is_anomalous'] = True
            analysis['anomaly_reasons'].append(f'high_deviation_in_{analysis["most_problematic_region"]}')
    
    return analysis
