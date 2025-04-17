"""
Модуль для расчета геометрических метрик лица
"""

import numpy as np
from app.config import LANDMARKS_INDICES

def calculate_3ddfa_metrics(landmarks, pose_info):
    """
    Рассчитывает геометрические метрики на основе landmarks от 3DDFA
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        pose_info (dict): Информация о ракурсе
        
    Returns:
        dict: Словарь с метриками
    """
    metrics = {}
    
    # Получаем список активных метрик для данного ракурса
    from app.utils.pose import get_active_metrics
    active_metrics = get_active_metrics(pose_info)
    
    # Рассчитываем только активные метрики
    for metric_name in active_metrics:
        if metric_name == 'eye_distance':
            metrics[metric_name] = calculate_eye_distance(landmarks)
        elif metric_name == 'eye_ratio':
            metrics[metric_name] = calculate_eye_ratio(landmarks)
        elif metric_name == 'nose_width':
            metrics[metric_name] = calculate_nose_width(landmarks)
        elif metric_name == 'mouth_width':
            metrics[metric_name] = calculate_mouth_width(landmarks)
        elif metric_name == 'face_width':
            metrics[metric_name] = calculate_face_width(landmarks)
        elif metric_name == 'face_height':
            metrics[metric_name] = calculate_face_height(landmarks)
        elif metric_name == 'jaw_width':
            metrics[metric_name] = calculate_jaw_width(landmarks)
        elif metric_name == 'symmetry':
            metrics[metric_name] = calculate_symmetry(landmarks)
        elif metric_name == 'visible_eye_size':
            metrics[metric_name] = calculate_visible_eye_size(landmarks, pose_info)
        elif metric_name == 'nose_projection':
            metrics[metric_name] = calculate_nose_projection(landmarks)
        elif metric_name == 'cheek_contour':
            metrics[metric_name] = calculate_cheek_contour(landmarks, pose_info)
        elif metric_name == 'jaw_angle':
            metrics[metric_name] = calculate_jaw_angle(landmarks, pose_info)
        elif metric_name == 'face_depth':
            metrics[metric_name] = calculate_face_depth(landmarks)
        elif metric_name == 'jaw_line':
            metrics[metric_name] = calculate_jaw_line(landmarks, pose_info)
        elif metric_name == 'forehead_curve':
            metrics[metric_name] = calculate_forehead_curve(landmarks, pose_info)
    
    return metrics

def calculate_eye_distance(landmarks):
    """
    Рассчитывает расстояние между центрами глаз
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Расстояние между центрами глаз
    """
    # Индексы для центров глаз
    left_eye_indices = LANDMARKS_INDICES['left_eye']
    right_eye_indices = LANDMARKS_INDICES['right_eye']
    
    # Вычисляем центры глаз
    left_eye_center = np.mean(landmarks[left_eye_indices], axis=0)
    right_eye_center = np.mean(landmarks[right_eye_indices], axis=0)
    
    # Вычисляем расстояние между центрами глаз
    distance = np.linalg.norm(left_eye_center - right_eye_center)
    
    return float(distance)

def calculate_eye_ratio(landmarks):
    """
    Рассчитывает соотношение ширины и высоты глаз
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Среднее соотношение ширины и высоты глаз
    """
    # Индексы для глаз
    left_eye_indices = LANDMARKS_INDICES['left_eye']
    right_eye_indices = LANDMARKS_INDICES['right_eye']
    
    # Вычисляем соотношение для левого глаза
    left_eye = landmarks[left_eye_indices]
    left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
    left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5])
    left_eye_ratio = left_eye_width / left_eye_height if left_eye_height > 0 else 0
    
    # Вычисляем соотношение для правого глаза
    right_eye = landmarks[right_eye_indices]
    right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
    right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5])
    right_eye_ratio = right_eye_width / right_eye_height if right_eye_height > 0 else 0
    
    # Вычисляем среднее соотношение
    eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
    
    return float(eye_ratio)

def calculate_nose_width(landmarks):
    """
    Рассчитывает ширину носа
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Ширина носа
    """
    # Индексы для крыльев носа
    nose_indices = LANDMARKS_INDICES['nose_tip']
    
    # Вычисляем ширину носа
    nose_width = np.linalg.norm(landmarks[nose_indices[0]] - landmarks[nose_indices[4]])
    
    return float(nose_width)

def calculate_mouth_width(landmarks):
    """
    Рассчитывает ширину рта
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Ширина рта
    """
    # Индексы для уголков рта
    outer_lips_indices = LANDMARKS_INDICES['outer_lips']
    
    # Вычисляем ширину рта
    mouth_width = np.linalg.norm(landmarks[outer_lips_indices[0]] - landmarks[outer_lips_indices[6]])
    
    return float(mouth_width)

def calculate_face_width(landmarks):
    """
    Рассчитывает ширину лица
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Ширина лица
    """
    # Индексы для контура лица
    jaw_indices = LANDMARKS_INDICES['jaw']
    
    # Вычисляем ширину лица
    face_width = np.linalg.norm(landmarks[jaw_indices[0]] - landmarks[jaw_indices[16]])
    
    return float(face_width)

def calculate_face_height(landmarks):
    """
    Рассчитывает высоту лица
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Высота лица
    """
    # Индексы для верхней и нижней точек лица
    jaw_indices = LANDMARKS_INDICES['jaw']
    nose_bridge_indices = LANDMARKS_INDICES['nose_bridge']
    
    # Вычисляем высоту лица
    face_height = np.linalg.norm(landmarks[nose_bridge_indices[0]] - landmarks[jaw_indices[8]])
    
    return float(face_height)

def calculate_jaw_width(landmarks):
    """
    Рассчитывает ширину челюсти
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Ширина челюсти
    """
    # Индексы для челюсти
    jaw_indices = LANDMARKS_INDICES['jaw']
    
    # Вычисляем ширину челюсти
    jaw_width = np.linalg.norm(landmarks[jaw_indices[3]] - landmarks[jaw_indices[13]])
    
    return float(jaw_width)

def calculate_symmetry(landmarks):
    """
    Рассчитывает симметрию лица
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Коэффициент симметрии (0-1)
    """
    # Индексы для симметричных точек
    symmetric_pairs = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (48, 54), (49, 53), (50, 52), (59, 55), (58, 56), (57, 57)
    ]
    
    # Вычисляем центральную линию лица
    center_line = (landmarks[8] + landmarks[27]) / 2
    center_direction = landmarks[8] - landmarks[27]
    center_direction = center_direction / np.linalg.norm(center_direction)
    
    # Вычисляем отклонения от симметрии
    deviations = []
    for left_idx, right_idx in symmetric_pairs:
        left_point = landmarks[left_idx]
        right_point = landmarks[right_idx]
        
        # Проекция на центральную линию
        left_proj = np.dot(left_point - center_line, center_direction)
        right_proj = np.dot(right_point - center_line, center_direction)
        
        # Расстояние до центральной линии
        left_dist = np.linalg.norm(left_point - center_line - left_proj * center_direction)
        right_dist = np.linalg.norm(right_point - center_line - right_proj * center_direction)
        
        # Отклонение от симметрии
        deviation = abs(left_dist - right_dist) / ((left_dist + right_dist) / 2)
        deviations.append(deviation)
    
    # Вычисляем средний коэффициент симметрии
    symmetry = 1.0 - min(1.0, sum(deviations) / len(deviations))
    
    return float(symmetry)

def calculate_visible_eye_size(landmarks, pose_info):
    """
    Рассчитывает размер видимого глаза для профильных ракурсов
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        pose_info (dict): Информация о ракурсе
        
    Returns:
        float: Размер видимого глаза
    """
    # Определяем, какой глаз видим лучше в зависимости от ракурса
    main_pose = pose_info['main']
    
    if 'left' in main_pose:
        # Для левого профиля виден правый глаз
        eye_indices = LANDMARKS_INDICES['right_eye']
    else:
        # Для правого профиля виден левый глаз
        eye_indices = LANDMARKS_INDICES['left_eye']
    
    # Вычисляем размер глаза (площадь)
    eye_points = landmarks[eye_indices]
    eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
    eye_height = np.linalg.norm(eye_points[1] - eye_points[5])
    eye_size = eye_width * eye_height
    
    return float(eye_size)

def calculate_nose_projection(landmarks):
    """
    Рассчитывает проекцию носа (выступание)
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Проекция носа
    """
    # Индексы для носа и лица
    nose_tip_indices = LANDMARKS_INDICES['nose_tip']
    nose_bridge_indices = LANDMARKS_INDICES['nose_bridge']
    
    # Вычисляем проекцию носа (выступание)
    nose_tip = landmarks[nose_tip_indices[2]]
    nose_bridge = landmarks[nose_bridge_indices[0]]
    
    # Проекция в направлении Z (глубина)
    nose_projection = nose_tip[2] - nose_bridge[2]
    
    return float(nose_projection)

def calculate_cheek_contour(landmarks, pose_info):
    """
    Рассчитывает контур щеки для профильных ракурсов
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        pose_info (dict): Информация о ракурсе
        
    Returns:
        float: Мера выпуклости щеки
    """
    # Определяем, какая щека видна лучше в зависимости от ракурса
    main_pose = pose_info['main']
    
    if 'left' in main_pose:
        # Для левого профиля видна правая щека
        cheek_indices = [1, 2, 3, 4, 31, 32, 33]
    else:
        # Для правого профиля видна левая щека
        cheek_indices = [13, 14, 15, 35, 42, 47]
    
    # Вычисляем контур щеки (кривизну)
    cheek_points = landmarks[cheek_indices]
    
    # Находим плоскость, проходящую через точки
    centroid = np.mean(cheek_points, axis=0)
    _, _, vh = np.linalg.svd(cheek_points - centroid)
    normal = vh[2]
    
    # Вычисляем отклонения от плоскости
    distances = np.abs(np.dot(cheek_points - centroid, normal))
    
    # Мера выпуклости - среднее отклонение от плоскости
    cheek_contour = np.mean(distances)
    
    return float(cheek_contour)

def calculate_jaw_angle(landmarks, pose_info):
    """
    Рассчитывает угол челюсти для профильных ракурсов
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        pose_info (dict): Информация о ракурсе
        
    Returns:
        float: Угол челюсти в градусах
    """
    # Определяем точки челюсти в зависимости от ракурса
    main_pose = pose_info['main']
    
    if 'left' in main_pose:
        # Для левого профиля
        jaw_points = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        # Для правого профиля
        jaw_points = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Вычисляем угол челюсти
    if len(jaw_points) >= 3:
        p1 = landmarks[jaw_points[0]]
        p2 = landmarks[jaw_points[len(jaw_points) // 2]]
        p3 = landmarks[jaw_points[-1]]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Нормализуем векторы
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Вычисляем угол между векторами
        cos_angle = np.dot(v1, v2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        
        return float(angle)
    else:
        return 0.0

def calculate_face_depth(landmarks):
    """
    Рассчитывает глубину лица (разница между передней и задней точками)
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        
    Returns:
        float: Глубина лица
    """
    # Вычисляем глубину лица (разница по Z)
    z_values = landmarks[:, 2]
    face_depth = np.max(z_values) - np.min(z_values)
    
    return float(face_depth)

def calculate_jaw_line(landmarks, pose_info):
    """
    Рассчитывает линию челюсти для профильных ракурсов
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        pose_info (dict): Информация о ракурсе
        
    Returns:
        float: Мера прямолинейности челюсти
    """
    # Определяем точки челюсти в зависимости от ракурса
    main_pose = pose_info['main']
    
    if 'left' in main_pose:
        # Для левого профиля
        jaw_points = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        # Для правого профиля
        jaw_points = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Вычисляем линию челюсти
    jaw_landmarks = landmarks[jaw_points]
    
    # Находим линию наилучшего приближения
    centroid = np.mean(jaw_landmarks, axis=0)
    _, _, vh = np.linalg.svd(jaw_landmarks - centroid)
    direction = vh[0]
    
    # Вычисляем отклонения от линии
    line_points = centroid + np.outer(np.dot(jaw_landmarks - centroid, direction), direction)
    distances = np.linalg.norm(jaw_landmarks - line_points, axis=1)
    
    # Мера прямолинейности - среднее отклонение от линии
    jaw_line = 1.0 / (1.0 + np.mean(distances))
    
    return float(jaw_line)

def calculate_forehead_curve(landmarks, pose_info):
    """
    Рассчитывает кривизну лба для профильных ракурсов
    
    Args:
        landmarks (numpy.ndarray): Массив landmarks размером (68, 3)
        pose_info (dict): Информация о ракурсе
        
    Returns:
        float: Мера кривизны лба
    """
    # Определяем точки лба в зависимости от ракурса
    main_pose = pose_info['main']
    
    if 'left' in main_pose:
        # Для левого профиля
        forehead_points = [17, 18, 19, 20, 21]
    else:
        # Для правого профиля
        forehead_points = [22, 23, 24, 25, 26]
    
    # Вычисляем кривизну лба
    forehead_landmarks = landmarks[forehead_points]
    
    if len(forehead_landmarks) >= 3:
        # Находим окружность, проходящую через точки
        centroid = np.mean(forehead_landmarks, axis=0)
        _, s, _ = np.linalg.svd(forehead_landmarks - centroid)
        
        # Мера кривизны - отношение сингулярных значений
        if s[1] > 0:
            forehead_curve = s[0] / s[1]
        else:
            forehead_curve = 1.0
    else:
        forehead_curve = 1.0
    
    return float(forehead_curve)
