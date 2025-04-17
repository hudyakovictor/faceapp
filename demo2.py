import warnings  # Импортируем модуль
import numpy as np
from metrics import add_metrics
from logger import logger, log_exception, log_function_call, LogEmoji, log_success, log_processing, log_face, log_metrics, log_save, log_time

# Отключаем все предупреждения
warnings.filterwarnings("ignore", category=Warning)


# Настройка для подавления вывода
import os
import sys
import io
import functools
import numpy
from core import compute_anomaly_score_v2
from analysis import run_anomaly_analysis_and_append

import sys
import argparse
import cv2
import yaml
import warnings
import json
import os
from PIL import Image
from datetime import datetime

# Отключаем сообщения от различных библиотек
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Контекстный менеджер для подавления вывода
class SuppressOutput:
    """
    Контекстный менеджер для подавления любого вывода в stdout и stderr.
    Перехватывает даже вывод из C/C++ кода.
    """
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.stdout_fd = None
        self.stderr_fd = None
        self.stdout_copy = None
        self.stderr_copy = None
        self.null_fd = None
        
    def __enter__(self):
        # Открываем null-устройство
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        
        # Сохраняем оригинальные файловые дескрипторы
        if self.suppress_stdout:
            self.stdout_fd = 1  # stdout file descriptor
            self.stdout_copy = os.dup(self.stdout_fd)
            os.dup2(self.null_fd, self.stdout_fd)
            
        if self.suppress_stderr:
            self.stderr_fd = 2  # stderr file descriptor
            self.stderr_copy = os.dup(self.stderr_fd)
            os.dup2(self.null_fd, self.stderr_fd)
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Восстанавливаем оригинальные файловые дескрипторы
        if self.suppress_stdout:
            os.dup2(self.stdout_copy, self.stdout_fd)
            os.close(self.stdout_copy)
            
        if self.suppress_stderr:
            os.dup2(self.stderr_copy, self.stderr_fd)
            os.close(self.stderr_copy)
            
        # Закрываем null-устройство
        os.close(self.null_fd)

# Декоратор для подавления вывода
def suppress_stdout(func):
    """
    Декоратор для подавления вывода функции.
    Перехватывает весь вывод, включая вывод из C/C++ кода.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with SuppressOutput():
            return func(*args, **kwargs)
    return wrapper

# Импортируем библиотеки с подавлением вывода
with SuppressOutput():
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





#-------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- ФУНКЦИИ -----------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------







@log_function_call
def load_image(img_path):
    """
    Загружает изображение по заданному пути.
    Возвращает изображение в формате BGR (cv2).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {img_path}")
    return img









@log_function_call
def extract_date_from_filename(filename):
    """
    Извлекает дату из имени файла в формате dd_mm_yy.jpg или dd_mm_yy-1.jpg
    
    Args:
        filename (str): Имя файла
        
    Returns:
        dict: Словарь с датой в разных форматах или None, если дата не найдена
    """
    import re
    from datetime import datetime
    
    pattern = r'(\d{2})_(\d{2})_(\d{2})(?:-(\d+))?\.jpg'
    match = re.search(pattern, filename)
    
    if match:
        day, month, year, photo_number = match.groups()
        photo_number = int(photo_number) if photo_number else 1
        
        # Предполагаем, что год в формате 20XX
        year_full = f"20{year}"
        date_str = f"{year_full}-{month}-{day}"
        
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            return {
                "day": int(day),
                "month": int(month),
                "year": int(year_full),
                "date_string": date_str,
                "datetime": date_obj,
                "photo_number": photo_number,
                "timestamp": int(date_obj.timestamp())
            }
        except ValueError:
            logger.error(f"{LogEmoji.ERROR} Некорректная дата в имени файла: {filename}")
    
    return None








@log_function_call
def analyze_date_intervals(file_dates):
    """
    Анализирует промежутки между датами
    
    Args:
        file_dates (list): Список словарей с датами, полученных из extract_date_from_filename
        
    Returns:
        dict: Словарь с метриками временных промежутков
    """
    if not file_dates or len(file_dates) < 2:
        return {"intervals": [], "average_interval": 0, "max_interval": 0, "min_interval": 0}
    
    # Сортируем даты
    sorted_dates = sorted(file_dates, key=lambda x: x["timestamp"])
    
    intervals = []
    total_days = 0
    
    for i in range(1, len(sorted_dates)):
        current_date = sorted_dates[i]["datetime"]
        prev_date = sorted_dates[i-1]["datetime"]
        
        days_diff = (current_date - prev_date).days
        
        intervals.append({
            "from_date": sorted_dates[i-1]["date_string"],
            "to_date": sorted_dates[i]["date_string"],
            "days": days_diff
        })
        
        total_days += days_diff
    
    avg_interval = total_days / len(intervals) if intervals else 0
    max_interval = max([i["days"] for i in intervals]) if intervals else 0
    min_interval = min([i["days"] for i in intervals]) if intervals else 0
    
    return {
        "intervals": intervals,
        "average_interval": avg_interval,
        "max_interval": max_interval,
        "min_interval": min_interval,
        "total_photos": len(file_dates),
        "date_range": {
            "from": sorted_dates[0]["date_string"],
            "to": sorted_dates[-1]["date_string"],
            "total_days": (sorted_dates[-1]["datetime"] - sorted_dates[0]["datetime"]).days
        }
    }









@log_function_call
def calculate_is_anomalous(shape_error, deviations, texture_uniformity):
    """
    Расчет флага is_anomalous на основе трех условий
    
    Args:
        shape_error (float): Ошибка формы от 3DDFA
        deviations (dict): Отклонения между FAN и 3DDFA
        texture_uniformity (float): Однородность текстуры лица
        
    Returns:
        tuple: (is_anomalous, anomaly_reasons) - флаг аномалии и причины
    """
    # Пороговые значения
    SHAPE_ERROR_THRESHOLD = 0.25
    DEVIATION_THRESHOLD = 20  # процент расхождения
    TEXTURE_UNIFORMITY_THRESHOLD = 0.85  # высокая однородность
    
    # Проверяем shape_error
    shape_error_anomaly = shape_error > SHAPE_ERROR_THRESHOLD
    
    # Проверяем расхождения между FAN и 3DDFA
    if not deviations:
        deviation_anomaly = False
    else:
        # Вычисляем процент расхождения
        total_points = len(deviations)
        deviation_count = sum(1 for d in deviations.values() if d > 5.0)
        deviation_percent = (deviation_count / total_points) * 100 if total_points else 0
        deviation_anomaly = deviation_percent > DEVIATION_THRESHOLD
    
    # Проверяем однородность текстуры
    texture_anomaly = texture_uniformity > TEXTURE_UNIFORMITY_THRESHOLD
    
    # Финальное решение - аномалия, если хотя бы одно условие выполняется
    is_anomalous = shape_error_anomaly or deviation_anomaly or texture_anomaly
    
    # Детальная информация о причинах
    anomaly_reasons = {
        "shape_error": shape_error_anomaly,
        "deviations": deviation_anomaly,
        "texture": texture_anomaly
    }
    
    return is_anomalous, anomaly_reasons








@log_function_call
def init_insightface():
    """
    Инициализирует модели InsightFace
    
    Returns:
        tuple: (face_recog, age_model) модели для распознавания лиц и оценки возраста
    """
    try:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        
        # Инициализация модели для распознавания лиц
        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Инициализация модели для оценки возраста
        age_model = get_model('buffalo_l', providers=['CPUExecutionProvider'])
        age_model.prepare(ctx_id=0)
        
        logger.info(f"{LogEmoji.PROCESSING} Модели InsightFace успешно инициализированы")
        return face_app, age_model
    except Exception as e:
        log_exception(e, "Ошибка при инициализации InsightFace")
        return None, None




@log_function_call
def extract_insightface_features(img, face_app):
    """
    Извлекает 512-мерный вектор embedding и другие признаки с помощью InsightFace
    
    Args:
        img (numpy.ndarray): Изображение лица
        face_app: Модель InsightFace
        
    Returns:
        dict: Словарь с embedding, возрастом и полом или None при ошибке
    """
    if face_app is None:
        return None
    
    try:
        # Преобразуем BGR в RGB для InsightFace
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Получаем результаты анализа
        faces = face_app.get(img_rgb)
        
        if not faces:
            logger.warning(f"{LogEmoji.WARNING} InsightFace не обнаружил лицо на изображении")
            return None
        
        # Берем первое (или самое большое) лицо
        face = sorted(faces, key=lambda x: x.bbox[2] - x.bbox[0], reverse=True)[0]
        
        # Извлекаем embedding
        embedding = face.embedding.tolist() if hasattr(face, 'embedding') else None
        
        # Получаем возраст и пол
        age = int(face.age) if hasattr(face, 'age') else None
        gender = face.gender if hasattr(face, 'gender') else None
        
        logger.info(f"{LogEmoji.FACE} InsightFace: возраст={age}, пол={gender}")
        return {
            "embedding": embedding,
            "age": age,
            "gender": gender
        }
    except Exception as e:
        log_exception(e, "Ошибка при извлечении признаков InsightFace")
        return None





@log_function_call
def get_putin_age_by_date(date_info):
    """
    Получает возраст Путина на определенную дату
    
    Args:
        date_info (dict): Словарь с информацией о дате
        
    Returns:
        float: Возраст Путина на указанную дату
    """
    from datetime import datetime
    
    if not date_info:
        return None
    
    # Дата рождения Путина - 7 октября 1952 года
    putin_birth_date = datetime(1952, 10, 7)
    
    # Вычисляем возраст на дату фотографии
    photo_date = date_info["datetime"]
    
    # Полных лет
    years_diff = photo_date.year - putin_birth_date.year
    
    # Корректировка, если день рождения в этом году еще не наступил
    if photo_date.month < putin_birth_date.month or \
       (photo_date.month == putin_birth_date.month and photo_date.day < putin_birth_date.day):
        years_diff -= 1
    
    # Вычисляем десятичную часть возраста (доли года)
    if photo_date.month > putin_birth_date.month or \
       (photo_date.month == putin_birth_date.month and photo_date.day >= putin_birth_date.day):
        # После дня рождения в текущем году
        days_since_birthday = (photo_date - datetime(photo_date.year, putin_birth_date.month, putin_birth_date.day)).days
    else:
        # До дня рождения в текущем году
        days_since_birthday = (photo_date - datetime(photo_date.year-1, putin_birth_date.month, putin_birth_date.day)).days
    
    # Примерное количество дней в году (учитываем високосные годы)
    days_in_year = 366 if (photo_date.year % 4 == 0 and photo_date.year % 100 != 0) or photo_date.year % 400 == 0 else 365
    
    decimal_part = days_since_birthday / days_in_year
    
    return years_diff + decimal_part






@log_function_call
def calculate_age_correlation(estimated_age, putin_age):
    """
    Расчет корреляции между оцененным возрастом и реальным возрастом Путина
    
    Args:
        estimated_age (float): Возраст, определенный по фотографии
        putin_age (float): Реальный возраст Путина на дату фотографии
        
    Returns:
        dict: Словарь с метриками корреляции
    """
    if estimated_age is None or putin_age is None:
        return {
            "correlation": 0.0,
            "age_difference": None,
            "is_plausible": False
        }
    
    # Вычисляем разницу в возрасте
    age_diff = abs(estimated_age - putin_age)
    
    # Вычисляем корреляцию (обратно пропорциональную разнице)
    # Чем меньше разница, тем выше корреляция
    max_acceptable_diff = 15.0  # максимально допустимая разница
    
    if age_diff <= max_acceptable_diff:
        correlation = 1.0 - (age_diff / max_acceptable_diff)
    else:
        correlation = 0.0
    
    # Определяем, является ли возраст правдоподобным
    is_plausible = age_diff <= 10.0
    
    return {
        "correlation": correlation,
        "age_difference": age_diff,
        "is_plausible": is_plausible,
        "estimated_age": estimated_age,
        "actual_age": putin_age
    }






@log_function_call
def build_structured_json(landmarks_data, embedding_data, date_data, texture_data=None):
    """
    Структурирует JSON согласно требованиям ТЗ
    
    Args:
        landmarks_data (dict): Данные ключевых точек и метрик
        embedding_data (dict): Данные embedding и возраста от InsightFace
        date_data (dict): Информация о дате фотографии
        texture_data (dict, optional): Данные текстурного анализа
        
    Returns:
        dict: Структурированный JSON
    """
    # Получаем данные о ракурсе и углах
    pose_type = landmarks_data.get('pose_type', 'unknown')
    pose_angles = landmarks_data.get('pose_angles', {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0})
    
    # Блок 1: параметры от 3DDFA
    block_1 = {
        "pose_type": pose_type,
        "pose_angles": pose_angles,
        "shape_error": landmarks_data.get('original_shape_error', 0.0),
        "landmarks_3ddfa": landmarks_data.get('3ddfa', []),
        "landmarks_fan": landmarks_data.get('fan', []),
        "landmarks_mediapipe": landmarks_data.get('mediapipe', [])
    }
    
    # Блок 2: метрики с confidence
    # Фильтруем метрики по confidence и добавляем префиксы
    all_metrics = landmarks_data.get('metrics', {})
    block_2 = {}
    
    for metric_name, value in all_metrics.items():
        confidence = landmarks_data.get('confidence', {}).get(metric_name, 1.0)
        
        # Определяем префикс по источнику метрики
        if metric_name.startswith('fn_'):
            prefix = f"fan_{pose_type}"
        elif metric_name.startswith('MP_'):
            prefix = f"mp_{pose_type}"
        else:
            prefix = f"3ddfa_{pose_type}"
        
        # Добавляем метрику с confidence
        block_2[f"{prefix}_{metric_name}"] = {
            "value": value,
            "confidence": confidence
        }
    
    # Блок 3: отклонения между FAN и 3DDFA
    block_3 = landmarks_data.get('deviations', {})
    
    # Блок 4: текстурные признаки
    block_4 = texture_data or {}
    
    # Блок 5: дополнительная информация (оставляем пустым)
    block_5 = {}
    
    # Блок 6: embedding и признаки из InsightFace
    block_6 = {}
    
    if embedding_data:
        # Добавляем embedding
        block_6["embedding"] = embedding_data.get("embedding", [])
        
        # Добавляем возраст
        block_6["age"] = embedding_data.get("age", None)
        
        # Добавляем корреляцию с возрастом Путина, если есть информация о дате
        if date_data and embedding_data.get("age") is not None:
            putin_age = get_putin_age_by_date(date_data)
            age_correlation = calculate_age_correlation(embedding_data.get("age"), putin_age)
            block_6["age_correlation"] = age_correlation
    
    # Добавляем флаг is_anomalous
    is_anomalous, anomaly_reasons = calculate_is_anomalous(
        landmarks_data.get('original_shape_error', 0.0),
        block_3,
        block_4.get('texture_uniformity', 0.0)
    )
    
    # Собираем финальный JSON
    final_json = {
        "version": "1.0",
        "filename": landmarks_data.get('filename', ''),
        "date_info": date_data,
        "is_anomalous": is_anomalous,
        "anomaly_reasons": anomaly_reasons,
        "block_1_3ddfa": block_1,
        "block_2_metrics": block_2,
        "block_3_deviations": block_3,
        "block_4_texture": block_4,
        "block_5_additional": block_5,
        "block_6_insightface": block_6
    }
    
    return final_json












@log_function_call
def detect_faces_3ddfa(img, face_boxes):
    """
    Детектирует лица на изображении с помощью FaceBoxes (для 3DDFA).
    Возвращает список ROI боксов.
    """
    boxes = face_boxes(img)
    if not boxes or len(boxes) == 0:
        raise ValueError("Лицо не найдено на изображении.")
    return boxes

@suppress_stdout
@log_function_call
def estimate_pose_3ddfa(param):
    """
    Возвращает углы наклона головы (yaw, pitch, roll) из параметров 3DDFA.
    Args:
        param: параметры 3DMM от 3DDFA
    Returns:
        tuple: (yaw, pitch, roll) в градусах
    """
    try:
        if param is None:
            return 0.0, 0.0, 0.0
        
        # Преобразуем в numpy array если нужно
        if isinstance(param, list):
            param = numpy.array(param)
            
        # Получаем углы из параметров через calc_pose
        with SuppressOutput():
            pose_values = calc_pose(param)
            
        # Преобразуем в numpy array если нужно
        if not isinstance(pose_values, numpy.ndarray):
            pose_values = numpy.array(pose_values)
            
        # Получаем значения углов
        pose_values = pose_values.ravel()
        
        if len(pose_values) >= 3:
            return float(pose_values[0]), float(pose_values[1]), float(pose_values[2])
        elif len(pose_values) == 2:
            return float(pose_values[0]), float(pose_values[1]), 0.0
        else:
            logger.warning(f"{LogEmoji.WARNING} Неожиданный формат данных поворота")
            return 0.0, 0.0, 0.0
            
    except Exception as e:
        log_exception(e, "Ошибка при расчете углов поворота")
        return 0.0, 0.0, 0.0


@log_function_call
def classify_pose_type(yaw):
    """
    Классифицирует ракурс головы по углу yaw
    Args:
        yaw (float): Угол поворота головы по yaw в градусах
    Returns:
        str: Тип ракурса ('frontal', 'profile_left', 'profile_right', 'semi_left', 'semi_right')
    """
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



# ------------------------------------------------------------------------------------------------------------------------

def analyze_normal_surface(normal_map, landmarks):
    """
    Анализирует микроперепады по карте нормалей для выявления масок
    
    Args:
        normal_map: карта нормалей
        landmarks: ключевые точки лица
    
    Returns:
        normal_zones: словарь с метриками нормалей по зонам
    """
    # Определяем зоны лица по индексам ключевых точек
    zones = {
        'jaw': list(range(0, 17)),
        'eyebrow_right': list(range(17, 22)),
        'eyebrow_left': list(range(22, 27)),
        'nose': list(range(27, 36)),
        'eye_right': list(range(36, 42)),
        'eye_left': list(range(42, 48)),
        'mouth': list(range(48, 68))
    }
    
    normal_zones = {}
    
    for zone_name, indices in zones.items():
        zone_normals = []
        
        for idx in indices:
            if idx < len(landmarks):
                x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                
                # Проверяем, что координаты в пределах изображения
                if 0 <= x < normal_map.shape[1] and 0 <= y < normal_map.shape[0]:
                    # Проверяем, что нормаль определена
                    if numpy.any(normal_map[y, x]):
                        zone_normals.append(normal_map[y, x])
        
        if zone_normals:
            zone_normals_array = numpy.array(zone_normals)
            
            # Вычисляем среднюю нормаль для зоны
            mean_normal = numpy.mean(zone_normals_array, axis=0)
            normal_length = numpy.linalg.norm(mean_normal)
            if normal_length > 0:
                mean_normal = mean_normal / normal_length
            
            # Вычисляем угол между нормалями и средней нормалью
            dot_products = numpy.sum(zone_normals_array * mean_normal, axis=1)
            dot_products = numpy.clip(dot_products, -1.0, 1.0)
            angles = numpy.arccos(dot_products)
            
            # Вычисляем метрики
            normal_zones[zone_name] = {
                'mean_normal': mean_normal.tolist(),
                'mean_angle': float(numpy.mean(angles)),
                'std_angle': float(numpy.std(angles)),
                'max_angle': float(numpy.max(angles)),
                'smoothness': float(1.0 - numpy.mean(angles) / numpy.pi)
            }
    
    # Вычисляем метрики для выявления масок
    
    # 1. Гладкость поверхности (маски обычно более гладкие)
    if 'jaw' in normal_zones:
        jaw_smoothness = normal_zones['jaw']['smoothness']
        
        # Значение близкое к 1 указывает на очень гладкую поверхность (возможно маска)
        normal_zones['mask_probability_smoothness'] = float(jaw_smoothness)
    
    # 2. Согласованность нормалей (маски имеют более согласованные нормали)
    if 'jaw' in normal_zones and 'nose' in normal_zones:
        jaw_std = normal_zones['jaw']['std_angle']
        nose_std = normal_zones['nose']['std_angle']
        
        # Если стандартное отклонение углов на подбородке намного меньше, чем на носу,
        # это может указывать на маску
        if nose_std > 0:
            normal_zones['mask_probability_consistency'] = float(1.0 - jaw_std / nose_std)
    
    return normal_zones






# ------------------------------------------------------------------------------------------------------------------------

@log_function_call



def param2lm(param, roi_box):
    """Преобразует параметры 3DMM в лицевые ландмарки.
    Args:
        param: параметры 3DMM
        roi_box: прямоугольник области интереса [sx, sy, ex, ey]
    Returns:
        landmarks: массив numpy размером (68, 2) с координатами ландмарок
    """
    R, offset, alpha_shp, alpha_exp = _parse_param(param)
    pts3d = R @ alpha_shp + offset
    return similar_transform(pts3d, roi_box, size=120)




# ------------------------------------------------------------------------------------------------------------------------

@log_function_call
def extract_shape_error_from_param(param):
    """
    Извлекает shape error из параметров 3DDFA
    Args:
        param: numpy array или список параметров 3DDFA
    Returns:
        float: значение shape error
    """
    try:
        if param is not None:
            # Преобразуем в numpy array если это еще не сделано
            if isinstance(param, list):
                param_array = numpy.array(param)
            else:
                param_array = param
                
            # Проверяем, что это не пустой массив
            if param_array.size > 0:
                # Берем последний элемент и преобразуем в float
                last_element = param_array[-1]
                if isinstance(last_element, list) or isinstance(last_element, numpy.ndarray):
                    return float(last_element[0]) if len(last_element) > 0 else 0.0
                else:
                    return float(last_element)
    except Exception as e:
        log_exception(e, "Ошибка при извлечении shape error")
    return 0.0





# ------------------------------------------------------------------------------------------------------------------------

@log_function_call
def get_shape_error_status(shape_error):
    """
    Определяет статус shape error
    Args:
        shape_error (float): значение shape error
    Returns:
        str: 'normal' или 'anomaly'
    """
    # Пороговые значения для определения аномалий
    SHAPE_ERROR_THRESHOLD = 0.1
    
    if shape_error is None:
        return 'unknown'
    
    if abs(shape_error) > SHAPE_ERROR_THRESHOLD:
        return 'anomaly'
    
    return 'normal'


@log_function_call
def draw_wireframe(img, ver_lst):
    # Индексы ключевых соединений на основе 68 точек
    connections = [
        list(range(0, 17)), # контур лица
        list(range(17, 22)), # бровь левая
        list(range(22, 27)), # бровь правая
        list(range(27, 31)), # переносица
        list(range(31, 36)), # нижняя часть носа
        [36, 37, 38, 39, 40, 41, 36], # левый глаз
        [42, 43, 44, 45, 46, 47, 42], # правый глаз
        list(range(48, 60)) + [48], # внешний контур рта
        list(range(60, 68)) + [60], # внутренний контур рта
    ]
    
    for ver in ver_lst:
        pts = ver[:2].T.astype(numpy.int32)
        for group in connections:
            for i in range(len(group) - 1):
                pt1 = tuple(pts[group[i]])
                pt2 = tuple(pts[group[i + 1]])
                cv2.line(img, pt1, pt2, (255, 255, 255), 2)




# ------------------------------------------------------------------------------------------------------------------------

@log_function_call
def get_largest_face(boxes):
    """
    Выбирает самое большое лицо из обнаруженных
    Args:
        boxes: список боксов в формате [x1, y1, x2, y2, score, ...]
    Returns:
        tuple: (box, idx) где box - координаты найденного бокса, idx - его индекс
    """
    if not boxes or len(boxes) == 0:
        return None, -1
    
    max_area = 0
    largest_idx = -1
    largest_box = None
    
    # Проверяем, является ли boxes списком списков
    if isinstance(boxes[0], list):
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = map(float, box[:4]) # Преобразуем явно в float
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_idx = i
                    largest_box = box
            except (ValueError, IndexError):
                continue
    else:
        # Обработка плоского списка
        for i in range(0, len(boxes), 5):
            try:
                x1, y1, x2, y2 = map(float, boxes[i:i+4]) # Преобразуем явно в float
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_idx = i // 5
                    largest_box = boxes[i:i+5]
            except (ValueError, IndexError):
                continue
    
    return largest_box, largest_idx
























# ------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- M A I N -------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

@log_function_call
def main(args):
    logger.info(f"{LogEmoji.CAMERA} Начало обработки изображения: {args.img_fp}")
    
    try:
        start_time = datetime.now()
        
        cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        logger.debug(f"{LogEmoji.PROCESSING} Загружена конфигурация из {args.config}")
        
        
      
 # ------------------------------------------------------------------------------------------------------------------------  
        

        # Инициализация FaceBoxes и TDDFA
        if args.onnx:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX
            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**cfg)
            logger.debug(f"{LogEmoji.PROCESSING} Инициализирован ONNX режим")
        else:
            gpu_mode = args.mode == 'gpu'
            tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
            face_boxes = FaceBoxes()
            logger.debug(f"{LogEmoji.PROCESSING} Инициализирован {'GPU' if gpu_mode else 'CPU'} режим")
        
        # Загрузка изображения
        img = cv2.imread(args.img_fp)
        if img is None:
            raise FileNotFoundError(f"{LogEmoji.ERROR} Не удалось загрузить изображение: {args.img_fp} <br>")
        
      
      
        
 # ------------------------------------------------------------------------------------------------------------------------ 
        
        # Копируем оригинальное изображение в папку results
        original_name = os.path.basename(args.img_fp)
        original_path = os.path.join('examples/results', original_name)
        cv2.imwrite(original_path, img)
        log_save(f'Сохранено оригинальное изображение: {original_path}')
        
        fa = face_alignment.FaceAlignment('2D', device='cpu', flip_input=False)
        
        # Инициализация mediapipe
        face_mesh = mediapipe.solutions.face_mesh
        drawing = mediapipe.solutions.drawing_utils
        face_detector = face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False
        )
        
        
# ------------------------------------------------------------------------------------------------------------------------       
        
        
        
        # Обнаружение лиц и получение параметров 3DMM и ROI
        boxes = face_boxes(img)
        n = len(boxes)
        if n == 0:
            logger.error(f"{LogEmoji.ERROR} Лицо не обнаружено в {args.img_fp}")
            return
        
        log_face(f"Обнаружено {n} лиц в изображении")
        param_lst, roi_box_lst = tddfa(img, boxes)
        
        # Визуализация и сериализация
        dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
        old_suffix = get_suffix(args.img_fp)
        new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'
        wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix
        
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        
        if args.opt == '2d_sparse':
            # Скопируем изображение для отрисовки
            h, w = img.shape[:2]
            debug_img = numpy.zeros((h, w, 4), dtype=numpy.uint8)
            mp_overlay = numpy.zeros((h, w, 3), dtype=numpy.uint8)
            debug_img[..., :3] = numpy.maximum(debug_img[..., :3], mp_overlay)
            alpha_mask = numpy.any(debug_img[..., :3] > 0, axis=-1)
            
            # Получение MediaPipe landmarks
            input_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.process(input_img_rgb)
            
            if results.multi_face_landmarks:
                mp_overlay = numpy.zeros((h, w, 3), dtype=numpy.uint8)
                for face_landmarks in results.multi_face_landmarks:
                    mp_pts = []
                    for lm in face_landmarks.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        mp_pts.append((x, y))
                        cv2.circle(debug_img, (x, y), 2, (255, 255, 255, 128), -1) # точки — белые, прозрачные, на переднем плане
                    
                    for start_idx, end_idx in face_mesh.FACEMESH_TESSELATION:
                        if start_idx < len(mp_pts) and end_idx < len(mp_pts):
                            pt1 = mp_pts[start_idx]
                            pt2 = mp_pts[end_idx]
                            cv2.line(mp_overlay, pt1, pt2, (255, 0, 0), thickness=1) # линии — чисто красные
                
                debug_img[..., :3] = numpy.maximum(debug_img[..., :3], mp_overlay)
                alpha_mask = numpy.any(debug_img[..., :3] > 0, axis=-1)
                debug_img[..., 3] = numpy.where(alpha_mask, 255, 0)
       
       
       
 # ------------------------------------------------------------------------------------------------------------------------      
            
            # Получение FAN landmarks
            preds_fan = fa.get_landmarks(input_img_rgb)
            
            if preds_fan:
                for (x, y) in preds_fan[0]:
                    cv2.circle(debug_img, (int(x), int(y)), 4, (0, 255, 0, 255), -1)
                
                fan_connections = [
                    list(range(0, 17)),
                    list(range(17, 22)),
                    list(range(22, 27)),
                    list(range(27, 31)),
                    list(range(31, 36)),
                    [36, 37, 38, 39, 40, 41, 36],
                    [42, 43, 44, 45, 46, 47, 42],
                    list(range(48, 60)) + [48],
                    list(range(60, 68)) + [60],
                ]
                
                fan_pts = numpy.array(preds_fan[0], dtype=numpy.int32)
                for group in fan_connections:
                    for i in range(len(group) - 1):
                        pt1 = tuple(fan_pts[group[i]])
                        pt2 = tuple(fan_pts[group[i + 1]])
                        cv2.line(debug_img, pt1, pt2, (0, 255, 0, 255), 2)
            
            # Wireframe 3DDFA
            wireframe_overlay = numpy.zeros((h, w, 3), dtype=numpy.uint8)
            draw_wireframe(wireframe_overlay, ver_lst)
            debug_img[..., :3] = numpy.maximum(debug_img[..., :3], wireframe_overlay)
            alpha_mask = numpy.any(debug_img[..., :3] > 0, axis=-1)
            debug_img[..., 3] = numpy.where(alpha_mask, 255, 0)
            
            
            
 # ------------------------------------------------------------------------------------------------------------------------           
                
            # Сохранение landmarks в JSON
            landmarks_data = {
                '3ddfa': [ver[:2].T.astype(int).tolist() for ver in ver_lst],
                'fan': [preds_fan[0].tolist()] if preds_fan else [],
                'mediapipe': []
            }
            
            if param_lst:
                shape_error = extract_shape_error_from_param(param_lst[0])
                landmarks_data['shape_error'] = {
                    'value': float(shape_error),
                    'status': get_shape_error_status(shape_error)
                }
                
                landmarks_data['3ddfa_param'] = param_lst[0].tolist()
                # Добавляем оригинальный shape error от 3DDFA
                landmarks_data['original_shape_error'] = extract_shape_error_from_param(param_lst[0])
            
            if roi_box_lst:
                landmarks_data['3ddfa_roi'] = list(map(int, roi_box_lst[0]))
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_points = []
                    for lm in face_landmarks.landmark:
                        x = int(lm.x * img.shape[1])
                        y = int(lm.y * img.shape[0])
                        face_points.append([x, y])
                    landmarks_data['mediapipe'].append(face_points)
            
                     
            
# ------------------------------------------------------------------------------------------------------------------------
           
            run_anomaly_analysis_and_append(landmarks_data, img, ver_lst, tddfa.tri)
            
            # Подготовка точек для расчёта аномалии
            if landmarks_data.get('3ddfa') and landmarks_data.get('fan'):
                try:
                    logger.debug(f"{LogEmoji.PROCESSING} Расчет показателя аномалии")
                    pts_3ddfa = numpy.array(landmarks_data['3ddfa'][0])[:68]
                    pts_fan = numpy.array(landmarks_data['fan'][0])[:68]
                    yaw, pitch, roll = estimate_pose_3ddfa(landmarks_data.get('3ddfa_param'))
                    logger.debug(f"{LogEmoji.FACE} Углы головы: yaw={yaw}, pitch={pitch}, roll={roll}")
                    
                    # Центр и масштаб
                    roi_box = landmarks_data.get('3ddfa_roi', [0, 0, img.shape[1], img.shape[0]])
                    x1, y1, x2, y2 = map(int, roi_box)
                    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    face_width = max(1, x2 - x1)
                    scale = face_width / 120.0
                    logger.debug(f"{LogEmoji.FACE} Центр лица: {face_center}, масштаб: {scale}")
                    
                    if len(pts_3ddfa) == 68 and len(pts_fan) == 68:
                        logger.debug(f"{LogEmoji.PROCESSING} Вызов compute_anomaly_score_v2")
                        anomaly = compute_anomaly_score_v2(
                            fan_pts=pts_fan,
                            ddfa_pts=pts_3ddfa,
                            mp_pts=pts_fan, # Используем FAN точки вместо MediaPipe
                            yaw=yaw,
                            pitch=pitch,
                            roll=roll,
                            shape_error_ddfa=float(landmarks_data.get('shape_error_ddfa', 0.1)),
                            face_center=face_center,
                            scale=scale
                        )
                        
                        # Проверяем тип anomaly и извлекаем значение
                        logger.debug(f"{LogEmoji.METRICS} Тип anomaly: {type(anomaly)}")
                        if isinstance(anomaly, dict):
                            anomaly_value = float(anomaly.get('A_face', 0.0))
                            landmarks_data['anomaly'] = anomaly_value
                            log_metrics(f"Рассчитан показатель аномалии: {anomaly_value:.3f}")
                        else:
                            logger.error(f"{LogEmoji.ERROR} Неожиданный тип результата anomaly: {type(anomaly)}")
                            landmarks_data['anomaly'] = 1.0
                    else:
                        logger.error(f"{LogEmoji.ERROR} Неверное количество точек: 3DDFA={len(pts_3ddfa)}, FAN={len(pts_fan)}")
                        landmarks_data['anomaly'] = 1.0
                except Exception as e:
                    log_exception(e, "Ошибка при расчете аномалии")
                    landmarks_data['anomaly'] = 1.0
                    landmarks_data['anomaly_details'] = {"error": str(e)}
            
            json_name = os.path.splitext(os.path.basename(args.img_fp))[0] + '.json'
            json_path = os.path.join('examples/results', json_name)
            logger.debug(f"{LogEmoji.SAVE} Сохранение JSON в {json_path}")
            
            with open(json_path, 'w') as f:
                json.dump(landmarks_data, f, indent=2)
            
            png_wfp = wfp.replace('.jpg', '.png')
            
            def apply_checkerboard(alpha_img, checker_size=10):
                h, w, _ = alpha_img.shape
                checker = numpy.indices((h, w)).sum(axis=0) % 2
                bg = numpy.where(checker[..., None], 200, 255).astype(numpy.uint8)
                bg = cv2.merge([bg, bg, bg])
                foreground = alpha_img[..., :3]
                alpha = alpha_img[..., 3:] / 255.0
                comp = (foreground * alpha + bg * (1 - alpha)).astype(numpy.uint8)
                return comp
            
            Image.fromarray(debug_img).save(png_wfp)
            log_save(f'Сохранен прозрачный PNG в {png_wfp}')
            
        elif args.opt == '2d_dense':
            logger.debug(f"{LogEmoji.PROCESSING} Режим 2d_dense")
            img = (img * 0.2).astype(numpy.uint8)
            draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
            log_save(f'Сохранены плотные ландмарки в {wfp}')
            
        elif args.opt == '3d':
            logger.debug(f"{LogEmoji.PROCESSING} Режим 3D рендеринга")
            try:
                # Проверяем наличие лиц
                if not boxes or len(boxes) == 0:
                    logger.error(f"{LogEmoji.ERROR} Лица не обнаружены в {args.img_fp}")
                    return
                
                # Выбираем самое большое лицо
                largest_box, idx = get_largest_face(boxes)
                if largest_box is None or idx < 0:
                    logger.error(f"{LogEmoji.ERROR} Не удалось определить самое большое лицо в {args.img_fp}")
                    return
                
                # Убедимся, что param_lst и roi_box_lst существуют и содержат данные
                if not param_lst or not roi_box_lst:
                    logger.error(f"{LogEmoji.ERROR} Нет данных параметров для {args.img_fp}")
                    return
                
                # Проверяем, что индекс не выходит за пределы списков
                if idx >= len(param_lst) or idx >= len(roi_box_lst):
                    logger.error(f"{LogEmoji.ERROR} Некорректный индекс лица для {args.img_fp}")
                    return
                
                # Преобразуем параметры в numpy массивы, если это списки
                if isinstance(param_lst[idx], list):
                    param = numpy.array(param_lst[idx])
                else:
                    param = param_lst[idx]
                
                if isinstance(roi_box_lst[idx], list):
                    roi_box = numpy.array(roi_box_lst[idx])
                else:
                    roi_box = roi_box_lst[idx]
                
                logger.debug(f"{LogEmoji.FACE} Выбрано лицо с индексом {idx}, ROI: {roi_box}")
                
                # Обновляем списки параметров и ROI
                param_lst = [param]
                roi_box_lst = [roi_box]
                
                # Реконструируем вершины
                logger.debug(f"{LogEmoji.PROCESSING} Реконструкция вершин для 3D рендеринга")
                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
                
                # Подготавливаем изображение для рендеринга
                img_render = (img * 0.9).astype(numpy.uint8)
                
                # Выполняем рендеринг
                logger.debug(f"{LogEmoji.PROCESSING} Выполнение 3D рендеринга")
                render(img_render, ver_lst, tddfa.tri, alpha=0.8, show_flag=False, wfp=wfp)
                log_save(f'Сохранен 3D рендер в {wfp}')
                
            except Exception as e:
                log_exception(e, f"Ошибка при обработке 3D рендера для {args.img_fp}")
                
        elif args.opt == 'depth':
            logger.debug(f"{LogEmoji.PROCESSING} Режим depth")
            img = (img * 0.5).astype(numpy.uint8)
            depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
            img = (img * 0.2).astype(numpy.uint8)
            log_save(f'Сохранена карта глубины в {wfp}')
            
        elif args.opt == 'pncc':
            logger.debug(f"{LogEmoji.PROCESSING} Режим PNCC")
            pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
            img = (img * 0.5).astype(numpy.uint8)
            log_save(f'Сохранена PNCC карта в {wfp}')
            
        elif args.opt == 'uv_tex':
            logger.debug(f"{LogEmoji.PROCESSING} Режим UV текстуры")
            uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
            log_save(f'Сохранена UV текстура в {wfp}')
            
        elif args.opt == 'pose':
            logger.debug(f"{LogEmoji.PROCESSING} Режим визуализации позы")
            viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
            img = (img * 0.5).astype(numpy.uint8)
            log_save(f'Сохранена визуализация позы в {wfp}')
            
        elif args.opt == 'ply':
            logger.debug(f"{LogEmoji.PROCESSING} Режим экспорта PLY")
            ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
            log_save(f'Сохранен PLY файл в {wfp}')
            
        elif args.opt == 'obj':
            logger.debug(f"{LogEmoji.PROCESSING} Режим экспорта OBJ")
            ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
            log_save(f'Сохранен OBJ файл в {wfp}')
            
        else:
            logger.error(f"{LogEmoji.ERROR} Неизвестный режим {args.opt}")
            raise ValueError(f'Unknown opt {args.opt}')
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log_time(f"Обработка изображения {args.img_fp} завершена за {duration:.2f} секунд")
        
    except Exception as e:
        log_exception(e, f"Критическая ошибка при обработке изображения {args.img_fp}")





# ------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, help='path to image file or directory')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                       choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true')
    parser.add_argument('--onnx', action='store_true', default=False)
    args = parser.parse_args()
    
    
# ------------------------------------------------------------------------------------------------------------------------  





    # Добавляем обработку директории
    if os.path.isdir(args.img_fp):
        # Получаем список поддерживаемых изображений
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [f for f in os.listdir(args.img_fp) 
                 if f.lower().endswith(image_extensions)]
        
        total = len(images)
        logger.info(f"{LogEmoji.CAMERA} Обработка директории {args.img_fp}, найдено {total} изображений")
        
        start_time = datetime.now()
        for idx, img_name in enumerate(images, 1):
            img_path = os.path.join(args.img_fp, img_name)
            logger.info(f"{LogEmoji.CAMERA} Обработка изображения [{idx}/{total}]: {img_name}")
            
            # Создаем копию аргументов для текущего изображения
            current_args = argparse.Namespace(**vars(args))
            current_args.img_fp = img_path
            
            try:
                main(current_args)
            except Exception as e:
                log_exception(e, f"Критическая ошибка при обработке {img_name}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log_time(f"Обработка всех {total} изображений завершена за {duration:.2f} секунд")
    else:
        # Обработка одного файла как раньше
        logger.info(f"{LogEmoji.CAMERA} Обработка одного файла: {args.img_fp}")
        main(args)
