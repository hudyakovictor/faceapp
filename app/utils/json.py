"""
Модуль для формирования и обработки JSON-результатов анализа лица
"""

import json
import numpy as np
import os
from datetime import datetime
from app.config import (
    SHAPE_ERROR_THRESHOLD, 
    DEVIATION_THRESHOLD, 
    TEXTURE_UNIFORMITY_THRESHOLD,
    CONFIDENCE_THRESHOLD
)

class NumpyEncoder(json.JSONEncoder):
    """
    Кастомный энкодер для сериализации numpy-типов в JSON
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)

def build_final_json(core_data, mp_data, texture_data, embedding_data, image_path=None):
    """
    Формирует итоговый JSON на основе данных от всех модулей
    
    Args:
        core_data (dict): Данные от core.py (3DDFA и FAN)
        mp_data (dict): Данные от MediaPipe
        texture_data (dict): Данные анализа текстуры
        embedding_data (dict): Данные от InsightFace (embedding)
        image_path (str, optional): Путь к исходному изображению
        
    Returns:
        dict: Итоговый JSON с результатами анализа
    """
    # Базовая структура JSON
    result = {
        "metadata": create_metadata(image_path),
        "3ddfa": extract_3ddfa_data(core_data),
        "metrics": merge_metrics(core_data, mp_data, texture_data),
        "deviations": core_data.get("deviations", {}),
        "texture": extract_texture_data(texture_data),
        "geometry": extract_geometry_data(core_data),
        "embedding": embedding_data.get("embedding", None),
        "is_anomalous": False  # Будет обновлено позже
    }
    
    # Определяем, является ли изображение аномальным
    result["is_anomalous"] = detect_anomaly(result)
    
    # Добавляем уровень достоверности анализа
    result["analysis_confidence"] = calculate_analysis_confidence(result)
    
    return result

def create_metadata(image_path=None):
    """
    Создает метаданные для JSON
    
    Args:
        image_path (str, optional): Путь к исходному изображению
        
    Returns:
        dict: Метаданные
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    if image_path:
        metadata["filename"] = os.path.basename(image_path)
        metadata["file_size"] = os.path.getsize(image_path) if os.path.exists(image_path) else None
        
        # Добавляем информацию о времени создания и изменения файла
        if os.path.exists(image_path):
            metadata["created_at"] = datetime.fromtimestamp(os.path.getctime(image_path)).isoformat()
            metadata["modified_at"] = datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat()
    
    return metadata

def extract_3ddfa_data(core_data):
    """
    Извлекает данные 3DDFA из core_data
    
    Args:
        core_data (dict): Данные от core.py
        
    Returns:
        dict: Данные 3DDFA
    """
    if "3ddfa" not in core_data:
        return {}
    
    tddfa_data = core_data["3ddfa"]
    
    # Извлекаем только необходимые данные
    result = {
        "landmarks": tddfa_data.get("landmarks", []),
        "angles": {
            "yaw": tddfa_data.get("angles", [0, 0, 0])[0],
            "pitch": tddfa_data.get("angles", [0, 0, 0])[1],
            "roll": tddfa_data.get("angles", [0, 0, 0])[2]
        },
        "shape_error": tddfa_data.get("shape_error", 0),
        "pose": tddfa_data.get("pose", {})
    }
    
    return result

def merge_metrics(core_data, mp_data, texture_data):
    """
    Объединяет метрики из разных источников
    
    Args:
        core_data (dict): Данные от core.py
        mp_data (dict): Данные от MediaPipe
        texture_data (dict): Данные анализа текстуры
        
    Returns:
        dict: Объединенные метрики
    """
    metrics = {}
    
    # Добавляем метрики 3DDFA
    if "metrics" in core_data:
        for key, value in core_data["metrics"].items():
            # Добавляем префикс, если его нет
            if not key.startswith("3ddfa_"):
                metrics[f"3ddfa_{key}"] = value
            else:
                metrics[key] = value
    
    # Добавляем метрики MediaPipe
    if mp_data and "metrics" in mp_data:
        for key, value in mp_data["metrics"].items():
            # Добавляем префикс, если его нет
            if not key.startswith("mp_"):
                metrics[f"mp_{key}"] = value
            else:
                metrics[key] = value
    
    # Добавляем базовые текстурные метрики
    if texture_data:
        for key, value in texture_data.items():
            if not key.startswith("texture_") and key not in ["global_uniformity", "global_color_variation", "global_contrast", "global_entropy", "global_saturation"]:
                metrics[f"texture_{key}"] = value
    
    # Фильтруем метрики по confidence
    filtered_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict) and "confidence" in value:
            if value["confidence"] >= CONFIDENCE_THRESHOLD:
                filtered_metrics[key] = value
        else:
            # Если нет confidence, добавляем метрику как есть
            filtered_metrics[key] = {"value": value, "confidence": 1.0}
    
    return filtered_metrics

def extract_texture_data(texture_data):
    """
    Извлекает данные текстуры
    
    Args:
        texture_data (dict): Данные анализа текстуры
        
    Returns:
        dict: Структурированные данные текстуры
    """
    if not texture_data:
        return {}
    
    # Группируем текстурные метрики по зонам
    zones = {}
    global_metrics = {}
    
    for key, value in texture_data.items():
        if key.startswith("global_"):
            global_metrics[key.replace("global_", "")] = value
        else:
            # Извлекаем название зоны из ключа (например, forehead_uniformity -> forehead)
            parts = key.split("_")
            if len(parts) >= 2:
                zone_name = parts[0]
                metric_name = "_".join(parts[1:])
                
                if zone_name not in zones:
                    zones[zone_name] = {}
                
                zones[zone_name][metric_name] = value
    
    # Формируем итоговую структуру
    result = {
        "global": global_metrics,
        "zones": zones
    }
    
    # Добавляем агрегированные метрики
    result["aggregated"] = {
        "mean_uniformity": np.mean([zone.get("uniformity", 0) for zone in zones.values() if "uniformity" in zone]),
        "std_uniformity": np.std([zone.get("uniformity", 0) for zone in zones.values() if "uniformity" in zone]),
        "mean_contrast": np.mean([zone.get("contrast", 0) for zone in zones.values() if "contrast" in zone]),
        "texture_anomaly_score": calculate_texture_anomaly_score(texture_data)
    }
    
    return result

def extract_geometry_data(core_data):
    """
    Извлекает геометрические данные из карт глубины и нормалей
    
    Args:
        core_data (dict): Данные от core.py
        
    Returns:
        dict: Геометрические данные
    """
    result = {
        "depth": {},
        "normals": {}
    }
    
    # Извлекаем данные карты глубины
    if "depth_map" in core_data and core_data["depth_map"] is not None:
        depth_map = core_data["depth_map"]
        
        # Рассчитываем статистики по карте глубины
        result["depth"] = {
            "mean": float(np.mean(depth_map)),
            "std": float(np.std(depth_map)),
            "min": float(np.min(depth_map)),
            "max": float(np.max(depth_map)),
            "range": float(np.max(depth_map) - np.min(depth_map)),
            "gradient_magnitude": calculate_gradient_magnitude(depth_map)
        }
    
    # Извлекаем данные карты нормалей
    if "normal_map" in core_data and core_data["normal_map"] is not None:
        normal_map = core_data["normal_map"]
        
        # Рассчитываем статистики по карте нормалей
        result["normals"] = {
            "mean_x": float(np.mean(normal_map[:, :, 0])),
            "mean_y": float(np.mean(normal_map[:, :, 1])),
            "mean_z": float(np.mean(normal_map[:, :, 2])),
            "std_x": float(np.std(normal_map[:, :, 0])),
            "std_y": float(np.std(normal_map[:, :, 1])),
            "std_z": float(np.std(normal_map[:, :, 2])),
            "normal_consistency": calculate_normal_consistency(normal_map)
        }
    
    # Добавляем общую оценку геометрической аномальности
    result["geometry_anomaly_score"] = calculate_geometry_anomaly_score(result)
    
    return result

def calculate_gradient_magnitude(depth_map):
    """
    Рассчитывает среднюю величину градиента карты глубины
    
    Args:
        depth_map (numpy.ndarray): Карта глубины
        
    Returns:
        float: Средняя величина градиента
    """
    if depth_map is None or depth_map.size == 0:
        return 0.0
    
    # Рассчитываем градиенты по осям x и y
    grad_y, grad_x = np.gradient(depth_map)
    
    # Рассчитываем величину градиента
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Возвращаем среднюю величину градиента
    return float(np.mean(gradient_magnitude))

def calculate_normal_consistency(normal_map):
    """
    Рассчитывает консистентность нормалей
    
    Args:
        normal_map (numpy.ndarray): Карта нормалей
        
    Returns:
        float: Мера консистентности нормалей
    """
    if normal_map is None or normal_map.size == 0:
        return 0.0
    
    # Рассчитываем локальную консистентность нормалей
    h, w, _ = normal_map.shape
    consistency = 0.0
    count = 0
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            # Текущая нормаль
            current = normal_map[i, j]
            
            # Соседние нормали
            neighbors = [
                normal_map[i-1, j],
                normal_map[i+1, j],
                normal_map[i, j-1],
                normal_map[i, j+1]
            ]
            
            # Рассчитываем косинусное сходство с соседями
            for neighbor in neighbors:
                dot_product = np.dot(current, neighbor)
                consistency += dot_product
                count += 1
    
    # Нормализуем результат
    if count > 0:
        consistency /= count
    
    return float(consistency)

def calculate_texture_anomaly_score(texture_data):
    """
    Рассчитывает оценку аномальности текстуры
    
    Args:
        texture_data (dict): Данные анализа текстуры
        
    Returns:
        float: Оценка аномальности текстуры (0-1)
    """
    if not texture_data:
        return 0.5
    
    # Извлекаем ключевые метрики для определения аномальности
    uniformity = texture_data.get("global_uniformity", 0.5)
    color_variation = texture_data.get("global_color_variation", 0.5)
    contrast = texture_data.get("global_contrast", 50)
    entropy = texture_data.get("global_entropy", 4)
    
    # Нормализуем метрики
    norm_uniformity = min(1.0, uniformity / 0.5)  # Высокая однородность подозрительна
    norm_color_variation = min(1.0, 0.5 / max(0.1, color_variation))  # Низкая вариация цвета подозрительна
    norm_contrast = min(1.0, 50 / max(10, contrast))  # Низкий контраст подозрителен
    norm_entropy = min(1.0, 4 / max(1, entropy))  # Низкая энтропия подозрительна
    
    # Рассчитываем взвешенную оценку аномальности
    anomaly_score = (
        0.4 * norm_uniformity +
        0.3 * norm_color_variation +
        0.2 * norm_contrast +
        0.1 * norm_entropy
    )
    
    return float(anomaly_score)

def calculate_geometry_anomaly_score(geometry_data):
    """
    Рассчитывает оценку аномальности геометрии
    
    Args:
        geometry_data (dict): Геометрические данные
        
    Returns:
        float: Оценка аномальности геометрии (0-1)
    """
    if not geometry_data or "depth" not in geometry_data or "normals" not in geometry_data:
        return 0.5
    
    depth_data = geometry_data["depth"]
    normals_data = geometry_data["normals"]
    
    # Извлекаем ключевые метрики для определения аномальности
    depth_std = depth_data.get("std", 0.1)
    depth_range = depth_data.get("range", 0.5)
    gradient_magnitude = depth_data.get("gradient_magnitude", 0.1)
    normal_std_x = normals_data.get("std_x", 0.1)
    normal_std_y = normals_data.get("std_y", 0.1)
    normal_std_z = normals_data.get("std_z", 0.1)
    normal_consistency = normals_data.get("normal_consistency", 0.9)
    
    # Нормализуем метрики
    norm_depth_std = min(1.0, 0.1 / max(0.01, depth_std))  # Низкое стандартное отклонение глубины подозрительно
    norm_depth_range = min(1.0, 0.5 / max(0.1, depth_range))  # Малый диапазон глубины подозрителен
    norm_gradient = min(1.0, 0.1 / max(0.01, gradient_magnitude))  # Низкий градиент подозрителен
    norm_normal_std = min(1.0, 0.1 / max(0.01, (normal_std_x + normal_std_y + normal_std_z) / 3))  # Низкое стандартное отклонение нормалей подозрительно
    norm_consistency = min(1.0, normal_consistency)  # Высокая консистентность подозрительна для масок
    
    # Рассчитываем взвешенную оценку аномальности
    anomaly_score = (
        0.25 * norm_depth_std +
        0.20 * norm_depth_range +
        0.25 * norm_gradient +
        0.15 * norm_normal_std +
        0.15 * norm_consistency
    )
    
    return float(anomaly_score)

def detect_anomaly(result):
    """
    Определяет, является ли изображение аномальным
    
    Args:
        result (dict): Итоговый JSON с результатами анализа
        
    Returns:
        bool: True, если изображение аномальное, иначе False
    """
    # Проверяем shape_error
    shape_error = result.get("3ddfa", {}).get("shape_error", 0)
    if shape_error > SHAPE_ERROR_THRESHOLD:
        return True
    
    # Проверяем отклонения между 3DDFA и FAN
    deviations = result.get("deviations", {})
    if deviations.get("percent_large", 0) > DEVIATION_THRESHOLD * 100:
        return True
    
    # Проверяем текстурные аномалии
    texture = result.get("texture", {})
    texture_anomaly_score = texture.get("aggregated", {}).get("texture_anomaly_score", 0)
    if texture_anomaly_score > 0.7:  # Пороговое значение для текстурных аномалий
        return True
    
    # Проверяем геометрические аномалии
    geometry = result.get("geometry", {})
    geometry_anomaly_score = geometry.get("geometry_anomaly_score", 0)
    if geometry_anomaly_score > 0.7:  # Пороговое значение для геометрических аномалий
        return True
    
    # Проверяем комбинированные признаки
    if texture_anomaly_score > 0.5 and geometry_anomaly_score > 0.5:
        return True
    
    # Проверяем симметрию
    symmetry_diff = deviations.get("symmetry_diff", 0)
    if symmetry_diff > 0.1:  # Пороговое значение для разницы в симметрии
        return True
    
    return False

def calculate_analysis_confidence(result):
    """
    Рассчитывает общую достоверность анализа
    
    Args:
        result (dict): Итоговый JSON с результатами анализа
        
    Returns:
        float: Достоверность анализа (0-1)
    """
    # Извлекаем углы головы
    angles = result.get("3ddfa", {}).get("angles", {"yaw": 0, "pitch": 0, "roll": 0})
    yaw = abs(angles.get("yaw", 0))
    pitch = abs(angles.get("pitch", 0))
    roll = abs(angles.get("roll", 0))
    
    # Рассчитываем базовую достоверность на основе углов головы
    # Чем ближе к фронтальному положению, тем выше достоверность
    yaw_confidence = max(0, 1 - yaw / 45)
    pitch_confidence = max(0, 1 - pitch / 30)
    roll_confidence = max(0, 1 - roll / 30)
    
    pose_confidence = 0.5 * yaw_confidence + 0.3 * pitch_confidence + 0.2 * roll_confidence
    
    # Учитываем shape_error
    shape_error = result.get("3ddfa", {}).get("shape_error", 0)
    shape_confidence = max(0, 1 - shape_error / SHAPE_ERROR_THRESHOLD)
    
    # Учитываем отклонения между 3DDFA и FAN
    deviations = result.get("deviations", {})
    deviation_percent = deviations.get("percent_large", 0) / 100
    deviation_confidence = max(0, 1 - deviation_percent / DEVIATION_THRESHOLD)
    
    # Рассчитываем итоговую достоверность
    confidence = 0.4 * pose_confidence + 0.3 * shape_confidence + 0.3 * deviation_confidence
    
    return float(confidence)

def save_json(data, output_path):
    """
    Сохраняет данные в JSON-файл
    
    Args:
        data (dict): Данные для сохранения
        output_path (str): Путь для сохранения файла
        
    Returns:
        bool: True, если сохранение успешно, иначе False
    """
    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохраняем JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False

def load_json(input_path):
    """
    Загружает данные из JSON-файла
    
    Args:
        input_path (str): Путь к JSON-файлу
        
    Returns:
        dict: Загруженные данные или None в случае ошибки
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def compare_jsons(json1, json2):
    """
    Сравнивает два JSON с результатами анализа
    
    Args:
        json1 (dict): Первый JSON
        json2 (dict): Второй JSON
        
    Returns:
        dict: Результаты сравнения
    """
    if not json1 or not json2:
        return {"error": "Invalid JSON data"}
    
    comparison = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "json1": json1.get("metadata", {}).get("filename", "unknown"),
            "json2": json2.get("metadata", {}).get("filename", "unknown")
        },
        "metrics_diff": {},
        "embedding_similarity": None,
        "is_same_person": None,
        "anomaly_diff": json1.get("is_anomalous", False) != json2.get("is_anomalous", False)
    }
    
    # Сравниваем метрики
    metrics1 = json1.get("metrics", {})
    metrics2 = json2.get("metrics", {})
    
    common_metrics = set(metrics1.keys()) & set(metrics2.keys())
    for metric in common_metrics:
        value1 = metrics1[metric].get("value", 0) if isinstance(metrics1[metric], dict) else metrics1[metric]
        value2 = metrics2[metric].get("value", 0) if isinstance(metrics2[metric], dict) else metrics2[metric]
        
        # Рассчитываем относительную разницу
        if value1 != 0:
            rel_diff = abs(value1 - value2) / abs(value1)
        else:
            rel_diff = abs(value2) if value2 != 0 else 0
        
        comparison["metrics_diff"][metric] = {
            "value1": value1,
            "value2": value2,
            "abs_diff": abs(value1 - value2),
            "rel_diff": rel_diff
        }
    
    # Сравниваем embedding
    embedding1 = json1.get("embedding")
    embedding2 = json2.get("embedding")
    
    if embedding1 is not None and embedding2 is not None:
        # Рассчитываем косинусное сходство
        similarity = cosine_similarity(embedding1, embedding2)
        comparison["embedding_similarity"] = similarity
        
        # Определяем, один ли это человек
        comparison["is_same_person"] = similarity > 0.6  # Пороговое значение для определения одного человека
    
    return comparison

def cosine_similarity(vec1, vec2):
    """
    Рассчитывает косинусное сходство между двумя векторами
    
    Args:
        vec1 (list): Первый вектор
        vec2 (list): Второй вектор
        
    Returns:
        float: Косинусное сходство (-1 до 1)
    """
    if not vec1 or not vec2:
        return 0.0
    
    # Преобразуем в numpy массивы
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Рассчитываем косинусное сходство
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def filter_metrics_by_confidence(metrics, threshold=CONFIDENCE_THRESHOLD):
    """
    Фильтрует метрики по уровню confidence
    
    Args:
        metrics (dict): Словарь метрик
        threshold (float): Пороговое значение confidence
        
    Returns:
        dict: Отфильтрованные метрики
    """
    filtered = {}
    
    for key, value in metrics.items():
        if isinstance(value, dict) and "confidence" in value:
            if value["confidence"] >= threshold:
                filtered[key] = value
        else:
            # Если нет confidence, добавляем метрику как есть
            filtered[key] = {"value": value, "confidence": 1.0}
    
    return filtered

def aggregate_jsons(json_list):
    """
    Агрегирует данные из списка JSON
    
    Args:
        json_list (list): Список JSON с результатами анализа
        
    Returns:
        dict: Агрегированные данные
    """
    if not json_list:
        return {"error": "Empty JSON list"}
    
    # Инициализируем агрегированные данные
    aggregated = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "count": len(json_list),
            "files": [j.get("metadata", {}).get("filename", "unknown") for j in json_list]
        },
        "metrics": {},
        "anomaly_stats": {
            "anomalous_count": sum(1 for j in json_list if j.get("is_anomalous", False)),
            "normal_count": sum(1 for j in json_list if not j.get("is_anomalous", False))
        }
    }
    
    # Собираем все метрики
    all_metrics = {}
    for json_data in json_list:
        metrics = json_data.get("metrics", {})
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            
            if isinstance(value, dict) and "value" in value:
                all_metrics[key].append(value["value"])
            else:
                all_metrics[key].append(value)
    
    # Рассчитываем статистики по метрикам
    for key, values in all_metrics.items():
        values = [v for v in values if v is not None]
        if not values:
            continue
        
        aggregated["metrics"][key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values)
        }
    
    # Группируем по embedding
    if all(j.get("embedding") is not None for j in json_list):
        clusters = cluster_by_embedding(json_list)
        aggregated["clusters"] = clusters
    
    return aggregated

def cluster_by_embedding(json_list, threshold=0.6):
    """
    Группирует JSON по сходству embedding
    
    Args:
        json_list (list): Список JSON с результатами анализа
        threshold (float): Пороговое значение сходства
        
    Returns:
        list: Список кластеров
    """
    if not json_list:
        return []
    
    # Инициализируем кластеры
    clusters = []
    processed = set()
    
    for i, json1 in enumerate(json_list):
        if i in processed:
            continue
        
        embedding1 = json1.get("embedding")
        if embedding1 is None:
            continue
        
        # Создаем новый кластер
        cluster = {
            "center": json1.get("metadata", {}).get("filename", f"item_{i}"),
            "items": [json1.get("metadata", {}).get("filename", f"item_{i}")],
            "anomalous_count": 1 if json1.get("is_anomalous", False) else 0
        }
        processed.add(i)
        
        # Ищем похожие JSON
        for j, json2 in enumerate(json_list):
            if j in processed or i == j:
                continue
            
            embedding2 = json2.get("embedding")
            if embedding2 is None:
                continue
            
            # Рассчитываем сходство
            similarity = cosine_similarity(embedding1, embedding2)
            
            # Если сходство выше порога, добавляем в кластер
            if similarity > threshold:
                cluster["items"].append(json2.get("metadata", {}).get("filename", f"item_{j}"))
                if json2.get("is_anomalous", False):
                    cluster["anomalous_count"] += 1
                processed.add(j)
        
        # Добавляем кластер в список
        cluster["count"] = len(cluster["items"])
        clusters.append(cluster)
    
    # Сортируем кластеры по размеру
    clusters.sort(key=lambda x: x["count"], reverse=True)
    
    return clusters
