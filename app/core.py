import os
import sys
import numpy as np
import cv2
import torch

# Импорт 3DDFA
from TDDFA import TDDFA

# Импорт face-alignment
import face_alignment

# Импорты из вашей структуры
from app.utils.pose import determine_pose, get_active_metrics
from app.utils.confidence import calculate_confidence
from app.metrics.geometry import calculate_3ddfa_metrics
from app.metrics.deviation import calculate_deviations
from app.config import CONFIDENCE_THRESHOLD

class FaceProcessor:
    def __init__(self):
        # Инициализация 3DDFA
        self.tddfa = TDDFA(gpu_mode=torch.cuda.is_available())
        
        # Инициализация FAN
        self.fan = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Устанавливаем устройство для вычислений
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
    
    def process(self, image_path):
        """
        Обрабатывает изображение с помощью 3DDFA и FAN
        
        Args:
            image_path (str): Путь к изображению
            
        Returns:
            dict: Результаты обработки
        """
        # Проверяем, существует ли файл
        if not os.path.exists(image_path):
            return {'error': f'File not found: {image_path}'}
        
        try:
            # Обработка с помощью 3DDFA
            tddfa_result = self.process_3ddfa(image_path)
            
            # Обработка с помощью FAN
            fan_result = self.process_fan(image_path)
            
            # Определяем ракурс на основе углов головы
            pose_info = determine_pose(tddfa_result['angles'])
            
            # Рассчитываем метрики с учетом ракурса
            metrics = calculate_3ddfa_metrics(tddfa_result['landmarks'], pose_info)
            
            # Добавляем confidence к метрикам
            metrics_with_confidence = self.add_confidence_to_metrics(metrics, tddfa_result['angles'], pose_info)
            
            # Рассчитываем отклонения между 3DDFA и FAN
            deviations = calculate_deviations(tddfa_result['landmarks'], fan_result['landmarks'])
            
            # Формируем результат
            result = {
                '3ddfa': {
                    'landmarks': tddfa_result['landmarks'].tolist(),
                    'angles': tddfa_result['angles'],
                    'shape_error': tddfa_result['shape_error'],
                    'pose': pose_info
                },
                'fan': {
                    'landmarks': fan_result['landmarks'].tolist()
                },
                'metrics': metrics_with_confidence,
                'deviations': deviations,
                'uv_texture': tddfa_result.get('uv_texture', None),
                'depth_map': tddfa_result.get('depth_map', None),
                'normal_map': tddfa_result.get('normal_map', None)
            }
            
            # Определяем, является ли лицо аномальным
            result['is_anomalous'] = self.check_anomaly(result)
            
            return result
        
        except Exception as e:
            return {'error': f'Error processing image: {str(e)}'}
    
    def process_3ddfa(self, image_path):
        """
        Обрабатывает изображение с помощью 3DDFA
        
        Args:
            image_path (str): Путь к изображению
            
        Returns:
            dict: Результаты обработки 3DDFA
        """
        # Обработка изображения с помощью 3DDFA
        param_lst, roi_box_lst = self.tddfa.detect(image_path)
        
        # Если лицо не обнаружено
        if len(param_lst) == 0:
            return {'error': 'No face detected by 3DDFA'}
        
        # Берем первое лицо
        param = param_lst[0]
        roi_box = roi_box_lst[0]
        
        # Получаем вершины 3D-модели
        vertices = self.tddfa.recon_vers(param, roi_box)
        
        # Получаем landmarks
        landmarks = self.tddfa.recon_landmarks(param, roi_box)
        
        # Получаем углы поворота головы
        angles = self.tddfa.calc_pose(param)
        
        # Получаем shape error
        shape_error = self.tddfa.calc_shape_error(param)
        
        # Получаем UV-текстуру
        uv_texture = self.tddfa.get_uv_texture(image_path, param, roi_box)
        
        # Получаем карту глубины
        depth_map = self.tddfa.get_depth_map(param, roi_box)
        
        # Получаем карту нормалей
        normal_map = self.tddfa.get_normal_map(param, roi_box)
        
        return {
            'landmarks': landmarks,
            'vertices': vertices,
            'angles': angles,
            'shape_error': shape_error,
            'roi_box': roi_box,
            'uv_texture': uv_texture,
            'depth_map': depth_map,
            'normal_map': normal_map
        }
    
    def process_fan(self, image_path):
        """
        Обрабатывает изображение с помощью FAN
        
        Args:
            image_path (str): Путь к изображению
            
        Returns:
            dict: Результаты обработки FAN
        """
        # Обработка изображения с помощью FAN
        landmarks = self.fan.get_landmarks(image_path)
        
        # Если лицо не обнаружено
        if landmarks is None or len(landmarks) == 0:
            return {'error': 'No face detected by FAN'}
        
        # Берем первое лицо
        landmarks = landmarks[0]
        
        return {
            'landmarks': landmarks
        }
    
    def add_confidence_to_metrics(self, metrics, angles, pose_info):
        """
        Добавляет confidence к метрикам
        
        Args:
            metrics (dict): Словарь с метриками
            angles (list): Углы поворота головы [yaw, pitch, roll]
            pose_info (dict): Информация о ракурсе
            
        Returns:
            dict: Словарь с метриками и confidence
        """
        metrics_with_confidence = {}
        
        for metric_name, value in metrics.items():
            # Рассчитываем confidence для метрики
            confidence = calculate_confidence(angles, metric_name, pose_info)
            
            # Добавляем метрику только если confidence выше порога
            if confidence >= CONFIDENCE_THRESHOLD:
                prefix = f"3ddfa_{pose_info['main']}"
                metrics_with_confidence[f"{prefix}_{metric_name}"] = {
                    "value": value,
                    "confidence": confidence
                }
        
        return metrics_with_confidence
    
    def check_anomaly(self, result):
        """
        Проверяет, является ли лицо аномальным
        
        Args:
            result (dict): Результаты обработки
            
        Returns:
            bool: True, если лицо аномальное, иначе False
        """
        from app.config import SHAPE_ERROR_THRESHOLD, DEVIATION_THRESHOLD, TEXTURE_UNIFORMITY_THRESHOLD
        
        # Проверяем shape error
        shape_error = result['3ddfa']['shape_error']
        if shape_error > SHAPE_ERROR_THRESHOLD:
            return True
        
        # Проверяем отклонения между 3DDFA и FAN
        if 'deviations' in result and 'percent_large' in result['deviations']:
            percent_large_deviations = result['deviations']['percent_large']
            if percent_large_deviations > DEVIATION_THRESHOLD * 100:
                return True
        
        # Проверяем однородность текстуры
        if 'metrics' in result:
            for key, metric in result['metrics'].items():
                if 'uniformity' in key and metric['value'] > TEXTURE_UNIFORMITY_THRESHOLD:
                    return True
        
        return False

def process_face(image_path):
    """
    Обрабатывает изображение с помощью FaceProcessor
    
    Args:
        image_path (str): Путь к изображению
        
    Returns:
        dict: Результаты обработки
    """
    processor = FaceProcessor()
    return processor.process(image_path)
