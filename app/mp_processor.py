"""
Модуль для обработки лица с помощью MediaPipe
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from app.utils.pose import get_mp_active_metrics
from app.utils.confidence import calculate_confidence
from app.config import MP_CONFIDENCE_THRESHOLD, MP_LANDMARKS_INDICES

class MediaPipeProcessor:
    def __init__(self):
        """
        Инициализация процессора MediaPipe
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Инициализация детектора лиц MediaPipe
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def process(self, image_path, pose_info):
        """
        Обрабатывает изображение с помощью MediaPipe
        
        Args:
            image_path (str): Путь к изображению
            pose_info (dict): Информация о ракурсе от 3DDFA
            
        Returns:
            dict: Результаты обработки MediaPipe
        """
        # Проверяем, существует ли файл
        if not os.path.exists(image_path):
            return {'error': f'File not found: {image_path}'}
        
        try:
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Failed to load image: {image_path}'}
            
            # Конвертируем BGR в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Обрабатываем изображение с помощью MediaPipe
            results = self.face_mesh.process(image_rgb)
            
            # Проверяем, найдено ли лицо
            if not results.multi_face_landmarks:
                return {'error': 'No face detected by MediaPipe'}
            
            # Получаем первое найденное лицо
            face_landmarks = results.multi_face_landmarks[0]
            
            # Преобразуем landmarks в массив numpy
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
            ])
            
            # Нормализуем координаты относительно размеров изображения
            h, w, _ = image.shape
            landmarks[:, 0] *= w
            landmarks[:, 1] *= h
            
            # Рассчитываем метрики с учетом ракурса
            metrics = self.calculate_metrics(landmarks, pose_info)
            
            # Формируем результат
            result = {
                'landmarks': landmarks.tolist(),
                'metrics': metrics
            }
            
            # Добавляем визуализацию (опционально)
            if False:  # Изменить на True для включения визуализации
                annotated_image = image_rgb.copy()
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                result['visualization'] = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            return result
            
        except Exception as e:
            return {'error': f'MediaPipe processing error: {str(e)}'}
    
    def calculate_metrics(self, landmarks, pose_info):
        """
        Рассчитывает метрики MediaPipe с учетом ракурса
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            pose_info (dict): Информация о ракурсе
            
        Returns:
            dict: Словарь с метриками MediaPipe
        """
        # Получаем активные метрики для данного ракурса
        active_metrics = get_mp_active_metrics(pose_info)
        
        # Получаем углы головы
        angles = pose_info['angles']
        
        # Словарь для хранения метрик
        metrics = {}
        
        # Рассчитываем метрики
        for metric_name in active_metrics:
            # Рассчитываем значение метрики
            value = self.calculate_metric(landmarks, metric_name, pose_info)
            
            # Рассчитываем confidence для метрики
            confidence = calculate_confidence(angles, metric_name, pose_info)
            
            # Добавляем метрику только если confidence выше порога
            if confidence >= MP_CONFIDENCE_THRESHOLD:
                metrics[f"mp_{pose_info['main']}_{metric_name}"] = {
                    "value": value,
                    "confidence": confidence
                }
        
        return metrics
    
    def calculate_metric(self, landmarks, metric_name, pose_info):
        """
        Рассчитывает значение конкретной метрики MediaPipe
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            metric_name (str): Название метрики
            pose_info (dict): Информация о ракурсе
            
        Returns:
            float: Значение метрики
        """
        # Определяем ракурс
        main_pose = pose_info['main']
        
        # Рассчитываем метрику в зависимости от её названия
        if metric_name == 'eyebrow_angle':
            return self.calculate_eyebrow_angle(landmarks, main_pose)
        elif metric_name == 'eye_aspect_ratio':
            return self.calculate_eye_aspect_ratio(landmarks, main_pose)
        elif metric_name == 'mouth_aspect_ratio':
            return self.calculate_mouth_aspect_ratio(landmarks)
        elif metric_name == 'face_oval_ratio':
            return self.calculate_face_oval_ratio(landmarks)
        elif metric_name == 'nose_tip_position':
            return self.calculate_nose_tip_position(landmarks)
        elif metric_name == 'visible_eye_ratio':
            return self.calculate_visible_eye_ratio(landmarks, main_pose)
        elif metric_name == 'cheek_contour':
            return self.calculate_cheek_contour(landmarks, main_pose)
        elif metric_name == 'jaw_projection':
            return self.calculate_jaw_projection(landmarks, main_pose)
        elif metric_name == 'nose_projection':
            return self.calculate_nose_projection(landmarks)
        elif metric_name == 'forehead_slope':
            return self.calculate_forehead_slope(landmarks, main_pose)
        elif metric_name == 'chin_projection':
            return self.calculate_chin_projection(landmarks, main_pose)
        else:
            # Если метрика не определена, возвращаем 0
            return 0.0
    
    def calculate_eyebrow_angle(self, landmarks, pose):
        """
        Рассчитывает угол наклона бровей
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            pose (str): Ракурс
            
        Returns:
            float: Угол наклона бровей в градусах
        """
        # Получаем индексы точек бровей
        if 'left' in pose:
            # Для левого профиля используем правую бровь
            eyebrow_indices = MP_LANDMARKS_INDICES['right_eyebrow']
        else:
            # Для правого профиля или фронтального ракурса используем левую бровь
            eyebrow_indices = MP_LANDMARKS_INDICES['left_eyebrow']
        
        # Получаем точки брови
        eyebrow_points = landmarks[eyebrow_indices]
        
        # Вычисляем линию наилучшего приближения
        x = eyebrow_points[:, 0]
        y = eyebrow_points[:, 1]
        
        if len(x) < 2:
            return 0.0
        
        # Линейная регрессия
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Угол наклона в градусах
        angle = np.arctan(m) * 180 / np.pi
        
        return float(angle)
    
    def calculate_eye_aspect_ratio(self, landmarks, pose):
        """
        Рассчитывает соотношение сторон глаза (ширина/высота)
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            pose (str): Ракурс
            
        Returns:
            float: Соотношение сторон глаза
        """
        # Определяем, какой глаз использовать в зависимости от ракурса
        if 'left' in pose:
            # Для левого профиля используем правый глаз
            eye_indices = MP_LANDMARKS_INDICES['right_eye']
        else:
            # Для правого профиля или фронтального ракурса используем левый глаз
            eye_indices = MP_LANDMARKS_INDICES['left_eye']
        
        # Получаем точки глаза
        eye_points = landmarks[eye_indices]
        
        # Находим крайние точки глаза
        min_x = np.min(eye_points[:, 0])
        max_x = np.max(eye_points[:, 0])
        min_y = np.min(eye_points[:, 1])
        max_y = np.max(eye_points[:, 1])
        
        # Вычисляем ширину и высоту глаза
        width = max_x - min_x
        height = max_y - min_y
        
        # Вычисляем соотношение сторон
        if height > 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 0.0
        
        return float(aspect_ratio)
    
    def calculate_mouth_aspect_ratio(self, landmarks):
        """
        Рассчитывает соотношение сторон рта (ширина/высота)
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            
        Returns:
            float: Соотношение сторон рта
        """
        # Получаем индексы точек рта
        lips_indices = MP_LANDMARKS_INDICES['lips']
        
        # Получаем точки рта
        lips_points = landmarks[lips_indices]
        
        # Находим крайние точки рта
        min_x = np.min(lips_points[:, 0])
        max_x = np.max(lips_points[:, 0])
        min_y = np.min(lips_points[:, 1])
        max_y = np.max(lips_points[:, 1])
        
        # Вычисляем ширину и высоту рта
        width = max_x - min_x
        height = max_y - min_y
        
        # Вычисляем соотношение сторон
        if height > 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 0.0
        
        return float(aspect_ratio)
    
    def calculate_face_oval_ratio(self, landmarks):
        """
        Рассчитывает соотношение сторон овала лица (ширина/высота)
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            
        Returns:
            float: Соотношение сторон овала лица
        """
        # Получаем индексы точек овала лица
        face_oval_indices = MP_LANDMARKS_INDICES['face_oval']
        
        # Получаем точки овала лица
        face_oval_points = landmarks[face_oval_indices]
        
        # Находим крайние точки овала лица
        min_x = np.min(face_oval_points[:, 0])
        max_x = np.max(face_oval_points[:, 0])
        min_y = np.min(face_oval_points[:, 1])
        max_y = np.max(face_oval_points[:, 1])
        
        # Вычисляем ширину и высоту овала лица
        width = max_x - min_x
        height = max_y - min_y
        
        # Вычисляем соотношение сторон
        if height > 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 0.0
        
        return float(aspect_ratio)
    
    def calculate_nose_tip_position(self, landmarks):
        """
        Рассчитывает положение кончика носа относительно центра лица
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            
        Returns:
            float: Положение кончика носа (отклонение от центра)
        """
        # Получаем индексы точек носа и овала лица
        nose_indices = MP_LANDMARKS_INDICES['nose']
        face_oval_indices = MP_LANDMARKS_INDICES['face_oval']
        
        # Получаем точки носа и овала лица
        nose_points = landmarks[nose_indices]
        face_oval_points = landmarks[face_oval_indices]
        
        # Находим кончик носа (средняя точка нижней части носа)
        nose_tip = nose_points[5]  # Индекс может отличаться в зависимости от модели
        
        # Находим центр лица
        face_center_x = np.mean(face_oval_points[:, 0])
        face_center_y = np.mean(face_oval_points[:, 1])
        face_center = np.array([face_center_x, face_center_y])
        
        # Вычисляем отклонение кончика носа от центра лица
        deviation = np.linalg.norm(nose_tip[:2] - face_center)
        
        # Нормализуем отклонение относительно ширины лица
        face_width = np.max(face_oval_points[:, 0]) - np.min(face_oval_points[:, 0])
        if face_width > 0:
            normalized_deviation = deviation / face_width
        else:
            normalized_deviation = 0.0
        
        return float(normalized_deviation)
    
    def calculate_visible_eye_ratio(self, landmarks, pose):
        """
        Рассчитывает соотношение видимого глаза для профильных ракурсов
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            pose (str): Ракурс
            
        Returns:
            float: Соотношение видимого глаза
        """
        # Определяем, какой глаз использовать в зависимости от ракурса
        if 'left' in pose:
            # Для левого профиля используем правый глаз
            eye_indices = MP_LANDMARKS_INDICES['right_eye']
            opposite_eye_indices = MP_LANDMARKS_INDICES['left_eye']
        else:
            # Для правого профиля используем левый глаз
            eye_indices = MP_LANDMARKS_INDICES['left_eye']
            opposite_eye_indices = MP_LANDMARKS_INDICES['right_eye']
        
        # Получаем точки видимого и невидимого глаза
        visible_eye_points = landmarks[eye_indices]
        opposite_eye_points = landmarks[opposite_eye_indices]
        
        # Вычисляем площадь видимого глаза
        visible_eye_area = self.calculate_eye_area(visible_eye_points)
        
        # Вычисляем площадь невидимого глаза
        opposite_eye_area = self.calculate_eye_area(opposite_eye_points)
        
        # Вычисляем соотношение площадей
        if opposite_eye_area > 0:
            ratio = visible_eye_area / opposite_eye_area
        else:
            ratio = 1.0
        
        return float(ratio)
    
    def calculate_eye_area(self, eye_points):
        """
        Вспомогательная функция для расчета площади глаза
        
        Args:
            eye_points (numpy.ndarray): Точки глаза
            
        Returns:
            float: Площадь глаза
        """
        # Находим крайние точки глаза
        min_x = np.min(eye_points[:, 0])
        max_x = np.max(eye_points[:, 0])
        min_y = np.min(eye_points[:, 1])
        max_y = np.max(eye_points[:, 1])
        
        # Вычисляем ширину и высоту глаза
        width = max_x - min_x
        height = max_y - min_y
        
        # Вычисляем площадь
        area = width * height
        
        return area
    
    def calculate_cheek_contour(self, landmarks, pose):
        """
        Рассчитывает контур щеки для профильных ракурсов
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            pose (str): Ракурс
            
        Returns:
            float: Мера выпуклости щеки
        """
        # Определяем точки щеки в зависимости от ракурса
        if 'left' in pose:
            # Для левого профиля используем правую щеку
            # Выбираем точки между правым глазом и правой частью челюсти
            cheek_indices = [
                MP_LANDMARKS_INDICES['right_eye'][0],
                MP_LANDMARKS_INDICES['right_eye'][1],
                MP_LANDMARKS_INDICES['right_eye'][2],
                MP_LANDMARKS_INDICES['face_oval'][4],
                MP_LANDMARKS_INDICES['face_oval'][5],
                MP_LANDMARKS_INDICES['face_oval'][6]
            ]
        else:
            # Для правого профиля используем левую щеку
            # Выбираем точки между левым глазом и левой частью челюсти
            cheek_indices = [
                MP_LANDMARKS_INDICES['left_eye'][0],
                MP_LANDMARKS_INDICES['left_eye'][1],
                MP_LANDMARKS_INDICES['left_eye'][2],
                MP_LANDMARKS_INDICES['face_oval'][10],
                MP_LANDMARKS_INDICES['face_oval'][11],
                MP_LANDMARKS_INDICES['face_oval'][12]
            ]
        
        # Получаем точки щеки
        cheek_points = landmarks[cheek_indices]
        
        # Находим плоскость, проходящую через точки
        if len(cheek_points) >= 3:
            centroid = np.mean(cheek_points, axis=0)
            _, s, _ = np.linalg.svd(cheek_points - centroid)
            
            # Мера выпуклости - отношение сингулярных значений
            if s[1] > 0:
                cheek_contour = s[0] / s[1]
            else:
                cheek_contour = 1.0
        else:
            cheek_contour = 1.0
        
        return float(cheek_contour)
    
    def calculate_jaw_projection(self, landmarks, pose):
        """
        Рассчитывает проекцию челюсти для профильных ракурсов
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            pose (str): Ракурс
            
        Returns:
            float: Проекция челюсти
        """
        # Определяем точки челюсти в зависимости от ракурса
        if 'left' in pose:
            # Для левого профиля используем правую часть челюсти
            jaw_indices = [
                MP_LANDMARKS_INDICES['face_oval'][4],
                MP_LANDMARKS_INDICES['face_oval'][5],
                MP_LANDMARKS_INDICES['face_oval'][6],
                MP_LANDMARKS_INDICES['face_oval'][7]
            ]
        else:
            # Для правого профиля используем левую часть челюсти
            jaw_indices = [
                MP_LANDMARKS_INDICES['face_oval'][10],
                MP_LANDMARKS_INDICES['face_oval'][11],
                MP_LANDMARKS_INDICES['face_oval'][12],
                MP_LANDMARKS_INDICES['face_oval'][13]
            ]
        
        # Получаем точки челюсти
        jaw_points = landmarks[jaw_indices]
        
        # Вычисляем проекцию челюсти (выступание)
        # Используем Z-координату для определения выступания
        jaw_projection = np.mean(jaw_points[:, 2])
        
        return float(jaw_projection)
    
    def calculate_nose_projection(self, landmarks):
        """
        Рассчитывает проекцию носа (выступание)
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            
        Returns:
            float: Проекция носа
        """
        # Получаем индексы точек носа
        nose_indices = MP_LANDMARKS_INDICES['nose']
        
        # Получаем точки носа
        nose_points = landmarks[nose_indices]
        
        # Вычисляем проекцию носа (выступание)
        # Используем Z-координату для определения выступания
        nose_projection = np.mean(nose_points[:, 2])
        
        return float(nose_projection)
    
    def calculate_forehead_slope(self, landmarks, pose):
        """
        Рассчитывает наклон лба для профильных ракурсов
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            pose (str): Ракурс
            
        Returns:
            float: Наклон лба в градусах
        """
        # Определяем точки лба в зависимости от ракурса
        if 'left' in pose:
            # Для левого профиля используем правую часть лба
            forehead_indices = [
                MP_LANDMARKS_INDICES['right_eyebrow'][0],
                MP_LANDMARKS_INDICES['right_eyebrow'][1],
                MP_LANDMARKS_INDICES['right_eyebrow'][2],
                MP_LANDMARKS_INDICES['face_oval'][0],
                MP_LANDMARKS_INDICES['face_oval'][1]
            ]
        else:
            # Для правого профиля используем левую часть лба
            forehead_indices = [
                MP_LANDMARKS_INDICES['left_eyebrow'][0],
                MP_LANDMARKS_INDICES['left_eyebrow'][1],
                MP_LANDMARKS_INDICES['left_eyebrow'][2],
                MP_LANDMARKS_INDICES['face_oval'][15],
                MP_LANDMARKS_INDICES['face_oval'][16]
            ]
        
        # Получаем точки лба
        forehead_points = landmarks[forehead_indices]
        
        # Вычисляем наклон лба
        x = forehead_points[:, 0]
        y = forehead_points[:, 1]
        
        if len(x) < 2:
            return 0.0
        
        # Линейная регрессия
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Угол наклона в градусах
        angle = np.arctan(m) * 180 / np.pi
        
        return float(angle)
    
    def calculate_chin_projection(self, landmarks, pose):
        """
        Рассчитывает проекцию подбородка для профильных ракурсов
        
        Args:
            landmarks (numpy.ndarray): Массив landmarks размером (468, 3)
            pose (str): Ракурс
            
        Returns:
            float: Проекция подбородка
        """
        # Определяем точки подбородка в зависимости от ракурса
        if 'left' in pose:
            # Для левого профиля используем правую часть подбородка
            chin_indices = [
                MP_LANDMARKS_INDICES['face_oval'][7],
                MP_LANDMARKS_INDICES['face_oval'][8],
                MP_LANDMARKS_INDICES['face_oval'][9]
            ]
        else:
            # Для правого профиля используем левую часть подбородка
            chin_indices = [
                MP_LANDMARKS_INDICES['face_oval'][7],
                MP_LANDMARKS_INDICES['face_oval'][8],
                MP_LANDMARKS_INDICES['face_oval'][9]
            ]
        
        # Получаем точки подбородка
        chin_points = landmarks[chin_indices]
        
        # Вычисляем проекцию подбородка (выступание)
        # Используем Z-координату для определения выступания
        chin_projection = np.mean(chin_points[:, 2])
        
        return float(chin_projection)
