"""
Модуль для извлечения embedding вектора лица с помощью InsightFace
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from app.config import INSIGHTFACE_MODEL_PATH

class FaceEmbedding:
    def __init__(self, model_path=None):
        """
        Инициализация модуля для извлечения embedding вектора лица
        
        Args:
            model_path (str, optional): Путь к модели InsightFace
        """
        # Если путь к модели не указан, используем путь из конфигурации
        if model_path is None:
            model_path = INSIGHTFACE_MODEL_PATH
        
        # Проверяем, существует ли файл модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"InsightFace model not found at: {model_path}")
        
        # Инициализация ONNX Runtime сессии
        self.session = ort.InferenceSession(model_path)
        
        # Получаем имена входных и выходных тензоров
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Размер входного изображения для модели
        self.input_size = (112, 112)
    
    def preprocess(self, image):
        """
        Предобработка изображения для InsightFace
        
        Args:
            image (numpy.ndarray): Входное изображение в формате BGR
            
        Returns:
            numpy.ndarray: Предобработанное изображение
        """
        # Изменяем размер изображения
        img = cv2.resize(image, self.input_size)
        
        # Преобразуем BGR в RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Нормализация
        img = img.astype(np.float32) / 255.0
        
        # Стандартизация
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = (img - mean) / std
        
        # Преобразуем в формат NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def extract_embedding(self, image):
        """
        Извлекает embedding вектор из изображения лица
        
        Args:
            image (numpy.ndarray): Изображение лица в формате BGR
            
        Returns:
            numpy.ndarray: Embedding вектор размерности 512
        """
        # Предобработка изображения
        preprocessed_img = self.preprocess(image)
        
        # Получаем embedding вектор
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed_img})
        embedding = outputs[0][0]
        
        # Нормализация вектора
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        
        return embedding
    
    def extract_embedding_from_file(self, image_path):
        """
        Извлекает embedding вектор из файла изображения
        
        Args:
            image_path (str): Путь к файлу изображения
            
        Returns:
            dict: Словарь с embedding вектором и дополнительной информацией
        """
        # Проверяем, существует ли файл
        if not os.path.exists(image_path):
            return {'error': f'File not found: {image_path}'}
        
        try:
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Failed to load image: {image_path}'}
            
            # Извлекаем embedding вектор
            embedding = self.extract_embedding(image)
            
            # Формируем результат
            result = {
                'embedding': embedding.tolist(),
                'embedding_norm': float(np.linalg.norm(embedding)),
                'embedding_dim': embedding.shape[0]
            }
            
            return result
        except Exception as e:
            return {'error': f'Error extracting embedding: {str(e)}'}
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Рассчитывает косинусное сходство между двумя embedding векторами
        
        Args:
            embedding1 (numpy.ndarray): Первый embedding вектор
            embedding2 (numpy.ndarray): Второй embedding вектор
            
        Returns:
            float: Косинусное сходство (от -1 до 1)
        """
        # Проверяем, что векторы имеют одинаковую размерность
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embedding vectors must have the same dimension")
        
        # Рассчитываем косинусное сходство
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def is_same_person(self, embedding1, embedding2, threshold=0.5):
        """
        Определяет, принадлежат ли два embedding вектора одному человеку
        
        Args:
            embedding1 (numpy.ndarray): Первый embedding вектор
            embedding2 (numpy.ndarray): Второй embedding вектор
            threshold (float, optional): Пороговое значение сходства
            
        Returns:
            bool: True, если векторы принадлежат одному человеку, иначе False
        """
        # Рассчитываем сходство
        similarity = self.calculate_similarity(embedding1, embedding2)
        
        # Сравниваем с пороговым значением
        return similarity > threshold

def get_embedding(image_path):
    """
    Получает embedding вектор из изображения
    
    Args:
        image_path (str): Путь к изображению
        
    Returns:
        dict: Словарь с embedding вектором и дополнительной информацией
    """
    # Инициализация модуля
    face_embedding = FaceEmbedding()
    
    # Извлечение embedding вектора
    result = face_embedding.extract_embedding_from_file(image_path)
    
    return result
