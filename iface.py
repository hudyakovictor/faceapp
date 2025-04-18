import cv2
import numpy as np
import insightface
import os
import hashlib
import json
import logging
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FaceEmbedder')

class FaceEmbedder:
    """
    Класс для работы с лицевыми эмбеддингами через InsightFace.
    Включает кэширование, многоуровневую детекцию и предобработку изображений.
    """
    
    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0, 
                 cache_dir: Optional[str] = None, use_multi_scale: bool = True):
        """
        Инициализирует модель InsightFace.
        
        Args:
            model_name: Название модели InsightFace
            ctx_id: ID контекста (0 для CPU, >0 для GPU)
            cache_dir: Директория для кэширования результатов
            use_multi_scale: Использовать ли многоуровневую детекцию
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.use_multi_scale = use_multi_scale
        self.detection_sizes = [(640, 640), (480, 480), (800, 800), (1000, 1000)]
        
        # Инициализация модели с фиксированным seed для воспроизводимости
        np.random.seed(42)
        self.model = insightface.app.FaceAnalysis(name=model_name)
        self.model.prepare(ctx_id=ctx_id, det_size=self.detection_sizes[0])
        
        # Настройка кэширования
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Кэш эмбеддингов будет сохраняться в {self.cache_dir}")
        
        # Счетчик успешных и неуспешных детекций
        self.stats = {"success": 0, "failed": 0}
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для улучшения детекции.
        
        Args:
            img: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        # Проверка на None
        if img is None:
            return None
            
        # Нормализация яркости и контраста
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Проверка размера изображения и масштабирование при необходимости
        height, width = enhanced.shape[:2]
        max_dim = 1280
        if max(height, width) > max_dim:
            scale = min(max_dim / width, max_dim / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            enhanced = cv2.resize(enhanced, (new_width, new_height))
            
        return enhanced
    
    def get_face_data_multi_scale(self, img: np.ndarray) -> Optional[List[Dict]]:
        """
        Многоуровневая детекция лиц с разными размерами.
        
        Args:
            img: Исходное изображение
            
        Returns:
            Список словарей с данными о лицах или None
        """
        if img is None:
            return None
        
        all_faces = []
        
        # Пробуем разные размеры детекции
        for det_size in self.detection_sizes:
            self.model.prepare(ctx_id=self.ctx_id, det_size=det_size)
            faces = self.model.get(img)
            if faces:
                all_faces.extend(faces)
                break  # Останавливаемся, если нашли лица
        
        # Возвращаем стандартный размер детекции
        self.model.prepare(ctx_id=self.ctx_id, det_size=self.detection_sizes[0])
        
        if not all_faces:
            return None
        
        # Удаляем дубликаты по IOU
        unique_faces = self._remove_duplicate_faces(all_faces)
        
        return [{
            "embedding": face.embedding.tolist(),
            "age": int(face.age),
            "bbox": face.bbox.tolist() if hasattr(face, 'bbox') else None,
            "det_score": float(face.det_score) if hasattr(face, 'det_score') else None
        } for face in unique_faces]
    
    def _remove_duplicate_faces(self, faces: List) -> List:
        """
        Удаляет дубликаты лиц на основе IOU (Intersection over Union).
        
        Args:
            faces: Список обнаруженных лиц
            
        Returns:
            Список уникальных лиц
        """
        if len(faces) <= 1:
            return faces
            
        def compute_iou(box1, box2):
            # Вычисление IOU между двумя боксами
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        # Сортируем лица по уверенности детекции
        sorted_faces = sorted(faces, key=lambda x: x.det_score if hasattr(x, 'det_score') else 0, reverse=True)
        
        unique_faces = []
        for face in sorted_faces:
            if not hasattr(face, 'bbox'):
                unique_faces.append(face)
                continue
                
            is_duplicate = False
            for unique_face in unique_faces:
                if not hasattr(unique_face, 'bbox'):
                    continue
                    
                iou = compute_iou(face.bbox, unique_face.bbox)
                if iou > 0.5:  # Порог IOU
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_faces.append(face)
                
        return unique_faces
    
    def get_face_data(self, image_path: str) -> Optional[List[Dict]]:
        """
        Получает данные о лицах из изображения с кэшированием результатов.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Список словарей с данными о лицах или None
        """
        # Проверяем кэш
        cache_result = self._check_cache(image_path)
        if cache_result is not None:
            self.stats["success"] += 1
            return cache_result
        
        # Загружаем и предобрабатываем изображение
        img = cv2.imread(image_path)
        if img is None:
            self.stats["failed"] += 1
            logger.error(f"Не удалось загрузить изображение: {image_path}")
            return None
        
        # Предобработка изображения
        img = self.preprocess_image(img)
        
        # Многоуровневая детекция или стандартная
        if self.use_multi_scale:
            result = self.get_face_data_multi_scale(img)
        else:
            faces = self.model.get(img)
            if not faces:
                self.stats["failed"] += 1
                logger.warning(f"Лица не обнаружены в изображении: {image_path}")
                return None
                
            result = [{
                "embedding": face.embedding.tolist(),
                "age": int(face.age),
                "bbox": face.bbox.tolist() if hasattr(face, 'bbox') else None,
                "det_score": float(face.det_score) if hasattr(face, 'det_score') else None
            } for face in faces]
        
        # Сохраняем в кэш
        if result:
            self._save_to_cache(image_path, result)
            self.stats["success"] += 1
        else:
            self.stats["failed"] += 1
            
        return result
    
    def _get_cache_path(self, image_path: str) -> str:
        """
        Получает путь к файлу кэша для изображения.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Путь к файлу кэша
        """
        # Создаем хэш изображения для уникального идентификатора
        img_hash = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
        return os.path.join(self.cache_dir, f"{img_hash}.json")
    
    def _check_cache(self, image_path: str) -> Optional[List[Dict]]:
        """
        Проверяет наличие результатов в кэше.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Данные из кэша или None
        """
        cache_file = self._get_cache_path(image_path)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Ошибка при чтении кэша: {e}")
                return None
        return None
    
    def _save_to_cache(self, image_path: str, result: List[Dict]) -> None:
        """
        Сохраняет результаты в кэш.
        
        Args:
            image_path: Путь к изображению
            result: Результаты для сохранения
        """
        cache_file = self._get_cache_path(image_path)
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Ошибка при сохранении в кэш: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Возвращает статистику работы детектора.
        
        Returns:
            Словарь со статистикой
        """
        return {
            "success": self.stats["success"],
            "failed": self.stats["failed"],
            "total": self.stats["success"] + self.stats["failed"],
            "success_rate": self.stats["success"] / (self.stats["success"] + self.stats["failed"]) * 100 if (self.stats["success"] + self.stats["failed"]) > 0 else 0
        }
    
    def compare_faces(self, face1: Union[str, List[float]], face2: Union[str, List[float]], 
                     threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Сравнивает два лица и определяет, принадлежат ли они одному человеку.
        
        Args:
            face1: Путь к изображению или эмбеддинг первого лица
            face2: Путь к изображению или эмбеддинг второго лица
            threshold: Порог сходства (0-1)
            
        Returns:
            Tuple[bool, float]: (Совпадение, Уровень сходства)
        """
        # Получаем эмбеддинги
        embedding1 = face1
        embedding2 = face2
        
        # Если переданы пути к изображениям, получаем эмбеддинги
        if isinstance(face1, str):
            face_data = self.get_face_data(face1)
            if not face_data:
                return False, 0.0
            embedding1 = face_data[0]["embedding"]
            
        if isinstance(face2, str):
            face_data = self.get_face_data(face2)
            if not face_data:
                return False, 0.0
            embedding2 = face_data[0]["embedding"]
        
        # Вычисляем косинусное сходство
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return False, 0.0
            
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Возвращаем результат сравнения
        return similarity > threshold, float(similarity)
    
    def get_average_embedding(self, image_paths: List[str]) -> Optional[List[float]]:
        """
        Вычисляет средний эмбеддинг для нескольких изображений одного человека.
        
        Args:
            image_paths: Список путей к изображениям
            
        Returns:
            Средний эмбеддинг или None
        """
        embeddings = []
        
        for path in image_paths:
            face_data = self.get_face_data(path)
            if face_data and len(face_data) > 0:
                embeddings.append(np.array(face_data[0]["embedding"]))
        
        if not embeddings:
            return None
            
        # Вычисляем средний эмбеддинг
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Нормализуем
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
            
        return avg_embedding.tolist()
    
    def clear_cache(self) -> None:
        """
        Очищает кэш эмбеддингов.
        """
        cache_files = os.listdir(self.cache_dir)
        for file in cache_files:
            if file.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, file))
        logger.info(f"Кэш очищен, удалено {len(cache_files)} файлов")

# Глобальный инстанс (загрузится при импорте)
face_embedder = FaceEmbedder()
