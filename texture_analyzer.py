import numpy as np
from skimage import io, color, feature, filters, exposure, measure, segmentation, util
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


class TextureAnalyzer:
    def __init__(self):
        """Инициализация анализатора текстур"""
        pass
    
    def analyze(self, image_path, landmarks=None, roi=None):
        """
        Анализирует текстуры кожи на изображении
        
        Параметры:
        - image_path: путь к изображению
        - landmarks: ключевые точки лица (опционально)
        - roi: область интереса (опционально)
        
        Возвращает:
        - словарь с результатами анализа текстур
        """
        # Загрузка изображения
        try:
            image = io.imread(image_path)
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")
            return None
        
        # Преобразование в оттенки серого
        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image
        
        # Анализ глянцевости (отражения света)
        gloss_data = self._analyze_gloss(gray_image)
        
        # Анализ пор
        pore_data = self._analyze_pores(gray_image)
        
        # Анализ цвета кожи
        color_data = self._analyze_skin_color(image)
        
        # Анализ текстуры
        texture_data = self._analyze_texture(gray_image)
        
        # Формирование результата
        result = {
            "gloss": gloss_data,
            "pores": pore_data,
            "color": color_data,
            "texture": texture_data
        }
        
        return result
    
    def _analyze_gloss(self, gray_image):
        """Анализирует глянцевость (отражение света) на коже"""
        # Находим яркие области на изображении
        thresh = filters.threshold_otsu(gray_image)
        binary = gray_image > (thresh * 1.5)  # Увеличиваем порог для выделения ярких областей
        
        # Анализируем статистику ярких областей
        gloss_ratio = np.sum(binary) / binary.size
        
        # Определяем максимальные значения яркости
        max_brightness = np.max(gray_image)
        
        # Вычисляем контраст между яркими и обычными областями
        if np.sum(binary) > 0:
            bright_mean = np.mean(gray_image[binary])
            normal_mean = np.mean(gray_image[~binary])
            brightness_contrast = bright_mean / normal_mean if normal_mean > 0 else 0
        else:
            brightness_contrast = 0
        
        return {
            "gloss_ratio": float(gloss_ratio),
            "max_brightness": float(max_brightness),
            "brightness_contrast": float(brightness_contrast),
            "is_glossy": gloss_ratio > 0.05  # Пороговое значение для определения глянцевости
        }
    
    def _analyze_pores(self, gray_image):
        """Анализирует поры на коже"""
        # Применяем фильтр Габора для выделения текстурных особенностей
        gabor_real, gabor_imag = filters.gabor(gray_image, frequency=0.6)
        
        # Применяем детектор пятен для обнаружения пор
        blobs = feature.blob_dog(gabor_real, min_sigma=1, max_sigma=3, threshold=0.01)
        
        # Вычисляем плотность пор
        pore_density = len(blobs) / (gray_image.shape[0] * gray_image.shape[1])
        
        # Вычисляем средний размер пор
        if len(blobs) > 0:
            avg_pore_size = np.mean(blobs[:, 2])  # Третий столбец содержит радиус
        else:
            avg_pore_size = 0
        
        return {
            "pore_count": int(len(blobs)),
            "pore_density": float(pore_density),
            "avg_pore_size": float(avg_pore_size),
            "has_pores": len(blobs) > 100  # Пороговое значение для определения наличия пор
        }
    
    def _analyze_skin_color(self, image):
        """Анализирует цвет кожи"""
        # Преобразуем изображение в цветовое пространство HSV
        if len(image.shape) == 3 and image.shape[2] >= 3:
            hsv_image = color.rgb2hsv(image)
            
            # Вычисляем средние значения H, S, V для всего изображения
            h_mean = np.mean(hsv_image[:, :, 0])
            s_mean = np.mean(hsv_image[:, :, 1])
            v_mean = np.mean(hsv_image[:, :, 2])
            
            # Определяем, является ли цвет кожи аномальным
            # Это упрощенная логика, которую нужно адаптировать
            is_abnormal = (h_mean < 0.05 or h_mean > 0.1 or 
                          s_mean < 0.2 or s_mean > 0.6 or 
                          v_mean < 0.3 or v_mean > 0.9)
            
            return {
                "hsv": {
                    "h_mean": float(h_mean),
                    "s_mean": float(s_mean),
                    "v_mean": float(v_mean)
                },
                "is_abnormal": bool(is_abnormal)
            }
        else:
            return {
                "hsv": {
                    "h_mean": 0,
                    "s_mean": 0,
                    "v_mean": 0
                },
                "is_abnormal": False
            }
    
    def _analyze_texture(self, gray_image):
        """Анализирует текстуру кожи"""
        # Вычисляем локальные бинарные шаблоны (LBP)
        lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
        
        # Применяем сегментацию SLIC для создания суперпикселей
        segments = segmentation.slic(util.img_as_float(gray_image), n_segments=100, compactness=10)
        num_segments = len(np.unique(segments))
        
        # Вычисляем GLCM (матрицу совместной встречаемости уровней серого)
        glcm = graycomatrix(
            (gray_image * 255).astype(np.uint8), 
            distances=[1, 3, 5], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=256, 
            symmetric=True, 
            normed=True
        )
        
        # Извлекаем признаки из GLCM
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Вычисляем статистические характеристики
        mean = np.mean(gray_image)
        std = np.std(gray_image)
        entropy = measure.shannon_entropy(gray_image)
        
        return {
            "glcm": {
                "contrast": float(contrast),
                "dissimilarity": float(dissimilarity),
                "homogeneity": float(homogeneity),
                "energy": float(energy),
                "correlation": float(correlation)
            },
            "lbp_histogram": lbp_hist.tolist(),
            "statistics": {
                "mean": float(mean),
                "std": float(std),
                "entropy": float(entropy)
            },
            "segmentation": {
                "num_segments": int(num_segments)
            }
        }
