"""
Модуль для анализа текстуры лица с использованием skimage
"""

import numpy as np
from skimage import color, feature, exposure, filters, measure, segmentation
from app.config import TEXTURE_ZONES

class TextureAnalyzer:
    def __init__(self):
        """
        Инициализация анализатора текстуры
        """
        self.texture_zones = TEXTURE_ZONES
    
    def analyze(self, uv_texture, depth_map=None, normal_map=None):
        """
        Анализирует текстуру лица и извлекает текстурные признаки
        
        Args:
            uv_texture (numpy.ndarray): UV-текстура лица размером (H, W, 3)
            depth_map (numpy.ndarray, optional): Карта глубины размером (H, W)
            normal_map (numpy.ndarray, optional): Карта нормалей размером (H, W, 3)
            
        Returns:
            dict: Словарь с текстурными признаками
        """
        if uv_texture is None or uv_texture.size == 0:
            return {'error': 'Invalid UV texture'}
        
        # Преобразуем текстуру в различные цветовые пространства для анализа
        lab_texture = color.rgb2lab(uv_texture)
        gray_texture = color.rgb2gray(uv_texture)
        
        # Словарь для хранения результатов
        texture_features = {
            'color_features': self._analyze_color(uv_texture, lab_texture),
            'texture_features': self._analyze_texture_patterns(gray_texture),
            'zone_features': self._analyze_zones(uv_texture, lab_texture, gray_texture),
            'uniformity_features': self._analyze_uniformity(lab_texture, gray_texture)
        }
        
        # Если предоставлены карты глубины и нормалей, анализируем их
        if depth_map is not None and depth_map.size > 0:
            texture_features['depth_features'] = self._analyze_depth(depth_map)
        
        if normal_map is not None and normal_map.size > 0:
            texture_features['normal_features'] = self._analyze_normals(normal_map)
        
        # Формируем плоский словарь признаков для удобства использования
        flat_features = self._flatten_features(texture_features)
        
        # Добавляем общие метрики и флаги
        flat_features['is_texture_uniform'] = self._is_texture_uniform(flat_features)
        flat_features['texture_anomaly_score'] = self._calculate_anomaly_score(flat_features)
        
        return flat_features
    
    def _analyze_color(self, rgb_texture, lab_texture):
        """
        Анализирует цветовые характеристики текстуры
        
        Args:
            rgb_texture (numpy.ndarray): RGB-текстура
            lab_texture (numpy.ndarray): LAB-текстура
            
        Returns:
            dict: Словарь с цветовыми признаками
        """
        features = {}
        
        # Извлекаем каналы LAB
        l_channel = lab_texture[:, :, 0]
        a_channel = lab_texture[:, :, 1]
        b_channel = lab_texture[:, :, 2]
        
        # Статистика по каналам LAB
        features['l_mean'] = float(np.mean(l_channel))
        features['l_std'] = float(np.std(l_channel))
        features['a_mean'] = float(np.mean(a_channel))
        features['a_std'] = float(np.std(a_channel))
        features['b_mean'] = float(np.mean(b_channel))
        features['b_std'] = float(np.std(b_channel))
        
        # Вычисляем насыщенность
        saturation = np.sqrt(a_channel**2 + b_channel**2)
        features['saturation_mean'] = float(np.mean(saturation))
        features['saturation_std'] = float(np.std(saturation))
        
        # Гистограммы цветов
        l_hist, _ = np.histogram(l_channel, bins=20, range=(0, 100), density=True)
        a_hist, _ = np.histogram(a_channel, bins=20, range=(-128, 127), density=True)
        b_hist, _ = np.histogram(b_channel, bins=20, range=(-128, 127), density=True)
        
        # Энтропия гистограмм
        features['l_entropy'] = float(-np.sum(l_hist * np.log2(l_hist + 1e-10)))
        features['a_entropy'] = float(-np.sum(a_hist * np.log2(a_hist + 1e-10)))
        features['b_entropy'] = float(-np.sum(b_hist * np.log2(b_hist + 1e-10)))
        
        # Анализ равномерности цвета
        features['color_uniformity'] = float(1.0 / (1.0 + features['l_std'] + features['a_std'] + features['b_std']))
        
        # Анализ контрастности
        features['l_contrast'] = float(np.max(l_channel) - np.min(l_channel))
        features['a_contrast'] = float(np.max(a_channel) - np.min(a_channel))
        features['b_contrast'] = float(np.max(b_channel) - np.min(b_channel))
        
        return features
    
    def _analyze_texture_patterns(self, gray_texture):
        """
        Анализирует текстурные паттерны
        
        Args:
            gray_texture (numpy.ndarray): Полутоновое изображение текстуры
            
        Returns:
            dict: Словарь с текстурными признаками
        """
        features = {}
        
        # Нормализуем изображение для текстурного анализа
        norm_texture = exposure.rescale_intensity(gray_texture, out_range=(0, 255)).astype(np.uint8)
        
        # Локальные бинарные паттерны (LBP)
        try:
            lbp = feature.local_binary_pattern(norm_texture, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
            features['lbp_hist'] = lbp_hist.tolist()
            features['lbp_entropy'] = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)))
            features['lbp_uniformity'] = float(np.sum(lbp_hist**2))
        except Exception as e:
            features['lbp_error'] = str(e)
        
        # GLCM (Gray-Level Co-occurrence Matrix)
        try:
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = feature.graycomatrix(norm_texture, distances, angles, 256, symmetric=True, normed=True)
            
            # Извлекаем признаки GLCM
            contrast = feature.graycoprops(glcm, 'contrast')
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
            homogeneity = feature.graycoprops(glcm, 'homogeneity')
            energy = feature.graycoprops(glcm, 'energy')
            correlation = feature.graycoprops(glcm, 'correlation')
            
            features['glcm_contrast'] = float(np.mean(contrast))
            features['glcm_dissimilarity'] = float(np.mean(dissimilarity))
            features['glcm_homogeneity'] = float(np.mean(homogeneity))
            features['glcm_energy'] = float(np.mean(energy))
            features['glcm_correlation'] = float(np.mean(correlation))
        except Exception as e:
            features['glcm_error'] = str(e)
        
        # Градиенты (для анализа микротекстуры)
        try:
            sobel_h = filters.sobel_h(gray_texture)
            sobel_v = filters.sobel_v(gray_texture)
            gradient_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
            
            features['gradient_mean'] = float(np.mean(gradient_magnitude))
            features['gradient_std'] = float(np.std(gradient_magnitude))
            features['gradient_entropy'] = float(-np.sum(np.histogram(
                gradient_magnitude, bins=20, density=True)[0] * np.log2(
                np.histogram(gradient_magnitude, bins=20, density=True)[0] + 1e-10)))
        except Exception as e:
            features['gradient_error'] = str(e)
        
        return features
    
    def _analyze_zones(self, rgb_texture, lab_texture, gray_texture):
        """
        Анализирует текстуру по зонам
        
        Args:
            rgb_texture (numpy.ndarray): RGB-текстура
            lab_texture (numpy.ndarray): LAB-текстура
            gray_texture (numpy.ndarray): Полутоновое изображение текстуры
            
        Returns:
            dict: Словарь с признаками по зонам
        """
        features = {}
        
        # Анализируем каждую зону
        for zone_name, (start, end) in self.texture_zones.items():
            # Вычисляем координаты зоны
            h, w = rgb_texture.shape[:2]
            x1 = int(start[0] * w)
            y1 = int(start[1] * h)
            x2 = int(end[0] * w)
            y2 = int(end[1] * h)
            
            # Проверяем, что координаты в пределах изображения
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Извлекаем зону
            zone_rgb = rgb_texture[y1:y2, x1:x2]
            zone_lab = lab_texture[y1:y2, x1:x2]
            zone_gray = gray_texture[y1:y2, x1:x2]
            
            # Проверяем, что зона не пустая
            if zone_rgb.size == 0:
                continue
            
            # Извлекаем каналы LAB для зоны
            zone_l = zone_lab[:, :, 0]
            zone_a = zone_lab[:, :, 1]
            zone_b = zone_lab[:, :, 2]
            
            # Статистика по каналам LAB для зоны
            features[f'{zone_name}_l_mean'] = float(np.mean(zone_l))
            features[f'{zone_name}_l_std'] = float(np.std(zone_l))
            features[f'{zone_name}_a_mean'] = float(np.mean(zone_a))
            features[f'{zone_name}_a_std'] = float(np.std(zone_a))
            features[f'{zone_name}_b_mean'] = float(np.mean(zone_b))
            features[f'{zone_name}_b_std'] = float(np.std(zone_b))
            
            # Вычисляем насыщенность для зоны
            zone_saturation = np.sqrt(zone_a**2 + zone_b**2)
            features[f'{zone_name}_saturation_mean'] = float(np.mean(zone_saturation))
            features[f'{zone_name}_saturation_std'] = float(np.std(zone_saturation))
            
            # Анализ равномерности цвета для зоны
            features[f'{zone_name}_color_uniformity'] = float(1.0 / (1.0 + features[f'{zone_name}_l_std'] + 
                                                              features[f'{zone_name}_a_std'] + 
                                                              features[f'{zone_name}_b_std']))
            
            # Градиенты для зоны
            try:
                sobel_h = filters.sobel_h(zone_gray)
                sobel_v = filters.sobel_v(zone_gray)
                gradient_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
                
                features[f'{zone_name}_gradient_mean'] = float(np.mean(gradient_magnitude))
                features[f'{zone_name}_gradient_std'] = float(np.std(gradient_magnitude))
            except Exception:
                pass
            
            # LBP для зоны
            try:
                if zone_gray.size > 100:  # Проверяем, что зона достаточно большая
                    norm_zone = exposure.rescale_intensity(zone_gray, out_range=(0, 255)).astype(np.uint8)
                    lbp = feature.local_binary_pattern(norm_zone, P=8, R=1, method='uniform')
                    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
                    features[f'{zone_name}_lbp_entropy'] = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)))
                    features[f'{zone_name}_lbp_uniformity'] = float(np.sum(lbp_hist**2))
            except Exception:
                pass
        
        return features
    
    def _analyze_uniformity(self, lab_texture, gray_texture):
        """
        Анализирует равномерность текстуры
        
        Args:
            lab_texture (numpy.ndarray): LAB-текстура
            gray_texture (numpy.ndarray): Полутоновое изображение текстуры
            
        Returns:
            dict: Словарь с признаками равномерности
        """
        features = {}
        
        # Извлекаем канал L
        l_channel = lab_texture[:, :, 0]
        
        # Сегментация на основе яркости
        try:
            # Применяем пороговую сегментацию
            thresh = filters.threshold_otsu(gray_texture)
            binary = gray_texture > thresh
            
            # Анализируем связные компоненты
            labeled, num_features = measure.label(binary, return_num=True)
            regions = measure.regionprops(labeled)
            
            # Вычисляем статистику по регионам
            if regions:
                areas = [region.area for region in regions]
                perimeters = [region.perimeter for region in regions]
                
                features['segment_count'] = int(num_features)
                features['segment_mean_area'] = float(np.mean(areas))
                features['segment_std_area'] = float(np.std(areas))
                features['segment_mean_perimeter'] = float(np.mean(perimeters))
                
                # Мера равномерности сегментации
                features['segmentation_uniformity'] = float(1.0 / (1.0 + features['segment_std_area'] / features['segment_mean_area']))
        except Exception as e:
            features['segmentation_error'] = str(e)
        
        # Анализ локальной вариации
        try:
            # Вычисляем локальную вариацию с помощью фильтра
            local_var = filters.rank.variance(gray_texture.astype(np.uint8), np.ones((5, 5)))
            
            features['local_variance_mean'] = float(np.mean(local_var))
            features['local_variance_std'] = float(np.std(local_var))
            
            # Мера равномерности локальной вариации
            features['local_variance_uniformity'] = float(1.0 / (1.0 + features['local_variance_mean']))
        except Exception as e:
            features['local_variance_error'] = str(e)
        
        # Анализ текстурной энтропии
        try:
            # Вычисляем локальную энтропию
            local_entropy = filters.rank.entropy(gray_texture.astype(np.uint8), np.ones((5, 5)))
            
            features['local_entropy_mean'] = float(np.mean(local_entropy))
            features['local_entropy_std'] = float(np.std(local_entropy))
            
            # Мера равномерности энтропии
            features['entropy_uniformity'] = float(1.0 / (1.0 + features['local_entropy_std'] / features['local_entropy_mean']))
        except Exception as e:
            features['entropy_error'] = str(e)
        
        return features
    
    def _analyze_depth(self, depth_map):
        """
        Анализирует карту глубины
        
        Args:
            depth_map (numpy.ndarray): Карта глубины
            
        Returns:
            dict: Словарь с признаками глубины
        """
        features = {}
        
        # Статистика по глубине
        features['depth_mean'] = float(np.mean(depth_map))
        features['depth_std'] = float(np.std(depth_map))
        features['depth_min'] = float(np.min(depth_map))
        features['depth_max'] = float(np.max(depth_map))
        features['depth_range'] = float(features['depth_max'] - features['depth_min'])
        
        # Градиенты глубины
        try:
            depth_sobel_h = filters.sobel_h(depth_map)
            depth_sobel_v = filters.sobel_v(depth_map)
            depth_gradient = np.sqrt(depth_sobel_h**2 + depth_sobel_v**2)
            
            features['depth_gradient_mean'] = float(np.mean(depth_gradient))
            features['depth_gradient_std'] = float(np.std(depth_gradient))
            
            # Мера равномерности глубины
            features['depth_uniformity'] = float(1.0 / (1.0 + features['depth_gradient_mean']))
        except Exception as e:
            features['depth_gradient_error'] = str(e)
        
        # Анализ глубины по зонам
        for zone_name, (start, end) in self.texture_zones.items():
            # Вычисляем координаты зоны
            h, w = depth_map.shape[:2]
            x1 = int(start[0] * w)
            y1 = int(start[1] * h)
            x2 = int(end[0] * w)
            y2 = int(end[1] * h)
            
            # Проверяем, что координаты в пределах изображения
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Извлекаем зону
            zone_depth = depth_map[y1:y2, x1:x2]
            
            # Проверяем, что зона не пустая
            if zone_depth.size == 0:
                continue
            
            # Статистика по глубине для зоны
            features[f'{zone_name}_depth_mean'] = float(np.mean(zone_depth))
            features[f'{zone_name}_depth_std'] = float(np.std(zone_depth))
            features[f'{zone_name}_depth_range'] = float(np.max(zone_depth) - np.min(zone_depth))
        
        return features
    
    def _analyze_normals(self, normal_map):
        """
        Анализирует карту нормалей
        
        Args:
            normal_map (numpy.ndarray): Карта нормалей размером (H, W, 3)
            
        Returns:
            dict: Словарь с признаками нормалей
        """
        features = {}
        
        # Проверяем, что карта нормалей имеет правильную форму
        if normal_map.ndim != 3 or normal_map.shape[2] != 3:
            return {'normal_error': 'Invalid normal map shape'}
        
        # Извлекаем компоненты нормалей
        nx = normal_map[:, :, 0]
        ny = normal_map[:, :, 1]
        nz = normal_map[:, :, 2]
        
        # Статистика по компонентам нормалей
        features['nx_mean'] = float(np.mean(nx))
        features['nx_std'] = float(np.std(nx))
        features['ny_mean'] = float(np.mean(ny))
        features['ny_std'] = float(np.std(ny))
        features['nz_mean'] = float(np.mean(nz))
        features['nz_std'] = float(np.std(nz))
        
        # Вычисляем угол наклона нормалей
        elevation = np.arcsin(nz)
        azimuth = np.arctan2(ny, nx)
        
        features['elevation_mean'] = float(np.mean(elevation))
        features['elevation_std'] = float(np.std(elevation))
        features['azimuth_mean'] = float(np.mean(azimuth))
        features['azimuth_std'] = float(np.std(azimuth))
        
        # Мера равномерности нормалей
        features['normal_uniformity'] = float(1.0 / (1.0 + features['nx_std'] + features['ny_std'] + features['nz_std']))
        
        # Анализ нормалей по зонам
        for zone_name, (start, end) in self.texture_zones.items():
            # Вычисляем координаты зоны
            h, w = normal_map.shape[:2]
            x1 = int(start[0] * w)
            y1 = int(start[1] * h)
            x2 = int(end[0] * w)
            y2 = int(end[1] * h)
            
            # Проверяем, что координаты в пределах изображения
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Извлекаем зону
            zone_normals = normal_map[y1:y2, x1:x2]
            
            # Проверяем, что зона не пустая
            if zone_normals.size == 0:
                continue
            
            # Извлекаем компоненты нормалей для зоны
            zone_nx = zone_normals[:, :, 0]
            zone_ny = zone_normals[:, :, 1]
            zone_nz = zone_normals[:, :, 2]
            
            # Статистика по компонентам нормалей для зоны
            features[f'{zone_name}_nx_std'] = float(np.std(zone_nx))
            features[f'{zone_name}_ny_std'] = float(np.std(zone_ny))
            features[f'{zone_name}_nz_std'] = float(np.std(zone_nz))
            
            # Мера равномерности нормалей для зоны
            features[f'{zone_name}_normal_uniformity'] = float(1.0 / (1.0 + features[f'{zone_name}_nx_std'] + 
                                                          features[f'{zone_name}_ny_std'] + 
                                                          features[f'{zone_name}_nz_std']))
        
        return features
    
    def _flatten_features(self, nested_features):
        """
        Преобразует вложенный словарь признаков в плоский словарь
        
        Args:
            nested_features (dict): Вложенный словарь признаков
            
        Returns:
            dict: Плоский словарь признаков
        """
        flat_features = {}
        
        def _flatten(d, prefix=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    _flatten(value, prefix + key + '_')
                else:
                    flat_features[prefix + key] = value
        
        _flatten(nested_features)
        return flat_features
    
    def _is_texture_uniform(self, features):
        """
        Определяет, является ли текстура слишком равномерной (признак маски)
        
        Args:
            features (dict): Словарь признаков
            
        Returns:
            bool: True, если текстура слишком равномерна
        """
        # Признаки, указывающие на равномерность текстуры
        uniformity_indicators = [
            features.get('color_uniformity', 0),
            features.get('glcm_energy', 0),
            features.get('local_variance_uniformity', 0),
            features.get('depth_uniformity', 0) if 'depth_uniformity' in features else 0,
            features.get('normal_uniformity', 0) if 'normal_uniformity' in features else 0
        ]
        
        # Среднее значение индикаторов равномерности
        mean_uniformity = np.mean([u for u in uniformity_indicators if u > 0])
        
        # Проверяем зональные признаки
        zone_uniformities = []
        for key, value in features.items():
            if 'color_uniformity' in key and 'global' not in key:
                zone_uniformities.append(value)
        
        # Если зональные признаки доступны, учитываем их
        if zone_uniformities:
            zone_mean_uniformity = np.mean(zone_uniformities)
            zone_std_uniformity = np.std(zone_uniformities)
            
            # Если стандартное отклонение равномерности по зонам низкое,
            # это может указывать на маску
            if zone_std_uniformity < 0.1 and zone_mean_uniformity > 0.7:
                return True
        
        # Если средняя равномерность выше порога, текстура слишком равномерна
        return mean_uniformity > 0.8
    
    def _calculate_anomaly_score(self, features):
        """
        Рассчитывает оценку аномальности текстуры
        
        Args:
            features (dict): Словарь признаков
            
        Returns:
            float: Оценка аномальности от 0 до 1
        """
        # Признаки, указывающие на аномальность текстуры
        anomaly_indicators = [
            features.get('color_uniformity', 0) * 1.0,  # Высокая равномерность цвета
            (1.0 - features.get('glcm_contrast', 0)) * 0.8,  # Низкий контраст
            features.get('local_variance_uniformity', 0) * 0.9,  # Высокая равномерность локальной вариации
            (1.0 - features.get('gradient_mean', 0)) * 0.7 if 'gradient_mean' in features else 0,  # Низкий градиент
            features.get('depth_uniformity', 0) * 1.2 if 'depth_uniformity' in features else 0,  # Высокая равномерность глубины
            features.get('normal_uniformity', 0) * 1.1 if 'normal_uniformity' in features else 0  # Высокая равномерность нормалей
        ]
        
        # Вычисляем взвешенное среднее индикаторов аномальности
        weights = [1.0, 0.8, 0.9, 0.7, 1.2, 1.1]
        valid_indicators = [ind for ind in anomaly_indicators if ind > 0]
        valid_weights = [weights[i] for i, ind in enumerate(anomaly_indicators) if ind > 0]
        
        if not valid_indicators:
            return 0.0
        
        anomaly_score = np.average(valid_indicators, weights=valid_weights)
        
        # Нормализуем оценку от 0 до 1
        return min(1.0, max(0.0, anomaly_score))

def analyze_texture(uv_texture, depth_map=None, normal_map=None):
    """
    Удобная функция для анализа текстуры лица
    
    Args:
        uv_texture (numpy.ndarray): UV-текстура лица
        depth_map (numpy.ndarray, optional): Карта глубины
        normal_map (numpy.ndarray, optional): Карта нормалей
        
    Returns:
        dict: Словарь с текстурными признаками
    """
    analyzer = TextureAnalyzer()
    return analyzer.analyze(uv_texture, depth_map, normal_map)
