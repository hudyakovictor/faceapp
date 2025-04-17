"""
Модуль для расчета текстурных метрик лица
"""

import numpy as np
from skimage import color, feature, exposure
from app.config import TEXTURE_ZONES

def calculate_texture_metrics(uv_texture):
    """
    Рассчитывает текстурные метрики на основе UV-текстуры
    
    Args:
        uv_texture (numpy.ndarray): UV-текстура лица
        
    Returns:
        dict: Словарь с текстурными метриками
    """
    metrics = {}
    
    # Преобразуем текстуру в LAB цветовое пространство
    lab_texture = color.rgb2lab(uv_texture)
    
    # Рассчитываем метрики для каждой зоны
    for zone_name, (start, end) in TEXTURE_ZONES.items():
        # Вычисляем координаты зоны
        x1 = int(start[0] * uv_texture.shape[1])
        y1 = int(start[1] * uv_texture.shape[0])
        x2 = int(end[0] * uv_texture.shape[1])
        y2 = int(end[1] * uv_texture.shape[0])
        
        # Извлекаем зону
        zone = uv_texture[y1:y2, x1:x2]
        zone_lab = lab_texture[y1:y2, x1:x2]
        
        # Рассчитываем текстурные метрики для зоны
        zone_metrics = calculate_zone_metrics(zone, zone_lab, zone_name)
        
        # Добавляем метрики зоны в общий словарь
        metrics.update(zone_metrics)
    
    # Рассчитываем глобальные текстурные метрики
    global_metrics = calculate_global_texture_metrics(uv_texture, lab_texture)
    metrics.update(global_metrics)
    
    return metrics

def calculate_zone_metrics(zone, zone_lab, zone_name):
    """
    Рассчитывает текстурные метрики для зоны
    
    Args:
        zone (numpy.ndarray): RGB-текстура зоны
        zone_lab (numpy.ndarray): LAB-текстура зоны
        zone_name (str): Название зоны
        
    Returns:
        dict: Словарь с текстурными метриками для зоны
    """
    metrics = {}
    
    # Проверяем, что зона не пустая
    if zone.size == 0:
        return metrics
    
    # Извлекаем каналы LAB
    l_channel = zone_lab[:, :, 0]
    a_channel = zone_lab[:, :, 1]
    b_channel = zone_lab[:, :, 2]
    
    # Вычисляем однородность текстуры (по каналу L)
    l_std = np.std(l_channel)
    metrics[f'{zone_name}_uniformity'] = float(1.0 / (1.0 + l_std))
    
    # Вычисляем вариацию цвета (по каналам A и B)
    a_std = np.std(a_channel)
    b_std = np.std(b_channel)
    metrics[f'{zone_name}_color_variation'] = float(a_std + b_std)
    
    # Вычисляем контрастность
    l_contrast = np.max(l_channel) - np.min(l_channel)
    metrics[f'{zone_name}_contrast'] = float(l_contrast)
    
    # Вычисляем текстурные признаки GLCM
    if l_channel.size > 100:  # Проверяем, что зона достаточно большая
        # Нормализуем канал L для GLCM
        l_norm = exposure.rescale_intensity(l_channel, out_range=(0, 255)).astype(np.uint8)
        
        # Вычисляем GLCM
        glcm = feature.graycomatrix(l_norm, [1], [0], 256, symmetric=True, normed=True)
        
        # Вычисляем признаки GLCM
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        energy = feature.graycoprops(glcm, 'energy')[0, 0]
        correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
        
        metrics[f'{zone_name}_glcm_contrast'] = float(contrast)
        metrics[f'{zone_name}_glcm_dissimilarity'] = float(dissimilarity)
        metrics[f'{zone_name}_glcm_homogeneity'] = float(homogeneity)
        metrics[f'{zone_name}_glcm_energy'] = float(energy)
        metrics[f'{zone_name}_glcm_correlation'] = float(correlation)
    
    # Вычисляем локальные бинарные паттерны (LBP)
    if l_channel.size > 100:
        l_norm = exposure.rescale_intensity(l_channel, out_range=(0, 255)).astype(np.uint8)
        lbp = feature.local_binary_pattern(l_norm, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
        
        # Энтропия LBP
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        metrics[f'{zone_name}_lbp_entropy'] = float(lbp_entropy)
    
    return metrics

def calculate_global_texture_metrics(texture, lab_texture):
    """
    Рассчитывает глобальные текстурные метрики
    
    Args:
        texture (numpy.ndarray): RGB-текстура
        lab_texture (numpy.ndarray): LAB-текстура
        
    Returns:
        dict: Словарь с глобальными текстурными метриками
    """
    metrics = {}
    
    # Извлекаем каналы LAB
    l_channel = lab_texture[:, :, 0]
    a_channel = lab_texture[:, :, 1]
    b_channel = lab_texture[:, :, 2]
    
    # Вычисляем глобальную однородность текстуры
    l_std = np.std(l_channel)
    metrics['global_uniformity'] = float(1.0 / (1.0 + l_std))
    
    # Вычисляем глобальную вариацию цвета
    a_std = np.std(a_channel)
    b_std = np.std(b_channel)
    metrics['global_color_variation'] = float(a_std + b_std)
    
    # Вычисляем глобальную контрастность
    l_contrast = np.max(l_channel) - np.min(l_channel)
    metrics['global_contrast'] = float(l_contrast)
    
    # Вычисляем энтропию изображения
    hist, _ = np.histogram(l_channel, bins=256, density=True)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    metrics['global_entropy'] = float(entropy)
    
    # Вычисляем среднюю насыщенность
    saturation = np.sqrt(a_channel**2 + b_channel**2)
    metrics['global_saturation'] = float(np.mean(saturation))
    
    # Вычисляем равномерность текстуры по зонам
    zone_uniformities = [v for k, v in metrics.items() if 'uniformity' in k and k != 'global_uniformity']
    if zone_uniformities:
        metrics['zone_uniformity_std'] = float(np.std(zone_uniformities))
    
    return metrics
