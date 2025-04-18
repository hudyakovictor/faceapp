#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



Модуль анализа текстур кожи с использованием маски ROI.
В данном файле реализованы функции:
  - apply_roi_mask: создание бинарной маски по координатам ROI и применение её к изображению.
  - analyze_gloss: расчет характеристик блеска (глянцевости) кожи.
  - analyze_pores: определение количества и характеристик пор с помощью бинаризации и поиска контуров.
  - analyze_color: анализ цветовых характеристик в цветовом пространстве HSV.
  - analyze_texture_features: вычисление текстурных показателей на основе GLCM, LBP и статистические метрики.
  - analyze_texture: объединяющая функция, которая по исходному изображению и ROI возвращает итоговый словарь с результатами.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage import color






def apply_roi_mask(image, roi):
    """
    Создаёт бинарную маску по ROI и применяет её к изображению.
    """
    import numpy as np
    import cv2
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x_min, y_min, x_max, y_max = roi
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.shape[1])
    y_max = min(y_max, image.shape[0])
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask






def TextureAnalyzer(image, roi):
    """
    Применяет маску ROI к исходному изображению.
    
    Параметры:
      image - исходное изображение (numpy-массив);
      roi - список координат [x_min, y_min, x_max, y_max].
    
    Возвращает:
      masked_image - изображение с нулевыми значениями вне ROI;
      mask - бинарная маска, где область ROI имеет значение 255.
    """
    
    
    
    
    
    
    
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x_min, y_min, x_max, y_max = roi
    # Гарантируем, что координаты не выходят за пределы изображения
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.shape[1])
    y_max = min(y_max, image.shape[0])
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

def analyze_gloss(image):
    """
    Вычисляет характеристики блеска (глянцевости) кожи на основе яркостных свойств.
    
    Если входное изображение цветное, оно преобразуется в оттенки серого.
    
    Расчёты:
      - Используется метод пороговой бинаризации с порогом Otsu, скорректированным коэффициентом;
      - Определяется доля ярких пикселей (gloss_ratio);
      - Вычисляются максимальная яркость и разница между средними значениями яркости ярких и темных областей.
    
    Возвращает словарь с параметрами:
      gloss_ratio, max_brightness, brightness_contrast, is_glossy.
    """
    # Если изображение цветное, переводим в оттенки серого
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Получаем порог по методу Оцу
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Корректировка порога (коэффициент можно настроить)
    adjusted_thresh = thresh_val * 1.2
    _, bright_mask = cv2.threshold(gray, adjusted_thresh, 255, cv2.THRESH_BINARY)
    gloss_ratio = np.sum(bright_mask > 0) / bright_mask.size
    max_brightness = np.max(gray)
    # Вычисление средних значений яркости для ярких и темных областей
    if np.sum(bright_mask > 0) > 0:
        bright_mean = np.mean(gray[bright_mask > 0])
    else:
        bright_mean = 0
    if np.sum(bright_mask == 0) > 0:
        dark_mean = np.mean(gray[bright_mask == 0])
    else:
        dark_mean = 0
    brightness_contrast = (bright_mean - dark_mean) if dark_mean > 0 else 0
    is_glossy = gloss_ratio > 0.1 and brightness_contrast > 15
    return {
        "gloss_ratio": float(gloss_ratio),
        "max_brightness": float(max_brightness),
        "brightness_contrast": float(brightness_contrast),
        "is_glossy": bool(is_glossy)
    }

def analyze_pores(image):
    """
    Анализирует поры на коже с использованием бинаризации и поиска контуров.
    
    Процесс:
      - Изображение переводится в оттенки серого (если оно цветное);
      - Применяется бинаризация с порогом 50 для выделения мелких деталей (пор);
      - Используется функция cv2.findContours для поиска контуров.
    
    Возвращает словарь с:
      pore_count, pore_density, avg_pore_size и флагом has_pores.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Бинаризация: инвертируем изображение для выделения темных областей (пор)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pore_count = len(contours)
    area = np.count_nonzero(gray)
    pore_density = pore_count / area if area > 0 else 0
    avg_pore_size = np.mean([cv2.contourArea(cnt) for cnt in contours]) if contours else 0
    has_pores = pore_count > 20  # значение порога можно скорректировать
    return {
        "pore_count": int(pore_count),
        "pore_density": float(pore_density),
        "avg_pore_size": float(avg_pore_size),
        "has_pores": bool(has_pores)
    }

def analyze_color(image):
    """
    Анализирует цветовые характеристики кожи в пространстве HSV.
    
    При расчётах используются только те пиксели, которые принадлежат ROI (не равны нулю).
    
    Возвращает:
      словарь с параметрами hsv (средние значения каналов H, S, V)
      и флаг is_abnormal, если значения выходят за типичные диапазоны.
    """
    if len(image.shape) != 3:
        return {"hsv": {"h_mean": 0.0, "s_mean": 0.0, "v_mean": 0.0}, "is_abnormal": True}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Выбираем только те пиксели, где хотя бы один канал не равен 0 (то есть внутри ROI)
    mask_nonzero = np.any(image != 0, axis=-1)
    hsv_nonzero = hsv[mask_nonzero]
    if hsv_nonzero.size == 0:
        return {"hsv": {"h_mean": 0.0, "s_mean": 0.0, "v_mean": 0.0}, "is_abnormal": True}
    h_mean = np.mean(hsv_nonzero[:, 0])
    s_mean = np.mean(hsv_nonzero[:, 1])
    v_mean = np.mean(hsv_nonzero[:, 2])
    # Пример допустимых диапазонов (можно скорректировать)
    is_abnormal = not (0 <= h_mean <= 180 and 50 <= s_mean <= 255 and 50 <= v_mean <= 255)
    return {
        "hsv": {"h_mean": float(h_mean), "s_mean": float(s_mean), "v_mean": float(v_mean)},
        "is_abnormal": bool(is_abnormal)
    }

def analyze_texture_features(image):
    """
    Вычисляет текстурные характеристики кожи:
      - GLCM: рассчитываются contrast, dissimilarity, homogeneity, energy и correlation;
      - LBP: вычисляется локально бинарный паттерн и строится его гистограмма;
      - Статистические метрики: среднее, стандартное отклонение и энтропия.
    
    Возвращает словарь с результатами анализа.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Вычисление матрицы GLCM
    glcm = graycomatrix(gray.astype(np.uint8), distances=[1], angles=[0],
                          levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Вычисление локально бинарного паттерна (LBP)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    # Статистические показатели
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    entropy_val = shannon_entropy(gray)

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
            "mean": float(mean_val),
            "std": float(std_val),
            "entropy": float(entropy_val)
        }
    }

def analyze_texture(original_image, roi):
    """
    Основная функция анализа текстурных характеристик.
    
    Принимает:
      original_image - исходное изображение;
      roi - координаты ROI: [x_min, y_min, x_max, y_max].
    
    Сначала создается маска ROI, затем все анализы (глянцевость, поры, цвет, текстура) выполняются
    по изображению с применённой маской.
    
    Возвращает итоговый словарь со следующими ключами:
      gloss, pores, color, texture.
    """
    masked_image, _ = apply_roi_mask(original_image, roi)
    gloss_data = analyze_gloss(masked_image)
    pores_data = analyze_pores(masked_image)
    color_data = analyze_color(masked_image)
    texture_feats = analyze_texture_features(masked_image)
    return {
        "gloss": gloss_data,
        "pores": pores_data,
        "color": color_data,
        "texture": texture_feats
    }

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Анализ текстур изображения с использованием ROI")
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--roi', type=int, nargs=4, required=True, metavar=('x_min', 'y_min', 'x_max', 'y_max'),
                        help='Координаты ROI: x_min, y_min, x_max, y_max')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print("Не удалось загрузить изображение:", args.image)
        exit(1)

    result = analyze_texture(image, args.roi)
    print(json.dumps(result, indent=4, ensure_ascii=False))
