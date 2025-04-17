"""
Конфигурационный файл с константами и настройками для системы анализа лиц
"""

# Пути к моделям (при необходимости настройте пути)
TDDFA_MODEL_PATH = './weights/mb1_120x120.pth'
TDDFA_BFM_PATH = './configs/bfm_noneck_v3.pkl'
FAN_MODEL_PATH = './weights/2DFAN4-11f355bf06.pth'
INSIGHTFACE_MODEL_PATH = './weights/buffalo_l/w600k_r50.onnx'

# Пороговые значения для confidence
CONFIDENCE_THRESHOLD = 0.6
MP_CONFIDENCE_THRESHOLD = 0.7

# Пороговые значения для определения аномалий
SHAPE_ERROR_THRESHOLD = 0.5
DEVIATION_THRESHOLD = 0.15
TEXTURE_UNIFORMITY_THRESHOLD = 0.8

# Углы для определения ракурса (в градусах)
YAW_THRESHOLDS = {
    'frontal': (-15, 15),
    'half_profile_left': (-45, -15),
    'half_profile_right': (15, 45),
    'profile_left': (-90, -45),
    'profile_right': (45, 90)
}

PITCH_THRESHOLDS = {
    'normal': (-20, 20),
    'up': (20, 45),
    'down': (-45, -20)
}

ROLL_THRESHOLDS = {
    'normal': (-15, 15),
    'tilted_left': (-45, -15),
    'tilted_right': (15, 45)
}

# Активные метрики для каждого ракурса
ACTIVE_METRICS = {
    'frontal': [
        'eye_distance', 'eye_ratio', 'nose_width', 'mouth_width', 
        'face_width', 'face_height', 'jaw_width', 'symmetry'
    ],
    'half_profile_left': [
        'visible_eye_size', 'nose_projection', 'cheek_contour', 
        'jaw_angle', 'face_depth'
    ],
    'half_profile_right': [
        'visible_eye_size', 'nose_projection', 'cheek_contour', 
        'jaw_angle', 'face_depth'
    ],
    'profile_left': [
        'nose_projection', 'face_depth', 'jaw_line', 'forehead_curve'
    ],
    'profile_right': [
        'nose_projection', 'face_depth', 'jaw_line', 'forehead_curve'
    ]
}

# Активные метрики MediaPipe для каждого ракурса
MP_ACTIVE_METRICS = {
    'frontal': [
        'eyebrow_angle', 'eye_aspect_ratio', 'mouth_aspect_ratio', 
        'face_oval_ratio', 'nose_tip_position'
    ],
    'half_profile_left': [
        'visible_eye_ratio', 'cheek_contour', 'jaw_projection'
    ],
    'half_profile_right': [
        'visible_eye_ratio', 'cheek_contour', 'jaw_projection'
    ],
    'profile_left': [
        'nose_projection', 'forehead_slope', 'chin_projection'
    ],
    'profile_right': [
        'nose_projection', 'forehead_slope', 'chin_projection'
    ]
}

# Индексы ключевых точек для 3DDFA/FAN (68 точек)
# Определение групп точек для различных частей лица
LANDMARKS_INDICES = {
    'jaw': list(range(0, 17)),
    'right_eyebrow': list(range(17, 22)),
    'left_eyebrow': list(range(22, 27)),
    'nose_bridge': list(range(27, 31)),
    'nose_tip': list(range(31, 36)),
    'right_eye': list(range(36, 42)),
    'left_eye': list(range(42, 48)),
    'outer_lips': list(range(48, 60)),
    'inner_lips': list(range(60, 68))
}

# Индексы ключевых точек для MediaPipe (сокращенный список из 468 точек)
MP_LANDMARKS_INDICES = {
    'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162],
    'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'left_eyebrow': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    'right_eyebrow': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    'nose': [1, 2, 3, 4, 5, 6, 168, 197, 195, 5, 4, 98, 97, 2, 326, 327],
    'lips': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
}

# Зоны для анализа текстуры
TEXTURE_ZONES = {
    'forehead': [(0.3, 0.1), (0.7, 0.2)],  # x1, y1, x2, y2 в UV-координатах
    'left_cheek': [(0.2, 0.3), (0.4, 0.6)],
    'right_cheek': [(0.6, 0.3), (0.8, 0.6)],
    'nose': [(0.4, 0.3), (0.6, 0.5)],
    'chin': [(0.4, 0.7), (0.6, 0.9)]
}

# Параметры для анализа карты глубины
DEPTH_ZONES = {
    'forehead': [(0.3, 0.1), (0.7, 0.2)],
    'left_cheek': [(0.2, 0.3), (0.4, 0.6)],
    'right_cheek': [(0.6, 0.3), (0.8, 0.6)],
    'nose': [(0.4, 0.3), (0.6, 0.5)],
    'chin': [(0.4, 0.7), (0.6, 0.9)]
}

# Параметры для анализа карты нормалей
NORMAL_ZONES = {
    'forehead': [(0.3, 0.1), (0.7, 0.2)],
    'left_cheek': [(0.2, 0.3), (0.4, 0.6)],
    'right_cheek': [(0.6, 0.3), (0.8, 0.6)],
    'nose': [(0.4, 0.3), (0.6, 0.5)],
    'chin': [(0.4, 0.7), (0.6, 0.9)]
}

# Параметры для определения аномалий
ANOMALY_THRESHOLDS = {
    'shape_error': 0.5,
    'deviation_total': 0.15,
    'deviation_symmetry_diff': 0.2,
    'texture_uniformity': 0.8,
    'texture_contrast': 0.3,
    'depth_variation': 0.2
}

# Параметры для InsightFace
INSIGHTFACE_PARAMS = {
    'det_size': (640, 640),
    'recognition_threshold': 0.5,
    'use_gpu': True
}

# Параметры для сохранения результатов
OUTPUT_PARAMS = {
    'save_visualization': True,
    'save_landmarks': True,
    'save_uv_texture': True,
    'save_depth_map': True,
    'save_normal_map': True,
    'save_obj': False,
    'save_ply': False
}

# Параметры для логирования
LOGGING_PARAMS = {
    'log_level': 'INFO',
    'log_file': 'face_analysis.log',
    'console_output': True
}

# Соответствие индексов между 3DDFA/FAN и MediaPipe
# Ключевые точки для сравнения (индексы 3DDFA/FAN -> индексы MediaPipe)
LANDMARK_CORRESPONDENCE = {
    0: 10,    # Челюсть (левый край)
    8: 152,   # Подбородок
    16: 323,  # Челюсть (правый край)
    27: 6,    # Переносица
    30: 4,    # Кончик носа
    36: 33,   # Правый глаз (внешний угол)
    39: 133,  # Правый глаз (внутренний угол)
    42: 362,  # Левый глаз (внутренний угол)
    45: 263,  # Левый глаз (внешний угол)
    48: 61,   # Рот (левый угол)
    51: 0,    # Верхняя губа (центр)
    54: 291,  # Рот (правый угол)
    57: 17    # Нижняя губа (центр)
}

# Параметры для обнаружения масок
MASK_DETECTION_PARAMS = {
    'texture_uniformity_threshold': 0.85,
    'depth_variation_threshold': 0.15,
    'normal_smoothness_threshold': 0.8,
    'symmetry_threshold': 0.9
}

# Параметры для обнаружения двойников
DOPPELGANGER_DETECTION_PARAMS = {
    'embedding_similarity_threshold': 0.7,
    'geometric_similarity_threshold': 0.8,
    'texture_similarity_threshold': 0.75
}
