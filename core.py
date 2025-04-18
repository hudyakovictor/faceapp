import numpy

def calculate_confidence_by_pose(metric_name, yaw, pitch, roll):
    """
    Расчет confidence для метрик на основе углов поворота головы
    Args:
        metric_name (str): Название метрики
        yaw (float): Угол поворота головы по оси Y (влево-вправо)
        pitch (float): Угол поворота головы по оси X (вверх-вниз)
        roll (float): Угол поворота головы по оси Z (наклон)
    Returns:
        float: Значение confidence от 0.0 до 1.0
    """
    # Оптимальные диапазоны для разных типов метрик
    optimal_ranges = {
        # Фронтальные метрики
        "face_asymmetry": {"yaw": (-15, 15), "pitch": (-20, 20), "roll": (-15, 15)},
        "eye_angle": {"yaw": (-20, 20), "pitch": (-15, 15), "roll": (-10, 10)},
        "eye_distance": {"yaw": (-25, 25), "pitch": (-20, 20), "roll": (-15, 15)},
        "eye_socket_depth": {"yaw": (-15, 15), "pitch": (-15, 15), "roll": (-10, 10)},
        "eye_asymmetry": {"yaw": (-10, 10), "pitch": (-15, 15), "roll": (-10, 10)},
        "face_width": {"yaw": (-20, 20), "pitch": (-25, 25), "roll": (-15, 15)},
        "nose_depth": {"yaw": (-15, 15), "pitch": (-10, 10), "roll": (-15, 15)},
        "cheek_width": {"yaw": (-20, 20), "pitch": (-20, 20), "roll": (-15, 15)},
        "mouth_width": {"yaw": (-20, 20), "pitch": (-15, 15), "roll": (-15, 15)},
        "nose_width": {"yaw": (-15, 15), "pitch": (-15, 15), "roll": (-15, 15)},
        "brow_height": {"yaw": (-20, 20), "pitch": (-15, 15), "roll": (-10, 10)},
        "brow_angle": {"yaw": (-15, 15), "pitch": (-15, 15), "roll": (-5, 5)},
        "chin_height": {"yaw": (-25, 25), "pitch": (-10, 10), "roll": (-15, 15)},
        "jaw_asymmetry": {"yaw": (-15, 15), "pitch": (-15, 15), "roll": (-10, 10)},
        "forehead_width": {"yaw": (-20, 20), "pitch": (-25, 25), "roll": (-15, 15)},
        
        # Метрики для профиля
        "left_chin_depth": {"yaw": (-60, -30), "pitch": (-15, 15), "roll": (-15, 15)},
        "left_nose_depth": {"yaw": (-60, -30), "pitch": (-15, 15), "roll": (-15, 15)},
        "left_nose_width": {"yaw": (-60, -30), "pitch": (-15, 15), "roll": (-15, 15)},
        "left_chin_height": {"yaw": (-60, -30), "pitch": (-15, 15), "roll": (-15, 15)},
        "right_chin_depth": {"yaw": (30, 60), "pitch": (-15, 15), "roll": (-15, 15)},
        "right_nose_depth": {"yaw": (30, 60), "pitch": (-15, 15), "roll": (-15, 15)},
        "right_nose_width": {"yaw": (30, 60), "pitch": (-15, 15), "roll": (-15, 15)},
        "right_chin_height": {"yaw": (30, 60), "pitch": (-15, 15), "roll": (-15, 15)},
        
        # Метрики для полупрофиля
        "semi_left_nose_depth": {"yaw": (-30, -15), "pitch": (-15, 15), "roll": (-15, 15)},
        "semi_left_skull_width": {"yaw": (-30, -15), "pitch": (-15, 15), "roll": (-15, 15)},
        "semi_right_nose_depth": {"yaw": (15, 30), "pitch": (-15, 15), "roll": (-15, 15)},
        "semi_right_skull_width": {"yaw": (15, 30), "pitch": (-15, 15), "roll": (-15, 15)},
        
        # FAN метрики (добавляем префикс fn_)
        "fn_inter_pupil_distance": {"yaw": (-20, 20), "pitch": (-20, 20), "roll": (-15, 15)},
        "fn_left_eye_openness": {"yaw": (-25, 5), "pitch": (-15, 15), "roll": (-15, 15)},
        "fn_right_eye_openness": {"yaw": (-5, 25), "pitch": (-15, 15), "roll": (-15, 15)},
        "fn_eye_symmetry_score": {"yaw": (-15, 15), "pitch": (-15, 15), "roll": (-10, 10)},
    }
    
    # Стандартный диапазон, если метрика не найдена в словаре
    default_range = {"yaw": (-30, 30), "pitch": (-20, 20), "roll": (-15, 15)}
    
    # Получаем оптимальный диапазон для метрики
    metric_range = optimal_ranges.get(metric_name, default_range)
    
    # Вычисляем confidence для каждого угла
    def calc_angle_confidence(angle, min_angle, max_angle):
        if min_angle <= angle <= max_angle:
            return 1.0
        elif angle < min_angle:
            # Линейное убывание при выходе за пределы диапазона
            # Полностью теряем confidence при отклонении на 20 градусов
            return max(0.0, 1.0 - abs(angle - min_angle) / 20.0)
        else:  # angle > max_angle
            return max(0.0, 1.0 - abs(angle - max_angle) / 20.0)
    
    yaw_confidence = calc_angle_confidence(yaw, metric_range["yaw"][0], metric_range["yaw"][1])
    pitch_confidence = calc_angle_confidence(pitch, metric_range["pitch"][0], metric_range["pitch"][1])
    roll_confidence = calc_angle_confidence(roll, metric_range["roll"][0], metric_range["roll"][1])
    
    # Итоговый confidence - минимальный из трех углов
    final_confidence = min(yaw_confidence, pitch_confidence, roll_confidence)
    
    return final_confidence

def filter_metrics_by_confidence(metrics, confidence_values, threshold=0.5):
    """
    Фильтрация метрик по порогу confidence
    Args:
        metrics (dict): Словарь с метриками {имя_метрики: значение}
        confidence_values (dict): Словарь с confidence {имя_метрики: confidence}
        threshold (float): Пороговое значение confidence (0.0-1.0)
    Returns:
        dict: Отфильтрованные метрики с их confidence
    """
    filtered_metrics = {}
    for metric_name, value in metrics.items():
        confidence = confidence_values.get(metric_name, 0.0)
        if confidence >= threshold:
            filtered_metrics[metric_name] = {
                "value": value,
                "confidence": confidence
            }
    return filtered_metrics

def compute_anomaly_score_v2(fan_pts, ddfa_pts, 
                            yaw=0.0, pitch=0.0, roll=0.0,
                            shape_error_ddfa=0.1,
                            face_center=(120, 110), scale=1.0,
                            zone_map=None, w_L=None):
    """
    Расчёт продвинутой метрики аномальности A_face с учётом:
    - shape_error (3DDFA + FAN)
    - симметрии (лево/право по ключевым точкам)
    - центра лица и масштаба
    - угла головы (yaw/pitch/roll)
    - доверия к библиотекам и норм геометрии
    
    Args:
        fan_pts: координаты 68 точек от FAN
        ddfa_pts: координаты 68 точек от 3DDFA
        yaw, pitch, roll: углы поворота головы
        shape_error_ddfa: ошибка формы от 3DDFA
        face_center: центр лица
        scale: масштаб лица
        zone_map: словарь маппинга точек по зонам лица
        w_L: веса для каждой библиотеки
        
    Возвращает словарь с финальной метрикой и деталями
    """
    n_points = len(fan_pts)
    assert len(ddfa_pts) == n_points, "Количество точек должно совпадать"
    
    # Приведение к numpy и проверка размерности
    fan_pts = numpy.array(fan_pts)[:, :2]  # Берем только x, y
    ddfa_pts = numpy.array(ddfa_pts)[:, :2]
    
    EPSILON = 1e-6
    scale = max(scale, EPSILON)
    
    # Нормализация по масштабу
    fan_pts = fan_pts / scale
    ddfa_pts = ddfa_pts / scale
    
    # ------------------- SHAPE ERROR -------------------
    mean_pts = (fan_pts + ddfa_pts) / 2  # Среднее между FAN и 3DDFA
    fan_shape_error = numpy.mean(numpy.linalg.norm(fan_pts - mean_pts, axis=1))
    
    # ------------------- СИММЕТРИЯ -------------------
    def get_mirrored_pairs(n_points):
        if n_points == 68:
            return [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
                    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
                    (31, 35), (32, 34), (48, 54), (49, 53), (50, 52)]
        else:
            return []
    
    mirrored_pairs = get_mirrored_pairs(n_points)
    center_x = numpy.mean(fan_pts[:, 0])
    
    def symmetry_error(points):
        return numpy.mean([
            abs(points[i][0] + points[j][0] - 2 * center_x)
            for i, j in mirrored_pairs
        ]) / max(1, (numpy.max(points[:, 0]) - numpy.min(points[:, 0])))
    
    sym_fan = symmetry_error(fan_pts)
    sym_ddfa = symmetry_error(ddfa_pts)
    sym_boost = 1 + (sym_fan + sym_ddfa) / 2
    
    # ------------------- SHAPE ERROR ПО ЗОНАМ -------------------
    shape_error_zones = {}
    if zone_map is None:
        zone_map = {}
        for i in range(n_points):
            if i <= 16:
                zone_map[i] = 'jaw'
            elif i <= 21:
                zone_map[i] = 'brow_left'
            elif i <= 26:
                zone_map[i] = 'brow_right'
            elif i <= 35:
                zone_map[i] = 'nose'
            elif i <= 47:
                zone_map[i] = 'eye'
            elif i <= 67:
                zone_map[i] = 'mouth'
            else:
                zone_map[i] = 'other'
    
    for z in set(zone_map.values()):
        indices = [i for i, zz in zone_map.items() if zz == z]
        err_fan = numpy.mean(numpy.linalg.norm(fan_pts[indices] - mean_pts[indices], axis=1))
        err_ddfa = numpy.mean(numpy.linalg.norm(ddfa_pts[indices] - mean_pts[indices], axis=1))
        shape_error_zones[z] = {
            'fan': round(err_fan, 3),
            'ddfa': round(err_ddfa, 3)
        }
    
    # Расчет shape_boost
    shape_boost = 1 + (shape_error_ddfa + fan_shape_error / (fan_shape_error + 1e-3)) / 2
    
    # ------------------- ПОЗА -------------------
    pose_boost = 1.0
    if abs(yaw) > 25:
        pose_boost *= 1.2
    if abs(roll) > 20:
        sym_boost *= 0.8  # понижаем силу симметрии при наклоне головы
    
    # ------------------- ЦЕНТР -------------------
    def center_deviation(pts):
        est_center = numpy.mean(pts, axis=0)
        return numpy.linalg.norm(est_center - numpy.array(face_center))
    
    center_dev_fan = center_deviation(fan_pts)
    center_dev_ddfa = center_deviation(ddfa_pts)
    center_boost = 1 + (center_dev_fan + center_dev_ddfa) / 200
    
    # ------------------- SCALE -------------------
    scale_boost = 1 + abs(1.0 - scale)
    
    # ------------------- ПРОПОРЦИИ -------------------
    eye_dist = numpy.linalg.norm(fan_pts[36] - fan_pts[45])
    height = numpy.linalg.norm(fan_pts[8] - fan_pts[27])
    prop_ratio = height / max(eye_dist, 1)
    golden_ratio = 1.618
    prop_boost = 1 + abs(prop_ratio - golden_ratio)
    
    # ------------------- БИБЛИОТЕЧНЫЕ ВЕСА -------------------
    if w_L is None:
        w_L = {'fan': 1.0, 'ddfa': 1.0}
    
    # ------------------- ГЕОМЕТРИЯ (A_geom базовая) -------------------
    A_geom = numpy.mean([
        w_L['fan'] * numpy.linalg.norm(fan_pts[i] - ddfa_pts[i])
        for i in range(n_points)
    ]) / n_points
    
    # ------------------- Финальный расчёт -------------------
    A_face = A_geom * shape_boost * sym_boost * pose_boost * center_boost * scale_boost * prop_boost
    
    # Проверка на NaN и бесконечность
    if numpy.isnan(A_face) or numpy.isinf(A_face):
        A_face = 1.0  # Значение по умолчанию
    
    # Ограничение максимального значения
    A_face = min(A_face, 1000.0)
    
    results = {
        'A_face': round(A_face, 3),
        'A_geom': round(A_geom, 3),
        'shape_error_ddfa': round(shape_error_ddfa, 3),
        'fan_shape_error': round(fan_shape_error, 3),
        'symmetry_fan': round(sym_fan, 3),
        'symmetry_ddfa': round(sym_ddfa, 3),
        'yaw': yaw, 'pitch': pitch, 'roll': roll,
        'pose_boost': round(pose_boost, 3),
        'center_dev_fan': round(center_dev_fan, 2),
        'center_dev_ddfa': round(center_dev_ddfa, 2),
        'center_boost': round(center_boost, 3),
        'scale': scale,
        'scale_boost': round(scale_boost, 3),
        'prop_boost': round(prop_boost, 3),
        'prop_ratio': round(prop_ratio, 3),
        'A_z': {},
        'anomaly_type': None
    }
    
    # ------------------- ЗОНИРОВАННЫЙ АНАЛИЗ -------------------
    zones = sorted(set(zone_map.values()))
    zone_deltas = {z: [] for z in zones}
    
    for i in range(n_points):
        z = zone_map.get(i, 'other')
        d1 = numpy.linalg.norm(fan_pts[i] - ddfa_pts[i])
        zone_deltas[z].append(d1)
    
    for z in zones:
        raw = numpy.mean(zone_deltas[z])
        # Байесовская логика: чем больше согласованных отклонений — тем сильнее множитель
        if raw > 10:
            B_z = 1.0
        elif raw > 5:
            B_z = 0.7
        elif raw > 3:
            B_z = 0.5
        else:
            B_z = 0.3
        results['A_z'][z] = round(raw * B_z, 2)
    
    # ------------------- КЛАССИФИКАЦИЯ ТИПА АНОМАЛИИ -------------------
    az = results['A_z']
    A_face_val = results['A_face']
    
    # Динамические пороги на основе значения аномалии
    if A_face_val < 12:
        results['anomaly_type'] = 'норма'
    elif az.get('jaw', 0) + az.get('mouth', 0) > 25 and az.get('eye', 0) < 6:
        results['anomaly_type'] = 'маска'
    elif all(v > 6 for v in az.values()):
        results['anomaly_type'] = 'цифровая'
    elif any(az.get(k, 0) > 10 for k in ['eye', 'nose', 'brow_left', 'brow_right']):
        results['anomaly_type'] = 'структурная'
    else:
        results['anomaly_type'] = 'неопределено'
    
    results['shape_error_zones'] = shape_error_zones
    
    return results

def universal_anomaly_detection(fan_pts, ddfa_pts, weights, confidence, zone_map, bias, angle_factors):
    total_weight = 0
    total_score = 0
    library_distances = {}
    zone_distances = {zone: [] for zone in set(zone_map.values())}
    
    for i in range(min(len(fan_pts), len(ddfa_pts))):
        zone = zone_map.get(i, 'other')
        zone_weight = weights.get(zone, 1.0)
        point_confidence = confidence.get(i, 1.0)
        zone_bias = bias.get(zone, 1.0)
        angle_correction = angle_factors.get(zone, 1.0)
        
        dist_fan_ddfa = numpy.linalg.norm(fan_pts[i] - ddfa_pts[i])
        
        library_distances[i] = dist_fan_ddfa
        zone_distances[zone].append(dist_fan_ddfa)
        
        bayesian_factor = 1.0
        if dist_fan_ddfa > 30:
            bayesian_factor = 3.0
        
        fractal_factor = 1.0  # зарезервировано
        chain_factor = 1.0  # зарезервировано
        
        adjusted_score = dist_fan_ddfa * bayesian_factor * fractal_factor * chain_factor * zone_weight * point_confidence * zone_bias * angle_correction
        
        total_score += adjusted_score
        total_weight += zone_weight * point_confidence
    
    zone_anomalies = {}
    for zone, distances in zone_distances.items():
        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
            variance = sum([(d - avg_distance)**2 for d in distances]) / len(distances)
            zone_anomalies[zone] = (avg_distance, variance)
    
    return {
        'total_score': total_score / total_weight if total_weight else 0,
        'zone_anomalies': zone_anomalies,
        'library_distances': library_distances
    }
