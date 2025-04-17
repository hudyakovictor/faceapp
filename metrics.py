import numpy as np

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


def get_optimal_metrics_for_pose(pose_type):
    """
    Получает список оптимальных метрик для конкретного ракурса
    
    Args:
        pose_type (str): Тип ракурса ('frontal', 'profile_left', 'profile_right', 'semi_left', 'semi_right')
        
    Returns:
        list: Список имен метрик, оптимальных для данного ракурса, или пустой список, если тип ракурса неизвестен.
    """
    # Оптимальные метрики для каждого ракурса
    optimal_metrics = {
        'frontal': [
            'face_asymmetry', 'eye_angle', 'eye_distance', 'eye_socket_depth', 'eye_asymmetry',
            'face_width', 'nose_depth', 'cheek_width', 'mouth_width', 'nose_width',
            'brow_height', 'brow_angle', 'chin_height', 'jaw_asymmetry', 'forehead_width',
            'fn_inter_pupil_distance', 'fn_left_eye_openness', 'fn_right_eye_openness', 
            'fn_eye_symmetry_score', 'fn_left_eye_corner_angle', 'fn_right_eye_corner_angle',
            'fn_eye_bulge_score', 'fn_left_eyebrow_height', 'fn_right_eyebrow_height',
            'fn_eyebrow_symmetry_score', 'fn_left_eyebrow_slope', 'fn_right_eyebrow_slope',
            'fn_nose_length', 'fn_nose_width', 'fn_nostril_asymmetry_score', 'fn_nose_angle',
            'fn_nose_tip_alignment', 'fn_mouth_width', 'fn_mouth_height', 'fn_lip_thickness_upper',
            'fn_lip_thickness_lower', 'fn_mouth_corner_angle', 'fn_mouth_asymmetry_score',
            'fn_jaw_width', 'fn_chin_height', 'fn_face_symmetry_score', 'fn_golden_ratio_score',
            'MP_eye_distance', 'MP_eye_angle', 'MP_eye_asymmetry', 'MP_face_width', 'MP_eye_socket_depth', 
            'MP_brow_height', 'MP_brow_angle', 'MP_forehead_width', 'MP_nose_width', 'MP_nose_depth', 
            'MP_cheek_width', 'MP_mouth_width', 'MP_chin_height', 'MP_jaw_asymmetry',
            'MP_face_asymmetry'
        ],
        'profile_left': [
            'left_chin_depth', 'left_nose_depth', 'left_nose_width', 'left_chin_height',
            'left_cheek_width', 'left_jaw_width', 'left_eye_distance', 'left_brow_height',
            'left_eye_socket_depth', 'left_nose_angle', 'left_jaw_angle',
            'fn_left_eye_openness', 'fn_left_eye_corner_angle', 'fn_left_eye_bulge_score',
            'fn_left_eyebrow_height', 'fn_left_eyebrow_slope', 'fn_nose_projection_left',
            'fn_nostril_visibility_left', 'fn_nose_angle_left', 'fn_mouth_corner_angle_left',
            'fn_left_cheek_offset', 'fn_jaw_angle_left', 'fn_jaw_curve_left', 'fn_chin_offset_left',
            'fn_mouth_asymmetry_left', 'fn_face_contour_visibility_left', 'fn_left_nose_projection',
            'fn_left_forehead_nose_angle', 'fn_left_lip_profile', 'fn_left_mouth_corner_angle',
            'fn_left_jaw_angle', 'fn_left_jaw_curve', 'fn_left_chin_shape', 'fn_left_cheek_curve',
            'fn_left_eye_profile_slit', 'fn_left_eyebrow_profile_angle', 'fn_left_nostril_contour_visibility', 
            'fn_left_face_curve_profile', 'fn_left_ear_to_nose_ratio', 'fn_left_face_silhouette_density', 
            'fn_left_face_angle_score',
            'MP_left_forehead_width', 'MP_left_nose_width', 'MP_left_nose_depth',
            'MP_left_jaw_angle', 'MP_left_jaw_width', 'MP_left_cheek_width',
            'MP_left_brow_angle', 'MP_left_face_width'
        ],
        'profile_right': [
            'right_chin_depth', 'right_chin_height', 'right_nose_depth', 'right_nose_width',
            'right_nose_angle', 'right_cheek_width', 'right_jaw_width', 'right_eye_distance',
            'right_jaw_angle',
            'fn_right_eye_openness', 'fn_right_eye_corner_angle', 'fn_right_eye_bulge_score',
            'fn_right_eyebrow_height', 'fn_right_eyebrow_slope', 'fn_nose_projection_right',
            'fn_nostril_visibility_right', 'fn_nose_angle_right', 'fn_mouth_corner_angle_right',
            'fn_right_cheek_offset', 'fn_jaw_angle_right', 'fn_jaw_curve_right', 'fn_chin_offset_right',
            'fn_mouth_asymmetry_right', 'fn_face_contour_visibility_right', 'fn_right_nose_projection',
            'fn_right_forehead_nose_angle', 'fn_right_lip_profile', 'fn_right_mouth_corner_angle',
            'fn_right_jaw_angle', 'fn_right_jaw_curve', 'fn_right_chin_shape', 'fn_right_cheek_curve',
            'fn_right_eye_profile_slit', 'fn_right_eyebrow_profile_angle', 'fn_right_nostril_contour_visibility',
            'fn_right_face_curve_profile', 'fn_right_ear_to_nose_ratio', 'fn_right_face_silhouette_density',
            'fn_right_face_angle_score',
            'MP_right_forehead_width', 'MP_right_nose_width', 'MP_right_nose_depth', 
            'MP_right_jaw_angle', 'MP_right_jaw_width', 'MP_right_cheek_width', 
            'MP_right_brow_angle', 'MP_right_face_width'
        ],
        'semi_left': [
            'semi_left_nose_depth', 'semi_left_nose_angle', 'semi_left_skull_width',
            'semi_left_cheek_width', 'semi_left_jaw_width', 'semi_left_jaw_angle',
            'semi_left_chin_width', 'semi_left_eye_socket_depth',
            'fn_left_eye_openness', 'fn_left_eye_corner_angle', 'fn_left_eye_bulge_score',
            'fn_left_eyebrow_height', 'fn_left_eyebrow_slope', 'fn_nose_projection_left',
            'fn_nostril_visibility_left', 'fn_nose_angle_left', 'fn_mouth_corner_angle_left',
            'fn_left_cheek_offset', 'fn_jaw_angle_left', 'fn_jaw_curve_left', 'fn_chin_offset_left',
            'fn_mouth_asymmetry_left', 'fn_face_contour_visibility_left',
            'MP_left_forehead_width', 'MP_left_nose_width', 'MP_left_nose_depth',
            'MP_left_jaw_angle', 'MP_left_jaw_width', 'MP_left_cheek_width',
            'MP_left_brow_angle', 'MP_left_face_width'
        ],
        'semi_right': [
            'semi_right_nose_depth', 'semi_right_nose_angle', 'semi_right_skull_width',
            'semi_right_cheek_width', 'semi_right_jaw_width', 'semi_right_jaw_angle', 
            'semi_right_chin_width', 'semi_right_eye_socket_depth',
            'fn_right_eye_openness', 'fn_right_eye_corner_angle', 'fn_right_eye_bulge_score',
            'fn_right_eyebrow_height', 'fn_right_eyebrow_slope', 'fn_nose_projection_right',
            'fn_nostril_visibility_right', 'fn_nose_angle_right', 'fn_mouth_corner_angle_right',
            'fn_right_cheek_offset', 'fn_jaw_angle_right', 'fn_jaw_curve_right', 'fn_chin_offset_right',
            'fn_mouth_asymmetry_right', 'fn_face_contour_visibility_right',
            'MP_right_forehead_width', 'MP_right_nose_width', 'MP_right_nose_depth',
            'MP_right_jaw_angle', 'MP_right_jaw_width', 'MP_right_cheek_width',
            'MP_right_brow_angle', 'MP_right_face_width'
        ]
    }






# Функция для добавления метрик в зависимости от ракурса
def add_metrics(pose_type, param, landmarks, mp_landmarks, delta_pose, img=None, roi_box=None):
    if landmarks is None or len(landmarks) == 0:
        print("⚠️ landmarks пустой — пропускаем расчёт метрик")
        return {}
        
    metrics = {}
    
    # Утилитарная функция для вычисления угла между тремя точками
    def calculate_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))
        cos_theta = numpy.clip(cos_theta, -1.0, 1.0)  # Избегаем ошибок округления
        return float(numpy.degrees(numpy.arccos(cos_theta)))
    
    # Утилитарная функция для вычисления симметрии между двумя массивами точек
    def symmetry_score(points1, points2):
        if len(points1) != len(points2):
            return 0.0
        diffs = numpy.linalg.norm(points1 - points2, axis=1)
        return float(numpy.mean(diffs))
    
    # Безопасное получение координат
    def get_point_coord(points, idx, coord=0):
        try:
            return points[idx][coord] if len(points[idx]) > coord else 0.0
        except:
            return 0.0

    # Метрики для фронтального ракурса (старые на основе 3DDFA)
    if pose_type == "frontal":
        try:
            metrics["face_asymmetry"] = float(numpy.abs(numpy.linalg.norm(landmarks[0] - landmarks[8]) - numpy.linalg.norm(landmarks[16] - landmarks[8])))
            metrics["eye_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[45][1] - landmarks[36][1], landmarks[45][0] - landmarks[36][0])))
            left_eye_center = numpy.mean(landmarks[36:42], axis=0)
            right_eye_center = numpy.mean(landmarks[42:48], axis=0)
            metrics["eye_distance"] = float(numpy.linalg.norm(left_eye_center - right_eye_center))
            eye_vector = right_eye_center - left_eye_center 
            metrics["eye_angle"] = float(numpy.degrees(numpy.arctan2(eye_vector[1], eye_vector[0])))
            metrics["eye_socket_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["eye_asymmetry"] = float(numpy.abs(numpy.linalg.norm(landmarks[36] - landmarks[39]) - numpy.linalg.norm(landmarks[42] - landmarks[45])))
            metrics["face_width"] = float(numpy.linalg.norm(landmarks[0] - landmarks[16]))
            metrics["nose_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["cheek_width"] = float(numpy.linalg.norm(landmarks[2] - landmarks[14]))
            metrics["mouth_width"] = float(numpy.linalg.norm(landmarks[48] - landmarks[54]))
            metrics["nose_width"] = float(numpy.linalg.norm(landmarks[31] - landmarks[35]))
            metrics["brow_height"] = float(numpy.linalg.norm(landmarks[19] - landmarks[36]))
            metrics["brow_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            metrics["chin_height"] = float(numpy.linalg.norm(landmarks[8] - landmarks[57]))
            metrics["jaw_asymmetry"] = float(numpy.abs(numpy.linalg.norm(landmarks[4] - landmarks[8]) - numpy.linalg.norm(landmarks[12] - landmarks[8])))
            metrics["forehead_width"] = float(numpy.linalg.norm(landmarks[0] - landmarks[16]))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых фронтальных метрик: {e}")

        # Новые метрики FAN для фронтального ракурса
        try:
            metrics["fn_inter_pupil_distance"] = float(numpy.linalg.norm(landmarks[42] - landmarks[39]))
            metrics["fn_left_eye_openness"] = float(numpy.linalg.norm(landmarks[37] - landmarks[41]))
            metrics["fn_right_eye_openness"] = float(numpy.linalg.norm(landmarks[44] - landmarks[48]))
            metrics["fn_eye_symmetry_score"] = symmetry_score(landmarks[36:42], landmarks[42:48])
            metrics["fn_left_eye_corner_angle"] = calculate_angle(landmarks[36], landmarks[39], landmarks[37])
            metrics["fn_right_eye_corner_angle"] = calculate_angle(landmarks[42], landmarks[45], landmarks[44])
            left_eye_bulge = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[39]) for i in [37, 38, 40, 41]])
            right_eye_bulge = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[45]) for i in [43, 44, 46, 47]])
            metrics["fn_eye_bulge_score"] = float(left_eye_bulge + right_eye_bulge) / 2
            metrics["fn_left_eyebrow_height"] = float(numpy.linalg.norm(landmarks[38] - landmarks[20]))
            metrics["fn_right_eyebrow_height"] = float(numpy.linalg.norm(landmarks[43] - landmarks[25]))
            metrics["fn_eyebrow_symmetry_score"] = symmetry_score(landmarks[17:22], landmarks[22:27])
            metrics["fn_left_eyebrow_slope"] = float(numpy.degrees(numpy.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            metrics["fn_right_eyebrow_slope"] = float(numpy.degrees(numpy.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_nose_length"] = float(numpy.linalg.norm(landmarks[27] - landmarks[30]))
            metrics["fn_nose_width"] = float(numpy.linalg.norm(landmarks[31] - landmarks[35]))
            metrics["fn_nostril_asymmetry_score"] = float(numpy.abs(landmarks[31][1] - landmarks[35][1]))
            metrics["fn_nose_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_nose_tip_alignment"] = float(numpy.abs(landmarks[32][0] - landmarks[34][0]))
            metrics["fn_mouth_width"] = float(numpy.linalg.norm(landmarks[48] - landmarks[54]))
            metrics["fn_mouth_height"] = float(numpy.linalg.norm(landmarks[51] - landmarks[57]))
            metrics["fn_lip_thickness_upper"] = float(numpy.linalg.norm(landmarks[50] - landmarks[52]))
            metrics["fn_lip_thickness_lower"] = float(numpy.linalg.norm(landmarks[58] - landmarks[56]))
            metrics["fn_mouth_corner_angle"] = calculate_angle(landmarks[48], landmarks[51], landmarks[54])
            metrics["fn_mouth_asymmetry_score"] = float(numpy.abs(landmarks[48][1] - landmarks[54][1]))
            metrics["fn_jaw_width"] = float(numpy.linalg.norm(landmarks[5] - landmarks[13]))
            metrics["fn_chin_height"] = float(numpy.linalg.norm(landmarks[57] - landmarks[8]))
            # Глобальная симметрия лица
            left_half = landmarks[1:17]
            right_half = landmarks[16:0:-1]
            metrics["fn_face_symmetry_score"] = symmetry_score(left_half, right_half)
            # Золотое сечение (примерное)
            eye_to_nose = numpy.linalg.norm(landmarks[36] - landmarks[30])
            nose_to_lip = numpy.linalg.norm(landmarks[30] - landmarks[51])
            lip_to_chin = numpy.linalg.norm(landmarks[51] - landmarks[8])
            metrics["fn_golden_ratio_score"] = float(abs(1.618 - (eye_to_nose + nose_to_lip) / lip_to_chin))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых фронтальных метрик FAN: {e}")

    # Метрики для левого профиля (старые на основе 3DDFA)
    elif pose_type == "profile_left":
        try:
            metrics["left_chin_depth"] = float(numpy.linalg.norm(landmarks[4] - landmarks[6]))
            metrics["left_nose_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["left_nose_width"] = float(numpy.linalg.norm(landmarks[31] - landmarks[30]))
            metrics["left_chin_height"] = float(numpy.linalg.norm(landmarks[8] - landmarks[57]))
            metrics["left_cheek_width"] = float(numpy.linalg.norm(landmarks[2] - landmarks[4]))
            metrics["left_jaw_width"] = float(numpy.linalg.norm(landmarks[0] - landmarks[4]))
            metrics["left_eye_distance"] = float(numpy.linalg.norm(numpy.mean(landmarks[36:42], axis=0) - numpy.mean(landmarks[42:48], axis=0)))
            metrics["left_brow_height"] = float(numpy.linalg.norm(landmarks[19] - landmarks[36]))
            metrics["left_eye_socket_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["left_nose_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["left_jaw_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[8][1] - landmarks[4][1], landmarks[8][0] - landmarks[4][0])))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых метрик profile_left: {e}")

        # Новые метрики FAN для левого профиля
        try:
            metrics["fn_left_eye_openness"] = float(numpy.linalg.norm(landmarks[37] - landmarks[41]))
            metrics["fn_left_eye_corner_angle"] = calculate_angle(landmarks[36], landmarks[39], landmarks[37])
            left_eye_bulge = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[39]) for i in [37, 38, 40, 41]])
            metrics["fn_left_eye_bulge_score"] = float(left_eye_bulge)
            metrics["fn_left_eyebrow_height"] = float(numpy.linalg.norm(landmarks[38] - landmarks[20]))
            metrics["fn_left_eyebrow_slope"] = float(numpy.degrees(numpy.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            
            # Убрали обращение к третьему элементу для проекции носа
            metrics["fn_nose_projection_left"] = float(numpy.linalg.norm(landmarks[30] - landmarks[27]))
            
            # Убрали обращение к третьему элементу для видимости ноздри
            metrics["fn_nostril_visibility_left"] = float(numpy.abs(landmarks[31][1] - landmarks[30][1]))  # используем только 2D координаты
            
            metrics["fn_nose_angle_left"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_mouth_corner_angle_left"] = calculate_angle(landmarks[48], landmarks[51], landmarks[54])
            metrics["fn_left_cheek_offset"] = float(numpy.linalg.norm(landmarks[3] - landmarks[30]))
            metrics["fn_jaw_angle_left"] = calculate_angle(landmarks[3], landmarks[5], landmarks[6])
            
            jaw_curve_left = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 5)])
            metrics["fn_jaw_curve_left"] = float(jaw_curve_left)
            metrics["fn_chin_offset_left"] = float(numpy.abs(landmarks[8][0] - landmarks[30][0]))
            metrics["fn_mouth_asymmetry_left"] = float(numpy.abs(landmarks[48][1] - landmarks[51][1]))
            
            contour_density_left = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 7)])
            metrics["fn_face_contour_visibility_left"] = float(contour_density_left)
            metrics["fn_left_nose_projection"] = float(numpy.linalg.norm(landmarks[27] - landmarks[30]))
            metrics["fn_left_forehead_nose_angle"] = calculate_angle(landmarks[19], landmarks[27], landmarks[30])
            
            lip_profile_left = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[51]) for i in [48, 54, 57]])
            metrics["fn_left_lip_profile"] = float(lip_profile_left)
            metrics["fn_left_mouth_corner_angle"] = calculate_angle(landmarks[48], landmarks[51], landmarks[54])
            metrics["fn_left_jaw_angle"] = calculate_angle(landmarks[3], landmarks[5], landmarks[6])
            metrics["fn_left_jaw_curve"] = float(jaw_curve_left)
            metrics["fn_left_chin_shape"] = float(numpy.linalg.norm(landmarks[8] - landmarks[57]))
            
            cheek_curve_left = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in [3, 4, 5]])
            metrics["fn_left_cheek_curve"] = float(cheek_curve_left)
            eye_slit_left = numpy.linalg.norm(landmarks[39] - landmarks[41])
            metrics["fn_left_eye_profile_slit"] = float(eye_slit_left)
            metrics["fn_left_eyebrow_profile_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых метрик FAN для profile_left: {e}")

    # Метрики для правого профиля (старые на основе 3DDFA)
    elif pose_type == "profile_right":
        try:
            metrics["right_chin_depth"] = float(numpy.linalg.norm(landmarks[10] - landmarks[12]))
            metrics["right_chin_height"] = float(numpy.linalg.norm(landmarks[8] - landmarks[57]))
            metrics["right_nose_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["right_nose_width"] = float(numpy.linalg.norm(landmarks[35] - landmarks[30]))
            metrics["right_nose_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["right_cheek_width"] = float(numpy.linalg.norm(landmarks[12] - landmarks[14]))
            metrics["right_jaw_width"] = float(numpy.linalg.norm(landmarks[12] - landmarks[16]))
            metrics["right_eye_distance"] = float(numpy.linalg.norm(numpy.mean(landmarks[36:42], axis=0) - numpy.mean(landmarks[42:48], axis=0)))
            metrics["right_jaw_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[8][1] - landmarks[12][1], landmarks[8][0] - landmarks[12][0])))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых метрик profile_right: {e}")

        # Новые метрики FAN для правого профиля
        try:
            metrics["fn_right_eye_openness"] = float(numpy.linalg.norm(landmarks[44][:2] - landmarks[48][:2]))
            metrics["fn_right_eye_corner_angle"] = calculate_angle(landmarks[42][:2], landmarks[45][:2], landmarks[44][:2])
            right_eye_bulge = numpy.mean([numpy.linalg.norm(landmarks[i][:2] - landmarks[45][:2]) for i in [43, 44, 46, 47]])
            metrics["fn_right_eye_bulge_score"] = float(right_eye_bulge)
            metrics["fn_right_eyebrow_height"] = float(numpy.linalg.norm(landmarks[43][:2] - landmarks[25][:2]))
            metrics["fn_right_eyebrow_slope"] = float(numpy.degrees(numpy.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_nose_projection_right"] = float(numpy.linalg.norm(landmarks[30][:2] - landmarks[27][:2]))
            metrics["fn_nostril_visibility_right"] = float(numpy.abs(landmarks[35][1] - landmarks[30][1]))  # Изменено с [2] на [1]
            metrics["fn_nose_angle_right"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_mouth_corner_angle_right"] = calculate_angle(landmarks[51][:2], landmarks[54][:2], landmarks[48][:2])
            metrics["fn_right_cheek_offset"] = float(numpy.linalg.norm(landmarks[13][:2] - landmarks[30][:2]))
            metrics["fn_jaw_angle_right"] = calculate_angle(landmarks[13][:2], landmarks[15][:2], landmarks[16][:2])
            jaw_curve_right = numpy.mean([numpy.linalg.norm(landmarks[i][:2] - landmarks[i+1][:2]) for i in range(10, 15)])
            metrics["fn_jaw_curve_right"] = float(jaw_curve_right)
            metrics["fn_chin_offset_right"] = float(numpy.abs(landmarks[8][0] - landmarks[30][0]))
            metrics["fn_mouth_asymmetry_right"] = float(numpy.abs(landmarks[54][1] - landmarks[51][1]))
            contour_density_right = numpy.mean([numpy.linalg.norm(landmarks[i][:2] - landmarks[i+1][:2]) for i in range(10, 16)])
            metrics["fn_face_contour_visibility_right"] = float(contour_density_right)
            metrics["fn_right_nose_projection"] = float(numpy.linalg.norm(landmarks[27][:2] - landmarks[30][:2]))
            metrics["fn_right_forehead_nose_angle"] = calculate_angle(landmarks[24][:2], landmarks[27][:2], landmarks[30][:2])
            lip_profile_right = numpy.mean([numpy.linalg.norm(landmarks[i][:2] - landmarks[51][:2]) for i in [48, 54, 57]])
            metrics["fn_right_lip_profile"] = float(lip_profile_right)
            metrics["fn_right_mouth_corner_angle"] = calculate_angle(landmarks[54][:2], landmarks[51][:2], landmarks[48][:2])
            metrics["fn_right_jaw_angle"] = calculate_angle(landmarks[13][:2], landmarks[15][:2], landmarks[16][:2])
            metrics["fn_right_jaw_curve"] = float(jaw_curve_right)
            metrics["fn_right_chin_shape"] = float(numpy.linalg.norm(landmarks[8][:2] - landmarks[57][:2]))
            cheek_curve_right = numpy.mean([numpy.linalg.norm(landmarks[i][:2] - landmarks[i+1][:2]) for i in [13, 14, 15]])
            metrics["fn_right_cheek_curve"] = float(cheek_curve_right)
            eye_slit_right = numpy.linalg.norm(landmarks[45][:2] - landmarks[47][:2])
            metrics["fn_right_eye_profile_slit"] = float(eye_slit_right)
            metrics["fn_right_eyebrow_profile_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_right_nostril_contour_visibility"] = float(numpy.abs(landmarks[35][1] - landmarks[30][1]))  # Изменено с [2] на [1]
            face_curve_right = numpy.mean([numpy.linalg.norm(landmarks[i][:2] - landmarks[i+1][:2]) for i in range(8, 16)])
            metrics["fn_right_face_curve_profile"] = float(face_curve_right)
            metrics["fn_right_ear_to_nose_ratio"] = float(numpy.linalg.norm(landmarks[16][:2] - landmarks[30][:2]))
            metrics["fn_right_face_silhouette_density"] = float(contour_density_right)
            metrics["fn_right_face_angle_score"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[16][1], landmarks[30][0] - landmarks[16][0])))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых метрик FAN для profile_right: {e}")

    # Метрики для полулевого ракурса (старые на основе 3DDFA)
    elif pose_type == "semi_left":
        try:
            metrics["semi_left_nose_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["semi_left_nose_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["semi_left_skull_width"] = float(numpy.linalg.norm(landmarks[0] - landmarks[4]))
            metrics["semi_left_cheek_width"] = float(numpy.linalg.norm(landmarks[2] - landmarks[4]))
            metrics["semi_left_jaw_width"] = float(numpy.linalg.norm(landmarks[0] - landmarks[4]))
            metrics["semi_left_jaw_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[8][1] - landmarks[4][1], landmarks[8][0] - landmarks[4][0])))
            metrics["semi_left_chin_width"] = float(numpy.linalg.norm(landmarks[4] - landmarks[8]))
            metrics["semi_left_eye_socket_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых метрик semi_left: {e}")

        # Новые метрики FAN для полулевого ракурса
        try:
            metrics["fn_left_eye_openness"] = float(numpy.linalg.norm(landmarks[37] - landmarks[41]))
            metrics["fn_left_eye_corner_angle"] = calculate_angle(landmarks[36], landmarks[39], landmarks[37])
            left_eye_bulge = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[39]) for i in [37, 38, 40, 41]])
            metrics["fn_left_eye_bulge_score"] = float(left_eye_bulge)
            metrics["fn_left_eyebrow_height"] = float(numpy.linalg.norm(landmarks[38] - landmarks[20]))
            metrics["fn_left_eyebrow_slope"] = float(numpy.degrees(numpy.arctan2(landmarks[21][1] - landmarks[17][1], landmarks[21][0] - landmarks[17][0])))
            metrics["fn_nose_projection_left"] = float(numpy.linalg.norm(landmarks[30] - landmarks[27]))
            metrics["fn_nostril_visibility_left"] = float(numpy.linalg.norm(landmarks[31] - landmarks[30]))  # Убираем z-координату
            metrics["fn_nose_angle_left"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_mouth_corner_angle_left"] = calculate_angle(landmarks[48], landmarks[51], landmarks[54])
            metrics["fn_left_cheek_offset"] = float(numpy.linalg.norm(landmarks[3] - landmarks[30]))
            metrics["fn_jaw_angle_left"] = calculate_angle(landmarks[3], landmarks[5], landmarks[6])
            jaw_curve_left = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 5)])
            metrics["fn_jaw_curve_left"] = float(jaw_curve_left)
            metrics["fn_chin_offset_left"] = float(numpy.abs(landmarks[8][0] - landmarks[30][0]))
            metrics["fn_mouth_asymmetry_left"] = float(numpy.abs(landmarks[48][1] - landmarks[51][1]))
            contour_density_left = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 7)])
            metrics["fn_face_contour_visibility_left"] = float(contour_density_left)
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых метрик FAN для semi_left: {e}")

    # Метрики для полуправого ракурса (старые на основе 3DDFA)
    elif pose_type == "semi_right":
        try:
            metrics["semi_right_nose_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
            metrics["semi_right_nose_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], 
                                                                        landmarks[30][0] - landmarks[27][0])))
            metrics["semi_right_skull_width"] = float(numpy.linalg.norm(landmarks[12] - landmarks[16]))
            metrics["semi_right_cheek_width"] = float(numpy.linalg.norm(landmarks[12] - landmarks[14]))
            metrics["semi_right_jaw_width"] = float(numpy.linalg.norm(landmarks[12] - landmarks[16]))
            metrics["semi_right_jaw_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[8][1] - landmarks[12][1], 
                                                                         landmarks[8][0] - landmarks[12][0])))
            metrics["semi_right_chin_width"] = float(numpy.linalg.norm(landmarks[8] - landmarks[12]))
            metrics["semi_right_eye_socket_depth"] = float(numpy.linalg.norm(landmarks[27] - landmarks[33]))
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте старых метрик semi_right: {e}")

        # Новые метрики FAN для полуправого ракурса
        try:
            metrics["fn_right_eye_openness"] = float(numpy.linalg.norm(landmarks[44] - landmarks[48]))
            metrics["fn_right_eye_corner_angle"] = calculate_angle(landmarks[42], landmarks[45], landmarks[44])
            right_eye_bulge = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[45]) for i in [43, 44, 46, 47]])
            metrics["fn_right_eye_bulge_score"] = float(right_eye_bulge)
            metrics["fn_right_eyebrow_height"] = float(numpy.linalg.norm(landmarks[43] - landmarks[25]))
            metrics["fn_right_eyebrow_slope"] = float(numpy.degrees(numpy.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_nose_projection_right"] = float(numpy.linalg.norm(landmarks[30] - landmarks[27]))
            metrics["fn_nostril_visibility_right"] = float(numpy.linalg.norm(landmarks[35] - landmarks[30]))  # Убираем z-координату
            metrics["fn_nose_angle_right"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[27][1], landmarks[30][0] - landmarks[27][0])))
            metrics["fn_mouth_corner_angle_right"] = calculate_angle(landmarks[51], landmarks[54], landmarks[48])
            metrics["fn_right_cheek_offset"] = float(numpy.linalg.norm(landmarks[13] - landmarks[30]))
            metrics["fn_jaw_angle_right"] = calculate_angle(landmarks[13], landmarks[15], landmarks[16])
            jaw_curve_right = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(10, 15)])
            metrics["fn_jaw_curve_right"] = float(jaw_curve_right)
            metrics["fn_chin_offset_right"] = float(numpy.abs(landmarks[8][0] - landmarks[30][0]))
            metrics["fn_mouth_asymmetry_right"] = float(numpy.abs(landmarks[54][1] - landmarks[51][1]))
            contour_density_right = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(10, 16)])
            metrics["fn_face_contour_visibility_right"] = float(contour_density_right)
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте новых метрик FAN для semi_right: {e}")

    # Новые метрики на основе MediaPipe (если mp_landmarks доступен)
    if mp_landmarks is not None and len(mp_landmarks) >= 68:
        if pose_type == "frontal":
            try:
                left_eye_center = numpy.mean(mp_landmarks[36:42], axis=0)
                right_eye_center = numpy.mean(mp_landmarks[42:48], axis=0)
                metrics["MP_eye_distance"] = float(numpy.linalg.norm(left_eye_center - right_eye_center))
                eye_vector = right_eye_center - left_eye_center
                metrics["MP_eye_angle"] = float(numpy.degrees(numpy.arctan2(eye_vector[1], eye_vector[0])))
                left_eye_width = numpy.linalg.norm(mp_landmarks[36] - mp_landmarks[39])
                right_eye_width = numpy.linalg.norm(mp_landmarks[42] - mp_landmarks[45])
                metrics["MP_eye_asymmetry"] = float(numpy.abs(left_eye_width - right_eye_width))
                metrics["MP_face_width"] = float(numpy.linalg.norm(mp_landmarks[0] - mp_landmarks[16]))
                metrics["MP_brow_height"] = float(numpy.linalg.norm(mp_landmarks[19] - mp_landmarks[36]))
                brow_vector = mp_landmarks[21] - mp_landmarks[17]
                metrics["MP_brow_angle"] = float(numpy.degrees(numpy.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_forehead_width"] = float(numpy.linalg.norm(mp_landmarks[0] - mp_landmarks[16]))
                metrics["MP_nose_width"] = float(numpy.linalg.norm(mp_landmarks[31] - mp_landmarks[35]))
                metrics["MP_cheek_width"] = float(numpy.linalg.norm(mp_landmarks[2] - mp_landmarks[14]))
                metrics["MP_mouth_width"] = float(numpy.linalg.norm(mp_landmarks[48] - mp_landmarks[54]))
                metrics["MP_chin_height"] = float(numpy.linalg.norm(mp_landmarks[8] - mp_landmarks[57]))
                metrics["MP_jaw_asymmetry"] = float(numpy.abs(numpy.linalg.norm(mp_landmarks[4] - mp_landmarks[8]) - 
                                                             numpy.linalg.norm(mp_landmarks[12] - mp_landmarks[8])))
                metrics["MP_face_asymmetry"] = float(numpy.abs(numpy.linalg.norm(mp_landmarks[0] - mp_landmarks[8]) - 
                                                              numpy.linalg.norm(mp_landmarks[16] - mp_landmarks[8])))
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для frontal: {e}")

        elif pose_type == "semi_left":
            try:
                metrics["MP_left_forehead_width"] = float(numpy.linalg.norm(mp_landmarks[0] - mp_landmarks[4]))
                metrics["MP_left_nose_width"] = float(numpy.linalg.norm(mp_landmarks[31] - mp_landmarks[30]))
                metrics["MP_left_nose_depth"] = float(numpy.abs(mp_landmarks[30][2] - mp_landmarks[27][2]))
                jaw_vector = mp_landmarks[8] - mp_landmarks[4]
                metrics["MP_left_jaw_angle"] = float(numpy.degrees(numpy.arctan2(jaw_vector[1], jaw_vector[0])))
                metrics["MP_left_jaw_width"] = float(numpy.linalg.norm(mp_landmarks[0] - mp_landmarks[4]))
                metrics["MP_left_cheek_width"] = float(numpy.linalg.norm(mp_landmarks[2] - mp_landmarks[4]))
                brow_vector = mp_landmarks[21] - mp_landmarks[17]
                metrics["MP_left_brow_angle"] = float(numpy.degrees(numpy.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_left_face_width"] = float(numpy.linalg.norm(mp_landmarks[0] - mp_landmarks[8]))
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для semi_left: {e}")

        elif pose_type == "semi_right":
            try:
                metrics["MP_right_forehead_width"] = float(numpy.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_nose_width"] = float(numpy.linalg.norm(mp_landmarks[30] - mp_landmarks[35]))
                # Убрали обращение к третьему элементу для глубины носа
                metrics["MP_right_nose_depth"] = float(numpy.linalg.norm(mp_landmarks[30] - mp_landmarks[27]))  # учитываем только xy координаты
                jaw_vector = mp_landmarks[8] - mp_landmarks[12]
                metrics["MP_right_jaw_angle"] = float(numpy.degrees(numpy.arctan2(jaw_vector[1], jaw_vector[0])))
                metrics["MP_right_jaw_width"] = float(numpy.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_cheek_width"] = float(numpy.linalg.norm(mp_landmarks[12] - mp_landmarks[14]))
                brow_vector = mp_landmarks[26] - mp_landmarks[22]
                metrics["MP_right_brow_angle"] = float(numpy.degrees(numpy.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_right_face_width"] = float(numpy.linalg.norm(mp_landmarks[8] - mp_landmarks[16]))
            
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для semi_right: {e}")

        elif pose_type == "profile_left":
            try:
                # Для вычислений используем только координаты x и y
                def get_xy(point):
                    return point[:2]
                
                # Расчёт ширины лба
                metrics["MP_left_forehead_width"] = float(numpy.linalg.norm(get_xy(mp_landmarks[0]) - get_xy(mp_landmarks[4])))
                
                # Расчёт ширины носа (между точками 31 и 30)
                metrics["MP_left_nose_width"] = float(numpy.linalg.norm(get_xy(mp_landmarks[31]) - get_xy(mp_landmarks[30])))
                
                # Расчёт глубины носа (разница по y, так как z нет)
                metrics["MP_left_nose_depth"] = float(abs(get_xy(mp_landmarks[30])[1] - get_xy(mp_landmarks[27])[1]))
                
                # Вектор челюсти (от 4 к 8)
                jaw_vector = get_xy(mp_landmarks[8]) - get_xy(mp_landmarks[4])
                metrics["MP_left_jaw_angle"] = float(numpy.degrees(numpy.arctan2(jaw_vector[1], jaw_vector[0])))
                
                # Ширина подбородка (от 0 к 4)
                metrics["MP_left_jaw_width"] = float(numpy.linalg.norm(get_xy(mp_landmarks[0]) - get_xy(mp_landmarks[4])))
                
                # Ширина щеки (от 2 к 4)
                metrics["MP_left_cheek_width"] = float(numpy.linalg.norm(get_xy(mp_landmarks[2]) - get_xy(mp_landmarks[4])))
                
                # Вектор бровей (от 17 к 21)
                brow_vector = get_xy(mp_landmarks[21]) - get_xy(mp_landmarks[17])
                metrics["MP_left_brow_angle"] = float(numpy.degrees(numpy.arctan2(brow_vector[1], brow_vector[0])))
                
                # Общая ширина лица (от 0 к 8)
                metrics["MP_left_face_width"] = float(numpy.linalg.norm(get_xy(mp_landmarks[0]) - get_xy(mp_landmarks[8])))
                
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для profile_left: {e}")

        elif pose_type == "profile_right":
            try:
                metrics["MP_right_forehead_width"] = float(numpy.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_nose_width"] = float(numpy.linalg.norm(mp_landmarks[30] - mp_landmarks[35]))
                jaw_vector = mp_landmarks[8] - mp_landmarks[12]
                metrics["MP_right_jaw_angle"] = float(numpy.degrees(numpy.arctan2(jaw_vector[1], jaw_vector[0])))
                metrics["MP_right_jaw_width"] = float(numpy.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_cheek_width"] = float(numpy.linalg.norm(mp_landmarks[12] - mp_landmarks[14]))
                brow_vector = mp_landmarks[26] - mp_landmarks[22]
                metrics["MP_right_brow_angle"] = float(numpy.degrees(numpy.arctan2(brow_vector[1], brow_vector[0])))
                metrics["MP_right_face_width"] = float(numpy.linalg.norm(mp_landmarks[8] - mp_landmarks[16]))
            except Exception as e:
                print(f"⚠️ Ошибка при расчёте MP_метрик для profile_right: {e}")

    return metrics