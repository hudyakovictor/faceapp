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
            metrics["fn_nose_projection_left"] = float(numpy.linalg.norm(landmarks[30] - landmarks[27]))
            metrics["fn_nostril_visibility_left"] = float(numpy.abs(landmarks[31][2] - landmarks[30][2]))
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
            metrics["fn_left_nostril_contour_visibility"] = float(numpy.abs(landmarks[31][2] - landmarks[30][2]))
            face_curve_left = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 7)])
            metrics["fn_left_face_curve_profile"] = float(face_curve_left)
            metrics["fn_left_ear_to_nose_ratio"] = float(numpy.linalg.norm(landmarks[0] - landmarks[30]))
            metrics["fn_left_face_silhouette_density"] = float(contour_density_left)
            metrics["fn_left_face_angle_score"] = float(numpy.degrees(numpy.arctan2(landmarks[30][1] - landmarks[0][1], landmarks[30][0] - landmarks[0][0])))
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
            metrics["fn_right_eye_openness"] = float(numpy.linalg.norm(landmarks[44] - landmarks[48]))
            metrics["fn_right_eye_corner_angle"] = calculate_angle(landmarks[42], landmarks[45], landmarks[44])
            right_eye_bulge = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[45]) for i in [43, 44, 46, 47]])
            metrics["fn_right_eye_bulge_score"] = float(right_eye_bulge)
            metrics["fn_right_eyebrow_height"] = float(numpy.linalg.norm(landmarks[43] - landmarks[25]))
            metrics["fn_right_eyebrow_slope"] = float(numpy.degrees(numpy.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_nose_projection_right"] = float(numpy.linalg.norm(landmarks[30] - landmarks[27]))
            metrics["fn_nostril_visibility_right"] = float(numpy.abs(landmarks[35][2] - landmarks[30][2]))
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
            metrics["fn_right_nose_projection"] = float(numpy.linalg.norm(landmarks[27] - landmarks[30]))
            metrics["fn_right_forehead_nose_angle"] = calculate_angle(landmarks[24], landmarks[27], landmarks[30])
            lip_profile_right = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[51]) for i in [48, 54, 57]])
            metrics["fn_right_lip_profile"] = float(lip_profile_right)
            metrics["fn_right_mouth_corner_angle"] = calculate_angle(landmarks[54], landmarks[51], landmarks[48])
            metrics["fn_right_jaw_angle"] = calculate_angle(landmarks[13], landmarks[15], landmarks[16])
            metrics["fn_right_jaw_curve"] = float(jaw_curve_right)
            metrics["fn_right_chin_shape"] = float(numpy.linalg.norm(landmarks[8] - landmarks[57]))
            cheek_curve_right = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in [13, 14, 15]])
            metrics["fn_right_cheek_curve"] = float(cheek_curve_right)
            eye_slit_right = numpy.linalg.norm(landmarks[45] - landmarks[47])
            metrics["fn_right_eye_profile_slit"] = float(eye_slit_right)
            metrics["fn_right_eyebrow_profile_angle"] = float(numpy.degrees(numpy.arctan2(landmarks[26][1] - landmarks[22][1], landmarks[26][0] - landmarks[22][0])))
            metrics["fn_right_nostril_contour_visibility"] = float(numpy.abs(landmarks[35][2] - landmarks[30][2]))
            face_curve_right = numpy.mean([numpy.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(8, 16)])
            metrics["fn_right_face_curve_profile"] = float(face_curve_right)
            metrics["fn_right_ear_to_nose_ratio"] = float(numpy.linalg.norm(landmarks[16] - landmarks[30]))
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
                metrics["MP_right_nose_depth"] = float(numpy.abs(mp_landmarks[30][2] - mp_landmarks[27][2]))
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
                print(f"⚠️ Ошибка при расчёте MP_метрик для profile_left: {e}")

        elif pose_type == "profile_right":
            try:
                metrics["MP_right_forehead_width"] = float(numpy.linalg.norm(mp_landmarks[12] - mp_landmarks[16]))
                metrics["MP_right_nose_width"] = float(numpy.linalg.norm(mp_landmarks[30] - mp_landmarks[35]))
                metrics["MP_right_nose_depth"] = float(numpy.abs(mp_landmarks[30][2] - mp_landmarks[27][2]))
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