import numpy as np

def get_depth_buffer(ver, tri, depth_buffer, h, w):
    """
    Программная растеризация для создания буфера глубины
    
    Args:
        ver: вершины 3D модели
        tri: треугольники модели
        depth_buffer: исходный буфер глубины
        h, w: высота и ширина изображения
        
    Returns:
        depth_buffer: обновленный буфер глубины
    """
    # Обрабатываем каждый треугольник в модели
    for i in range(len(tri)):
        # Получаем индексы вершин треугольника
        idx1, idx2, idx3 = tri[i]
        
        # Получаем координаты вершин треугольника
        v1 = ver[idx1]
        v2 = ver[idx2]
        v3 = ver[idx3]
        
        # Проверяем, что координаты в пределах изображения
        x1, y1, z1 = int(v1[0]), int(v1[1]), v1[2]
        x2, y2, z2 = int(v2[0]), int(v2[1]), v2[2]
        x3, y3, z3 = int(v3[0]), int(v3[1]), v3[2]
        
        # Определяем границы треугольника
        min_x = max(0, min(x1, x2, x3))
        max_x = min(w - 1, max(x1, x2, x3))
        min_y = max(0, min(y1, y2, y3))
        max_y = min(h - 1, max(y1, y2, y3))
        
        # Если треугольник находится за пределами изображения, пропускаем его
        if min_x >= max_x or min_y >= max_y:
            continue
        
        # Вычисляем барицентрические координаты для каждого пикселя в треугольнике
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Вычисляем барицентрические координаты
                area = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                if area == 0:
                    continue
                
                w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / area
                w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / area
                w3 = 1 - w1 - w2
                
                # Проверяем, находится ли точка внутри треугольника
                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                    # Вычисляем глубину точки
                    z = w1 * z1 + w2 * z2 + w3 * z3
                    
                    # Если точка ближе к камере, чем предыдущая, обновляем буфер глубины
                    if z > depth_buffer[y, x]:
                        depth_buffer[y, x] = z
    
    return depth_buffer

def get_normal_buffer(ver, tri, normal_buffer, depth_buffer, h, w):
    """
    Создает буфер нормалей для 3D модели
    
    Args:
        ver: вершины 3D модели
        tri: треугольники модели
        normal_buffer: исходный буфер нормалей
        depth_buffer: буфер глубины для проверки видимости
        h, w: высота и ширина изображения
        
    Returns:
        normal_buffer: обновленный буфер нормалей
    """
    # Вычисляем нормали для каждой вершины
    vertex_normals = np.zeros_like(ver)
    vertex_counts = np.zeros((ver.shape[0],), dtype=np.int32)
    
    # Для каждого треугольника
    for i in range(len(tri)):
        # Получаем индексы вершин треугольника
        idx1, idx2, idx3 = tri[i]
        
        # Получаем координаты вершин треугольника
        v1 = ver[idx1][:3]
        v2 = ver[idx2][:3]
        v3 = ver[idx3][:3]
        
        # Вычисляем нормаль треугольника
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        normal_length = np.linalg.norm(normal)
        
        if normal_length > 0:
            normal = normal / normal_length
            
            # Добавляем нормаль к каждой вершине треугольника
            vertex_normals[idx1][:3] += normal
            vertex_normals[idx2][:3] += normal
            vertex_normals[idx3][:3] += normal
            
            vertex_counts[idx1] += 1
            vertex_counts[idx2] += 1
            vertex_counts[idx3] += 1
    
    # Нормализуем нормали вершин
    for i in range(ver.shape[0]):
        if vertex_counts[i] > 0:
            vertex_normals[i][:3] = vertex_normals[i][:3] / vertex_counts[i]
            normal_length = np.linalg.norm(vertex_normals[i][:3])
            if normal_length > 0:
                vertex_normals[i][:3] = vertex_normals[i][:3] / normal_length
    
    # Обрабатываем каждый треугольник в модели
    for i in range(len(tri)):
        # Получаем индексы вершин треугольника
        idx1, idx2, idx3 = tri[i]
        
        # Получаем координаты вершин треугольника
        v1 = ver[idx1]
        v2 = ver[idx2]
        v3 = ver[idx3]
        
        # Получаем нормали вершин треугольника
        n1 = vertex_normals[idx1][:3]
        n2 = vertex_normals[idx2][:3]
        n3 = vertex_normals[idx3][:3]
        
        # Проверяем, что координаты в пределах изображения
        x1, y1, z1 = int(v1[0]), int(v1[1]), v1[2]
        x2, y2, z2 = int(v2[0]), int(v2[1]), v2[2]
        x3, y3, z3 = int(v3[0]), int(v3[1]), v3[2]
        
        # Определяем границы треугольника
        min_x = max(0, min(x1, x2, x3))
        max_x = min(w - 1, max(x1, x2, x3))
        min_y = max(0, min(y1, y2, y3))
        max_y = min(h - 1, max(y1, y2, y3))
        
        # Если треугольник находится за пределами изображения, пропускаем его
        if min_x >= max_x or min_y >= max_y:
            continue
        
        # Вычисляем барицентрические координаты для каждого пикселя в треугольнике
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Вычисляем барицентрические координаты
                area = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                if area == 0:
                    continue
                
                w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / area
                w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / area
                w3 = 1 - w1 - w2
                
                # Проверяем, находится ли точка внутри треугольника
                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                    # Вычисляем глубину точки
                    z = w1 * z1 + w2 * z2 + w3 * z3
                    
                    # Если точка ближе к камере, чем предыдущая, обновляем буфер нормалей
                    if z > depth_buffer[y, x] - 1e-5:
                        # Интерполируем нормаль
                        nx = w1 * n1[0] + w2 * n2[0] + w3 * n3[0]
                        ny = w1 * n1[1] + w2 * n2[1] + w3 * n3[1]
                        nz = w1 * n1[2] + w2 * n2[2] + w3 * n3[2]
                        
                        # Нормализуем нормаль
                        norm = np.sqrt(nx*nx + ny*ny + nz*nz)
                        if norm > 0:
                            nx /= norm
                            ny /= norm
                            nz /= norm
                        
                        # Записываем нормаль в буфер нормалей
                        normal_buffer[y, x, 0] = nx
                        normal_buffer[y, x, 1] = ny
                        normal_buffer[y, x, 2] = nz
    
    return normal_buffer

def get_depth_buffer(ver, tri, depth_buffer, h, w):
    """
    Программная растеризация для создания буфера глубины
    
    Args:
        ver: вершины 3D модели
        tri: треугольники модели
        depth_buffer: исходный буфер глубины
        h, w: высота и ширина изображения
        
    Returns:
        depth_buffer: обновленный буфер глубины
    """
    # Проверяем максимальный индекс в треугольниках
    max_idx = tri.max() if hasattr(tri, 'max') else max(map(max, tri))
    
    # Проверяем, что все индексы в пределах размера массива вершин
    if max_idx >= len(ver):
        print(f"Предупреждение: максимальный индекс {max_idx} превышает размер массива вершин {len(ver)}")
        # Создаем маску для фильтрации треугольников с корректными индексами
        valid_triangles = []
        for i in range(len(tri)):
            idx1, idx2, idx3 = tri[i]
            if idx1 < len(ver) and idx2 < len(ver) and idx3 < len(ver):
                valid_triangles.append(tri[i])
        
        # Если нет корректных треугольников, возвращаем исходный буфер глубины
        if not valid_triangles:
            return depth_buffer
        
        # Используем только корректные треугольники
        tri = valid_triangles
    
    # Обрабатываем каждый треугольник в модели
    for i in range(len(tri)):
        # Получаем индексы вершин треугольника
        idx1, idx2, idx3 = tri[i]
        
        # Проверяем, что индексы в пределах массива вершин
        if idx1 >= len(ver) or idx2 >= len(ver) or idx3 >= len(ver):
            continue
        
        # Получаем координаты вершин треугольника
        v1 = ver[idx1]
        v2 = ver[idx2]
        v3 = ver[idx3]
        
        # Проверяем, что координаты в пределах изображения
        x1, y1, z1 = int(v1[0]), int(v1[1]), v1[2]
        x2, y2, z2 = int(v2[0]), int(v2[1]), v2[2]
        x3, y3, z3 = int(v3[0]), int(v3[1]), v3[2]
        
        # Определяем границы треугольника
        min_x = max(0, min(x1, x2, x3))
        max_x = min(w - 1, max(x1, x2, x3))
        min_y = max(0, min(y1, y2, y3))
        max_y = min(h - 1, max(y1, y2, y3))
        
        # Если треугольник находится за пределами изображения, пропускаем его
        if min_x >= max_x or min_y >= max_y:
            continue
        
        # Вычисляем барицентрические координаты для каждого пикселя в треугольнике
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Вычисляем барицентрические координаты
                area = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                if area == 0:
                    continue
                
                w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / area
                w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / area
                w3 = 1 - w1 - w2
                
                # Проверяем, находится ли точка внутри треугольника
                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                    # Вычисляем глубину точки
                    z = w1 * z1 + w2 * z2 + w3 * z3
                    
                    # Если точка ближе к камере, чем предыдущая, обновляем буфер глубины
                    if z > depth_buffer[y, x]:
                        depth_buffer[y, x] = z
    
    return depth_buffer
