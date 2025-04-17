import os
import numpy as np
import cv2
import dlib
import scipy.io
import tensorflow as tf
from PRNet.utils.render import render_texture, render
from PRNet.utils.load import load_net, load_img, resize_n_crop_img, process_uv, predict_dense
from PRNet.utils.estimate_pose import estimate_pose
from PRNet.api import PRN

# Настройка
prn = PRN(is_dlib=True)
img_path = '/Users/victorkhudyakov/sx/3DDFA_V2/1.jpeg'
img_ori = cv2.imread(img_path)
img, img_landmark, box = resize_n_crop_img(img_ori, dlib.get_frontal_face_detector(), dlib.shape_predictor('PRNet/Data/net-data/shape_predictor_68_face_landmarks.dat'))
pos = prn.process(img)

# Если позиционная карта не найдена
if pos is None:
    raise RuntimeError('PRNet не смог обработать изображение.')

# Папка для вывода
output_dir = './prnet_outputs'
os.makedirs(output_dir, exist_ok=True)

# Базовая сетка
uv_coords = prn.get_uv_coords()
colors = img / 255.

# 10 ВАРИАНТОВ
for i in range(10):
    canvas = img.copy()

    if i == 0:
        rendered = render_texture(pos, img, prn.get_triangles())  # текстурная сетка
    elif i == 1:
        rendered = render(pos, colors, prn.get_triangles())  # без текстуры
    elif i == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rendered = render(pos, gray_colored / 255., prn.get_triangles())
    elif i == 3:
        inverted = 255 - img
        rendered = render(pos, inverted / 255., prn.get_triangles())
    elif i == 4:
        sharp = cv2.Laplacian(img, cv2.CV_64F)
        sharp = cv2.convertScaleAbs(sharp)
        rendered = render(pos, sharp / 255., prn.get_triangles())
    elif i == 5:
        uniform_color = np.ones_like(img) * 120
        rendered = render(pos, uniform_color / 255., prn.get_triangles())
    elif i == 6:
        color_jitter = img.copy()
        color_jitter[:, :, 0] = cv2.equalizeHist(color_jitter[:, :, 0])
        rendered = render(pos, color_jitter / 255., prn.get_triangles())
    elif i == 7:
        overlay = render(pos, colors, prn.get_triangles())
        rendered = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    elif i == 8:
        mesh = render(pos, np.ones_like(img), prn.get_triangles())
        rendered = cv2.bitwise_and(img, mesh)
    elif i == 9:
        textured = render_texture(pos, img, prn.get_triangles())
        rendered = cv2.addWeighted(img, 0.7, textured, 0.3, 0)
    
    out_path = os.path.join(output_dir, f'prnet_visualization_{i+1}.jpg')
    cv2.imwrite(out_path, (rendered * 255).astype(np.uint8))
    print(f"✅ Сохранено: {out_path}")