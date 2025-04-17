# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np

from Sim3DR import RenderPipeline
from utils.functions import plot_image
from .tddfa_util import _to_ctype

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (0.6, 0.6, 0.6),  # Серый фон
    'intensity_directional': 0.3,
    'color_directional': (1.0, 1.0, 1.0),  # Чистый белый свет
    'intensity_specular': 0.1,
    'specular_exp': 20,  # Точечные резкие блики
    'light_pos': (0, 0, 2),  # Фронтальный свет
    'view_pos': (0, 0, 5)
}
render_app = RenderPipeline(**cfg)


def render(img, ver_lst, tri, alpha=0.6, show_flag=False, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = img.copy()

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap)

    res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res

def render_color(ver_lst, tri, h, w, c=3):
    """
    Render 3D face on a blank canvas.

    Args:
        ver_lst: list of vertex arrays
        tri: triangle indices
        h: height of the output image
        w: width of the output image
        c: number of channels (default 3 for RGB or 4 for RGBA)

    Returns:
        img: rendered image
    """
    img = np.zeros((h, w, c), dtype=np.uint8)

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)
        img = render_app(ver, tri, img)

    return img