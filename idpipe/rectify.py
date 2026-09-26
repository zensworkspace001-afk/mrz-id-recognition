"""透視校正：用四角點（左上、右上、右下、左下）把證件拉正成標準大小。規則式，不需訓練。"""
from __future__ import annotations

import cv2
import numpy as np


def rectify(image, corners, size=(1250, 880)):
    w, h = size
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def order_corners(pts):
    """把任意順序的四點排成 左上、右上、右下、左下。"""
    p = np.array(pts, dtype=np.float32)
    s, d = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return [p[np.argmin(s)].tolist(), p[np.argmin(d)].tolist(), p[np.argmax(s)].tolist(), p[np.argmax(d)].tolist()]
