"""傳統（不需訓練）的證件四角點偵測。之後可用訓練好的模型取代，介面相同：
detect_corners(image) -> [[x,y] x4]（左上、右上、右下、左下）或 None

方法一：以影像四邊的像素估計背景，與背景色差大的區域視為證件。背景先用平面（隨位置線性變化的顏色）
        擬合，可處理陰影造成的明暗漸層；再退回單一中位數色。
方法二：Canny 邊緣輪廓（背景複雜時的備援）。
找到區域後，不直接取多邊形頂點，而是去掉貼著影像邊界的線段、對證件四邊各擬合一條直線再求交點，
所以證件有角落超出畫面時，也能把超出的角點推算出來（座標可能在影像外）。
最後檢查形狀像不像證件（長寬比、內角）；都不像就回傳 None，由呼叫端改用整張圖。"""
from __future__ import annotations

import cv2
import numpy as np

from .rectify import order_corners

# 證件長寬比：護照 ID-3 約 1.42、卡片 ID-1 約 1.59；留空間給透視變形
ASPECT_RANGE = (1.25, 1.8)
ANGLE_RANGE = (65.0, 115.0)
FRAME_SNAP = 0.12  # 四角都在影像角落這個比例內：視為照片已裁切到證件，直接用整張圖的四角
MAX_OUTSIDE = 0.35  # 推算出的角點最多可超出影像邊界的比例


def plausible_document(quad) -> bool:
    """四邊形像不像拍到的證件：長寬比在 ASPECT_RANGE、四個內角在 ANGLE_RANGE。
    例如照片已裁切到只剩居留證時，背景色估計會失準，可能框到證件中間的文字區塊（長寬比 2 以上）。"""
    p = np.array(order_corners(quad), dtype=np.float64)
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    if min(w, h) < 1 or not ASPECT_RANGE[0] <= max(w, h) / min(w, h) <= ASPECT_RANGE[1]:
        return False
    for i in range(4):
        v1, v2 = p[i - 1] - p[i], p[(i + 1) % 4] - p[i]
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
        if not ANGLE_RANGE[0] <= ang <= ANGLE_RANGE[1]:
            return False
    return True


def _snap_to_frame(corners, w, h):
    """照片已裁切到只剩證件：四角都在影像內、且都貼近影像四角時，改用整張圖。
    推算出超出畫面的角點不套用（那是證件超出畫面，不是已裁切）。"""
    frame = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    inside = all(0 <= c[0] <= w - 1 and 0 <= c[1] <= h - 1 for c in corners)
    if inside and all(abs(c[0] - f[0]) <= FRAME_SNAP * w and abs(c[1] - f[1]) <= FRAME_SNAP * h
                      for c, f in zip(corners, frame)):
        return [[float(x), float(y)] for x, y in frame]
    return corners


def _intersect(l1, l2):
    """兩條直線 (點, 方向) 的交點；幾乎平行時回傳 None。"""
    (p1, d1), (p2, d2) = l1, l2
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def _fit_side(contour, a, b):
    """用輪廓上落在線段 a-b 附近的點擬合直線（比只用兩個頂點準，模糊時尤其明顯）。"""
    d = b - a
    n = np.linalg.norm(d)
    rel = contour - a
    t = rel @ d / (n * n)
    off = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / n
    pts = contour[(t > 0.1) & (t < 0.9) & (off < 4)]
    if len(pts) < 5:
        return a, d / n
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _quad_from_mask(mask, min_area):
    h, w = mask.shape[:2]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    hull = cv2.convexHull(c)
    contour = c.reshape(-1, 2).astype(np.float64)
    peri = cv2.arcLength(hull, True)
    for eps in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04):
        poly = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2).astype(np.float64)
        if len(poly) <= 8:
            break
    n = len(poly)
    if n < 4:
        return None

    tol = 3
    def on_border(p, q):
        return ((p[0] <= tol and q[0] <= tol) or (p[0] >= w - 1 - tol and q[0] >= w - 1 - tol) or
                (p[1] <= tol and q[1] <= tol) or (p[1] >= h - 1 - tol and q[1] >= h - 1 - tol))

    # 證件的邊：不貼著影像邊界、且夠長的線段；取最長的四條，依多邊形順序排列
    sides = [(i, poly[i], poly[(i + 1) % n]) for i in range(n)
             if not on_border(poly[i], poly[(i + 1) % n])
             and np.linalg.norm(poly[(i + 1) % n] - poly[i]) >= 0.08 * min(w, h)]
    if len(sides) < 4:
        return None
    sides = sorted(sorted(sides, key=lambda s: -np.linalg.norm(s[2] - s[1]))[:4], key=lambda s: s[0])
    lines = [_fit_side(contour, a, b) for _, a, b in sides]
    corners = [_intersect(lines[k], lines[(k + 1) % 4]) for k in range(4)]
    if any(p is None for p in corners):
        return None
    q = np.array(corners, dtype=np.float32)
    if (q[:, 0] < -MAX_OUTSIDE * w).any() or (q[:, 0] > (1 + MAX_OUTSIDE) * w).any() or \
            (q[:, 1] < -MAX_OUTSIDE * h).any() or (q[:, 1] > (1 + MAX_OUTSIDE) * h).any():
        return None
    return q


def _plane_background(img, band=6, iters=3):
    """以影像四邊 band 像素寬的區域，對每個色版擬合 c = a + b·x + c·y（陰影多為線性漸層）。
    反覆去掉殘差最大的 30%（多半是超出畫面、壓到邊界的證件本身）再擬合。"""
    h, w = img.shape[:2]
    m = np.zeros((h, w), bool)
    m[:band], m[-band:], m[:, :band], m[:, -band:] = True, True, True, True
    ys, xs = np.nonzero(m)
    A = np.stack([np.ones_like(xs), xs / w, ys / h], 1).astype(np.float64)
    y = img[ys, xs].astype(np.float64)
    keep = np.ones(len(xs), bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        res = np.linalg.norm(A @ coef - y, axis=1)
        keep = res <= np.percentile(res, 70)
    gy, gx = np.mgrid[0:h, 0:w]
    full = np.stack([np.ones((h, w)), gx / w, gy / h], -1) @ coef
    return full.astype(np.float32)


def detect_corners(image, min_area_ratio: float = 0.15):
    h, w = image.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    sh, sw = small.shape[:2]
    min_area = min_area_ratio * sh * sw
    blur = cv2.GaussianBlur(small, (7, 7), 0)

    def accept(q):
        if q is not None and plausible_document(q):
            return _snap_to_frame(order_corners((q / scale).tolist()), w, h)
        return None

    # 方法一：背景色差（先平面背景，再單一中位數色）
    border = np.concatenate([blur[:5].reshape(-1, 3), blur[-5:].reshape(-1, 3),
                             blur[:, :5].reshape(-1, 3), blur[:, -5:].reshape(-1, 3)])
    backgrounds = [_plane_background(blur), np.median(border, axis=0).astype(np.float32)]
    for bg in backgrounds:
        dist = np.linalg.norm(blur.astype(np.float32) - bg, axis=2)
        for thr in (35, 25, 50):
            q = accept(_quad_from_mask((dist > thr).astype(np.uint8) * 255, min_area))
            if q is not None:
                return q

    # 方法二：邊緣
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    for lo, hi in ((30, 90), (15, 50), (50, 150)):
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((3, 3), np.uint8), iterations=2)
        q = accept(_quad_from_mask(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), min_area))
        if q is not None:
            return q
    return None
