"""直接在照片中找 MRZ，再依 ICAO 9303 的版面比例推算資料頁範圍。

四角點偵測（detect.py）需要看到證件的四個邊；護照整本打開拍攝（上面一頁、手指、背景）時常找不到。
MRZ 則是 2～3 行等長、平行、緊鄰的深色等寬字，不論照片怎麼旋轉都很好找：
1. blackhat 凸顯淺底上的深色小字，以數種門檻（Otsu 與高百分位）各二值化一次。
2. 以 ROTATION_STEP 為間隔旋轉二值圖，每個角度用水平閉運算把同一行的字連成長條，
   找出 2 條（TD3 護照）或 3 條（TD1 卡片）長度相近、左緣對齊、間距合理，且每條由至少 MIN_GLYPHS 個
   獨立字形組成的長條（排除護照疊起來的頁緣線這類連續線條）。
3. 字距固定每字 2.54 mm，所以由 MRZ 寬度得到「每毫米幾像素」，再依頁面尺寸推算資料頁四角。

MRZ 與頁面邊緣的距離是用實拍台灣護照與移民署居留證樣張量測的（見 PAGE_GEOMETRY），
各國實際版面可能略有差異；推算出的頁面只用來拉正與裁切，MRZ 本身另有精確範圍。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ROTATION_STEP = 5  # 度；每個角度的閉運算可容忍約 ±2.5° 的傾斜，再由 minAreaRect 修正
BLACKHAT_PERCENTILES = (90, 94, 97)  # 加上 Otsu；防偽底紋多時 Otsu 門檻太低，低解析度卡片則反過來
WORK_SIZE = 1200  # 長邊縮放到這個尺寸再找

# 格式 -> (頁寬 mm, 頁高 mm, MRZ 墨跡寬 mm, 最後一行中心到頁面底邊 mm)
# 墨跡寬 = 字數 × 2.54 mm 扣掉最後一格的空白；底邊距離為實測值
PAGE_GEOMETRY = {
    "TD3": (125.0, 88.0, 111.0, 7.8),
    "TD1": (85.6, 54.0, 75.4, 7.7),
}


@dataclass
class MrzLocation:
    format: str  # TD3 | TD1
    lines: list  # 每行的四角（原圖座標，沿文字方向：左上、右上、右下、左下）
    u: np.ndarray  # 文字行方向（單位向量，由左到右，假設證件正向）
    v: np.ndarray  # 垂直於行、指向頁面上方（假設證件正向）
    left: np.ndarray  # MRZ 左緣、最後一行中心的點（原圖座標）
    width: float  # MRZ 墨跡寬（像素）
    line_height: float  # 字高（像素）

    def page_quad(self, flipped: bool = False) -> list:
        """資料頁四角（左上、右上、右下、左下）。flipped=True 表示證件實際上是倒的（第一行在下面）。"""
        page_w, page_h, ink_w, bottom = PAGE_GEOMETRY[self.format]
        px = self.width / ink_w
        u, v, origin = self.u, self.v, self.left
        if flipped:  # 倒過來：最後一行其實是第一行，頁面在另一側
            origin = self.left + u * self.width + v * self._first_to_last()
            u, v = -u, -v
        margin = (page_w - ink_w) / 2 * px
        bl = origin - u * margin - v * bottom * px
        br = bl + u * page_w * px
        tl = bl + v * page_h * px
        tr = br + v * page_h * px
        return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]

    def crop(self, image, char_h: int = 32, upright: bool = False, pad: float = 0.3):
        """把 MRZ 區塊轉成水平並裁切（灰階），縮放到字高 char_h 像素，四周留 pad 倍字高的邊。
        方向同 u／v（可能上下顛倒）；upright=True 表示影像已經轉正（例如拉正後的頁面），強制文字由左到右。"""
        pts = np.vstack([np.array(l) for l in self.lines])
        u, v = (self.u, self.v) if not upright or self.u[0] >= 0 else (-self.u, -self.v)
        pu, pv = pts @ u, pts @ v
        pad = pad * self.line_height
        x0, x1, y0, y1 = pu.min() - pad, pu.max() + pad, pv.min() - pad, pv.max() + pad
        k = char_h / max(self.line_height, 1.0)
        W, H = max(1, int((x1 - x0) * k)), max(1, int((y1 - y0) * k))
        src = np.array([u * x0 + v * y1, u * x1 + v * y1, u * x1 + v * y0], np.float32)
        dst = np.array([[0, 0], [W, 0], [W, H]], np.float32)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.warpAffine(gray, cv2.getAffineTransform(src, dst), (W, H), borderValue=255)

    def _first_to_last(self) -> float:
        """第一行中心到最後一行中心的距離（像素）。"""
        c = [np.mean(np.array(l), axis=0) for l in self.lines]
        return float(np.linalg.norm(c[-1] - c[0]))


MIN_GLYPHS = 8  # 一條 MRZ 長條內至少要有的獨立字形數；模糊時字會黏在一起，頁緣等連續線條只有 1～2 個元件


def _glyph_count(binary, cx, cy, length, thick):
    """長條範圍內、閉運算前的獨立字形數。MRZ 是一個個字；頁緣線、桌緣等連續線條只有一兩個元件。"""
    x0, x1 = int(max(0, cx - length / 2)), int(cx + length / 2) + 1
    y0, y1 = int(max(0, cy - thick / 2)), int(cy + thick / 2) + 1
    sub = binary[y0:y1, x0:x1]
    if sub.size == 0:
        return 0
    n, _, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    return sum(1 for i in range(1, n) if st[i][3] >= 0.3 * thick and st[i][2] <= 2.5 * thick)


def _candidates(closed, binary, min_len):
    """閉運算後的長條：回傳 [(cx, cy, length, thickness, angle)]（旋轉後座標）。
    binary 為閉運算前的二值圖，用來確認長條是由一個個字組成（排除頁緣、桌緣等連續線條）。"""
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        (cx, cy), (a, b), ang = cv2.minAreaRect(c)
        length, thick = max(a, b), min(a, b)
        if a < b:
            ang += 90
        ang = (ang + 90) % 180 - 90  # 轉成 -90～90，接近 0 表示水平
        if length >= min_len and thick > 0 and length / thick >= 10 and abs(ang) <= 8 \
                and _glyph_count(binary, cx, cy, length, thick) >= MIN_GLYPHS:
            out.append((cx, cy, length, thick, ang))
    return out


def _best_group(cands):
    """找 2 或 3 條長度相近、左緣對齊、間距合理的平行長條；回傳 (分數, 群組)。"""
    cands = sorted(cands, key=lambda c: c[1])
    best = (0.0, None)
    for n in (3, 2):
        for i in range(len(cands) - n + 1):
            g = cands[i:i + n]
            lens = [c[2] for c in g]
            thick = float(np.median([c[3] for c in g]))
            lefts = [c[0] - c[2] / 2 for c in g]
            gaps = np.diff([c[1] for c in g])
            if min(lens) / max(lens) < 0.85 or max(lefts) - min(lefts) > 0.06 * max(lens):
                continue
            if not all(1.1 * thick <= d <= 3.0 * thick for d in gaps):
                continue
            # MRZ 各行字高相同、行距固定；避免把上方的視覺區文字行算進來
            if max(c[3] for c in g) > 1.5 * min(c[3] for c in g) or (len(gaps) > 1 and max(gaps) > 1.25 * min(gaps)):
                continue
            if max(c[4] for c in g) - min(c[4] for c in g) > 3:
                continue
            # 長度 × 行數；TD1 三行的每行比 TD3 短，所以以行數加權
            score = sum(lens) * (1.15 if n == 3 else 1.0)
            if score > best[0]:
                best = (score, g)
    return best


def locate_mrz(image):
    """在照片中找 MRZ；找不到回傳 None。"""
    h, w = image.shape[:2]
    s = min(1.0, WORK_SIZE / max(h, w))
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape
    k = max(9, int(0.02 * max(sh, sw)) | 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    binaries = [cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
    binaries += [((bh > np.percentile(bh, p)) * 255).astype(np.uint8) for p in BLACKHAT_PERCENTILES]

    diag = int(np.hypot(sh, sw)) + 2
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3))
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    best = (0.0, None, None)
    for deg in range(0, 180, ROTATION_STEP):
        R = cv2.getRotationMatrix2D((sw / 2, sh / 2), deg, 1.0)
        R[:, 2] += [(diag - sw) / 2, (diag - sh) / 2]
        for binary in binaries:
            rb = cv2.warpAffine(binary, R, (diag, diag))
            closed = cv2.morphologyEx(cv2.morphologyEx(rb, cv2.MORPH_CLOSE, close_k), cv2.MORPH_OPEN, open_k)
            score, group = _best_group(_candidates(closed, rb, 0.18 * max(sh, sw)))
            if group and score > best[0]:
                best = (score, group, R)
    if best[1] is None:
        return None

    _, group, R = best
    return _finish(group, R, s)


def _finish(group, R, s):
    """把旋轉後座標的長條換回原圖座標。"""
    Rinv = cv2.invertAffineTransform(R)
    lines = []
    for cx, cy, length, thick, ang in group:
        box = cv2.boxPoints(((cx, cy), (length, thick), ang))  # 旋轉後座標
        box = sorted(box.tolist(), key=lambda p: p[0])
        left = sorted(box[:2], key=lambda p: p[1])
        right = sorted(box[2:], key=lambda p: p[1])
        quad = np.array([left[0], right[0], right[1], left[1]], dtype=np.float64)  # 左上、右上、右下、左下
        quad = (np.hstack([quad, np.ones((4, 1))]) @ Rinv.T) / s
        lines.append(quad)
    d = sum(l[1] - l[0] for l in lines)
    u = d / np.linalg.norm(d)
    v = np.array([u[1], -u[0]])  # 影像 y 軸向下：u 朝右時 v 朝上
    # 正反方向在這裡無法可靠判斷（MRZ 下方可能是條碼、手指或有紋路的背景），
    # 呼叫端用 page_quad(flipped=False/True) 各拉正一次，以 MRZ 檢查碼決定。
    height = float(np.mean([np.linalg.norm(l[3] - l[0]) for l in lines]))
    lines.sort(key=lambda l: -float(np.mean(l, 0) @ v))  # 第一行在最上方
    lines = [_reorder(l, u, v) for l in lines]
    pts = np.vstack(lines)
    lo, hi = float((pts @ u).min()), float((pts @ u).max())
    last_c = np.mean(lines[-1], 0)
    left = last_c + u * (lo - float(last_c @ u))
    return MrzLocation(format="TD1" if len(lines) == 3 else "TD3", lines=[l.tolist() for l in lines],
                       u=u, v=v, left=left, width=hi - lo, line_height=height)


def _reorder(quad, u, v):
    """四角依文字方向排成左上、右上、右下、左下。"""
    c = quad.mean(0)
    def key(p):
        return (float((p - c) @ u) > 0, float((p - c) @ v) < 0)
    order = {(False, False): 0, (True, False): 1, (True, True): 2, (False, True): 3}
    out = [None] * 4
    for p in quad:
        out[order[key(p)]] = p
    return np.array(out) if all(o is not None for o in out) else quad
