"""精簡的 PP-OCR 推論（文字偵測＋辨識），用來讀視覺區。

模型：PaddleOCR 的 PP-OCRv6 small（中英文，Apache-2.0），取自 RapidOCR 3.9.2 的發行檔，放在 models/ppocr/。
推論只用 onnxruntime；前後處理照 RapidOCR 的預設參數重寫（DB 文字偵測 + CTC 解碼），
不依賴 RapidOCR 套件本身——它會連帶安裝 pip 版 opencv-python（含 GPL 的 FFmpeg）與 shapely（含 LGPL 的 GEOS）。
唯一改寫的步驟是文字框外擴：RapidOCR 用 shapely + pyclipper 把矩形往外擴，
對矩形而言等同於「最小外接矩形長寬各加 2 倍擴張距離」，這裡直接這樣算。

不做 180° 轉向分類：這裡只處理已經拉正的頁面。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "ppocr"
DET_MODEL = "PP-OCRv6_det_small.onnx"
REC_MODEL = "PP-OCRv6_rec_small.onnx"

# RapidOCR 3.9.2 的預設參數
DET_LIMIT_SIDE = 736  # 短邊至少縮放到這個長度（limit_type = min）
DET_THRESH, BOX_THRESH, UNCLIP_RATIO, MAX_CANDIDATES = 0.3, 0.5, 1.6, 1000
REC_HEIGHT, REC_WIDTH, REC_BATCH = 48, 320, 6
TEXT_SCORE = 0.5
BOX_SORT_Y = 10


@dataclass
class TextLine:
    text: str
    score: float
    box: np.ndarray  # 4 x 2，左上、右上、右下、左下（輸入影像座標）

    @property
    def x0(self) -> float:
        return float(self.box[:, 0].min())

    @property
    def x1(self) -> float:
        return float(self.box[:, 0].max())

    @property
    def y0(self) -> float:
        return float(self.box[:, 1].min())

    @property
    def y1(self) -> float:
        return float(self.box[:, 1].max())

    @property
    def yc(self) -> float:
        return float(self.box[:, 1].mean())


class PPOCR:
    def __init__(self, model_dir: Path = MODEL_DIR, threads: int = 0):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        if threads:
            opts.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(model_dir / DET_MODEL), opts, providers=providers)
        self.rec = ort.InferenceSession(str(model_dir / REC_MODEL), opts, providers=providers)
        chars = self.rec.get_modelmeta().custom_metadata_map["character"].splitlines()
        self.chars = ["blank"] + chars + [" "]

    def __call__(self, img: np.ndarray) -> list:
        """img：BGR。回傳分數達 TEXT_SCORE 的文字行，依閱讀順序（上到下、同一列左到右）。"""
        boxes = self._detect(img)
        if not boxes:
            return []
        crops = [_crop(img, b) for b in boxes]
        results = self._recognize(crops)
        return [TextLine(t, s, b) for b, (t, s) in zip(boxes, results) if s >= TEXT_SCORE and t]

    # ---- 偵測（DB） ----
    def _detect(self, img: np.ndarray) -> list:
        h, w = img.shape[:2]
        ratio = DET_LIMIT_SIDE / min(h, w) if min(h, w) < DET_LIMIT_SIDE else 1.0
        rh = max(32, int(round(int(h * ratio) / 32) * 32))
        rw = max(32, int(round(int(w * ratio) / 32) * 32))
        x = cv2.resize(img, (rw, rh)).astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        pred = self.det.run(None, {self.det.get_inputs()[0].name: x})[0][0, 0]
        mask = cv2.dilate((pred > DET_THRESH).astype(np.uint8), np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ph, pw = pred.shape
        boxes = []
        for c in contours[:MAX_CANDIDATES]:
            pts, sside = _mini_box(c)
            if sside < 3 or _box_score(pred, pts) < BOX_THRESH:
                continue
            (cx, cy), (bw, bh), ang = cv2.minAreaRect(pts.astype(np.float32))
            d = bw * bh * UNCLIP_RATIO / (2 * (bw + bh))  # 矩形外擴距離（面積 × 倍率 / 周長）
            pts, sside = _mini_box(cv2.boxPoints(((cx, cy), (bw + 2 * d, bh + 2 * d), ang)))
            if sside < 5:
                continue
            pts[:, 0] = np.clip(np.round(pts[:, 0] / pw * w), 0, w - 1)
            pts[:, 1] = np.clip(np.round(pts[:, 1] / ph * h), 0, h - 1)
            box = _order_clockwise(pts)
            if np.linalg.norm(box[0] - box[1]) <= 3 or np.linalg.norm(box[0] - box[3]) <= 3:
                continue
            boxes.append(box)
        boxes.sort(key=lambda b: (b[0, 1], b[0, 0]))
        # 同一列（y 差距小於 BOX_SORT_Y）依 x 排序
        rows, row = [], []
        for b in boxes:
            if row and b[0, 1] - row[-1][0, 1] >= BOX_SORT_Y:
                rows.append(row)
                row = []
            row.append(b)
        if row:
            rows.append(row)
        return [b for r in rows for b in sorted(r, key=lambda b: b[0, 0])]

    # ---- 辨識（CTC） ----
    def _recognize(self, crops: list) -> list:
        ratios = [c.shape[1] / c.shape[0] for c in crops]
        order = np.argsort(ratios)
        out = [("", 0.0)] * len(crops)
        for i in range(0, len(crops), REC_BATCH):
            idx = order[i:i + REC_BATCH]
            max_ratio = max(REC_WIDTH / REC_HEIGHT, max(ratios[j] for j in idx))
            width = int(REC_HEIGHT * max_ratio)
            batch = np.zeros((len(idx), 3, REC_HEIGHT, width), np.float32)
            for k, j in enumerate(idx):
                c = crops[j]
                rw = min(width, int(math.ceil(REC_HEIGHT * c.shape[1] / c.shape[0])))
                r = cv2.resize(c, (rw, REC_HEIGHT)).astype(np.float32).transpose(2, 0, 1) / 255
                batch[k, :, :, :rw] = (r - 0.5) / 0.5
            probs = self.rec.run(None, {self.rec.get_inputs()[0].name: batch})[0]
            for k, j in enumerate(idx):
                out[j] = self._ctc_decode(probs[k])
        return out

    def _ctc_decode(self, p: np.ndarray):
        ids, conf = p.argmax(axis=1), p.max(axis=1)
        keep = np.ones(len(ids), bool)
        keep[1:] = ids[1:] != ids[:-1]
        keep &= ids != 0
        text = "".join(self.chars[i] for i in ids[keep])
        score = float(np.mean(conf[keep])) if keep.any() else 0.0
        return text, round(score, 5)


def _mini_box(contour):
    rect = cv2.minAreaRect(np.asarray(contour, np.float32))
    p = sorted(list(cv2.boxPoints(rect)), key=lambda q: q[0])
    a, d = (p[0], p[1]) if p[1][1] > p[0][1] else (p[1], p[0])
    b, c = (p[2], p[3]) if p[3][1] > p[2][1] else (p[3], p[2])
    return np.array([a, b, c, d], np.float32), min(rect[1])


def _box_score(pred, box):
    h, w = pred.shape
    x0 = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1)); x1 = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    y0 = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1)); y1 = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [(box - [x0, y0]).astype(np.int32)], 1)
    return cv2.mean(pred[y0:y1 + 1, x0:x1 + 1], mask)[0]


def _order_clockwise(pts):
    xs = pts[np.argsort(pts[:, 0])]
    tl, bl = xs[:2][np.argsort(xs[:2, 1])]
    tr, br = xs[2:][np.argsort(xs[2:, 1])]
    return np.array([tl, tr, br, bl], np.float32)


def _crop(img, box):
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    return np.rot90(out) if out.shape[0] / max(out.shape[1], 1) >= 1.5 else out


_ENGINE = None


def engine() -> PPOCR:
    """共用的推論實例（載入模型約需 0.1～0.5 秒，只做一次）。"""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PPOCR()
    return _ENGINE
