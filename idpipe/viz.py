"""視覺區（VIZ）欄位抽取：依版型 region 切出各欄位，逐欄 OCR，回傳文字、信心與框位置。

region 是拉正後影像的相對座標 (x0, y0, x1, y1)。版型未標定（region 為 None）的欄位略過。
目前只用 Tesseract 英文模型；漢字/韓文等原文欄位（name_native）待接多語 OCR。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import re

import cv2
import numpy as np

from . import tess

from .normalize import parse_date
from .ocr import _otsu, _sharpen, _up

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
# 各欄位允許的字元。Tesseract whitelist 不能含空白，含空白的欄位（姓名、日期）不設限、事後清理。
WHITELIST = {
    "nationality": _LETTERS,
    "sex": "MFX",
    "document_number": _LETTERS + _DIGITS,
    "personal_number": _LETTERS + _DIGITS,
    "arc_number": _LETTERS + _DIGITS,
}
SKIP = {"name_native"}  # 需要非英文 OCR


def _is_date(name: str) -> bool:
    return "date" in name or name in ("valid_from", "valid_until")


def _valid(name: str, text: str) -> bool:
    """欄位格式檢查，作用類似 MRZ 檢查碼：不通過就換下一種前處理再試。"""
    if not text:
        return False
    if _is_date(name):
        return parse_date(text) is not None
    if name == "sex":
        return text in ("M", "F", "X")
    if name == "nationality":
        return bool(re.fullmatch(r"[A-Z]{3}", text))
    if name in ("surname", "given_names", "middle_name"):
        return bool(re.fullmatch(r"[A-Z][A-Z' -]*", text))
    return True


def _clean(name: str, text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().upper()
    if name in ("surname", "given_names", "middle_name"):
        text = re.sub(r"[^A-Z' -]", "", text).strip()
    return text


def _flatten(gray):
    """除以大範圍背景亮度，消除反光與陰影造成的明暗不均。"""
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    return cv2.normalize(cv2.divide(gray, bg, scale=255), None, 0, 255, cv2.NORM_MINMAX)


def _drop_color(bgr):
    """每個像素取最亮的色版：彩色的浮水印、底紋變淺，黑色墨水維持深色。"""
    return bgr.max(axis=2)


FIELD_VARIANTS = [
    lambda c: _sharpen(_up(_flatten(_drop_color(c)), 2.0)),
    lambda c: _otsu(_up(_flatten(_drop_color(c)), 2.0)),
    lambda c: _sharpen(_up(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), 2.0)),
]


def ocr_field(crop_bgr, name: str):
    """回傳 (文字, 信心 0~1)。依序嘗試 FIELD_VARIANTS，取第一個通過格式檢查的結果；
    都不通過則取信心最高者。信心為各字詞 Tesseract 信心的最小值。"""
    cfg = "--psm 7" + (f" -c tessedit_char_whitelist={WHITELIST[name]}" if name in WHITELIST else "")
    best = ("", 0.0)
    for prep in FIELD_VARIANTS:
        d = tess.image_to_data(prep(crop_bgr), cfg)
        words = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
        if not words:
            continue
        text = _clean(name, " ".join(t for t, _ in words))
        conf = min(c for _, c in words) / 100
        if _valid(name, text):
            return text, conf
        if conf > best[1]:
            best = (text, conf)
    return best


def make_viz_extractor(layout: dict | None = None):
    """layout 給定時用它取代版型的 region（例如 templates.SYNTH_LAYOUT），只抽版型本身有的欄位。"""
    def extract(rectified, tpl) -> dict:
        h, w = rectified.shape[:2]
        out = {}
        for name in tpl.fields:
            region = (layout or {}).get(name) or tpl.fields[name]
            if region is None or name in SKIP:
                continue
            x0, y0, x1, y1 = region
            box = [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)]
            text, conf = ocr_field(rectified[box[1]:box[3], box[0]:box[2]], name)
            if text:
                out[name] = {"raw": text, "confidence": conf, "bbox": box}
        return out
    return extract
