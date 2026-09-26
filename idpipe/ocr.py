"""Tesseract OCR 包裝。MRZ 用專用設定（只允許 A-Z 0-9 <），全文用一般設定。"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # Tesseract 多執行緒反而慢很多

import cv2
import numpy as np

from . import tess

MRZ_CONFIG = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def _up(gray, scale):
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _otsu(g):
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _adaptive(g):
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


def _sharpen(g, amount=1.5, sigma=2):
    """Unsharp mask：模糊照片的字元邊緣補回來。"""
    return cv2.addWeighted(g, 1 + amount, cv2.GaussianBlur(g, (0, 0), sigma), -amount, 0)


def _clahe(g):
    """局部對比增強：暗、低對比的照片把字拉出來。"""
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)


def _flatten(g):
    """除以大範圍模糊的背景，壓平光照（逆光：證件一側亮、一側暗）。"""
    bg = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 20)
    return cv2.normalize(cv2.divide(g, bg, scale=200), None, 0, 255, cv2.NORM_MINMAX)


def _prep(gray, scale=2.0):
    return _otsu(cv2.GaussianBlur(_up(gray, scale), (3, 3), 0))


# MRZ 前處理組合，依序嘗試，取第一個檢查碼通過的。順序依合成資料上的貪婪選擇：
# 銳化灰階與自適應二值化對模糊照片最有效（blur 5~7 檢查碼通過率 10/27 -> 24/27）。
MRZ_VARIANTS = [
    lambda g: _sharpen(_up(g, 1.5)),
    lambda g: _adaptive(_up(g, 1.5)),
    lambda g: _sharpen(_up(g, 2.0)),
    lambda g: _adaptive(_sharpen(_up(g, 2.0))),
    lambda g: _adaptive(_sharpen(_up(g, 1.5))),
    lambda g: _prep(g, 2.0),
    lambda g: _prep(g, 3.0),
    # 光線不佳時的補救，放在最後：前面的讀法已經足夠時不會執行，清楚的照片不會變慢。
    # 模擬逆光／偏暗（14 張 × 5 種光線）：實拍護照逆光從檢查碼失敗變成全對，偏暗的讀錯欄位變少。
    lambda g: _adaptive(_clahe(_up(g, 2.0))),
    lambda g: _sharpen(_clahe(_flatten(_up(g, 1.5)))),
    lambda g: _prep(_flatten(g), 2.0),
]


MRZ_CROP_CHAR_H = (40, 30)  # 精確裁切的 MRZ 縮放到的字高（像素）；實拍護照在 40 讀得最好，30 與拉正頁面上的字高相近


def _ocr_mrz_text(img) -> str:
    """MRZ 用的 OCR：以 image_to_data 同時取得文字與行框，像 MRZ 的行（24 字以上）再依字形校正
    （'<' 誤讀成 K/S/E、多讀或漏讀字；見 chevron.py）。回傳與 image_to_string 相同格式的多行文字。"""
    from .chevron import correct_line
    d = tess.image_to_data(img, MRZ_CONFIG)
    words = [(d["text"][i].strip(), d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i])
             for i in range(len(d["text"])) if d["text"][i].strip()]
    # 依垂直位置組行（Tesseract 的 block/line 編號有時把同一行拆到不同區塊，例如姓名與後面的 '<' 串）
    rows = []
    for w in sorted(words, key=lambda w: (w[2] + w[4]) / 2):
        yc, hh = (w[2] + w[4]) / 2, w[4] - w[2]
        if rows and abs(yc - rows[-1]["yc"]) < 0.5 * max(hh, rows[-1]["h"]):
            rows[-1]["words"].append(w)
        else:
            rows.append({"yc": yc, "h": hh, "words": [w]})
    binary = None
    out = []
    for row in rows:
        ws = sorted(row["words"], key=lambda w: w[1])
        text = "".join(w[0] for w in ws)
        box = (min(w[1] for w in ws), min(w[2] for w in ws), max(w[3] for w in ws), max(w[4] for w in ws))
        if len(text) >= 24:
            if binary is None:
                binary = img if len(np.unique(img)) <= 2 else _otsu(img)
            text = correct_line(binary, box, text)
        out.append(text)
    return "\n".join(out)


class MrzText(str):
    """ocr_mrz_region 的回傳值：就是 MRZ 文字，另外帶 uncertain_name（姓名行各次辨識不一致，應送複核）。"""
    uncertain_name: bool = False


def vote_name(reads: list) -> tuple:
    """姓名行（沒有檢查碼）逐格多數決。reads：同一格式、同長度的多次辨識結果（已經過字形校正、格子對齊）。
    先取 '<' 位置相同的最大一群，再逐格投票。回傳 (姓名行, 是否不確定)：
    只有一次讀法、或任一格票數打平時視為不確定。"""
    from collections import Counter
    patterns = Counter(tuple(i for i, c in enumerate(r) if c == "<") for r in reads)
    top = patterns.most_common(1)[0][0]
    group = [r for r in reads if tuple(i for i, c in enumerate(r) if c == "<") == top]
    out, uncertain = [], len(group) < 2
    for pos in range(len(group[0])):
        ranked = Counter(r[pos] for r in group).most_common(2)
        if len(ranked) == 2 and ranked[0][1] == ranked[1][1]:
            uncertain = True
        out.append(ranked[0][0])
    return "".join(out), uncertain


def ocr_mrz_region(rectified, bottom_ratios: tuple = (0.32, 0.45), min_reads: int = 3, quick: bool = False) -> str:
    """只辨識拉正後影像的下方區域（MRZ 所在），依序用各裁切範圍 × MRZ_VARIANTS 各種前處理辨識。
    - 有檢查碼的行：取第一個檢查碼全過的結果。
    - 姓名行（沒有檢查碼）：收集各次讀法逐格多數決（vote_name）；讀到 min_reads 次、或 2 次完全相同才停止，
      不夠就繼續用下一個裁切範圍讀（實拍護照：第一種讀法把姓的 I 讀成 L，其餘讀法正確）。
      各次讀法打平或只有一次時，回傳值的 uncertain_name 為 True。
    - 先用 MRZ 定位（locate.py）把 MRZ 精確裁出、轉水平、統一字高（上方的視覺區文字、照片、頁面外的背景都會干擾
      Tesseract，實拍護照在「下方 32%」裡大多數前處理都找不到 MRZ）；定位不到或讀不夠時，
      再讀下方 32%（護照）與 45%（卡片 TD1 三行位置較高）。
    - 都沒有檢查碼全過的結果時，回傳通過最多檢查碼的那組（交給下游標記哪個檢查碼失敗）；
      連 MRZ 格式都找不到才回傳第一次的原始文字。
    - quick=True（拉正時探測方向用）：只讀精確裁切（定位不到才讀下方裁切），不讀備援。"""
    from .locate import locate_mrz
    from .mrz import find_mrz_lines, name_line_index, parse_mrz
    h = rectified.shape[0]
    first, checked, best, reads = None, None, None, {}
    crops = []
    loc = locate_mrz(rectified)
    if loc is not None:
        crops += [loc.crop(rectified, char_h=ch, upright=True, pad=0.8) for ch in MRZ_CROP_CHAR_H]
    if not (quick and crops):
        crops += [cv2.cvtColor(rectified[int(h * (1 - r)):], cv2.COLOR_BGR2GRAY) for r in bottom_ratios]

    def enough():
        rs = reads.get(len(checked), []) if checked else []
        return len(rs) >= min_reads or (len(rs) >= 2 and len(set(rs)) == 1)

    for gray in crops:
        for prep in MRZ_VARIANTS:
            text = _ocr_mrz_text(prep(gray))
            first = first if first is not None else text
            lines = find_mrz_lines(text)
            if not lines:
                continue
            reads.setdefault(len(lines), []).append(lines[name_line_index(lines)])
            if checked is None:
                try:
                    r = parse_mrz(lines)
                except ValueError:
                    continue
                if r.checksum_valid:
                    checked = lines
                elif best is None or sum(r.checks.values()) > best[0]:
                    best = (sum(r.checks.values()), lines)
            if enough():
                break
        if enough():
            break

    target = checked or (best[1] if best else None)
    if not target:
        return MrzText(first or "")
    ni = name_line_index(target)
    name, uncertain = vote_name([x for x in reads[len(target)] if len(x) == len(target[ni])])
    out = MrzText("\n".join(name if i == ni else l for i, l in enumerate(target)))
    out.uncertain_name = uncertain
    return out


def ocr_full(rectified) -> str:
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    return tess.image_to_string(gray, "--psm 6")
