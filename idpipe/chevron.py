"""依字形校正 MRZ 一行的內容，專門處理填充符號 '<'。

問題：Tesseract 英文模型遇到一長串 '<'（姓名行後面的填充）常讀成 K、S、E，也會多讀或漏讀字，
在護照防偽底紋上尤其嚴重（實拍護照：第一行 23 個 '<' 幾乎全讀成 K，姓名還多讀了一個字母）。
姓名行沒有檢查碼，這種錯只能靠字形抓。

做法（MRZ 是等寬字型、一行固定 44 或 30 字）：
1. 以連通元件找出一行的每個字形，逐一判斷是不是 '<'：'<' 的上、下緣墨跡都在右半邊，中段只有靠左的尖端；
   K、E 左側有直線，S、C 上下緣延伸到左側，4 的中段延伸到最右邊。
2. 字形數剛好是 44 或 30 時，以動態規劃把 OCR 字串對齊到這些格子：多讀的字跳過、'<' 格一律填 '<'。
3. 每一段連續字母再單獨裁切重新辨識，字數與格數相符才採用；避開長串 '<' 對 Tesseract 語言模型的干擾。
含數字的行（TD3 第二行）改為依位置的字元類型分段重讀，只有通過的檢查碼變多才採用。
字形數不對、或仍有格子對不到字時，回傳原本的 OCR 字串，不做校正。
"""
from __future__ import annotations

import cv2
import numpy as np

from . import tess

MRZ_LENGTHS = (44, 30)
CONFUSED_WITH_CHEVRON = set("KSECX")
RUN_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def is_chevron(glyph) -> bool:
    """glyph：單一字形的布林墨跡陣列（True = 墨）。"""
    ys, xs = np.nonzero(glyph)
    if len(ys) < 10:
        return False
    c = glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = c.shape
    if h < 4 or w < 4:
        return False

    def extent(rows):
        cols = np.nonzero(c[rows].any(axis=0))[0]
        return (cols.min() / w, (cols.max() + 1) / w) if len(cols) else (1.0, 0.0)

    band = max(1, h // 6)
    top = extent(slice(0, band))[0]
    bottom = extent(slice(h - band, h))[0]
    mid_left, mid_right = extent(slice(h // 2 - max(1, h // 10), h // 2 + max(1, h // 10) + 1))
    # 中段只有左側的尖端；數字 4 形狀相近，但中段有橫槓和右側直線延伸到最右邊
    return top > 0.45 and bottom > 0.45 and mid_left < 0.25 and mid_right < 0.7


def glyphs(binary, box) -> list:
    """binary：白底黑字影像；box：(x0, y0, x1, y1) 一行的範圍（Tesseract 的行框，常被雜訊撐大）。
    回傳依 x 排序的字形 [{"box": (x0, y0, x1, y1), "chevron": bool}]。
    以字高（第 90 百分位，不是行高）篩掉雜點與底紋碎片；'<' 在有些字型比字母矮很多，所以下限放寬。"""
    x0, y0, x1, y1 = box
    ink = (binary[y0:y1, x0:x1] < 128).astype(np.uint8)
    num, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    cand = [i for i in range(1, num) if st[i][3] >= 0.2 * (y1 - y0) and st[i][4] >= 30]
    if not cand:
        return []
    hm = float(np.percentile([st[i][3] for i in cand], 90))
    keep = sorted((i for i in cand if 0.35 * hm <= st[i][3] <= 1.25 * hm and st[i][2] <= 1.3 * hm),
                  key=lambda i: cen[i][0])
    out = []
    for i in keep:
        x, y, w, h, _ = st[i]
        out.append({"box": (x0 + x, y0 + y, x0 + x + w, y0 + y + h),
                    "chevron": is_chevron(lab[y:y + h, x:x + w] == i)})
    return out


def align(slots: list, text: str) -> str:
    """slots：每格是否為 '<'；text：OCR 字串。回傳長度 len(slots) 的字串：
    '<' 格填 '<'，其他格填對齊到的 OCR 字元，對不到（OCR 漏字）填 '?'。"""
    n, m = len(slots), len(text)
    gap = -1.0

    def score(chev, ch):
        if chev:
            return 2.0 if ch == "<" else (0.5 if ch in CONFUSED_WITH_CHEVRON else -2.0)
        return -2.0 if ch == "<" else 1.0

    D = np.zeros((n + 1, m + 1))
    D[0, :] = np.arange(m + 1) * gap
    D[:, 0] = np.arange(n + 1) * gap
    B = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = (D[i - 1, j - 1] + score(slots[i - 1], text[j - 1]), D[i - 1, j] + gap, D[i, j - 1] + gap)
            k = int(np.argmax(opts))
            D[i, j], B[i, j] = opts[k], k
    out, i, j = [], n, m
    while i > 0:
        k = B[i, j] if j > 0 else 1
        if k == 0:
            out.append("<" if slots[i - 1] else text[j - 1])
            i, j = i - 1, j - 1
        elif k == 1:
            out.append("<" if slots[i - 1] else "?")
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))


def _reread_runs(binary, gl: list, line: str) -> str:
    """姓名行：每段連續字母單獨裁切重新辨識，字數相符才取代對齊結果。"""
    chars = list(line)
    i = 0
    while i < len(gl):
        if gl[i]["chevron"]:
            i += 1
            continue
        j = i
        while j < len(gl) and not gl[j]["chevron"]:
            j += 1
        if j - i >= 2:  # 單一字元的段落用 psm 7 讀不穩，保留對齊結果
            xa = min(g["box"][0] for g in gl[i:j]); ya = min(g["box"][1] for g in gl[i:j])
            xb = max(g["box"][2] for g in gl[i:j]); yb = max(g["box"][3] for g in gl[i:j])
            pad = max(4, (yb - ya) // 3)
            crop = binary[max(0, ya - pad):yb + pad, max(0, xa - pad):xb + pad]
            crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            t = tess.image_to_string(crop, RUN_CONFIG).strip().replace(" ", "")
            if len(t) == j - i:
                chars[i:j] = list(t)
        i = j
    return "".join(chars)


# TD3 第二行各位置的字元類型：A 英數、D 數字、L 字母、S 性別
TD3_LINE2_CLASSES = "A" * 9 + "D" + "L" * 3 + "D" * 7 + "S" + "D" * 7 + "A" * 14 + "D" * 2
_WHITELIST = {"A": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "D": "0123456789",
              "L": "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "S": "MFX"}


def _reread_by_class(binary, gl: list, line: str, classes: str) -> str:
    """依位置的字元類型分段（遇到 '<' 或類型改變就切開）裁切重新辨識，每段只允許該類型的字元，
    字數相符才取代。用在有檢查碼的行，由呼叫端比較檢查碼決定要不要採用。"""
    chars = list(line)
    i = 0
    while i < len(gl):
        if gl[i]["chevron"]:
            i += 1
            continue
        j = i
        while j < len(gl) and not gl[j]["chevron"] and classes[j] == classes[i]:
            j += 1
        xa = min(g["box"][0] for g in gl[i:j]); ya = min(g["box"][1] for g in gl[i:j])
        xb = max(g["box"][2] for g in gl[i:j]); yb = max(g["box"][3] for g in gl[i:j])
        pad = max(4, (yb - ya) // 3)
        crop = binary[max(0, ya - pad):yb + pad, max(0, xa - pad):xb + pad]
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        psm = 10 if j - i == 1 else 7
        t = tess.image_to_string(crop, f"--psm {psm} -c tessedit_char_whitelist={_WHITELIST[classes[i]]}")
        t = t.strip().replace(" ", "")
        if len(t) == j - i:
            chars[i:j] = list(t)
        i = j
    return "".join(chars)


def _td3_line2_score(line: str) -> int:
    """TD3 第二行自己就能驗證的檢查碼通過數（號碼、生日、到期日、個人號碼、總檢查碼都只用到第二行）。"""
    from .mrz import _fix_line2, _rebuild_line2, parse_td3, repair_document_number
    try:
        l2 = repair_document_number(_fix_line2(_rebuild_line2(line)[:44].ljust(44, "<")))
        return sum(parse_td3("P<XXX" + "<" * 39, l2).checks.values())
    except (ValueError, IndexError):
        return -1


def correct_line(binary, box, text: str) -> str:
    """依字形校正一行 MRZ 的 OCR 結果；無法可靠校正時回傳原字串。"""
    gl = glyphs(binary, box)
    if len(gl) not in MRZ_LENGTHS:
        return text
    line = align([g["chevron"] for g in gl], text)
    if any(ch.isdigit() for ch in line):
        # 含數字的行有檢查碼：TD3 第二行依位置類型分段重讀，只有通過的檢查碼比原 OCR 結果多才採用。
        # （不直接用對齊結果：對齊可能把號碼整段位移、又碰巧通過檢查碼，合成資料實測過）
        if len(gl) == 44 and "?" not in line:
            cand = _reread_by_class(binary, gl, line, TD3_LINE2_CLASSES)
            if _td3_line2_score(cand) > _td3_line2_score(text):
                return cand
        return text
    line = _reread_runs(binary, gl, line)
    # 姓名行一定以證件類別字母開頭、且含有字母；不符合表示字形找錯（例如只找到 '<'），不採用
    if "?" in line or not line[0].isalpha() or line.count("<") > len(line) - 2:
        return text
    return line
