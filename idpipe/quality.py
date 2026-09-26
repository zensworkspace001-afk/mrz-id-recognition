"""照片品質快速檢查（約 1 秒，不做 OCR）：辨識前先告訴使用者這張照片能不能用、該怎麼重拍。

門檻依合成資料（已知模糊程度與 MRZ 辨識結果）與實拍照片量測：
- 清晰度：MRZ 縮放到字高 32 像素後的 Laplacian 變異數。清晰的照片約 450～2900；最模糊的合成圖 18～350，
  其中多數仍讀得到 MRZ，所以偏低只警告，極低才判定不適用。
- 反光：MRZ 區塊內過曝（≥250）像素比例。讀不到 MRZ 的合成圖中，反光案例的過曝比例約 0.5。
- 字高：移民署居留證樣張 MRZ 字高 12 像素仍讀得到；實拍護照約 34～40 像素。
"""
from __future__ import annotations

import cv2
import numpy as np

from .locate import locate_mrz

MIN_CHAR_H = (10, 18)  # (不適用, 警告) 像素
MIN_SHARPNESS = (15, 60)
MAX_GLARE = (0.25, 0.05)
MIN_BRIGHTNESS = 70
EDGE_MARGIN = 0.01  # MRZ 距離畫面邊緣少於這個比例：可能被切到


def check_photo(image) -> dict:
    """回傳 {"usable": bool, "checks": [{"key", "label", "status": ok|warn|fail, "message"}], "mrz_lines": [...]}。
    usable 為 False 表示有任何一項 fail，建議重拍（仍可強制辨識）。"""
    checks = []

    def add(key, label, status, message):
        checks.append({"key": key, "label": label, "status": status, "message": message})

    h, w = image.shape[:2]
    loc = locate_mrz(image)
    if loc is None:
        add("mrz", "MRZ", "fail", "找不到 MRZ（證件底部兩三行英數字）。請確認 MRZ 完整入鏡、沒有被手指或反光遮住。")
        return {"usable": False, "checks": checks, "mrz_lines": []}
    kind = "卡片（3 行）" if loc.format == "TD1" else "護照（2 行）"
    add("mrz", "MRZ", "ok", f"找到 MRZ：{kind}")

    pts = np.vstack([np.array(l) for l in loc.lines])
    mx, my = EDGE_MARGIN * w, EDGE_MARGIN * h
    if pts[:, 0].min() < mx or pts[:, 1].min() < my or pts[:, 0].max() > w - mx or pts[:, 1].max() > h - my:
        add("frame", "完整入鏡", "warn", "MRZ 貼近畫面邊緣，可能有字被切掉。")
    else:
        add("frame", "完整入鏡", "ok", "MRZ 完整在畫面內")

    ch = loc.line_height
    if ch < MIN_CHAR_H[0]:
        add("size", "解析度", "fail", f"MRZ 字高只有 {ch:.0f} 像素，太小。請靠近一點或用較高解析度拍攝。")
    elif ch < MIN_CHAR_H[1]:
        add("size", "解析度", "warn", f"MRZ 字高 {ch:.0f} 像素，偏小，可能讀錯。")
    else:
        add("size", "解析度", "ok", f"MRZ 字高 {ch:.0f} 像素")

    crop = loc.crop(image, 32)
    sharp = cv2.Laplacian(crop, cv2.CV_64F).var()
    if sharp < MIN_SHARPNESS[0]:
        add("sharpness", "清晰度", "fail", f"影像很模糊（{sharp:.0f}）。請對焦在 MRZ、拿穩再拍。")
    elif sharp < MIN_SHARPNESS[1]:
        add("sharpness", "清晰度", "warn", f"影像偏模糊（{sharp:.0f}），可能讀錯。")
    else:
        add("sharpness", "清晰度", "ok", f"清晰（{sharp:.0f}）")

    glare = float((crop >= 250).mean())
    if glare > MAX_GLARE[0]:
        add("glare", "反光", "fail", f"MRZ 有 {glare:.0%} 過曝。請換個角度避開燈光反射。")
    elif glare > MAX_GLARE[1]:
        add("glare", "反光", "warn", f"MRZ 有 {glare:.0%} 過曝，部分字可能讀不到。")
    else:
        add("glare", "反光", "ok", "沒有明顯反光")

    if float(crop.mean()) < MIN_BRIGHTNESS:
        add("light", "亮度", "warn", "MRZ 偏暗，建議在光線充足處拍攝。")
    else:
        add("light", "亮度", "ok", "亮度足夠")

    return {"usable": all(c["status"] != "fail" for c in checks), "checks": checks, "mrz_lines": loc.lines}
