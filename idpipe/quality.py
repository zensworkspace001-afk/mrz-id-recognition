"""照片品質快速檢查（約 1 秒，不做 OCR）：辨識前先告訴使用者這張照片能不能用、該怎麼重拍。

門檻依合成資料（已知模糊程度與 MRZ 辨識結果）與實拍照片量測：
- 清晰度：MRZ 縮放到字高 32 像素後的 Laplacian 變異數。清晰的照片約 450～2900；最模糊的合成圖 18～350，
  其中多數仍讀得到 MRZ，所以偏低只警告，極低才判定不適用。
- 反光：MRZ 區塊內過曝（≥250）像素比例。讀不到 MRZ 的合成圖中，反光案例的過曝比例約 0.5。
- 光線均勻：MRZ 背景（去掉字）最暗處與最亮處的比例，依模擬逆光、陰影的照片量測。
- 字高：移民署居留證樣張 MRZ 字高 12 像素仍讀得到；實拍護照約 34～40 像素。
- 斜拍：由 MRZ 字形量測（見 perspective）。門檻依兩本正拍台灣護照模擬各種傾斜後跑完整流程：
  左右斜 30° 時 MRZ 姓氏讀錯一字而且沒被發現、上下斜 45° 起欄位開始讀錯，所以在這之前就判定不適用。
"""
from __future__ import annotations

import cv2
import numpy as np

from .locate import locate_mrz

MIN_CHAR_H = (10, 18)  # (不適用, 警告) 像素
MIN_SHARPNESS = (15, 60)
MAX_GLARE = (0.25, 0.05)
MIN_BRIGHTNESS = 70
# MRZ 背景亮度最暗處 ÷ 最亮處（去掉字之後）。正常照片與合成資料 ≥ 0.66；逆光約 0.6～0.7（辨識時會補救）；
# 陰影斜切過 MRZ 約 0.3～0.5，實拍護照在這種情況下 MRZ 讀不出來
MIN_EVEN_LIGHT = (0.45, 0.65)  # (不適用, 警告)
EDGE_MARGIN = 0.01  # MRZ 距離畫面邊緣少於這個比例：可能被切到
OCRB_ASPECT = 1.08  # OCR-B 字高 ÷ 字距（實拍台灣護照 1.076／1.079、居留證樣張 1.10）
MIN_ASPECT = (0.66, 0.80)  # (不適用, 警告)；上下斜拍約 43°／30°
MAX_PITCH_CHANGE = (1.65, 1.45)  # (不適用, 警告)；左右斜拍約 25～30°／20°


def perspective(image, loc):
    """由 MRZ 字形估計斜拍程度，回傳 (字高字距比, 字距漸變)；字形不夠時回傳 None。
    - 字高字距比：OCR-B 的字高與字距（每字 2.54 mm）是國際標準規定的，各國相同；除以 OCRB_ASPECT 後正拍約 1.0。
      上下方向斜拍時字被壓扁，比值變小。（合成資料不是 OCR-B 字型、比值偏大，不會被誤判成斜拍。）
    - 字距漸變：MRZ 字距固定，左右方向斜拍時近端字距大、遠端小；取一行兩端字距的比值，正拍約 1.0。與字型無關。
    模糊會讓字形變粗、字高字距比偏大，所以只用它判斷「變小」（上下斜拍）。"""
    crop = loc.crop(image, 40, pad=0.4)
    bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, _, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    comps = [(st[i][3], cen[i]) for i in range(1, n) if st[i][4] > 15 and 10 < st[i][3] < 80 and st[i][2] < 60]
    if len(comps) < 20:
        return None
    char_h = float(np.percentile([h for h, _ in comps], 80))
    comps.sort(key=lambda c: c[1][1])
    rows, row = [], [comps[0]]
    for c in comps[1:]:  # 依 y 分行
        if c[1][1] - row[-1][1][1] < 0.5 * char_h:
            row.append(c)
        else:
            rows.append(row)
            row = [c]
    rows.append(row)
    gaps, changes = [], []
    for r in rows:
        if len(r) < 12:
            continue
        x = np.sort([c[1][0] for c in r])
        g, mid = np.diff(x), (x[1:] + x[:-1]) / 2
        ok = (g > 0.7 * np.median(g)) & (g < 1.3 * np.median(g))  # 排除字黏在一起或斷開造成的異常間距
        gaps += list(g[ok])
        if ok.sum() >= 10:
            k, b = np.polyfit(mid[ok], g[ok], 1)
            ends = (k * x[0] + b, k * x[-1] + b)
            if min(ends) > 0:
                changes.append(max(ends) / min(ends))
    if len(gaps) < 10 or not changes:
        return None
    return char_h / float(np.median(gaps)) / OCRB_ASPECT, float(np.median(changes))


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

    tilt = perspective(image, loc)
    if tilt is not None:
        aspect, change = tilt
        if aspect < MIN_ASPECT[0] or change > MAX_PITCH_CHANGE[0]:
            way = "上下" if aspect < MIN_ASPECT[0] else "左右"
            add("tilt", "拍攝角度", "fail", f"{way}方向斜拍太嚴重，容易讀錯而且不一定會被發現。請把手機放平、正對資料頁重拍。")
        elif aspect < MIN_ASPECT[1] or change > MAX_PITCH_CHANGE[1]:
            add("tilt", "拍攝角度", "warn", "有點斜拍，可能讀錯；可以的話請正對資料頁重拍。")
        else:
            add("tilt", "拍攝角度", "ok", "角度正常")

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

    bg = cv2.blur(cv2.morphologyEx(loc.crop(image, 32, pad=0.5), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)), (15, 15))
    lo, hi = np.percentile(bg, [3, 97])
    even = float(lo / max(hi, 1.0))
    if even < MIN_EVEN_LIGHT[0]:
        add("shadow", "光線均勻", "fail", "MRZ 上有明顯陰影或一半很暗，容易讀錯。請避開手或手機的影子，在光線均勻的地方重拍。")
    elif even < MIN_EVEN_LIGHT[1]:
        add("shadow", "光線均勻", "warn", "MRZ 光線不均（逆光或局部陰影），系統會嘗試補救，但可能讀錯。")
    else:
        add("shadow", "光線均勻", "ok", "光線均勻")

    if float(crop.mean()) < MIN_BRIGHTNESS:
        add("light", "亮度", "warn", "MRZ 偏暗，建議在光線充足處拍攝。")
    else:
        add("light", "亮度", "ok", "亮度足夠")

    return {"usable": all(c["status"] != "fail" for c in checks), "checks": checks, "mrz_lines": loc.lines}
