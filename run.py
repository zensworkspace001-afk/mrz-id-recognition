"""批次辨識：python run.py <照片資料夾> [--out out]

流程：偵測四角點 -> 拉正 -> OCR 讀 MRZ -> 解析驗證 -> 寫入 JSONL 與 CSV。
各版型 region 尚未用官方樣本標定，所以預設只有 MRZ 內的欄位有值。
--synth-layout 改用合成資料的通用版面抽取視覺區（只適用 synth/ 產生的圖）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from idpipe.detect import detect_corners
from idpipe.locate import locate_mrz
from idpipe.mrz import find_mrz_lines, parse_mrz
from idpipe.ocr import ocr_mrz_region
from idpipe.pipeline import Components, Pipeline
from idpipe.rectify import rectify
from idpipe.storage import JsonlStorage, export_csv
from idpipe.templates import SYNTH_LAYOUT
from idpipe.viz import make_viz_extractor

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _mrz_score(img) -> int:
    """快速看拉正後的影像能讀出幾個 MRZ 檢查碼（只取第一個全過的結果，不做姓名投票）。
    全部通過回傳 99；讀不到 MRZ 回傳 -1。"""
    lines = find_mrz_lines(ocr_mrz_region(img, min_reads=1))
    if not lines:
        return -1
    try:
        r = parse_mrz(lines)
    except ValueError:
        return -1
    return 99 if r.checksum_valid else sum(r.checks.values())


def _mrz_horizontal_in(corners, loc) -> bool:
    """以四角點拉正後，MRZ 是否大致水平（否則四角點的方向錯了，例如直式照片被拉成橫式）。"""
    src = np.array(corners, np.float32)
    dst = np.array([[0, 0], [1250, 0], [1250, 880], [0, 880]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    a, b = loc.left, loc.left + loc.u * loc.width
    p = cv2.perspectiveTransform(np.array([[a, b]], np.float32), H)[0]
    ang = abs(np.degrees(np.arctan2(p[1][1] - p[0][1], p[1][0] - p[0][0])))
    return ang < 30 or ang > 150


def rectify_details(image):
    """拉正證件。回傳 (拉正影像, 頁面四角或 None, 是否轉了 180 度, 方法)。
    候選：四角點偵測（detect.py）的 0/180 度，以及直接找 MRZ（locate.py）推算出的資料頁正反兩個方向。
    依序拉正，第一個讀得出檢查碼全過 MRZ 的採用；都沒有全過時，取通過最多檢查碼的（至少 2 個）。
    MRZ 在四角點拉正後是直的，表示四角點方向錯了（例如直式照片），跳過四角點候選。
    都不行：用整張圖（找得到四角時用四角）。"""
    corners = detect_corners(image)
    loc = locate_mrz(image)
    cands = []
    if corners and (loc is None or _mrz_horizontal_in(corners, loc)):
        cands += [(corners, False, "corners"), (corners, True, "corners")]
    if loc is not None:
        cands += [(loc.page_quad(False), False, "mrz"), (loc.page_quad(True), False, "mrz")]
    best = None
    for quad, rot180, method in cands:
        flat = rectify(image, quad)
        if rot180:
            flat = cv2.rotate(flat, cv2.ROTATE_180)
        score = _mrz_score(flat)
        if score == 99:
            return flat, quad, rot180, method
        if best is None or score > best[0]:
            best = (score, flat, quad, rot180, method)
    if best is not None and best[0] >= 2:
        return best[1:]
    if corners:
        return rectify(image, corners), corners, False, "corners"
    return cv2.resize(image, (1250, 880)), None, False, "whole"


def auto_rectify(image):
    return rectify_details(image)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="out")
    ap.add_argument("--synth-layout", action="store_true", help="用合成資料的通用版面抽取視覺區欄位")
    a = ap.parse_args()

    out = Path(a.out)
    store = JsonlStorage(out / "records.jsonl")
    comps = Components(rectify=auto_rectify, ocr_text=ocr_mrz_region)
    if a.synth_layout:
        comps.viz_extract = make_viz_extractor(SYNTH_LAYOUT)
    pipe = Pipeline(comps, store)

    files = sorted(p for p in Path(a.folder).rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print("資料夾內沒有圖片")
        return
    ok = 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"讀不到圖片: {p}")
            continue
        rec = pipe.process(img, p.read_bytes(), source=str(p))
        ok += not rec.needs_review
        flag = "OK    " if not rec.needs_review else "REVIEW"
        print(f"{flag} {p.name}  {rec.template_key}  {rec.surname or ''} {rec.given_names or ''}  "
              f"{'; '.join(rec.review_reasons)}")
    n = export_csv(store.all(), out / "results.csv")
    print(f"\n完成：{len(files)} 張，{ok} 張免複核；輸出 {out}/records.jsonl 與 {out}/results.csv（{n} 筆）")


if __name__ == "__main__":
    main()
