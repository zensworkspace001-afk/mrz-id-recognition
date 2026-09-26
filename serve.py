"""本機驗證網站：python serve.py [--port 8000]，再用瀏覽器開 http://127.0.0.1:8000

- 照片檢查：辨識前約 1 秒先檢查照片能不能用（找得到 MRZ、清晰度、反光、解析度），不能用就說明怎麼重拍。
- 照片辨識：可一次上傳多張，依序處理，結果先放在暫存區；每張顯示拉正影像、MRZ 各檢查碼、欄位與複核原因。
- 全部處理完後，勾選要入庫的筆數匯出 SQL（PostgreSQL／MySQL／SQL Server／SQLite；沒有資料表會自動建立）或 CSV。
- 手動輸入 MRZ：直接貼上 MRZ 文字驗證檢查碼，並顯示每個檢查碼「應為」多少。

只監聽 127.0.0.1；上傳的照片只在記憶體中處理，不寫入磁碟、不存入 out/。
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from idpipe.mrz import check_digit, parse_mrz
from idpipe.ocr import ocr_mrz_region
from idpipe.pipeline import Components, Pipeline
from idpipe.quality import check_photo
from idpipe.sqlexport import DIALECTS, row_of, to_sql
from idpipe.templates import SYNTH_LAYOUT
from idpipe.viz import default_extractor, make_viz_extractor
from run import rectify_details

INDEX = Path(__file__).with_name("web") / "index.html"
SAMPLES = Path(__file__).with_name("web") / "samples"  # 合成範例圖（不含真實個資），公開網站一鍵試用
MAX_UPLOAD = 25 * 1024 * 1024
FIELD_LABEL = {"document_number": "證件號碼", "date_of_birth": "出生日期", "date_of_expiry": "到期日",
               "personal_number": "個人號碼", "composite": "總檢查碼"}


def check_digit_detail(lines) -> list:
    """每個檢查碼：所在行與位置、印的值、依資料算出的應有值、是否通過。"""
    def item(name, data, line, pos):
        printed = lines[line][pos]
        try:
            expected = check_digit(data)
        except ValueError:
            expected = "?"
        if printed == "<" and set(data) <= {"<"}:
            expected = "<"
        return {"field": name, "label": FIELD_LABEL[name], "line": line, "pos": pos,
                "printed": printed, "expected": expected, "ok": printed == expected}

    if len(lines) == 3:
        l1, l2, _ = lines
        doc_data, doc_pos = l1[5:14], 14
        if l1[14] == "<" and l1[15] != "<":  # 證件號碼超過 9 碼
            ext = l1[15:].split("<", 1)[0]
            doc_data, doc_pos = l1[5:14] + ext[:-1], 15 + len(ext) - 1
        return [item("document_number", doc_data, 0, doc_pos),
                item("date_of_birth", l2[0:6], 1, 6),
                item("date_of_expiry", l2[8:14], 1, 14),
                item("composite", l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29], 1, 29)]
    _, l2 = lines
    return [item("document_number", l2[0:9], 1, 9),
            item("date_of_birth", l2[13:19], 1, 19),
            item("date_of_expiry", l2[21:27], 1, 27),
            item("personal_number", l2[28:42], 1, 42),
            item("composite", l2[0:10] + l2[13:20] + l2[21:43], 1, 43)]


def mrz_summary(m) -> dict:
    return {"format": m.format, "lines": list(m.lines), "checksum_valid": m.checksum_valid,
            "checks": check_digit_detail(m.lines),
            "parsed": {"證件類別": m.doc_code, "簽發國": m.issuing_country, "證件號碼": m.document_number,
                       "姓": m.surname, "名": m.given_names, "國籍": m.nationality, "性別": m.sex,
                       "出生日期": m.date_of_birth, "到期日": m.date_of_expiry,
                       "個人號碼 / 選用資料": m.personal_number or " | ".join(o.strip("<") for o in m.optional_data)}}


def _jpeg(img, max_w=960) -> str:
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


BAD_IMAGE = "無法讀取圖片。請上傳 JPG、PNG 或 WebP（iPhone 的 HEIC 請先轉成 JPG）。"


def check(data: bytes) -> dict:
    """照片品質快速檢查，附上標出 MRZ 位置的預覽圖。"""
    image = _decode(data)
    if image is None:
        return {"error": BAD_IMAGE}
    t0 = time.time()
    q = check_photo(image)
    overlay = image.copy()
    th = max(3, image.shape[1] // 300)
    for line in q.pop("mrz_lines"):
        cv2.polylines(overlay, [np.array(line, np.int32)], True, (40, 110, 230), th)
    return {**q, "elapsed": round(time.time() - t0, 1), "size": [image.shape[1], image.shape[0]],
            "preview": _jpeg(overlay)}


def recognize(data: bytes, synth_layout: bool) -> dict:
    image = _decode(data)
    if image is None:
        return {"error": BAD_IMAGE}
    t0 = time.time()
    info, ocr_texts = {}, []

    def rect(img):
        flat, corners, rotated, method = rectify_details(img)
        info.update(flat=flat, corners=corners, rotated=rotated, method=method)
        return flat

    def ocr(img):
        text = ocr_mrz_region(img)
        ocr_texts.append(text)
        return text

    comps = Components(rectify=rect, ocr_text=ocr, viz_extract=default_extractor)
    if synth_layout:
        comps.viz_extract = make_viz_extractor(SYNTH_LAYOUT)
    rec = Pipeline(comps).process(image, data, source="upload")

    overlay = image.copy()
    if info.get("corners"):
        pts = np.array(info["corners"], np.int32)
        th = max(3, image.shape[1] // 300)
        cv2.polylines(overlay, [pts], True, (60, 200, 60), th)
        for p in pts:
            cv2.circle(overlay, tuple(int(v) for v in p), th * 3, (60, 200, 60), -1)
    mrz = None
    if rec.mrz.get("line1"):
        lines = [rec.mrz["line1"], rec.mrz["line2"]] + ([rec.mrz["line3"]] if rec.mrz.get("line3") else [])
        mrz = mrz_summary(parse_mrz(lines))
    d = rec.to_dict()
    return {"elapsed": round(time.time() - t0, 1), "size": [image.shape[1], image.shape[0]],
            "corners_found": bool(info.get("corners")), "rotated": info.get("rotated", False),
            "method": info.get("method"),
            "original": _jpeg(overlay), "rectified": _jpeg(info["flat"]) if "flat" in info else None,
            "ocr_text": ocr_texts[0] if ocr_texts else "", "mrz": mrz, "record": d}


def export_sql(payload: dict) -> dict:
    """暫存區勾選的結果打包成 SQL。payload：{dialect, table, items: [{record, source_file, reviewed}]}。"""
    dialect, table = payload.get("dialect", "postgresql"), payload.get("table", "id_documents")
    if dialect not in DIALECTS:
        return {"error": f"不支援的資料庫：{dialect}"}
    rows = [row_of(it["record"], it.get("source_file", ""), it.get("reviewed", False)) for it in payload.get("items", [])]
    if not rows:
        return {"error": "沒有勾選任何一筆"}
    try:
        return {"sql": to_sql(rows, dialect, table), "count": len(rows)}
    except ValueError as e:
        return {"error": str(e)}


def verify_text(text: str) -> dict:
    lines = [l.replace(" ", "").replace("\t", "").upper() for l in text.splitlines()]
    lines = [l for l in lines if l]
    lens = [len(l) for l in lines]
    if not ((len(lines) == 2 and lens == [44, 44]) or (len(lines) == 3 and lens == [30, 30, 30])):
        return {"error": f"MRZ 應為護照 2 行 × 44 字元，或卡片 3 行 × 30 字元；目前是 {len(lines)} 行，長度 {lens}。",
                "lengths": lens}
    bad = sorted({c for l in lines for c in l if not (c.isdigit() or "A" <= c <= "Z" or c == "<")})
    if bad:
        return {"error": f"MRZ 只允許 A-Z、0-9 與 <，出現了：{' '.join(bad)}"}
    return {"mrz": mrz_summary(parse_mrz(lines))}


def route(raw_path: str):
    """回傳 (路徑, 查詢參數)。Vercel 把所有請求轉到 api/index.py 時，handler 看到的是改寫後的路徑，
    原始路徑放在查詢參數 __route（見 vercel.json）；本機直接執行時沒有這個參數。"""
    url = urlparse(raw_path)
    query = parse_qs(url.query)
    path = "/" + query.pop("__route")[0].lstrip("/") if "__route" in query else url.path
    return path, query


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path, _ = route(self.path)
        if path.startswith("/samples/"):
            name = path.rsplit("/", 1)[-1]
            f = SAMPLES / name
            if name.endswith(".jpg") and f.parent == SAMPLES and f.is_file():  # 只提供 samples/ 內的檔案
                return self._send(200, f.read_bytes(), "image/jpeg")
            return self._send(404, b"not found", "text/plain")
        if path == "/api/samples":
            return self._json(sorted(p.name for p in SAMPLES.glob("*.jpg")))
        self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        path, query = route(self.path)
        endpoint = path.rstrip("/").rsplit("/", 1)[-1]  # /api/check -> check
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            return self._json({"error": "檔案超過 25 MB"}, 413)
        body = self.rfile.read(n)
        try:
            if endpoint == "recognize":
                synth = query.get("synth", ["0"])[0] == "1"
                return self._json(recognize(body, synth))
            if endpoint == "check":
                return self._json(check(body))
            if endpoint == "sql":
                return self._json(export_sql(json.loads(body.decode("utf-8"))))
            if endpoint == "mrz":
                return self._json(verify_text(body.decode("utf-8", "replace")))
            self._json({"error": "not found"}, 404)
        except Exception as e:  # 本機除錯工具：把錯誤顯示在頁面上
            self._json({"error": f"處理失敗：{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):  # 不記錄請求內容
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"開啟 http://127.0.0.1:{a.port}  （Ctrl+C 結束）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
