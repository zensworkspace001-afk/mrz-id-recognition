"""Tesseract 呼叫的轉接層：本機用 pytesseract（呼叫 tesseract 指令），雲端（例如 Vercel，無法安裝系統程式）
改用 tesserocr（pip 套件內含 Tesseract 引擎）。兩者是同一個引擎，辨識邏輯不需要知道用的是哪一個。

選擇順序：環境變數 IDPIPE_OCR_BACKEND=pytesseract|tesserocr；未設定時，找得到 tesseract 指令就用 pytesseract，
否則用 tesserocr。tesserocr 的語言資料取自 TESSDATA_PREFIX，未設定時用專案內的 tessdata/。

對外只提供本專案用到的兩個函式，介面與 pytesseract 相同：
image_to_string(image, config) 與 image_to_data(image, config)（回傳 dict，鍵同 pytesseract.Output.DICT）。
"""
from __future__ import annotations

import os
import re
import shutil
import threading
from pathlib import Path

import numpy as np

_BACKEND = None
_local = threading.local()
TESSDATA = Path(__file__).resolve().parent.parent / "tessdata"


def backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        forced = os.environ.get("IDPIPE_OCR_BACKEND")
        if forced:
            _BACKEND = forced
        else:
            try:
                import pytesseract  # noqa: F401
                _BACKEND = "pytesseract" if shutil.which("tesseract") else "tesserocr"
            except ImportError:
                _BACKEND = "tesserocr"
    return _BACKEND


def _parse_config(config: str):
    psm = int(m.group(1)) if (m := re.search(r"--psm\s+(\d+)", config)) else 3
    variables = dict(re.findall(r"-c\s+(\w+)=(\S+)", config))
    return psm, variables


def _api(config: str):
    """每個執行緒、每種設定各保留一個 tesserocr API（初始化要載入模型，重複建立很慢）。"""
    import tesserocr
    cache = getattr(_local, "apis", None)
    if cache is None:
        cache = _local.apis = {}
    if config not in cache:
        psm, variables = _parse_config(config)
        path = os.environ.get("TESSDATA_PREFIX") or str(TESSDATA)
        api = tesserocr.PyTessBaseAPI(path=path.rstrip("/") + "/", lang="eng", psm=psm)
        for k, v in variables.items():
            api.SetVariable(k, v)
        cache[config] = api
    return cache[config]


def _pil(image):
    from PIL import Image
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = image[:, :, ::-1]  # BGR -> RGB
        return Image.fromarray(np.ascontiguousarray(image))
    return image


def image_to_string(image, config: str = "") -> str:
    if backend() == "pytesseract":
        import pytesseract
        return pytesseract.image_to_string(image, lang="eng", config=config)
    api = _api(config)
    api.SetImage(_pil(image))
    return api.GetUTF8Text()


def image_to_data(image, config: str = "") -> dict:
    """逐字詞的文字、位置與信心（-1 表示非文字的版面元素，與 pytesseract 一致）。"""
    if backend() == "pytesseract":
        import pytesseract
        return pytesseract.image_to_data(image, lang="eng", config=config, output_type=pytesseract.Output.DICT)
    import tesserocr
    api = _api(config)
    api.SetImage(_pil(image))
    api.Recognize()
    out = {k: [] for k in ("text", "left", "top", "width", "height", "conf", "block_num", "par_num", "line_num")}
    it = api.GetIterator()
    level = tesserocr.RIL.WORD
    block = par = line = 0
    if it is not None:
        while True:
            if it.IsAtBeginningOf(tesserocr.RIL.BLOCK):
                block, par, line = block + 1, 0, 0
            if it.IsAtBeginningOf(tesserocr.RIL.PARA):
                par, line = par + 1, 0
            if it.IsAtBeginningOf(tesserocr.RIL.TEXTLINE):
                line += 1
            box = it.BoundingBox(level)
            if box is not None:
                x0, y0, x1, y1 = box
                out["text"].append(it.GetUTF8Text(level) or "")
                out["left"].append(x0); out["top"].append(y0)
                out["width"].append(x1 - x0); out["height"].append(y1 - y0)
                out["conf"].append(float(it.Confidence(level)))
                out["block_num"].append(block); out["par_num"].append(par); out["line_num"].append(line)
            if not it.Next(level):
                break
    return out
