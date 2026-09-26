"""Vercel 的 Python 函式入口：沿用 serve.py 的處理邏輯（照片檢查、辨識、SQL 匯出、MRZ 驗證與網頁本身）。

Vercel 無法安裝 tesseract 指令，所以強制使用 tesserocr（pip 套件內含 Tesseract 引擎），
語言資料用專案內的 tessdata/。所有路徑都轉到這個函式（見 vercel.json）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("IDPIPE_OCR_BACKEND", "tesserocr")
os.environ.setdefault("TESSDATA_PREFIX", os.path.join(ROOT, "tessdata"))
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# tesserocr 依賴的 cysignals 在 import 時註冊訊號處理程式，只能在主執行緒執行；
# Vercel 在工作執行緒處理請求，所以要在模組載入時（主執行緒）先 import，不能等第一次辨識時才載入。
import tesserocr  # noqa: E402,F401

from serve import Handler  # noqa: E402


class handler(Handler):
    pass
