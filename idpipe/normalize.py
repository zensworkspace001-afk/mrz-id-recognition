"""日期、姓名、國碼的正規化與比對。"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Optional

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def strip_diacritics(s: str) -> str:
    """去除變音符號；Đ/đ 轉成 D/d，Ñ 轉成 N（MRZ 規則）。"""
    s = s.replace("Đ", "D").replace("đ", "d")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def name_key(s: Optional[str]) -> str:
    """姓名比對用：去變音符號、大寫、只留字母。"""
    if not s:
        return ""
    return re.sub(r"[^A-Z]", "", strip_diacritics(s).upper())


def names_match(a: Optional[str], b: Optional[str], allow_prefix: bool = True) -> bool:
    """allow_prefix：MRZ 姓名欄過長會被截斷，允許前綴相符。兩邊都沒有名字（單名）視為相符。"""
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return ka == kb
    return ka == kb or (allow_prefix and (ka.startswith(kb) or kb.startswith(ka)))


def _valid(y: int, m: int, d: int) -> Optional[str]:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def mrz_date_to_iso(yymmdd: str, kind: str, today: Optional[date] = None) -> Optional[str]:
    """MRZ 兩位數年份轉 ISO。kind='birth'：不可晚於今天；kind='expiry'：預設 2000 年代。"""
    if not re.fullmatch(r"\d{6}", yymmdd or ""):
        return None
    today = today or date.today()
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    if kind == "birth":
        year = 2000 + yy
        if year > today.year:
            year -= 100
    else:
        year = 2000 + yy
    return _valid(year, mm, dd)


def parse_date(text: Optional[str]) -> Optional[str]:
    """視覺區日期 -> YYYY-MM-DD。支援 '31 JAN 1990'、'31 JAN/JAN 1990'、
    '31/01/1990'（日/月/年，印度等）、'1990-01-31'、'1990.01.31'、'1990年1月31日'。"""
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?", t)
    if m:
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"\b(\d{4})[-./](\d{1,2})[-./](\d{1,2})\b", t)
    if m:
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"\b(\d{1,2})[-./](\d{1,2})[-./](\d{4})\b", t)
    if m:
        return _valid(int(m[3]), int(m[2]), int(m[1]))  # 日/月/年
    m = re.search(r"\b(\d{1,2})\s*([A-Z]{3})[A-Z/ ]*?\s*(\d{4})\b", t)
    if m and m[2] in MONTHS:
        return _valid(int(m[3]), MONTHS[m[2]], int(m[1]))
    return None


# MRZ 用的國碼與 ISO 3166-1 alpha-3 大致相同，但少數為特例
COUNTRY_NAMES = {
    "JPN": "Japan", "CHN": "China", "HKG": "Hong Kong SAR", "KOR": "Korea",
    "VNM": "Viet Nam", "IDN": "Indonesia", "PHL": "Philippines", "IND": "India",
    "TWN": "Taiwan",
}


# 國民身分證統一編號／新式外來人口統一證號（1 英文 + 9 數字）的英文字母代碼
_TW_LETTER = {c: v for c, v in zip("ABCDEFGHJKLMNPQRSTUVXYWZIO", range(10, 36))}


def taiwan_id_valid(s: Optional[str]) -> bool:
    """台灣國民身分證統一編號與新式外來人口統一證號（第 2 碼 8/9）的檢查碼。
    字母轉成兩位數後，權重 1,9,8,7,6,5,4,3,2,1,1 加總須為 10 的倍數。舊式居留證號（2 英文 + 8 數字）不適用。"""
    if not s or not re.fullmatch(r"[A-Z][1289]\d{8}", s):
        return False
    v = _TW_LETTER[s[0]]
    digits = [v // 10, v % 10] + [int(c) for c in s[1:]]
    return sum(d * w for d, w in zip(digits, (1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1))) % 10 == 0
