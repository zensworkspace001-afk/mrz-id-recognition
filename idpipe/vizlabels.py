"""以欄位標籤定位讀取視覺區（不需要事先量測欄位位置）。

ICAO 9303 要求護照資料頁以英文印出欄位標籤（Surname、Date of birth、Authority…），台灣護照另有中文標籤，
日本護照另有日文標籤（「旅券番号/Passport No.」，姓與名分成兩個標籤，名是單數 Given name）。
PP-OCR（ppocr.py）連標籤一起讀出來，所以：找到標籤行 → 取正下方、左緣對齊的那一行當作值。
斜拍造成的壓縮、位置偏移都不影響，因為找的是標籤與值的相對位置，不是固定座標。

實測（台灣護照，正拍、斜拍、低解析截圖各一）：英文姓名、護照號碼、身分證字號、三個日期、性別、出生地、發照機關
都讀得到；中文姓名在斜拍時可能讀錯字（丞 → 永），且無法由 MRZ 驗證，需人工確認。
"""
from __future__ import annotations

import difflib
import re

import numpy as np

from .ppocr import TextLine, engine

# 欄位 -> (英文標籤, 中文標籤)；英文比對忽略大小寫與非字母字元
LABELS = {
    "document_number": ("passport no", "護照號碼"),
    "name": ("name", "姓名"),
    "surname": ("surname", "姓"),
    "given_names": ("given name", "名"),  # 台灣 Given names、日本 Given name（部分比對，兩者都符合）
    "personal_number": ("personal id no", "身分證統一編號"),
    "date_of_birth": ("date of birth", "出生日期"),
    "date_of_issue": ("date of issue", "發照日期"),
    "date_of_expiry": ("date of expiry", "效期截止日期"),
    "place_of_birth": ("place of birth", "出生地"),
    "authority": ("authority", "發照機關"),
    "sex": ("sex", "性別"),
    "nationality": ("nationality", "國籍"),
    "registered_domicile": ("registered domicile", "本籍"),  # 日本護照（都道府縣），日本護照沒有出生地
}
# 其他寫法：OCR 常讀成簡體字（PP-OCR 的字典簡繁混合），以及日本護照的日文標籤（含讀成繁／簡體的寫法）
ZH_ALIASES = {"sex": ("性别",), "nationality": ("国籍",),
              "document_number": ("护照号码", "旅券番号", "旅券番號"),
              "date_of_birth": ("生年月日",),
              "date_of_issue": ("発行年月日", "發行年月日", "发行年月日"),
              "date_of_expiry": ("有効期間満了日", "有效期間滿了日", "有效期间满了日"),
              "authority": ("発行官庁", "發行官廳", "发行官厅")}
LABEL_MIN_SCORE = 0.75
CJK = re.compile(r"[㐀-鿿]")


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _partial_ratio(key: str, text: str) -> float:
    """key 與 text 中最相近片段的相似度（0～1）。text 比 key 短很多時不算相似
    （左上角紅字「PASSPORT」不應被當成「Passport No.」標籤）。"""
    if not key or not text or len(text) < 0.85 * len(key):
        return 0.0
    if len(text) <= len(key):
        return difflib.SequenceMatcher(None, key, text).ratio()
    return max(difflib.SequenceMatcher(None, key, text[i:i + len(key)]).ratio()
               for i in range(len(text) - len(key) + 1))


def _word_ratio(key: str, text: str) -> float:
    """單字標籤：和 text 中每個字比對整個字的相似度（name 不應匹配到 named）。"""
    words = re.findall(r"[a-z]+", text.lower())
    return max((difflib.SequenceMatcher(None, key, w).ratio() for w in words), default=0.0)


def label_of(line: TextLine):
    """這一行是哪個欄位的標籤；不是標籤回傳 (None, 0)。
    英文：多字標籤以相似度比對（容許 OCR 錯字），單字標籤比對整個字；標籤行最多 6 個英文字
    （避免上一頁「NOTIFY THE INDIVIDUAL NAMED BELOW」這類句子被當成標籤）。
    中文：標籤要完整出現（2 字以上），避免「姓」「名」單字誤判。
    同時含 Surname 與 Given names 的是合併的姓名標籤（台灣護照：姓名/Name (Surname, Given names)）。"""
    en = _letters(line.text)
    n_words = len(re.findall(r"[A-Za-z]+", line.text))
    if n_words > 6:
        return None, 0.0
    if _partial_ratio("surname", en) >= LABEL_MIN_SCORE and _partial_ratio("givennames", en) >= LABEL_MIN_SCORE:
        return "name", 1.0
    scores = {}
    for field, (key_en, key_zh) in LABELS.items():
        k = _letters(key_en)
        score = _partial_ratio(k, en) if " " in key_en else _word_ratio(k, line.text)
        if any(len(z) >= 2 and z in line.text for z in (key_zh, *ZH_ALIASES.get(field, ()))):
            score = max(score, 0.95)
        scores[field] = score
    best = max(scores.items(), key=lambda kv: kv[1])
    # 「名/Given name」也整字符合單字標籤 name：多字標籤符合時以多字標籤為準
    if best[0] == "name" and scores["given_names"] >= LABEL_MIN_SCORE:
        best = ("given_names", scores["given_names"])
    # 「Name (Surname, Given names)」讀壞時可能只認得出其中一部分；有括號的就是合併的姓名標籤
    if best[0] in ("surname", "given_names") and "(" in line.text:
        best = ("name", best[1])
    return best if best[1] >= LABEL_MIN_SCORE else (None, 0.0)


def _same_column(a: TextLine, b: TextLine, page_w: float) -> bool:
    return abs(a.x0 - b.x0) < 0.06 * page_w or (a.x0 < b.x1 and a.x1 > b.x0)


def _value_below(label: TextLine, lines: list, label_lines: list, page_w: float, h: float):
    """標籤正下方、同一欄的第一行，且要在同一欄的下一個標籤之上（值沒被偵測到時，不會抓到下一個欄位的值）。
    同一列上緊接的片段（間距不到 4 倍字高，或起點仍在標籤範圍內）一併收進來，例如中文姓名與英文姓名分開偵測時。"""
    # h：一般文字行高（中位數）。不用標籤自己的高度：斜拍時標籤框可能和相鄰文字黏在一起、高出一倍
    below = [l.yc for l in label_lines if l is not label and l.yc > label.yc + 0.4 * h and _same_column(l, label, page_w)]
    limit = min(below, default=label.yc + 4 * h)
    label_ids = {id(l) for l in label_lines}
    cands = [l for l in lines if id(l) not in label_ids and l.yc > label.yc and l.y1 > label.y1
             and l.yc < min(limit, label.yc + 4 * h) and _same_column(l, label, page_w)]
    if not cands:
        return []
    row = [min(cands, key=lambda l: l.yc)]
    for l in sorted((l for l in lines if id(l) not in label_ids), key=lambda l: l.x0):
        if l is row[-1] or l in row or abs(l.yc - row[0].yc) >= 0.6 * h:
            continue
        # 間距可略為負：同一個值被切成兩段時，兩個框常重疊一兩個像素（「15」「APR 1990」）
        if -0.5 * h <= l.x0 - row[-1].x1 < 4 * h or (l.x0 > row[-1].x0 and l.x0 < label.x1):
            row.append(l)
    return row


def _field(text, lines):
    x0 = min(l.x0 for l in lines); y0 = min(l.y0 for l in lines)
    x1 = max(l.x1 for l in lines); y1 = max(l.y1 for l in lines)
    return {"raw": text, "confidence": round(min(l.score for l in lines), 3),
            "bbox": [int(x0), int(y0), int(x1), int(y1)]}


def _plausible(field: str, text: str, formats: dict | None) -> bool:
    from .normalize import parse_date
    t = text.strip()
    if field == "sex":
        return t.upper() in ("M", "F", "X")
    if field.startswith("date_"):
        return parse_date(normalize_date(t)) is not None
    if formats and field in formats:
        return re.fullmatch(formats[field], t) is not None
    return any(ch.isalnum() for ch in t)


def normalize_date(text: str) -> str:
    """「01SEP2014」「01 SEP2014」-> 「01 SEP 2014」（數字與月份之間補空格，常見 OCR 結果）。"""
    t = re.sub(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)", " ", text.strip())
    return re.sub(r"\s+", " ", t)


def split_name(text: str):
    """台灣護照「江丞恩CHIANG，CHENG-EN」-> (中文姓名, 英文姓, 英文名)。沒有逗號時整段當作姓。"""
    t = text.replace("，", ",").replace("、", ",")
    native = "".join(CJK.findall(t)) or None
    latin = re.sub(r"[^A-Za-z,' -]", " ", CJK.sub(" ", t)).upper()
    surname, _, given = latin.partition(",")
    return native, re.sub(r"\s+", " ", surname).strip(), re.sub(r"\s+", " ", given).strip()


def extract(rectified, formats: dict | None = None, mrz_top: float | None = None) -> dict:
    """讀取拉正後頁面的視覺區。回傳 {欄位: {"raw", "confidence", "bbox"}}，格式同 viz.make_viz_extractor。
    formats：{欄位: 正規表示式}，標籤沒認出來時，改找視覺區中唯一完全符合此格式的一行（例如台灣身分證字號）。
    mrz_top：MRZ 最上緣的 y（之下的文字不當作視覺區）；未提供時排除含 '<' 的行。
    有標籤卻沒讀到值時（彩色浮水印、底紋蓋住值，偵測不到），改用去色的影像再讀一次，只補缺的欄位；
    第一次讀到的欄位維持原樣。"""
    out, missing = _extract(rectified, formats, mrz_top)
    if missing and rectified.ndim == 3:
        gray = rectified.max(axis=2)  # 每個像素取最亮的色版：紅色浮水印、彩色底紋變淺，黑字不變
        more, _ = _extract(np.repeat(gray[:, :, None], 3, axis=2), formats, mrz_top)
        for field, v in more.items():
            out.setdefault(field, v)
    return out


def _extract(rectified, formats, mrz_top):
    """回傳 (欄位結果, 有標籤但沒讀到值的欄位集合)。"""
    page_w = rectified.shape[1]
    lines = [l for l in engine()(rectified)
             if "<" not in l.text and (mrz_top is None or l.y1 < mrz_top)]
    tagged = [(l, *label_of(l)) for l in lines]
    label_lines = [l for l, f, _ in tagged if f]
    h = max(8.0, float(np.median([l.y1 - l.y0 for l in lines]))) if lines else 20.0
    out, seen = {}, set()
    for label, field, score in sorted((t for t in tagged if t[1]), key=lambda t: -t[2]):
        seen.add(field)
        if field in out or (field == "name" and "surname" in out):
            continue
        row = _value_below(label, lines, label_lines, page_w, h)
        if not row:
            continue
        text = " ".join(l.text for l in row)
        if field == "name":
            native, surname, given = split_name(text)
            if native:
                out["name_native"] = _field(native, [l for l in row if CJK.search(l.text)])
            latin_rows = [l for l in row if re.search(r"[A-Za-z]", l.text)] or row
            if surname:
                out["surname"] = _field(surname, latin_rows)
            if given:
                out["given_names"] = _field(given, latin_rows)
            continue
        if field in ("document_number", "personal_number"):
            text = re.sub(r"\s+", "", text)
        if not _plausible(field, text, formats):
            continue  # 抓到的不像這個欄位（例如值沒被偵測到、抓到別的東西）：當作沒讀到，不拿去比對
        out[field] = _field(text, row)
    for field, pattern in (formats or {}).items():
        if field in out:
            continue
        hits = [l for l in lines if re.fullmatch(pattern, re.sub(r"\s+", "", l.text))]
        if len(hits) == 1:
            out[field] = _field(re.sub(r"\s+", "", hits[0].text), hits)
    for field in list(out):
        if field.startswith("date_"):
            out[field]["raw"] = normalize_date(out[field]["raw"])
    names = {"surname", "given_names"}
    missing = {f for f in seen if f not in out and not (f == "name" and names & out.keys())}
    return out, missing
