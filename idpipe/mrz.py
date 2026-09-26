"""ICAO 9303 MRZ 解析、檢查碼、產生：TD3（護照，2 x 44）與 TD1（卡片型證件如居留證，3 x 30）。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .normalize import mrz_date_to_iso

WEIGHTS = (7, 3, 1)


def char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - 55
    if c == "<":
        return 0
    raise ValueError(f"MRZ 不允許的字元: {c!r}")


def check_digit(s: str) -> str:
    total = sum(char_value(c) * WEIGHTS[i % 3] for i, c in enumerate(s))
    return str(total % 10)


def _cd_ok(field: str, cd: str) -> bool:
    if cd == "<" and set(field) <= {"<"}:
        return True
    try:
        return check_digit(field) == cd
    except ValueError:
        return False


def _clean(s: str) -> str:
    return s.replace("<", " ").strip()


def parse_names(field: str) -> tuple[str, str]:
    """姓 << 名（名字間以單個 < 分隔）。單名時 given 為空字串。"""
    parts = field.split("<<", 1)
    surname = _clean(parts[0])
    given = _clean(parts[1]) if len(parts) > 1 else ""
    return surname, given


@dataclass
class MrzResult:
    line1: str
    line2: str
    doc_code: str
    issuing_country: str
    surname: str
    given_names: str
    document_number: str
    nationality: str
    date_of_birth: Optional[str]
    sex: str
    date_of_expiry: Optional[str]
    personal_number: str
    checks: dict  # 各欄位檢查碼是否通過
    checksum_valid: bool
    line3: str = ""
    format: str = "TD3"  # TD3 | TD1
    optional_data: tuple = ()  # TD1 兩段選用資料欄（原樣保留，含 <）

    @property
    def lines(self) -> tuple:
        return (self.line1, self.line2, self.line3) if self.format == "TD1" else (self.line1, self.line2)

    @property
    def name_field(self) -> str:
        """姓名欄（沒有檢查碼）：TD3 在第一行 5 之後，TD1 是整個第三行。"""
        return self.line3 if self.format == "TD1" else self.line1[5:44]


def parse_td3(line1: str, line2: str) -> MrzResult:
    line1, line2 = line1.strip().upper(), line2.strip().upper()
    if len(line1) != 44 or len(line2) != 44:
        raise ValueError(f"TD3 每行必須為 44 字元，得到 {len(line1)} 與 {len(line2)}")
    surname, given = parse_names(line1[5:44])
    docnum, cd_doc = line2[0:9], line2[9]
    dob, cd_dob = line2[13:19], line2[19]
    exp, cd_exp = line2[21:27], line2[27]
    pers, cd_pers = line2[28:42], line2[42]
    composite = line2[0:10] + line2[13:20] + line2[21:43]
    checks = {
        "document_number": _cd_ok(docnum, cd_doc),
        "date_of_birth": _cd_ok(dob, cd_dob),
        "date_of_expiry": _cd_ok(exp, cd_exp),
        "personal_number": _cd_ok(pers, cd_pers),
        "composite": _cd_ok(composite, line2[43]),
    }
    return MrzResult(
        line1=line1, line2=line2,
        doc_code=line1[0:2].replace("<", ""),
        issuing_country=line1[2:5].replace("<", ""),
        surname=surname, given_names=given,
        document_number=docnum.replace("<", ""),
        nationality=line2[10:13].replace("<", ""),
        date_of_birth=mrz_date_to_iso(dob, "birth"),
        sex={"M": "M", "F": "F"}.get(line2[20], "X"),
        date_of_expiry=mrz_date_to_iso(exp, "expiry"),
        personal_number=pers.replace("<", ""),
        checks=checks,
        checksum_valid=all(checks.values()),
    )


def parse_td1(line1: str, line2: str, line3: str) -> MrzResult:
    """TD1：第一行 證件類別(2) 簽發國(3) 證件號碼(9) 檢查碼(1) 選用資料(15)；
    第二行 生日(6)+檢查碼 性別 到期日(6)+檢查碼 國籍(3) 選用資料(11) 總檢查碼；第三行 姓名。
    證件號碼超過 9 碼時，檢查碼位置填 <，其餘號碼與檢查碼接在第一行選用資料開頭（ICAO 9303 Part 5）。"""
    line1, line2, line3 = (l.strip().upper() for l in (line1, line2, line3))
    if not (len(line1) == len(line2) == len(line3) == 30):
        raise ValueError(f"TD1 每行必須為 30 字元，得到 {len(line1)}、{len(line2)}、{len(line3)}")
    docnum, cd_doc, opt1 = line1[5:14], line1[14], line1[15:30]
    if cd_doc == "<" and opt1[0] != "<":
        ext = opt1.split("<", 1)[0]
        docnum, cd_doc, opt1 = docnum + ext[:-1], ext[-1], opt1[len(ext):]
    dob, cd_dob = line2[0:6], line2[6]
    exp, cd_exp = line2[8:14], line2[14]
    composite = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    checks = {
        "document_number": _cd_ok(docnum, cd_doc),
        "date_of_birth": _cd_ok(dob, cd_dob),
        "date_of_expiry": _cd_ok(exp, cd_exp),
        "composite": _cd_ok(composite, line2[29]),
    }
    surname, given = parse_names(line3)
    return MrzResult(
        line1=line1, line2=line2, line3=line3, format="TD1",
        doc_code=line1[0:2].replace("<", ""),
        issuing_country=line1[2:5].replace("<", ""),
        surname=surname, given_names=given,
        document_number=docnum.replace("<", ""),
        nationality=line2[15:18].replace("<", ""),
        date_of_birth=mrz_date_to_iso(dob, "birth"),
        sex={"M": "M", "F": "F"}.get(line2[7], "X"),
        date_of_expiry=mrz_date_to_iso(exp, "expiry"),
        personal_number="",
        optional_data=(opt1, line2[18:29]),
        checks=checks,
        checksum_valid=all(checks.values()),
    )


def parse_mrz(lines) -> MrzResult:
    """依行數分派：3 行為 TD1，2 行為 TD3。"""
    return parse_td1(*lines) if len(lines) == 3 else parse_td3(*lines)


def name_line_index(lines) -> int:
    """沒有檢查碼的姓名行在第幾行（TD3 第一行、TD1 第三行）。"""
    return 2 if len(lines) == 3 else 0


# ---- 從 OCR 文字找 MRZ ----
_TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"})
_TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"})


def _fix_line2(l2: str) -> str:
    """依欄位位置把常見 OCR 混淆的字元（O/0、I/1 等）修正回去。"""
    c = list(l2)
    numeric = list(range(9, 10)) + list(range(13, 20)) + list(range(21, 28)) + [42, 43]
    for i in numeric:
        c[i] = c[i].translate(_TO_DIGIT)
    for i in list(range(10, 13)):  # 國籍為字母
        c[i] = c[i].translate(_TO_ALPHA)
    return "".join(c)


def _rebuild_line2(b: str) -> str:
    """OCR 常漏掉或多讀連續的 '<'，使第二行長度不是 44。
    前 28 字元（號碼、國籍、生日、性別、到期日及其檢查碼）與最後 1 字元（總檢查碼）位置固定，
    中間的個人號碼欄重新補齊到 15 字元；能讓檢查碼通過的候選優先。"""
    if len(b) == 44:
        return b
    if len(b) < 29:
        return b.ljust(44, "<")
    head, tail, middle = b[:28], b[-1], b[28:-1]
    m = middle.strip("<")
    cands = []
    if not m:
        cands.append(head + "<" * 15 + tail)
    else:
        cands.append(head + m[:-1].ljust(14, "<") + m[-1] + tail)  # 個人號碼 + 其檢查碼
        cands.append(head + m.ljust(14, "<")[:14] + "<" + tail)
    for c in cands:
        try:
            if parse_td3("P<XXX" + "<" * 39, repair_document_number(_fix_line2(c))).checksum_valid:
                return c
        except ValueError:
            pass
    return cands[0]


def find_mrz_lines(ocr_text: str) -> Optional[tuple]:
    """從一段 OCR 文字中找出 MRZ（容許 '<' 數量被 OCR 弄錯）。
    TD3 回傳 2 行、TD1 回傳 3 行；兩種都像時以 TD3 優先。"""
    cands = []
    for raw in ocr_text.splitlines():
        s = re.sub(r"\s+", "", raw.upper()).replace("«", "<").replace("‹", "<")
        if len(s) >= 8 and re.fullmatch(r"[A-Z0-9<]+", s):
            cands.append(s)
    td3 = [s for s in cands if len(s) >= 12]  # 短行只可能是 TD1 的姓名行（例如 LIN<MAY<LEE 漏讀尾端 <）
    for i in range(len(td3) - 1):
        a, b = td3[i], td3[i + 1]
        # 第二個字元的 '<' 常被讀成 S/K/E/C（實拍護照：P<TWN… 讀成 PSTWN…），只要第 3～5 字是國家碼字母就接受
        first_ok = "<" in a[:6] or (len(a) >= 5 and a[1] in "SKEC" and a[2:5].isalpha())
        if a[0] == "P" and first_ok and len(b) >= 29 and any(ch.isdigit() for ch in b):
            if a[1] in "SKEC" and "<" not in a[:6]:
                a = a[0] + "<" + a[2:]
            a = a.ljust(44, "<")[:44]
            a = a[:5].translate(_TO_ALPHA) + a[5:].translate(_TO_ALPHA)  # 第一行只有字母，數字必為 OCR 誤讀（檢查碼不涵蓋姓名）
            b = repair_document_number(_fix_line2(_rebuild_line2(b)[:44].ljust(44, "<")))
            return _reconcile_country(a, b)
    for i in range(len(cands) - 2):
        a, b, c = cands[i], cands[i + 1], cands[i + 2]
        if a[0] in "IAC" and 24 <= len(a) <= 36 and 24 <= len(b) <= 36 and 10 <= len(c) <= 36 \
                and sum(ch.isdigit() for ch in b[:15]) >= 8:
            return _fix_td1(a, b, c)
    return None


def _fit(s: str, fixed: int, total: int) -> str:
    """前 fixed 字元位置固定，其後的選用資料欄去掉尾端 < 再補齊到 total 長度（OCR 常漏讀或多讀 <）。"""
    return (s[:fixed] + s[fixed:].rstrip("<")).ljust(total, "<")[:total]


def _fix_td1(a: str, b: str, c: str) -> tuple[str, str, str]:
    a = _fit(a, 15, 30)
    a = a[:5].translate(_TO_ALPHA) + a[5:]
    # 第二行：最後一字為總檢查碼，中間選用資料欄 11 字元
    head, tail = b[:18], b[-1] if len(b) > 18 else "<"
    b = head.ljust(18, "<") + b[18:-1].rstrip("<").ljust(11, "<")[:11] + tail
    cb = list(b)
    for i in list(range(0, 7)) + list(range(8, 15)) + [29]:
        cb[i] = cb[i].translate(_TO_DIGIT)
    for i in range(15, 18):
        cb[i] = cb[i].translate(_TO_ALPHA)
    b = "".join(cb)
    c = c.translate(_TO_ALPHA).ljust(30, "<")[:30]
    return repair_document_number(a, start=5), b, c


def _reconcile_country(line1: str, line2: str) -> tuple[str, str]:
    """簽發國（第一行）與國籍（第二行）都不在檢查碼範圍內。兩者只差一個字元、
    且其中一個是支援的國家碼時，視為 OCR 誤讀（例如 JPN 讀成 3PN、IND 讀成 TND），以支援的那個為準。"""
    from .templates import COUNTRY_TO_PASSPORT
    issuer, nat = line1[2:5], line2[10:13]
    if issuer == nat or sum(x != y for x, y in zip(issuer, nat)) != 1:
        return line1, line2
    if issuer in COUNTRY_TO_PASSPORT and nat not in COUNTRY_TO_PASSPORT:
        return line1, line2[:10] + issuer + line2[13:]
    if nat in COUNTRY_TO_PASSPORT and issuer not in COUNTRY_TO_PASSPORT:
        return line1[:2] + nat + line1[5:], line2
    return line1, line2


_SWAPS = {"O": "0", "0": "O", "I": "1", "1": "I", "B": "8", "8": "B", "S": "5", "5": "S", "Z": "2", "2": "Z"}


def repair_document_number(line: str, start: int = 0) -> str:
    """證件號碼欄可含字母也可含數字，OCR 混淆無法靠位置判斷。
    若檢查碼不符，嘗試單一字元的 O/0、I/1 等互換，只接受能讓檢查碼通過的結果。
    start：號碼在該行的起始位置（TD3 第二行為 0，TD1 第一行為 5）。"""
    docnum, cd = line[start:start + 9], line[start + 9]
    if _cd_ok(docnum, cd):
        return line
    for i, ch in enumerate(docnum):
        if ch in _SWAPS:
            cand = docnum[:i] + _SWAPS[ch] + docnum[i + 1:]
            if _cd_ok(cand, cd):
                return line[:start] + cand + line[start + 9:]
    return line


# ---- 產生（合成資料與測試用） ----
def _mrz_text(s: str, n: int) -> str:
    from .normalize import strip_diacritics
    s = strip_diacritics(s).upper()
    s = re.sub(r"[^A-Z0-9]+", "<", s).strip("<")
    return s[:n].ljust(n, "<")


def build_td3(country: str, surname: str, given: str, doc_number: str, nationality: str,
              dob_yymmdd: str, sex: str, exp_yymmdd: str, personal: str = "", doc_code: str = "P") -> tuple[str, str]:
    from .normalize import strip_diacritics
    sn = re.sub(r"[^A-Z0-9]+", "<", strip_diacritics(surname).upper()).strip("<")
    gn = re.sub(r"[^A-Z0-9]+", "<", strip_diacritics(given).upper()).strip("<")
    names = (sn + "<<" + gn) if gn else sn
    line1 = (doc_code.ljust(2, "<") + country.ljust(3, "<") + names)[:44].ljust(44, "<")
    docn = _mrz_text(doc_number, 9)
    pers = _mrz_text(personal, 14) if personal else "<" * 14
    cd_pers = check_digit(pers) if personal else "<"
    body = (docn + check_digit(docn) + nationality.ljust(3, "<") + dob_yymmdd + check_digit(dob_yymmdd)
            + sex + exp_yymmdd + check_digit(exp_yymmdd) + pers + cd_pers)
    composite = body[0:10] + body[13:20] + body[21:43]
    return line1, body + check_digit(composite)


def build_td1(country: str, name: str, doc_number: str, nationality: str, dob_yymmdd: str, sex: str,
              exp_yymmdd: str, optional1: str = "", optional2: str = "", doc_code: str = "I") -> tuple[str, str, str]:
    """name 直接放第三行（姓與名的分隔依證件慣例，台灣居留證樣張整個姓名只用單個 < 分隔）。"""
    docn = _mrz_text(doc_number, 9)
    line1 = (doc_code.ljust(2, "<") + country.ljust(3, "<") + docn + check_digit(docn)
             + _mrz_text(optional1, 15) if optional1 else doc_code.ljust(2, "<") + country.ljust(3, "<") + docn
             + check_digit(docn) + "<" * 15)
    body = (dob_yymmdd + check_digit(dob_yymmdd) + sex + exp_yymmdd + check_digit(exp_yymmdd)
            + nationality.ljust(3, "<") + (_mrz_text(optional2, 11) if optional2 else "<" * 11))
    composite = line1[5:30] + body[0:7] + body[8:15] + body[18:29]
    line3 = _mrz_text(name, 30)
    return line1, body + check_digit(composite), line3
