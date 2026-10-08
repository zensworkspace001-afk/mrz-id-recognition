"""版型設定。欄位區域（region）為拉正後影像的相對座標 (x0, y0, x1, y1)，範圍 0~1。

注意：所有 region 目前都是「空模板」，需要拿到官方樣本後，量測實際位置再填入。
這裡只定義每個版型有哪些欄位、用什麼 OCR 語言、日期格式與特殊規則。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Region = tuple  # (x0, y0, x1, y1)


@dataclass
class Template:
    key: str
    doc_type: str  # passport | arc
    country: str
    description: str
    ocr_langs: list  # 各欄位需要的 OCR 語言
    date_format: str  # 視覺區日期格式提示
    fields: dict = field(default_factory=dict)  # 欄位名 -> Region | None（None 表示尚未標定）
    rules: dict = field(default_factory=dict)
    calibrated: bool = False  # 是否已用樣本標定過 region


def _f(*names):
    return {n: None for n in names}


PASSPORT_COMMON = ["surname", "given_names", "nationality", "sex", "date_of_birth",
                   "place_of_birth", "document_number", "date_of_issue", "date_of_expiry", "authority"]

# 日本護照「本籍」欄印的都道府縣（羅馬拼音、大寫）
JPN_PREFECTURES = (
    "HOKKAIDO", "AOMORI", "IWATE", "MIYAGI", "AKITA", "YAMAGATA", "FUKUSHIMA", "IBARAKI", "TOCHIGI", "GUNMA",
    "SAITAMA", "CHIBA", "TOKYO", "KANAGAWA", "NIIGATA", "TOYAMA", "ISHIKAWA", "FUKUI", "YAMANASHI", "NAGANO",
    "GIFU", "SHIZUOKA", "AICHI", "MIE", "SHIGA", "KYOTO", "OSAKA", "HYOGO", "NARA", "WAKAYAMA", "TOTTORI",
    "SHIMANE", "OKAYAMA", "HIROSHIMA", "YAMAGUCHI", "TOKUSHIMA", "KAGAWA", "EHIME", "KOCHI", "FUKUOKA", "SAGA",
    "NAGASAKI", "KUMAMOTO", "OITA", "MIYAZAKI", "KAGOSHIMA", "OKINAWA")

TEMPLATES: dict[str, Template] = {
    "JPN-passport-current": Template(
        "JPN-passport-current", "passport", "JPN",
        "日本護照（現行版）。標籤日英對照（旅券番号/Passport No.、姓/Surname、名/Given name…）；"
        "沒有出生地，改印本籍（Registered Domicile，都道府縣）。尚未用實拍樣本驗證",
        ["en"], "DD MMM YYYY",
        _f(*[n for n in PASSPORT_COMMON if n != "place_of_birth"], "registered_domicile"),
        {"date_month_names": "en",
         "viz": "labels",
         # 護照號碼 2 英文 + 7 數字（例如 TK1234567）；本籍只會是 47 都道府縣之一（只有視覺區有，不能讓雜訊混進來）
         "viz_formats": {"document_number": r"[A-Z]{2}\d{7}", "nationality": r"JAPAN",
                         "registered_domicile": "|".join(JPN_PREFECTURES)},
         "viz_canonical": {"authority": "MINISTRY OF FOREIGN AFFAIRS", "nationality": "JAPAN"}}),
    # 不設 validity_years：發照日與到期日的對應關係（是否為同月同日）尚未用實拍樣本確認
    "CHN-passport-current": Template(
        "CHN-passport-current", "passport", "CHN", "中國護照，中英對照，含漢字姓名",
        ["en", "ch"], "DD MMM YYYY / 中英對照", _f(*PASSPORT_COMMON, "name_native"),
        {"native_name_script": "han", "native_excluded_from_matching": True}),
    "HKG-passport-current": Template(
        "HKG-passport-current", "passport", "HKG", "香港特區護照，與中國護照為不同版型",
        ["en", "ch"], "DD MMM YYYY", _f(*PASSPORT_COMMON, "name_native"),
        {"native_name_script": "han", "native_excluded_from_matching": True}),
    "KOR-passport-current": Template(
        "KOR-passport-current", "passport", "KOR", "韓國護照，含韓文姓名",
        ["en", "korean"], "DD MMM YYYY", _f(*PASSPORT_COMMON, "name_native"),
        {"native_name_script": "hangul", "native_excluded_from_matching": True}),
    "VNM-passport-current": Template(
        "VNM-passport-current", "passport", "VNM", "越南護照，姓名含變音符號，可能有個人識別號",
        ["en", "vi"], "DD/MM/YYYY", _f(*PASSPORT_COMMON, "personal_number", "name_latin"),
        {"strip_diacritics_for_matching": True}),
    "IDN-passport-current": Template(
        "IDN-passport-current", "passport", "IDN", "印尼護照，單名常見",
        ["en"], "DD MMM YYYY", _f(*PASSPORT_COMMON),
        {"mononym_possible": True}),
    "PHL-passport-current": Template(
        "PHL-passport-current", "passport", "PHL", "菲律賓護照，有 middle name，MRZ 無獨立欄位",
        ["en"], "DD MMM YYYY", _f(*PASSPORT_COMMON, "middle_name"),
        {"enye_maps_to_n_in_mrz": True}),
    "IND-passport-current": Template(
        "IND-passport-current", "passport", "IND", "印度護照，標籤含印地語，部分欄位在最後一頁",
        ["en"], "DD/MM/YYYY", _f(*PASSPORT_COMMON, "personal_number"),
        {"mononym_possible": True, "some_fields_on_last_page": True}),
    "TWN-passport-current": Template(
        "TWN-passport-current", "passport", "TWN", "中華民國護照，含漢字姓名；個人號碼欄為國民身分證統一編號（依實拍樣本）",
        ["en", "ch"], "DD MMM YYYY", _f(*PASSPORT_COMMON, "name_native", "personal_number"),
        {"native_name_script": "han", "native_excluded_from_matching": True, "personal_number_format": "taiwan_id",
         # 視覺區以欄位標籤定位（vizlabels.py，不需量測位置）；標籤讀壞時改找唯一符合格式的一行
         "viz": "labels",
         "viz_formats": {"document_number": r"\d{9}", "personal_number": r"[A-Z][12]\d{8}"},
         "viz_canonical": {"authority": "MINISTRY OF FOREIGN AFFAIRS"},
         "validity_years": (5, 10)}),  # 未滿 14 歲 5 年、其餘 10 年
    "TWN-arc-current": Template(
        "TWN-arc-current", "arc", "TWN",
        "台灣居留證（ARC，卡片；含就業金卡）。背面為 TD1 三行 MRZ（依移民署樣張）："
        "證件號碼欄是護照號碼，統一證號放在第一行選用資料欄，姓名行不以 << 分隔姓與名",
        ["en", "ch"], "YYYY/MM/DD 或 DD MMM YYYY",
        _f("surname", "given_names", "name_native", "nationality", "sex", "date_of_birth",
           "arc_number", "valid_from", "valid_until", "employer", "purpose"),
        {"arc_number_pattern": r"[A-Z]{2}\d{8}|[A-Z]\d{9}",  # 舊式 2 英文 + 8 數字、新式 1 英文 + 9 數字
         "mrz_format": "TD1", "mrz_document_number_is_passport": True, "mrz_name_unsplit": True}),
}

# MRZ 簽發國碼 -> 護照版型 key
COUNTRY_TO_PASSPORT = {t.country: t.key for t in TEMPLATES.values() if t.doc_type == "passport"}
# 卡片型證件（TD1）簽發國碼 -> 版型 key
COUNTRY_TO_CARD = {t.country: t.key for t in TEMPLATES.values() if t.rules.get("mrz_format") == "TD1"}


def template_for_mrz(mrz_format: str, country: Optional[str]) -> Optional[str]:
    return (COUNTRY_TO_CARD if mrz_format == "TD1" else COUNTRY_TO_PASSPORT).get(country)


def _px(x0, y0, x1, y1, w=1250, h=880):
    return (x0 / w, y0 / h, x1 / w, y1 / h)


# 合成資料（synth/render.py）的通用示意版面，所有國家共用，座標取自合成標注並往外留邊。
# 只用來在沒有官方樣本前把視覺區抽取流程跑通，不是任何真實版型。
SYNTH_LAYOUT = {
    "surname": _px(370, 110, 1000, 166),
    "given_names": _px(370, 200, 1000, 256),
    "nationality": _px(370, 290, 1000, 346),
    "date_of_birth": _px(370, 380, 790, 436),
    "sex": _px(790, 380, 900, 436),
    "document_number": _px(370, 470, 1000, 526),
    "date_of_issue": _px(370, 560, 790, 616),
    "date_of_expiry": _px(790, 560, 1230, 616),
}
