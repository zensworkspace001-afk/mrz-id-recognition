"""主流程。各階段用可替換的 callable 注入，模型還沒訓練好時可先用假實作跑通。

階段：拉正 -> OCR 找 MRZ -> 路由 -> 視覺區抽取 -> MRZ/VIZ 交叉比對 -> 正規化 -> needs_review 判斷
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from .mrz import find_mrz_lines, fix_filler_noise, parse_mrz
from .normalize import names_match, parse_date, strip_diacritics, taiwan_id_valid
from .router import route
from .schema import DocRecord, image_hash
from .templates import TEMPLATES

# 介面定義
Rectifier = Callable[[object], object]  # 影像 -> 拉正後影像
OcrFn = Callable[[object], str]  # 影像 -> 全文 OCR 文字（用來找 MRZ）
# 視覺區抽取：(拉正影像, 版型) -> {欄位名: {"raw": str, "confidence": float, "bbox": list}}
VizExtractor = Callable[[object, object], dict]

CONF_THRESHOLD = 0.80


@dataclass
class Components:
    rectify: Rectifier = lambda img: img
    ocr_text: OcrFn = lambda img: ""
    viz_extract: VizExtractor = lambda img, tpl: {}
    classifier: Optional[Callable] = None


class Pipeline:
    def __init__(self, components: Components, storage=None, conf_threshold: float = CONF_THRESHOLD):
        self.c = components
        self.storage = storage
        self.conf_threshold = conf_threshold

    def process(self, image, image_bytes: bytes, source: str = "") -> DocRecord:
        rec = DocRecord(doc_id=image_hash(image_bytes), source_image=source,
                        processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        reasons = rec.review_reasons

        rect = self.c.rectify(image)
        mrz = None
        ocr_text = self.c.ocr_text(rect)
        if getattr(ocr_text, "uncertain_name", False):
            reasons.append("MRZ 姓名行各次辨識結果不一致")
        lines = find_mrz_lines(ocr_text)
        fixes = []
        if lines:
            lines, fixes = fix_filler_noise(lines)
            try:
                mrz = parse_mrz(lines)
            except ValueError as e:
                reasons.append(f"MRZ 解析失敗: {e}")

        rt = route(rect, mrz.issuing_country if mrz else None, self.c.classifier, mrz.format if mrz else "TD3")
        reasons.extend(rt.reasons)
        rec.template_key = rt.template_key
        tpl = TEMPLATES.get(rt.template_key)
        rec.doc_type = tpl.doc_type if tpl else ("passport" if mrz else "unknown")

        viz = self.c.viz_extract(rect, tpl) if tpl else {}
        self._merge(rec, mrz, viz, tpl)
        if mrz and fixes:
            rec.mrz["auto_fixes"] = fixes
        self._validate(rec, mrz, tpl)
        rec.needs_review = bool(reasons)
        if self.storage is not None:
            self.storage.save(rec.to_dict())
        return rec

    # ---- 合併 ----
    def _merge(self, rec: DocRecord, mrz, viz: dict, tpl):
        def put(name, value, src, v=None):
            rec.fields[name] = {"value": value, "raw": (v or {}).get("raw"),
                                "confidence": (v or {}).get("confidence"),
                                "bbox": (v or {}).get("bbox"), "source": src}

        if mrz:
            rec.issuing_country = mrz.issuing_country
            rec.nationality = mrz.nationality
            rec.sex = mrz.sex
            rec.date_of_birth = mrz.date_of_birth
            rec.date_of_expiry = mrz.date_of_expiry
            rec.document_number = mrz.document_number
            rec.personal_number = mrz.personal_number or None
            rec.surname, rec.given_names = mrz.surname, mrz.given_names
            rec.name_mrz = f"{mrz.surname} {mrz.given_names}".strip()
            rec.mrz = {"line1": mrz.line1, "line2": mrz.line2, "checksum_valid": mrz.checksum_valid,
                       "checks": mrz.checks}
            if mrz.format == "TD1":
                rec.mrz.update({"line3": mrz.line3, "format": "TD1", "optional_data": list(mrz.optional_data)})
            for n in ("document_number", "date_of_birth", "date_of_expiry", "nationality", "sex"):
                put(n, getattr(rec, n), "mrz")
                rec.fields[n]["confidence"] = 1.0 if mrz.checks.get(n, mrz.checksum_valid) else None
            for n in ("surname", "given_names"):  # 姓名行沒有檢查碼，信心未知
                put(n, getattr(rec, n), "mrz")
            if rec.personal_number:  # 以 MRZ 為準；視覺區讀到的只用來比對（否則會覆蓋 MRZ 的值）
                put("personal_number", rec.personal_number, "mrz")
                rec.fields["personal_number"]["confidence"] = 1.0 if mrz.checks.get("personal_number", True) else None
            if tpl and tpl.doc_type == "arc":
                # 居留證 MRZ：證件號碼欄是護照號碼；統一證號在選用資料欄（只受總檢查碼保護）
                rec.residence.valid_until = mrz.date_of_expiry
                m = re.search(f"(?:{tpl.rules['arc_number_pattern']})", "".join(mrz.optional_data))
                if m:
                    rec.residence.arc_number = m.group(0)
                    put("arc_number", m.group(0), "mrz")
                    rec.fields["arc_number"]["confidence"] = 1.0 if mrz.checks.get("composite") else None
        elif tpl:
            rec.issuing_country = tpl.country

        # MRZ 有的欄位：值與信心以 MRZ 為準，視覺區結果另存（viz_value / viz_confidence）供交叉比對。
        # MRZ 沒有的欄位、或 MRZ 不存在時，採用視覺區。
        for name, v in viz.items():
            raw = v.get("raw")
            if name in ("date_of_birth", "date_of_issue", "date_of_expiry", "valid_from", "valid_until"):
                val = parse_date(raw)
            else:
                val = raw.strip() if isinstance(raw, str) else raw
            src = "mrz+viz" if name in rec.fields else "viz"
            if name in rec.fields:
                rec.fields[name].update({"raw": raw, "bbox": v.get("bbox"), "source": src,
                                         "viz_value": val, "viz_confidence": v.get("confidence")})
                continue
            put(name, val, src, v)
            if name in ("place_of_birth", "date_of_issue", "authority", "middle_name", "name_native",
                        "name_latin", "personal_number"):
                setattr(rec, name, val)
            elif name in ("surname", "given_names") and not mrz:
                setattr(rec, name, val)
            elif name in ("document_number", "nationality", "sex", "date_of_birth", "date_of_expiry") and not mrz:
                setattr(rec, name, val)
            elif name == "arc_number":
                rec.residence.arc_number = val
            elif name in ("valid_from", "valid_until", "employer", "purpose"):
                setattr(rec.residence, name, val)

    # ---- 驗證與交叉比對 ----
    def _validate(self, rec: DocRecord, mrz, tpl):
        r = rec.review_reasons
        if mrz and not mrz.checksum_valid:
            bad = [k for k, ok in mrz.checks.items() if not ok]
            r.append("MRZ 檢查碼失敗: " + ",".join(bad))
        # 姓名欄沒有檢查碼；連續填充符號後又出現文字，通常是雜訊被讀成字母
        if mrz and re.search(r"<{3,}[A-Z]", mrz.name_field):
            r.append("MRZ 姓名欄有雜訊")

        # 與 MRZ 交叉比對（視覺區值 vs MRZ 值）
        if mrz:
            for n in ("document_number", "personal_number", "date_of_birth", "date_of_expiry", "sex", "arc_number"):
                f = rec.fields.get(n, {})
                vv = f.get("viz_value")
                if vv is None:
                    continue
                # MRZ 檢查碼通過、視覺區信心又低：多半是視覺區讀錯，以 MRZ 為準
                if mrz.checks.get(n, mrz.checks.get("composite")) and (f.get("viz_confidence") or 0) < self.conf_threshold:
                    continue
                if str(vv).replace(" ", "") != str(f.get("value")):
                    r.append(f"{n} 視覺區與 MRZ 不一致")
            # 姓名：第一行沒有檢查碼，只能靠視覺區比對。越南文先去變音符號；中/韓文原文不參與比對。
            # MRZ 姓名欄寫滿（結尾不是 <）才可能被截斷，只有這時允許前綴相符。
            truncated = not mrz.name_field.endswith("<")
            if tpl and tpl.rules.get("mrz_name_unsplit"):
                # 居留證 MRZ 姓名行不分姓與名：以視覺區姓 + 名的全名比對
                vs = [rec.fields.get(n, {}).get("viz_value") for n in ("surname", "given_names")]
                if any(vs) and not names_match(" ".join(v for v in vs if v), rec.name_mrz, allow_prefix=truncated):
                    r.append("姓名視覺區與 MRZ 不一致")
            else:
                for n in ("surname", "given_names"):
                    vv = rec.fields.get(n, {}).get("viz_value")
                    if vv is not None and not names_match(vv, getattr(rec, n), allow_prefix=truncated):
                        r.append(f"{n} 視覺區與 MRZ 不一致")
                # 視覺區的姓與名都讀到、而且和 MRZ 一致：姓名已有獨立來源確認，MRZ 各次讀法的分歧不必再送複核
                viz_names = [rec.fields.get(n, {}).get("viz_value") for n in ("surname", "given_names")]
                if all(v is not None for v in viz_names) and not any("視覺區與 MRZ 不一致" in x and ("surname" in x or "given_names" in x) for x in r):
                    r[:] = [x for x in r if x != "MRZ 姓名行各次辨識結果不一致"]

        # 低信心
        for n, f in rec.fields.items():
            c = f.get("confidence")
            if c is not None and c < self.conf_threshold:
                r.append(f"{n} 信心低 ({c:.2f})")

        # 發照日期（MRZ 沒有）：要早於到期日、晚於出生日；版型有規定效期年數時，到期日應為發照日加上該年數
        issue = rec.fields.get("date_of_issue", {}).get("value")
        if issue and rec.date_of_expiry:
            years = tpl.rules.get("validity_years") if tpl else None
            ok = rec.date_of_birth is None or issue >= rec.date_of_birth
            ok = ok and issue < rec.date_of_expiry
            if ok and years:
                ok = any(_add_years(issue, y) == rec.date_of_expiry for y in years)
            if not ok:
                r.append("發照日期與出生日期或到期日不符" + (f"（效期應為 {'或'.join(map(str, years))} 年）" if years else ""))

        # 台灣居留證：ARC 號碼格式
        if tpl and tpl.key == "TWN-arc-current":
            pat = tpl.rules["arc_number_pattern"]
            if not rec.residence.arc_number:
                r.append("未讀到居留證號")
            elif not re.fullmatch(pat, rec.residence.arc_number.replace(" ", "")):
                r.append("居留證號格式不符")
            elif re.fullmatch(r"[A-Z]\d{9}", rec.residence.arc_number) and not taiwan_id_valid(rec.residence.arc_number):
                r.append("統一證號檢查碼不符")

        # 台灣護照：個人號碼欄為身分證字號，有自己的檢查碼（MRZ 檢查碼之外的第二道檢查）
        if tpl and tpl.rules.get("personal_number_format") == "taiwan_id" and rec.personal_number:
            if not taiwan_id_valid(rec.personal_number):
                r.append("身分證統一編號格式或檢查碼不符")

        # 無 MRZ 的護照要複核；未知版型要複核
        if rec.doc_type == "passport" and not mrz:
            r.append("護照未讀到 MRZ")
        if rec.template_key == "unknown":
            r.append("版型未知")
        if tpl and not tpl.calibrated and not rec.fields:
            r.append("此版型尚未標定欄位位置，僅 MRZ 可用")


def _add_years(iso_date: str, years: int) -> str:
    """ISO 日期加上整數年（2 月 29 日遇到非閏年時取 2 月 28 日）。"""
    y, m, d = map(int, iso_date.split("-"))
    y += years
    if m == 2 and d == 29 and not (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
        d = 28
    return f"{y:04d}-{m:02d}-{d:02d}"
