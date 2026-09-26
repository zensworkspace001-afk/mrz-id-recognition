"""輸出記錄的 schema。JSONL 是主要資料來源，日後 DB 資料表照這些欄位建立。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from . import SCHEMA_VERSION


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class FieldValue:
    """單一欄位：正規化值、原始辨識文字、信心、框位置、來源。"""
    value: Optional[str] = None
    raw: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[list] = None
    source: str = ""  # "mrz" | "viz" | "mrz+viz"


@dataclass
class ResidenceInfo:
    """台灣居留證（ARC）專用欄位。"""
    arc_number: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    employer: Optional[str] = None
    purpose: Optional[str] = None  # 居留事由 / 工作類別
    remarks: Optional[str] = None


@dataclass
class DocRecord:
    doc_id: str
    source_image: str
    doc_type: str = "unknown"  # passport | arc | unknown
    template_key: str = "unknown"
    issuing_country: Optional[str] = None
    surname: Optional[str] = None
    given_names: Optional[str] = None
    middle_name: Optional[str] = None
    mononym: bool = False
    name_native: Optional[str] = None  # 漢字 / 韓文 / 帶變音符號的原文姓名
    name_latin: Optional[str] = None
    name_mrz: Optional[str] = None
    nationality: Optional[str] = None
    sex: Optional[str] = None
    date_of_birth: Optional[str] = None
    place_of_birth: Optional[str] = None
    document_number: Optional[str] = None
    personal_number: Optional[str] = None
    date_of_issue: Optional[str] = None
    date_of_expiry: Optional[str] = None
    authority: Optional[str] = None
    residence: ResidenceInfo = field(default_factory=ResidenceInfo)
    mrz: dict = field(default_factory=lambda: {"line1": None, "line2": None, "checksum_valid": None})
    fields: dict = field(default_factory=dict)  # 欄位名 -> FieldValue 的 dict（raw / confidence / bbox）
    needs_review: bool = False
    review_reasons: list = field(default_factory=list)
    reviewed_by_human: bool = False
    processed_at: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def min_confidence(self) -> Optional[float]:
        confs = [f["confidence"] for f in self.fields.values() if f.get("confidence") is not None]
        return min(confs) if confs else None
