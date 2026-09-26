"""儲存層 adapter：JSONL（現在）與 SQLite（日後換 DB 時參考）。CSV 為攤平匯出。"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Protocol

CSV_COLUMNS = [
    "doc_id", "source_image", "doc_type", "template_key", "issuing_country",
    "surname", "given_names", "middle_name", "mononym", "name_native", "name_mrz",
    "nationality", "sex", "date_of_birth", "place_of_birth", "document_number",
    "personal_number", "date_of_issue", "date_of_expiry", "authority",
    "arc_number", "residence_valid_from", "residence_valid_until", "employer", "purpose",
    "mrz_line1", "mrz_line2", "mrz_line3", "mrz_checksum_valid",
    "min_confidence", "needs_review", "review_reason", "processed_at", "schema_version",
]


class Storage(Protocol):
    def save(self, record: dict) -> None: ...
    def get(self, doc_id: str) -> Optional[dict]: ...
    def all(self) -> Iterable[dict]: ...


class JsonlStorage:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def all(self):
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def get(self, doc_id):
        for r in self.all():
            if r["doc_id"] == doc_id:
                return r
        return None

    def save(self, record):
        """以 doc_id 去重：已存在則取代（保留最新處理結果）。"""
        rows = [r for r in self.all() if r["doc_id"] != record["doc_id"]]
        rows.append(record)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(self.path)


class SqliteStorage:
    """整筆 JSON 放一欄，加上常用查詢欄位。日後換 Postgres 可照此拆成正式資料表。"""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("""CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY, doc_type TEXT, issuing_country TEXT,
            document_number TEXT, date_of_expiry TEXT, needs_review INTEGER,
            schema_version TEXT, record_json TEXT NOT NULL)""")
        self.db.commit()

    def save(self, record):
        self.db.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
            (record["doc_id"], record["doc_type"], record["issuing_country"],
             record["document_number"], record["date_of_expiry"], int(record["needs_review"]),
             record["schema_version"], json.dumps(record, ensure_ascii=False)))
        self.db.commit()

    def get(self, doc_id):
        row = self.db.execute("SELECT record_json FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self):
        for (j,) in self.db.execute("SELECT record_json FROM documents"):
            yield json.loads(j)


def _txt(v):
    """證件號碼等以文字寫出，避免 Excel 吃掉前導 0 或轉成科學記號：加上製表符前綴。"""
    return v


def flatten(record: dict) -> dict:
    res = record.get("residence") or {}
    mrz = record.get("mrz") or {}
    confs = [f["confidence"] for f in (record.get("fields") or {}).values() if f.get("confidence") is not None]
    row = {
        **{k: record.get(k) for k in CSV_COLUMNS if k in record},
        "arc_number": res.get("arc_number"),
        "residence_valid_from": res.get("valid_from"),
        "residence_valid_until": res.get("valid_until"),
        "employer": res.get("employer"),
        "purpose": res.get("purpose"),
        "mrz_line1": mrz.get("line1"),
        "mrz_line2": mrz.get("line2"),
        "mrz_line3": mrz.get("line3"),
        "mrz_checksum_valid": mrz.get("checksum_valid"),
        "min_confidence": round(min(confs), 3) if confs else "",
        "review_reason": "; ".join(record.get("review_reasons") or []),
    }
    for k in ("document_number", "personal_number", "arc_number"):
        if row.get(k):
            row[k] = "\t" + str(row[k])  # 前置製表符，Excel 會當文字處理
    return row


def export_csv(records: Iterable[dict], path) -> int:
    """整份重寫 CSV（以 utf-8-sig 寫入，Excel 不亂碼）。回傳筆數。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(flatten(r))
            n += 1
    return n
