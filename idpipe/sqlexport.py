"""把辨識結果打包成可直接在資料庫執行的 .sql 檔：資料表不存在就建立，已存在就寫入；
同一筆（doc_id，照片內容的 SHA-256）重複匯入時更新為最新結果，不會重複新增。

支援 PostgreSQL、MySQL、SQL Server、SQLite。欄位與 CSV 匯出（storage.flatten）一致，
另加 reviewed_by_human、source_file 與完整 JSON（record_json），日後要補欄位可從 JSON 取。
已存在的資料表若欄位不同（例如舊版匯出建的），INSERT 會失敗，需先調整資料表。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

DIALECTS = ("postgresql", "mysql", "sqlserver", "sqlite")

# (欄位, 型別)；型別：id | text | long | date | bool | ts | json
COLUMNS = [
    ("doc_id", "id"), ("source_file", "text"), ("doc_type", "text"), ("template_key", "text"),
    ("issuing_country", "text"), ("surname", "text"), ("given_names", "text"), ("middle_name", "text"),
    ("name_native", "text"), ("name_mrz", "text"), ("nationality", "text"), ("sex", "text"),
    ("date_of_birth", "date"), ("place_of_birth", "text"), ("document_number", "text"),
    ("personal_number", "text"), ("date_of_issue", "date"), ("date_of_expiry", "date"), ("authority", "text"),
    ("arc_number", "text"), ("residence_valid_from", "date"), ("residence_valid_until", "date"),
    ("employer", "text"), ("purpose", "text"),
    ("mrz_line1", "text"), ("mrz_line2", "text"), ("mrz_line3", "text"), ("mrz_checksum_valid", "bool"),
    ("needs_review", "bool"), ("review_reasons", "long"), ("reviewed_by_human", "bool"),
    ("processed_at", "ts"), ("schema_version", "text"), ("record_json", "json"),
]

TYPES = {
    "postgresql": {"id": "VARCHAR(64) PRIMARY KEY", "text": "TEXT", "long": "TEXT", "date": "DATE",
                   "bool": "BOOLEAN", "ts": "TIMESTAMPTZ", "json": "JSONB"},
    "mysql": {"id": "VARCHAR(64) NOT NULL PRIMARY KEY", "text": "VARCHAR(255)", "long": "TEXT", "date": "DATE",
              "bool": "BOOLEAN", "ts": "DATETIME", "json": "JSON"},
    "sqlserver": {"id": "NVARCHAR(64) NOT NULL PRIMARY KEY", "text": "NVARCHAR(255)", "long": "NVARCHAR(MAX)",
                  "date": "DATE", "bool": "BIT", "ts": "DATETIME2", "json": "NVARCHAR(MAX)"},
    "sqlite": {"id": "TEXT PRIMARY KEY", "text": "TEXT", "long": "TEXT", "date": "TEXT",
               "bool": "INTEGER", "ts": "TEXT", "json": "TEXT"},
}


def row_of(record: dict, source_file: str = "", reviewed: bool = False) -> dict:
    """一筆辨識結果（DocRecord.to_dict()）轉成資料表的一列。"""
    res = record.get("residence") or {}
    mrz = record.get("mrz") or {}
    rec = dict(record, reviewed_by_human=bool(reviewed or record.get("reviewed_by_human")))
    return {
        **{k: rec.get(k) for k, _ in COLUMNS if k in rec},
        "source_file": source_file or record.get("source_image"),
        "arc_number": res.get("arc_number"),
        "residence_valid_from": res.get("valid_from"),
        "residence_valid_until": res.get("valid_until"),
        "employer": res.get("employer"),
        "purpose": res.get("purpose"),
        "mrz_line1": mrz.get("line1"), "mrz_line2": mrz.get("line2"), "mrz_line3": mrz.get("line3"),
        "mrz_checksum_valid": mrz.get("checksum_valid"),
        "review_reasons": "; ".join(record.get("review_reasons") or []) or None,
        "record_json": json.dumps(rec, ensure_ascii=False, sort_keys=True),
    }


def _ident(name: str, dialect: str) -> str:
    return {"mysql": f"`{name}`", "sqlserver": f"[{name}]"}.get(dialect, f'"{name}"')


def _literal(value, kind: str, dialect: str) -> str:
    if value is None or value == "":
        return "NULL"
    if kind == "bool":
        return ("TRUE" if value else "FALSE") if dialect == "postgresql" else ("1" if value else "0")
    s = str(value)
    if kind == "ts" and dialect in ("mysql", "sqlserver"):
        s = s.replace("T", " ")[:19]  # 2026-09-26T08:00:00+00:00 -> 2026-09-26 08:00:00（UTC）
    s = s.replace("'", "''")
    if dialect == "mysql":
        s = s.replace("\\", "\\\\")  # MySQL 預設把反斜線當跳脫字元
    return ("N'" if dialect == "sqlserver" else "'") + s + "'"


def to_sql(rows: list, dialect: str = "postgresql", table: str = "id_documents") -> str:
    """rows：row_of() 的結果。回傳完整的 SQL 腳本（建表 + 寫入，包在同一個交易裡）。"""
    if dialect not in DIALECTS:
        raise ValueError(f"不支援的資料庫：{dialect}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", table):
        raise ValueError("資料表名稱只能用英文字母、數字與底線，且不能以數字開頭")
    t = _ident(table, dialect)
    cols = [c for c, _ in COLUMNS]
    col_list = ", ".join(_ident(c, dialect) for c in cols)
    defs = ",\n  ".join(f"{_ident(c, dialect)} {TYPES[dialect][k]}" for c, k in COLUMNS)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = [f"-- 證件辨識結果：{len(rows)} 筆，{dialect}，產生於 {now}",
           "-- 含個人資料，請妥善保管。資料表不存在會自動建立；同一 doc_id 重複匯入時以本檔內容更新。"]
    out += {"postgresql": ["-- 匯入方式：pgAdmin 請用 Query Tool 開啟本檔後執行（F5），不要用 Restore（Restore 只接受 pg_dump 備份檔）；",
                           "--          或命令列：psql -d 資料庫名稱 -f 本檔.sql"],
            "mysql": ["-- 匯入方式：mysql -u 使用者 -p 資料庫名稱 < 本檔.sql，或在 MySQL Workbench 開啟後執行"],
            "sqlserver": ["-- 匯入方式：在 SSMS 開啟本檔、選好資料庫後執行，或 sqlcmd -d 資料庫名稱 -i 本檔.sql"],
            "sqlite": ["-- 匯入方式：sqlite3 資料庫檔 < 本檔.sql"]}[dialect] + [""]

    if dialect == "sqlserver":
        out += [f"IF OBJECT_ID(N'{table}', N'U') IS NULL", f"CREATE TABLE {t} (\n  {defs}\n);", "GO", "",
                "BEGIN TRANSACTION;"]
    else:
        suffix = " DEFAULT CHARSET=utf8mb4" if dialect == "mysql" else ""
        out += [f"CREATE TABLE IF NOT EXISTS {t} (\n  {defs}\n){suffix};", "",
                "START TRANSACTION;" if dialect == "mysql" else "BEGIN;"]

    kinds = dict(COLUMNS)
    for row in rows:
        vals = ", ".join(_literal(row.get(c), kinds[c], dialect) for c in cols)
        others = [c for c in cols if c != "doc_id"]
        if dialect == "postgresql" or dialect == "sqlite":
            sets = ", ".join(f"{_ident(c, dialect)} = excluded.{_ident(c, dialect)}" for c in others)
            out.append(f"INSERT INTO {t} ({col_list}) VALUES ({vals})\n  ON CONFLICT ({_ident('doc_id', dialect)}) DO UPDATE SET {sets};")
        elif dialect == "mysql":
            sets = ", ".join(f"{_ident(c, dialect)} = VALUES({_ident(c, dialect)})" for c in others)
            out.append(f"INSERT INTO {t} ({col_list}) VALUES ({vals})\n  ON DUPLICATE KEY UPDATE {sets};")
        else:
            src = ", ".join(f"{_literal(row.get(c), kinds[c], dialect)} AS {_ident(c, dialect)}" for c in cols)
            sets = ", ".join(f"t.{_ident(c, dialect)} = s.{_ident(c, dialect)}" for c in others)
            ins = ", ".join(f"s.{_ident(c, dialect)}" for c in cols)
            out.append(f"MERGE {t} AS t USING (SELECT {src}) AS s ON t.[doc_id] = s.[doc_id]\n"
                       f"  WHEN MATCHED THEN UPDATE SET {sets}\n"
                       f"  WHEN NOT MATCHED THEN INSERT ({col_list}) VALUES ({ins});")
    out += ["COMMIT;", ""]
    return "\n".join(out)
