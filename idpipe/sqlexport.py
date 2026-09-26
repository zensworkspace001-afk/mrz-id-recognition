"""把辨識結果打包成可直接在資料庫執行的 .sql 檔：資料表不存在就建立，已存在就寫入；
同一筆（doc_id，照片內容的 SHA-256）重複匯入時更新為最新結果，不會重複新增。

支援 PostgreSQL、MySQL、SQL Server、SQLite。欄位與 CSV 匯出（storage.flatten）一致，
另加 reviewed_by_human、source_file 與完整 JSON（record_json），日後要補欄位可從 JSON 取。
已存在的資料表若欄位不同（例如舊版匯出建的），INSERT 會失敗，需先調整資料表。

網頁上人工修改過的欄位另外寫入「<資料表>_changes」修改紀錄表（每次修改一列：欄位、原值、新值、時間、當時的系統提示），
主表存修改後的值。修改紀錄只新增、不覆蓋：同一筆紀錄（doc_id＋欄位＋修改時間）重複匯入時略過。
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

# 修改紀錄表 (欄位, 型別)
CHANGE_COLUMNS = [
    ("doc_id", "key"), ("source_file", "text"), ("field_name", "keyshort"), ("old_value", "text"),
    ("new_value", "text"), ("system_note", "long"), ("changed_at", "tskey"),
]

TYPES = {
    "postgresql": {"id": "VARCHAR(64) PRIMARY KEY", "text": "TEXT", "long": "TEXT", "date": "DATE",
                   "bool": "BOOLEAN", "ts": "TIMESTAMPTZ", "json": "JSONB",
                   "key": "VARCHAR(64) NOT NULL", "keyshort": "VARCHAR(64) NOT NULL", "tskey": "TIMESTAMPTZ NOT NULL"},
    "mysql": {"id": "VARCHAR(64) NOT NULL PRIMARY KEY", "text": "VARCHAR(255)", "long": "TEXT", "date": "DATE",
              "bool": "BOOLEAN", "ts": "DATETIME", "json": "JSON",
              "key": "VARCHAR(64) NOT NULL", "keyshort": "VARCHAR(64) NOT NULL", "tskey": "DATETIME(3) NOT NULL"},
    "sqlserver": {"id": "NVARCHAR(64) NOT NULL PRIMARY KEY", "text": "NVARCHAR(255)", "long": "NVARCHAR(MAX)",
                  "date": "DATE", "bool": "BIT", "ts": "DATETIME2", "json": "NVARCHAR(MAX)",
                  "key": "NVARCHAR(64) NOT NULL", "keyshort": "NVARCHAR(64) NOT NULL", "tskey": "DATETIME2 NOT NULL"},
    "sqlite": {"id": "TEXT PRIMARY KEY", "text": "TEXT", "long": "TEXT", "date": "TEXT",
               "bool": "INTEGER", "ts": "TEXT", "json": "TEXT",
               "key": "TEXT NOT NULL", "keyshort": "TEXT NOT NULL", "tskey": "TEXT NOT NULL"},
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
    if kind in ("ts", "tskey") and dialect in ("mysql", "sqlserver"):
        # 2026-09-26T08:00:00.123Z -> 2026-09-26 08:00:00.123（UTC）；主表只到秒
        s = s.replace("T", " ")[:23 if kind == "tskey" else 19].rstrip("Z")
    s = s.replace("'", "''")
    if dialect == "mysql":
        s = s.replace("\\", "\\\\")  # MySQL 預設把反斜線當跳脫字元
    return ("N'" if dialect == "sqlserver" else "'") + s + "'"


def change_rows(record: dict, source_file: str = "") -> list:
    """record["change_log"]（網頁人工修改紀錄）轉成修改紀錄表的列。"""
    return [{"doc_id": record.get("doc_id"), "source_file": source_file or record.get("source_image"),
             "field_name": c.get("field"), "old_value": c.get("old"), "new_value": c.get("new"),
             "system_note": c.get("note"), "changed_at": c.get("at")}
            for c in record.get("change_log") or []]


def to_sql(rows: list, dialect: str = "postgresql", table: str = "id_documents", changes: list | None = None) -> str:
    """rows：row_of() 的結果；changes：change_rows() 的結果（有人工修改時才建修改紀錄表）。
    回傳完整的 SQL 腳本（建表 + 寫入，包在同一個交易裡）。"""
    if dialect not in DIALECTS:
        raise ValueError(f"不支援的資料庫：{dialect}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,54}", table):
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

    changes = changes or []
    ct = _ident(table + "_changes", dialect)
    ccols = [c for c, _ in CHANGE_COLUMNS]
    cdefs = ",\n  ".join(f"{_ident(c, dialect)} {TYPES[dialect][k]}" for c, k in CHANGE_COLUMNS)
    cdefs += f",\n  PRIMARY KEY ({', '.join(_ident(c, dialect) for c in ('doc_id', 'field_name', 'changed_at'))})"
    if dialect == "sqlserver":
        out += [f"IF OBJECT_ID(N'{table}', N'U') IS NULL", f"CREATE TABLE {t} (\n  {defs}\n);", "GO", ""]
        if changes:
            out += [f"IF OBJECT_ID(N'{table}_changes', N'U') IS NULL", f"CREATE TABLE {ct} (\n  {cdefs}\n);", "GO", ""]
        out += ["BEGIN TRANSACTION;"]
    else:
        suffix = " DEFAULT CHARSET=utf8mb4" if dialect == "mysql" else ""
        out += [f"CREATE TABLE IF NOT EXISTS {t} (\n  {defs}\n){suffix};", ""]
        if changes:
            out += [f"-- 人工修改紀錄", f"CREATE TABLE IF NOT EXISTS {ct} (\n  {cdefs}\n){suffix};", ""]
        out += ["START TRANSACTION;" if dialect == "mysql" else "BEGIN;"]

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
    ckinds = dict(CHANGE_COLUMNS)
    ccol_list = ", ".join(_ident(c, dialect) for c in ccols)
    for row in changes:
        vals = ", ".join(_literal(row.get(c), ckinds[c], dialect) for c in ccols)
        if dialect in ("postgresql", "sqlite"):
            out.append(f"INSERT INTO {ct} ({ccol_list}) VALUES ({vals})\n  ON CONFLICT DO NOTHING;")
        elif dialect == "mysql":
            out.append(f"INSERT IGNORE INTO {ct} ({ccol_list}) VALUES ({vals});")
        else:
            cond = " AND ".join(f"{_ident(c, dialect)} = {_literal(row.get(c), ckinds[c], dialect)}"
                                for c in ("doc_id", "field_name", "changed_at"))
            out.append(f"IF NOT EXISTS (SELECT 1 FROM {ct} WHERE {cond})\n  INSERT INTO {ct} ({ccol_list}) VALUES ({vals});")
    out += ["COMMIT;", ""]
    return "\n".join(out)
