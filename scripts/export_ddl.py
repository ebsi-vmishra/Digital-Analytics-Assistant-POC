# scripts/export_ddl.py
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import List, Optional, Dict, Any

import yaml
from sqlalchemy import create_engine, text

# Works whether run as module or script
try:
    from utils.db import resolve_mssql_url
except Exception:
    # fallback if run as a plain script
    import sys, os
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.db import resolve_mssql_url


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _schema_filter_list(cfg: dict) -> List[str]:
    inc = (cfg.get("ingest") or {}).get("schema_include") or []
    if isinstance(inc, str):
        inc = [inc]
    return [str(x).strip() for x in inc if str(x).strip()]


def _len_spec(row) -> str:
    dt = str(row["data_type"]).lower()
    if dt in ("nvarchar", "nchar"):
        ml = row["max_length"]
        if ml is None or ml < 0:
            return "(MAX)"
        return f"({int(ml) // 2})"
    if dt in ("varchar", "char", "varbinary"):
        ml = row["max_length"]
        if ml is None or ml < 0:
            return "(MAX)"
        return f"({int(ml)})"
    if dt in ("decimal", "numeric"):
        return f"({row['precision']},{row['scale']})"
    return ""


def _quote_ident(name: str) -> str:
    return f"[{name}]"


def _build_column_line(row) -> str:
    parts = [f"{_quote_ident(row['column_name'])} {row['data_type']}{_len_spec(row)}"]
    if row.get("is_identity"):
        parts.append(f"IDENTITY({row['identity_seed']},{row['identity_increment']})")
    if row.get("default_definition"):
        parts.append(f"DEFAULT {row['default_definition']}")
    parts.append("NOT NULL" if not row["is_nullable"] else "NULL")
    return " ".join(parts)


def _gather_metadata(conn, schema_filter: Optional[List[str]]):
    # NOTE: CAST everything to ODBC-friendly types to avoid pyodbc type -16 issues.
    where = ""
    params: Dict[str, Any] = {}
    if schema_filter:
        placeholders = ",".join([f":s{i}" for i, _ in enumerate(schema_filter)])
        where = f"WHERE sch.name IN ({placeholders})"
        params = {f"s{i}": s for i, s in enumerate(schema_filter)}

    tables_sql = f"""
        SELECT CAST(sch.name AS NVARCHAR(128)) AS schema_name,
               CAST(t.name  AS NVARCHAR(128)) AS table_name,
               CAST(t.object_id AS INT)       AS object_id
        FROM sys.tables t
        JOIN sys.schemas sch ON sch.schema_id = t.schema_id
        {where}
        ORDER BY sch.name, t.name
    """
    tables = conn.execute(text(tables_sql), params).fetchall()

    cols_sql = f"""
        SELECT CAST(sch.name AS NVARCHAR(128)) AS schema_name,
               CAST(t.name  AS NVARCHAR(128)) AS table_name,
               CAST(t.object_id AS INT)       AS object_id,
               CAST(c.column_id AS INT)       AS column_id,
               CAST(c.name  AS NVARCHAR(128)) AS column_name,
               CAST(ty.name AS NVARCHAR(128)) AS data_type,
               CAST(c.max_length AS INT)      AS max_length,
               CAST(c.precision  AS INT)      AS precision,
               CAST(c.scale      AS INT)      AS scale,
               CAST(c.is_nullable AS TINYINT) AS is_nullable,
               CAST(c.is_identity AS TINYINT) AS is_identity,
               CAST(ISNULL(ic.seed_value,1)      AS BIGINT)         AS identity_seed,
               CAST(ISNULL(ic.increment_value,1) AS BIGINT)         AS identity_increment,
               CAST(dc.definition AS NVARCHAR(4000)) AS default_definition
        FROM sys.tables t
        JOIN sys.schemas sch ON sch.schema_id = t.schema_id
        JOIN sys.columns c ON c.object_id = t.object_id
        JOIN sys.types ty ON ty.user_type_id = c.user_type_id
        LEFT JOIN sys.default_constraints dc 
               ON dc.parent_object_id = c.object_id AND dc.parent_column_id = c.column_id
        LEFT JOIN sys.identity_columns ic 
               ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        {where}
        ORDER BY sch.name, t.name, c.column_id
    """
    cols = conn.execute(text(cols_sql), params).fetchall()

    pk_sql = f"""
        SELECT CAST(sch.name AS NVARCHAR(128)) AS schema_name,
               CAST(t.name  AS NVARCHAR(128)) AS table_name,
               CAST(kc.name AS NVARCHAR(128)) AS constraint_name,
               CAST(ic.key_ordinal AS INT)    AS key_ordinal,
               CAST(col.name AS NVARCHAR(128)) AS column_name,
               CAST(kc.type AS NVARCHAR(2))   AS type
        FROM sys.key_constraints kc
        JOIN sys.tables t ON t.object_id = kc.parent_object_id
        JOIN sys.schemas sch ON sch.schema_id = t.schema_id
        JOIN sys.index_columns ic ON ic.object_id = t.object_id AND ic.index_id = kc.unique_index_id
        JOIN sys.columns col ON col.object_id = t.object_id AND col.column_id = ic.column_id
        {where}
        ORDER BY sch.name, t.name, kc.name, ic.key_ordinal
    """
    pks = conn.execute(text(pk_sql), params).fetchall()

    fk_sql = f"""
        SELECT CAST(schp.name AS NVARCHAR(128)) AS schema_name,
               CAST(tp.name  AS NVARCHAR(128))  AS table_name,
               CAST(fk.name  AS NVARCHAR(128))  AS constraint_name,
               CAST(scol.name AS NVARCHAR(128)) AS fk_column,
               CAST(schr.name AS NVARCHAR(128)) AS ref_schema,
               CAST(tr.name   AS NVARCHAR(128)) AS ref_table,
               CAST(rcol.name AS NVARCHAR(128)) AS ref_column,
               CAST(fkc.constraint_column_id AS INT) AS ord
        FROM sys.foreign_keys fk
        JOIN sys.tables tp ON tp.object_id = fk.parent_object_id
        JOIN sys.schemas schp ON schp.schema_id = tp.schema_id
        JOIN sys.tables tr ON tr.object_id = fk.referenced_object_id
        JOIN sys.schemas schr ON schr.schema_id = tr.schema_id
        JOIN sys.foreign_key_columns fkc ON fkc.constraint_object_id = fk.object_id
        JOIN sys.columns scol ON scol.object_id = tp.object_id AND scol.column_id = fkc.parent_column_id
        JOIN sys.columns rcol ON rcol.object_id = tr.object_id AND rcol.column_id = fkc.referenced_column_id
        {where.replace('sch.name','schp.name')}
        ORDER BY schp.name, tp.name, fk.name, fkc.constraint_column_id
    """
    fks = conn.execute(text(fk_sql), params).fetchall()

    return tables, cols, pks, fks


def _assemble_ddls(tables, cols, pks, fks):
    cols_by_tbl = defaultdict(list)
    for r in cols:
        cols_by_tbl[(r.schema_name, r.table_name)].append({k: getattr(r, k) for k in r._mapping.keys()})

    pk_by_tbl = defaultdict(list)
    uq_by_tbl = defaultdict(list)
    for r in pks:
        if r.type == 'PK':
            pk_by_tbl[(r.schema_name, r.table_name)].append((r.key_ordinal, r.column_name, r.constraint_name))
        elif r.type == 'UQ':
            uq_by_tbl[(r.schema_name, r.table_name)].append((r.key_ordinal, r.column_name, r.constraint_name))

    fk_by_tbl = defaultdict(list)
    for r in fks:
        fk_by_tbl[(r.schema_name, r.table_name)].append({
            "constraint_name": r.constraint_name,
            "ord": r.ord,
            "fk_column": r.fk_column,
            "ref_schema": r.ref_schema,
            "ref_table": r.ref_table,
            "ref_column": r.ref_column
        })

    ddls = OrderedDict()
    for t in tables:
        key = (t.schema_name, t.table_name)
        lines = [f"CREATE TABLE [{t.schema_name}].[{t.table_name}] ("]
        col_lines = []
        for row in cols_by_tbl[key]:
            # normalize Python types
            row["is_nullable"] = bool(int(row["is_nullable"] or 0))
            row["is_identity"] = bool(int(row["is_identity"] or 0))
            col_lines.append(_build_column_line(row))

        pk_cols = sorted(pk_by_tbl.get(key, []))
        if pk_cols:
            cname = pk_cols[0][2]
            cols_str = ", ".join(f"[{c[1]}]" for c in pk_cols)
            col_lines.append(f"CONSTRAINT [{cname}] PRIMARY KEY ({cols_str})")

        if not col_lines:
            col_lines.append("[__dummy] INT NULL")

        lines.append("  " + ",\n  ".join(col_lines))
        lines.append(");\nGO\n")

        # UQ constraints
        if uq_by_tbl.get(key):
            grp = defaultdict(list)
            for ord_, col, cname in uq_by_tbl[key]:
                grp[cname].append((ord_, col))
            for cname, items in grp.items():
                items = sorted(items)
                cols_str = ", ".join(f"[{c}]" for _, c in items)
                lines.append(
                    f"ALTER TABLE [{t.schema_name}].[{t.table_name}] "
                    f"ADD CONSTRAINT [{cname}] UNIQUE ({cols_str});\nGO\n"
                )

        # FK constraints
        if fk_by_tbl.get(key):
            grp = defaultdict(list)
            for rec in fk_by_tbl[key]:
                grp[rec['constraint_name']].append(rec)
            for cname, items in grp.items():
                items = sorted(items, key=lambda x: x['ord'])
                fk_cols = ", ".join(f"[{x['fk_column']}]" for x in items)
                ref_cols = ", ".join(f"[{x['ref_column']}]" for x in items)
                ref = items[0]
                lines.append(
                    f"ALTER TABLE [{t.schema_name}].[{t.table_name}] "
                    f"ADD CONSTRAINT [{cname}] FOREIGN KEY ({fk_cols}) "
                    f"REFERENCES [{ref['ref_schema']}].[{ref['ref_table']}] ({ref_cols});\nGO\n"
                )

        ddls[f"{t.schema_name}.{t.table_name}"] = "".join(lines)

    return ddls


def export(cfg_path: str):
    cfg = _load_cfg(cfg_path)
    db_url = resolve_mssql_url(cfg)
    out_dir = Path(cfg.get("outputs", {}).get("ddl_dir", "artifacts/ddl"))
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_filter = _schema_filter_list(cfg)
    with create_engine(db_url).connect() as conn:
        tables, cols, pks, fks = _gather_metadata(conn, schema_filter)
    ddls = _assemble_ddls(tables, cols, pks, fks)

    for full, ddl in ddls.items():
        sch, tbl = full.split(".", 1)
        (out_dir / f"{sch}.{tbl}.sql").write_text(ddl, encoding="utf-8")

    with open(out_dir / "all_tables.sql", "w", encoding="utf-8") as f:
        for full, ddl in ddls.items():
            f.write(f"-- {full}\n{ddl}\n")

    print(f"[export_ddl] Wrote {len(ddls)} tables to {out_dir}")
    return str(out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    export(args.config)
