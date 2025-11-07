# scripts/ssms_ingest.py
import json
import yaml
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from urllib.parse import quote_plus
from typing import Dict, List, Tuple
from utils.db import resolve_mssql_url


def _resolve_db_url(cfg: dict) -> str:
    dbc = (cfg.get("database") or {})
    raw = (dbc.get("odbc_connection_string") or "").strip()
    dsn = (dbc.get("dsn") or "").strip()
    if raw:
        if raw.lower().startswith("mssql+pyodbc://"):
            return raw
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(raw)}"
    if dsn:
        return f"mssql+pyodbc://@{dsn}"
    raise ValueError("No database connection info found. Provide database.odbc_connection_string or database.dsn in YAML.")


def _schema_include(cfg: dict) -> List[str]:
    inc = (cfg.get("ingest") or {}).get("schema_include") or []
    if isinstance(inc, str):
        inc = [inc]
    # strip empties and normalize to UPPER for case-insensitive match
    return [str(s).strip().upper() for s in inc if str(s).strip()]


def _in_clause_params(values: List[str], param_prefix: str) -> Tuple[str, Dict[str, str]]:
    """
    Returns an IN (...) clause for UPPER(schema), so pass UPPERCASE values.
    """
    if not values:
        return "", {}
    names = [f"{param_prefix}{i}" for i in range(len(values))]
    clause = "IN (" + ", ".join([f":{n}" for n in names]) + ")"
    params = {names[i]: values[i] for i in range(len(values))}
    return clause, params


def run(db_url: str,
        out_schema_path: str = "artifacts/schema.json",
        out_rels_path: str = "artifacts/relationships.json",
        schema_filter: List[str] | None = None):
    Path(out_schema_path).parent.mkdir(parents=True, exist_ok=True)

    eng = create_engine(db_url, fast_executemany=False)
    meta = {"tables": {}, "relationships": {"foreign_keys": []}, "summary": {}}

    # Make filter explicit & uppercase for case-insensitive comparisons
    schema_filter = [s.upper() for s in (schema_filter or [])]

    # Debug: show what filter is being used
    if schema_filter:
        print(f"[ingest] Filtering to schemas (case-insensitive): {schema_filter}")
    else:
        print("[ingest] No schema filter provided — ingesting ALL schemas")

    with eng.connect() as conn:
        # ----------------- TABLES -----------------
        schema_clause = ""
        params: Dict[str, str] = {}

        if schema_filter:
            in_sql, in_params = _in_clause_params(schema_filter, "s")
            # Use UPPER() for case-insensitive match
            schema_clause = f"AND UPPER(TABLE_SCHEMA) {in_sql}"
            params.update(in_params)

        tables_sql = f"""
            SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE IN ('BASE TABLE','VIEW')
            {schema_clause}
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        tables = pd.read_sql(text(tables_sql), conn, params=params)

        # Columns
        cols_sql = f"""
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE 1=1
            {schema_clause}
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        """
        cols = pd.read_sql(text(cols_sql), conn, params=params)

        # Build table->columns
        for _, r in tables.iterrows():
            schema, tname, ttype = str(r.TABLE_SCHEMA), str(r.TABLE_NAME), str(r.TABLE_TYPE)
            full = f"{schema}.{tname}"
            meta["tables"][full] = {"columns": {}, "table_type": ttype}

        for _, c in cols.iterrows():
            full = f"{c.TABLE_SCHEMA}.{c.TABLE_NAME}"
            if full in meta["tables"]:
                meta["tables"][full]["columns"][str(c.COLUMN_NAME)] = {
                    "data_type": str(c.DATA_TYPE),
                    "is_nullable": str(c.IS_NULLABLE)
                }

        # ----------------- RELATIONSHIPS -----------------
        rel_clause = ""
        rel_params: Dict[str, str] = {}
        if schema_filter:
            in_sql1, in_params1 = _in_clause_params(schema_filter, "r1_")
            in_sql2, in_params2 = _in_clause_params(schema_filter, "r2_")
            # Use UPPER() for both sides
            rel_clause = f"WHERE UPPER(sch1.name) {in_sql1} AND UPPER(sch2.name) {in_sql2}"
            rel_params.update(in_params1)
            rel_params.update(in_params2)

        fks_sql = f"""
            SELECT 
              sch1.name AS FK_SCHEMA, t1.name AS FK_TABLE, c1.name AS FK_COLUMN,
              sch2.name AS PK_SCHEMA, t2.name AS PK_TABLE, c2.name AS PK_COLUMN,
              fk.name  AS FK_NAME
            FROM sys.foreign_key_columns fkc
            INNER JOIN sys.tables t1 ON fkc.parent_object_id = t1.object_id
            INNER JOIN sys.schemas sch1 ON t1.schema_id = sch1.schema_id
            INNER JOIN sys.columns c1 ON fkc.parent_object_id = c1.object_id AND fkc.parent_column_id = c1.column_id
            INNER JOIN sys.tables t2 ON fkc.referenced_object_id = t2.object_id
            INNER JOIN sys.schemas sch2 ON t2.schema_id = sch2.schema_id
            INNER JOIN sys.columns c2 ON fkc.referenced_object_id = c2.object_id AND fkc.referenced_column_id = c2.column_id
            INNER JOIN sys.foreign_keys fk ON fk.object_id = fkc.constraint_object_id
            {rel_clause}
            ORDER BY sch1.name, t1.name, fk.name
        """
        fks = pd.read_sql(text(fks_sql), conn, params=rel_params)

        for _, r in fks.iterrows():
            meta["relationships"]["foreign_keys"].append({
                "fk_table": f"{r.FK_SCHEMA}.{r.FK_TABLE}",
                "fk_column": str(r.FK_COLUMN),
                "pk_table": f"{r.PK_SCHEMA}.{r.PK_TABLE}",
                "pk_column": str(r.PK_COLUMN),
                "name": str(r.FK_NAME)
            })

    Path(out_schema_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    Path(out_rels_path).write_text(json.dumps(meta["relationships"], indent=2), encoding="utf-8")

    # Extra visibility
    schemas_seen = sorted({t.split(".", 1)[0] for t in meta["tables"].keys()})
    print(f"[ingest] Schemas included in output: {schemas_seen}")
    print(f"Wrote schema for {len(meta['tables'])} objects → {out_schema_path}")


if __name__ == "__main__":
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_url = resolve_mssql_url(cfg)
    out_schema = cfg["inputs"]["schema_json"]
    out_rels = cfg["inputs"]["relationships_json"]
    schema_inc = _schema_include(cfg)
    run(db_url, out_schema, out_rels, schema_filter=schema_inc)
