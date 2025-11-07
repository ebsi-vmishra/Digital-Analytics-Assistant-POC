# scripts/quick_analysis.py
import os, json, yaml, math, pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Dict
from utils.db import resolve_mssql_url

BOOLEAN_TOKENS = {"y","n","yes","no","1","0","true","false","t","f"}


def _resolve_db_url(cfg: dict) -> str:
    """
    Accept either:
      1) database.odbc_connection_string as a *raw ODBC string*
         -> URL-encoded and wrapped as "mssql+pyodbc:///?odbc_connect=<ENCODED>"
      2) database.odbc_connection_string as a *full SQLAlchemy URL* (starts with "mssql+pyodbc://")
         -> used as-is
      3) database.dsn set (e.g., "ReportingDevCo") -> DSN URL "mssql+pyodbc://@ReportingDevCo"
    """
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
    # prefer explicit profile.schema_include; fallback to ingest.schema_include
    inc = (cfg.get("profile") or {}).get("schema_include")
    if inc is None:
        inc = (cfg.get("ingest") or {}).get("schema_include") or []
    if isinstance(inc, str):
        inc = [inc]
    return [s.strip() for s in inc if str(s).strip()]


def _safe_examples(series: pd.Series, k: int = 5):
    out = []
    s = series.dropna().head(k)
    for v in s:
        try:
            if isinstance(v, memoryview):
                v = bytes(v)
            if isinstance(v, (bytes, bytearray)):
                b = bytes(v)
                preview = b[:16].hex()
                out.append(f"<{len(b)} bytes: {preview}…>")
            else:
                out.append(str(v))
        except Exception:
            try:
                if isinstance(v, (bytes, bytearray)):
                    out.append(bytes(v).decode("utf-8", errors="replace"))
                else:
                    out.append("<unprintable>")
            except Exception:
                out.append("<unprintable>")
    return out


def _infer_boolish_counts(series: pd.Series, topk=5):
    try:
        s = series.dropna().astype(str).str.strip().str.lower()
        vals = s.value_counts().head(topk)
        uniq = set(vals.index)
        if len(uniq) <= 6 and (uniq.issubset(BOOLEAN_TOKENS) or len(uniq.intersection(BOOLEAN_TOKENS)) >= max(1, len(uniq) - 1)):
            return {k: int(v) for k, v in vals.items()}
    except Exception:
        pass
    return None


def _is_numeric(series: pd.Series):
    return pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)


def _is_datetime(series: pd.Series):
    return pd.api.types.is_datetime64_any_dtype(series)


def _len_stats(series: pd.Series):
    try:
        s = series.dropna().astype(str)
        lens = s.map(len)
        if lens.empty:
            return {"avg_len": None, "min_len": None, "max_len": None}
        return {
            "avg_len": float(lens.mean()),
            "min_len": int(lens.min()),
            "max_len": int(lens.max())
        }
    except Exception:
        return {"avg_len": None, "min_len": None, "max_len": None}


def _random_sample_sql(schema: str, name: str, n: int, use_newid=True):
    base = f"[{schema}].[{name}] WITH (NOLOCK)"
    if use_newid:
        return f"SELECT TOP {n} * FROM {base} ORDER BY NEWID()"
    return f"SELECT TOP {n} * FROM {base}"


def _top_values(series: pd.Series, max_unique=20, topk=10):
    try:
        s = series.dropna()
        if s.nunique(dropna=True) <= max_unique:
            vc = s.astype(str).value_counts().head(topk)
            return [{"value": k, "count": int(v)} for k, v in vc.items()]
    except Exception:
        pass
    return None


def _profile_column(s: pd.Series, dtype_hint: str):
    non_null = int(s.notna().sum())
    nulls = int(s.isna().sum())
    total = non_null + nulls
    null_pct = round(nulls / total * 100.0, 2) if total else None

    distinct = int(s.nunique(dropna=True))
    distinct_pct = round(distinct / non_null * 100.0, 2) if non_null else None

    col_meta = {
        "non_null": non_null, "nulls": nulls, "null_pct": null_pct,
        "distinct": distinct, "distinct_pct": distinct_pct,
        "examples": _safe_examples(s, k=5),
        "numeric": False, "datetime": False, "text": False
    }

    if _is_numeric(s):
        col_meta["numeric"] = True
        try:
            col_meta.update({
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)) if non_null > 1 else 0.0
            })
        except Exception:
            pass
    elif _is_datetime(s):
        col_meta["datetime"] = True
        try:
            col_meta.update({
                "min": str(pd.to_datetime(s.min())),
                "max": str(pd.to_datetime(s.max()))
            })
        except Exception:
            pass
    else:
        col_meta["text"] = True
        col_meta.update(_len_stats(s))
        boolish = _infer_boolish_counts(s)
        if boolish:
            col_meta["booleanish_counts"] = boolish

    if dtype_hint:
        col_meta["sql_type"] = dtype_hint

    pk_cand = (nulls == 0) and (distinct_pct is not None and distinct_pct >= 95.0)
    enum_cand = (distinct <= 20) and not col_meta["datetime"]
    col_meta["pk_candidate"] = bool(pk_cand)
    col_meta["enum_candidate"] = bool(enum_cand)

    tv = _top_values(s, max_unique=20, topk=10)
    if tv:
        col_meta["top_values"] = tv

    return col_meta


def profile_table(conn, full_name: str, sample_rows: int, dtype_map: dict):
    schema, name = full_name.split(".", 1)
    try:
        rc = pd.read_sql(text(f"SELECT COUNT_BIG(1) AS row_count FROM [{schema}].[{name}] WITH (NOLOCK)"), conn)
        row_count = int(rc.iloc[0, 0])
    except Exception:
        row_count = None

    n = max(1, min(int(sample_rows), 2000))
    try:
        sql = _random_sample_sql(schema, name, n, use_newid=True)
        sample = pd.read_sql(text(sql), conn)
    except Exception:
        try:
            sql = _random_sample_sql(schema, name, n, use_newid=False)
            sample = pd.read_sql(text(sql), conn)
        except Exception:
            sample = pd.DataFrame()

    # Coerce obvious datetime columns based on SQL type hints
    for c in sample.columns:
        if dtype_map.get(f"{schema}.{name}.{c}", "").startswith(("date", "datetime", "time")):
            try:
                sample[c] = pd.to_datetime(sample[c], errors="coerce")
            except Exception:
                pass

    cols = {}
    for col in sample.columns:
        try:
            dtype_hint = dtype_map.get(f"{schema}.{name}.{col}", "")
            cols[col] = _profile_column(sample[col], dtype_hint)
        except Exception as e:
            cols[col] = {"error": str(e)}

    pk_candidates = [c for c, m in cols.items() if m.get("pk_candidate")]
    date_columns = [c for c, m in cols.items() if m.get("datetime")]
    id_like = [c for c in sample.columns if c.lower() in ("id", "guid", "uuid") or c.lower().endswith("_id")]

    return {
        "table": full_name,
        "row_count": row_count,
        "columns": cols,
        "hints": {
            "pk_candidates": pk_candidates,
            "date_columns": date_columns,
            "id_like": id_like
        }
    }


def run(db_url: str, schema_json_path: str, profiles_dir: str, profiles_summary_csv: str, sample_rows: int, schema_filter: List[str] | None = None):
    # Ensure output directory
    Path(profiles_dir).mkdir(parents=True, exist_ok=True)

    # Load schema.json (already filtered by ingest if schema_include was set)
    schema_path = Path(schema_json_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"{schema_json_path} not found. Run ssms_ingest first.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    # Optionally re-filter here too (defensive in case ingest ran w/o filter)
    schema_filter = schema_filter or []
    tables_all = list((schema.get("tables") or {}).keys())
    if schema_filter:
        allowed = {s.lower() for s in schema_filter}
        tables_all = [t for t in tables_all if t.split(".", 1)[0].lower() in allowed]

    eng = create_engine(db_url)

    # Build dtype map
    dtype_map = {}
    for full, tdef in (schema.get("tables") or {}).items():
        # respect table selection
        if full not in tables_all:
            continue
        sch, tbl = full.split(".", 1)
        for c, meta in (tdef.get("columns") or {}).items():
            dtype_map[f"{sch}.{tbl}.{c}"] = (meta.get("data_type") or "").lower()

    flat_rows = []
    with eng.connect() as conn:
        for full in tables_all:
            out_base = Path(profiles_dir) / f"{full.replace('.', '__')}"
            try:
                prof = profile_table(conn, full, sample_rows, dtype_map)
                (out_base.with_suffix(".json")).write_text(json.dumps(prof, indent=2), encoding="utf-8")

                t = prof["table"]; rc = prof.get("row_count")
                for c, meta in (prof.get("columns") or {}).items():
                    flat_rows.append({
                        "table": t, "column": c, "row_count": rc,
                        "non_null": meta.get("non_null"), "nulls": meta.get("nulls"),
                        "null_pct": meta.get("null_pct"),
                        "distinct": meta.get("distinct"), "distinct_pct": meta.get("distinct_pct"),
                        "numeric": meta.get("numeric"), "datetime": meta.get("datetime"), "text": meta.get("text"),
                        "min": meta.get("min"), "max": meta.get("max"),
                        "mean": meta.get("mean"), "std": meta.get("std"),
                        "min_len": meta.get("min_len"), "max_len": meta.get("max_len"), "avg_len": meta.get("avg_len"),
                        "pk_candidate": meta.get("pk_candidate"), "enum_candidate": meta.get("enum_candidate"),
                        "sql_type": meta.get("sql_type")
                    })
            except Exception as e:
                (out_base.parent / (out_base.name + "_ERROR.txt")).write_text(str(e), encoding="utf-8")

    if flat_rows:
        df = pd.DataFrame(flat_rows)
        Path(profiles_summary_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(profiles_summary_csv, index=False)

    print(f"Profiles written to {profiles_dir}/*.json and {profiles_summary_csv}")


if __name__ == "__main__":
    # Use the same config file and keys as the pipeline & ingest
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_url = resolve_mssql_url(cfg)
    sample_rows = int(cfg.get("sample_rows_per_table", 500))

    schema_inc = _schema_include(cfg)

    run(
        db_url=db_url,
        schema_json_path=cfg["inputs"]["schema_json"],
        profiles_dir=cfg["inputs"]["profiles_dir"],
        profiles_summary_csv=cfg["inputs"]["profiles_summary_csv"],
        sample_rows=sample_rows,
        schema_filter=schema_inc
    )
