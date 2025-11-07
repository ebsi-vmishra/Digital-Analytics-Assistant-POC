from urllib.parse import quote_plus

def resolve_mssql_url(cfg: dict) -> str:
    dbc = (cfg.get("database") or {})
    raw = (dbc.get("odbc_connection_string") or "").strip()
    dsn = (dbc.get("dsn") or "").strip()
    if raw:
        return raw if raw.lower().startswith("mssql+pyodbc://") else f"mssql+pyodbc:///?odbc_connect={quote_plus(raw)}"
    if dsn:
        return f"mssql+pyodbc://@{dsn}"
    raise ValueError("No database connection info found. Provide database.odbc_connection_string or database.dsn in YAML.")
