# scripts/auto_docs_llm.py
"""
Fast documentation builder (table + column prompts), aligned to your last working version,
but wired to the v2 config paths and LLMClient. Uses plain-text completions (no JSON forcing)
with robust logging and heuristic fallbacks.

Inputs (from YAML):
  inputs.schema_json
  inputs.relationships_json
  inputs.profiles_dir
  outputs.docs_jsonl
  outputs.attribute_map_json (optional, if produced already)
  inputs.heuristic_attribute_map_json (optional fallback)

Config llm.* is reused (model, timeouts, TLS/proxy, logging).
Supports both limits.llm_tables_max and llm.table_limit for selecting a subset of tables.
"""

# --- project root import shim ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------

import os
import re
import json
import glob
from typing import Dict, List, Any

import pandas as pd

from libs.log_utils import setup_logger
from libs.llm_client import LLMClient

# -------------------------
# Optional helper: pick a subset of tables (kept local to avoid extra deps)
# -------------------------
def pick_tables(schema_json: Dict[str, Any], limit: int | None) -> List[str]:
    tables = sorted(list((schema_json.get("tables") or {}).keys()))
    if limit and isinstance(limit, int) and limit > 0:
        return tables[:limit]
    return tables

# -------------------------
# Tokenization helpers
# -------------------------
WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
def _tokens(name: str):
    s = name.split(".")[-1]
    s = re.sub(r"[_\\-]+"," ", s)
    return [p.lower() for p in WORD_RE.findall(s) if p]

def _clip_list(xs, k):
    xs = [x for x in xs if x]
    return xs[:k]

def _clip_text(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else (s[: max(0, n-1)] + "…")

# -------------------------
# Load artifacts
# -------------------------
def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _load_profiles_map(profiles_dir: str) -> Dict[str, dict]:
    out = {}
    for p in glob.glob(str(Path(profiles_dir) / "*.json")):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            if "table" in d:
                out[d["table"]] = d
        except Exception:
            pass
    return out

def _load_attribute_map(cfg: Dict) -> List[Dict]:
    """
    Use attribute_map (if available) to surface common aliases in docs.
    Priority: outputs.attribute_map_json -> inputs.heuristic_attribute_map_json -> []
    """
    # try LLM-produced first
    path1 = (cfg.get("outputs") or {}).get("attribute_map_json")
    if path1 and Path(path1).exists():
        try:
            return json.loads(Path(path1).read_text(encoding="utf-8"))
        except Exception:
            pass
    # try heuristic fallback
    path2 = (cfg.get("inputs") or {}).get("heuristic_attribute_map_json")
    if path2 and Path(path2).exists():
        try:
            return json.loads(Path(path2).read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

# -------------------------
# Alias helpers
# -------------------------
def top_aliases_for_target(amap_rows: List[Dict], target: str, k=5, min_conf=0.80,
                           allowed_sources=("exact","acronym","tokens","type")) -> List[str]:
    rows = [
        r for r in amap_rows
        if r.get("target") == target
        and float(r.get("confidence", 0)) >= float(min_conf)
        and (r.get("source", "tokens") in allowed_sources)
    ]
    rows.sort(key=lambda r: (-float(r.get("confidence",0)), r.get("alias","") or ""))

    uniq = []
    seen = set()
    for r in rows:
        a = (r.get("alias") or "").strip()
        if a and a.lower() not in seen:
            uniq.append(a); seen.add(a.lower())
        if len(uniq) >= k:
            break
    return uniq

# -------------------------
# Heuristic descriptions (table + column)
# -------------------------
def nlg_table_desc(full: str, columns: list, prof: dict | None, relationships: dict, amap_rows: List[Dict]):
    col_ct = len(columns)
    parts = _tokens(full)
    base_noun = " ".join(parts) if parts else full.split(".")[-1]
    row_ct = (prof or {}).get("row_count")

    hints = (prof or {}).get("hints") or {}
    pk_candidates = hints.get("pk_candidates", [])
    date_columns = hints.get("date_columns", [])
    enum_cols = [c for c, meta in (prof or {}).get("columns", {}).items() if meta.get("enum_candidate")]

    fks = (relationships or {}).get("foreign_keys", [])
    neighbors = set(); child_edges = 0; parent_edges = 0
    for fk in fks:
        if fk.get("fk_table")==full:
            neighbors.add(fk.get("pk_table")); child_edges += 1
        if fk.get("pk_table")==full:
            neighbors.add(fk.get("fk_table")); parent_edges += 1
    neighbor_list = sorted(list(neighbors))[:3]

    t_aliases = top_aliases_for_target(amap_rows, full, k=4, min_conf=0.80, allowed_sources=("exact","acronym","tokens"))

    bits = []
    if row_ct is not None:
        bits.append(f"**{full}** stores approximately {row_ct:,} rows with {col_ct} columns, capturing {base_noun}.")
    else:
        bits.append(f"**{full}** stores {col_ct} columns, capturing {base_noun}.")
    more = []
    if pk_candidates:
        more.append("key fields such as " + ", ".join(pk_candidates[:3]))
    if date_columns:
        more.append("date attributes like " + ", ".join(date_columns[:3]))
    if enum_cols:
        more.append("several enumerations/flags (e.g., " + ", ".join(enum_cols[:3]) + ")")
    if more:
        bits.append("It includes " + "; ".join(more) + ".")
    if child_edges or parent_edges:
        desc_rel = []
        if child_edges: desc_rel.append(f"{child_edges} outbound FK(s)")
        if parent_edges: desc_rel.append(f"{parent_edges} inbound FK(s)")
        if neighbor_list:
            desc_rel.append("links with " + ", ".join(neighbor_list))
        bits.append("Relationships: " + "; ".join(desc_rel) + ".")
    if t_aliases:
        bits.append("Common aliases: " + ", ".join(t_aliases) + ".")
    return " ".join(bits)

def nlg_col_desc(full: str, col: str, dtype: str, prof: dict | None, amap_rows: List[Dict]):
    cmeta = (prof or {}).get("columns", {}).get(col, {}) if prof else {}
    role = []
    if cmeta.get("pk_candidate"): role.append("primary key candidate")
    if cmeta.get("enum_candidate"): role.append("enumeration/flag-like")
    if cmeta.get("numeric"): role.append("numeric")
    if cmeta.get("datetime"): role.append("datetime")
    if cmeta.get("text"): role.append("text")
    role_s = f" ({', '.join(role)})" if role else ""

    nn, nulls = cmeta.get("non_null"), cmeta.get("nulls")
    distinct = cmeta.get("distinct")
    stats = []
    if nn is not None and nulls is not None:
        stats.append(f"non-null {nn}, nulls {nulls}")
    if distinct is not None:
        stats.append(f"distinct {distinct}")
    if cmeta.get("avg_len") is not None:
        stats.append(f"avg length {int(cmeta['avg_len'])}")
    stats_s = ("; " + "; ".join(stats)) if stats else ""

    bc = cmeta.get("booleanish_counts")
    boolish_s = ""
    if isinstance(bc, dict) and bc:
        tops = sorted(bc.items(), key=lambda kv: -kv[1])[:3]
        tops_s = ", ".join([f"{k}:{v}" for k,v in tops])
        boolish_s = f"; common values → {tops_s}"

    ex = cmeta.get("examples") or []
    ex_clean = []
    for v in ex[:3]:
        s = str(v)
        ex_clean.append(s if len(s)<=40 else s[:37]+"…")
    ex_s = f"; e.g., " + "; ".join(ex_clean) if ex_clean else ""

    key = f"{full}.{col}"
    c_aliases = top_aliases_for_target(amap_rows, key, k=4, min_conf=0.80, allowed_sources=("exact","acronym","tokens","type"))
    alias_s = f"; aliases: " + ", ".join(c_aliases) if c_aliases else ""

    dtype_s = dtype or ""
    return f"{dtype_s}{role_s}{stats_s}{boolish_s}{ex_s}{alias_s}".strip("; ").strip()

# -------------------------
# LLM prompt shapers (per-object)
# -------------------------
def format_table_llm_prompt(full: str, columns: list, prof: dict | None, relationships: dict, amap_rows: List[Dict], max_cols=20):
    cols = _clip_list(columns, max_cols)
    row_ct = (prof or {}).get("row_count")
    hints = (prof or {}).get("hints") or {}
    pk_candidates = _clip_list(hints.get("pk_candidates", []), 3)
    date_columns = _clip_list(hints.get("date_columns", []), 3)

    enum_cols = []
    for c, meta in (prof or {}).get("columns", {}).items():
        if meta.get("enum_candidate"):
            enum_cols.append(c)
    enum_cols = _clip_list(enum_cols, 6)

    fks = (relationships or {}).get("foreign_keys", [])
    neighbors = set(); child_edges = 0; parent_edges = 0
    for fk in fks:
        if fk.get("fk_table")==full:
            neighbors.add(fk.get("pk_table")); child_edges += 1
        if fk.get("pk_table")==full:
            neighbors.add(fk.get("fk_table")); parent_edges += 1
    neighbor_list = _clip_list(sorted(list(neighbors)), 5)

    t_aliases = _clip_list(top_aliases_for_target(amap_rows, full, k=6, min_conf=0.80, allowed_sources=("exact","acronym","tokens")), 6)

    parts = []
    parts.append(f"OBJECT: {full}")
    if row_ct is not None: parts.append(f"ROWS: ~{row_ct}")
    if cols: parts.append(f"COLUMNS: {', '.join(cols)}")
    if pk_candidates: parts.append(f"PK_CANDIDATES: {', '.join(pk_candidates)}")
    if date_columns: parts.append(f"DATE_COLUMNS: {', '.join(date_columns)}")
    if enum_cols: parts.append(f"ENUM_COLUMNS: {', '.join(enum_cols)}")
    if (child_edges or parent_edges) or neighbor_list:
        parts.append(f"RELATIONSHIPS: outbound={child_edges}, inbound={parent_edges}, neighbors={', '.join(neighbor_list)}")
    if t_aliases: parts.append(f"ALIASES: {', '.join(t_aliases)}")

    guidance = (
        "Write 1–2 sentences in plain business language describing what this table represents, "
        "what kinds of records it stores, and any notable keys/enumerations or relationships. "
        "Prefer clarity and avoid jargon."
    )
    return guidance + "\n" + "\n".join(parts)

def format_col_llm_prompt(full: str, col: str, dtype: str, prof: dict | None, amap_rows: List[Dict], max_examples=3):
    cmeta = (prof or {}).get("columns", {}).get(col, {}) if prof else {}
    role = []
    if cmeta.get("pk_candidate"): role.append("primary key candidate")
    if cmeta.get("enum_candidate"): role.append("enumeration/flag-like")
    if cmeta.get("numeric"): role.append("numeric")
    if cmeta.get("datetime"): role.append("datetime")
    if cmeta.get("text"): role.append("text")

    nn, nulls = cmeta.get("non_null"), cmeta.get("nulls")
    distinct = cmeta.get("distinct")
    avg_len = cmeta.get("avg_len")
    ex = [ _clip_text(x, 40) for x in (cmeta.get("examples") or [])[:max_examples] ]

    bc = (cmeta.get("booleanish_counts") or {})
    top_boolish = sorted(bc.items(), key=lambda kv: -kv[1])[:3]
    boolish_str = ", ".join([f"{k}:{v}" for k,v in top_boolish])

    key = f"{full}.{col}"
    c_aliases = top_aliases_for_target(amap_rows, key, k=6, min_conf=0.80, allowed_sources=("exact","acronym","tokens","type"))

    parts = []
    parts.append(f"COLUMN: {full}.{col}")
    if dtype: parts.append(f"TYPE: {dtype}")
    if role: parts.append(f"ROLE_HINTS: {', '.join(role)}")
    if nn is not None and nulls is not None: parts.append(f"NULLS: non_null={nn}, nulls={nulls}")
    if distinct is not None: parts.append(f"DISTINCT: {distinct}")
    if avg_len is not None: parts.append(f"AVG_LEN: {int(avg_len)}")
    if boolish_str: parts.append(f"BOOLEANISH: {boolish_str}")
    if ex: parts.append("EXAMPLES: " + "; ".join(ex))
    if c_aliases: parts.append("ALIASES: " + ", ".join(c_aliases))

    guidance = (
        "Write one concise sentence describing this column’s likely business meaning and how it is used. "
        "Prefer business language; avoid restating raw stats."
    )
    return guidance + "\n" + "\n".join(parts)

# -------------------------
# Runner
# -------------------------
def generate_docs(cfg: Dict):
    # Resolve logging directory: prefer llm.log_dir if provided; else default under outputs/run_id
    run_id = cfg.get("run_id", "run")
    out_base = Path(cfg.get("output_dir", "outputs")) / run_id
    default_logs = out_base / "artifacts" / "logs"
    (default_logs).mkdir(parents=True, exist_ok=True)

    log_dir = (cfg.get("llm") or {}).get("log_dir") or str(default_logs)
    log_level = (cfg.get("llm") or {}).get("log_level", "INFO")
    logger = setup_logger("auto_docs_llm", log_dir, log_level)

    # Inputs/outputs
    schema_path         = (cfg.get("inputs") or {}).get("schema_json", "artifacts/schema.json")
    relationships_path  = (cfg.get("inputs") or {}).get("relationships_json", "artifacts/relationships.json")
    profiles_dir        = (cfg.get("inputs") or {}).get("profiles_dir", "artifacts/profiles")
    out_docs_jsonl      = (cfg.get("outputs") or {}).get("docs_jsonl", "artifacts/schema_docs.jsonl")

    if not Path(schema_path).exists():
        raise FileNotFoundError(f"Schema JSON not found: {schema_path}")
    schema = _load_json(schema_path)

    relationships = _load_json(relationships_path) if Path(relationships_path).exists() else {"foreign_keys":[]}
    profiles_map  = _load_profiles_map(profiles_dir) if Path(profiles_dir).exists() else {}
    attribute_map = _load_attribute_map(cfg)

    # LLM client (only if enabled)
    llm_cfg = (cfg.get("llm") or {})
    llm_enabled = bool(llm_cfg.get("enabled", True))
    if llm_enabled:
        llm = LLMClient(
            model=llm_cfg.get("model", "gpt-4.1"),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            request_timeout_sec=int(llm_cfg.get("request_timeout_sec", 60)),
            connect_timeout_sec=int(llm_cfg.get("connect_timeout_sec", 30)),
            read_timeout_sec=int(llm_cfg.get("read_timeout_sec", 120)),
            max_retries=int(llm_cfg.get("max_retries", 4)),
            json_strict=False,  # <-- plain text for summaries
            api_key=llm_cfg.get("api_key"),
            verify_ssl=bool(llm_cfg.get("verify_ssl", True)),
            ca_bundle_path=llm_cfg.get("ca_bundle_path"),
            proxies=llm_cfg.get("proxies"),
            base_url=llm_cfg.get("base_url"),
            log_dir=log_dir,
            log_level=log_level,
            log_prompts=bool(llm_cfg.get("log_prompts", True)),
            redact_values=llm_cfg.get("redact_values") or [],
            max_input_bytes=int(llm_cfg.get("max_input_bytes", 180000)),
        )
    else:
        llm = None

    SYS_TABLE = (
        "You are a data catalog and business meaning writer. "
        "Write 1–2 concise sentences in plain language. "
        "Explain what the table represents and notable keys/enums/relationships. "
        "Avoid code and avoid restating raw numbers unless relevant."
    )
    SYS_COLUMN = (
        "You are a data catalog and business meaning writer. "
        "Write one concise sentence in plain language that explains the column’s likely business meaning and usage. "
        "Avoid restating raw stats; prefer what it means."
    )

    # Table selection: support new limits.llm_tables_max or older llm.table_limit
    limits = (cfg.get("limits") or {})
    table_limit = limits.get("llm_tables_max") or (cfg.get("llm") or {}).get("table_limit") or 0
    try:
        table_limit = int(table_limit)
    except Exception:
        table_limit = 0

    tables_all = pick_tables(schema, table_limit)

    logger.info(
        f"Docs generation: {len(tables_all)} tables "
        + (f"(limited by {('limits.llm_tables_max' if limits.get('llm_tables_max') else 'llm.table_limit')}={table_limit})"
           if table_limit > 0 else "(full)")
    )

    rows_out: List[Dict] = []
    for ti, full in enumerate(tables_all, start=1):
        tdef = schema["tables"][full]
        cols = list((tdef.get("columns") or {}).keys())
        prof = profiles_map.get(full)

        # --- Table doc ---
        table_desc = ""
        if llm_enabled:
            up = format_table_llm_prompt(full, cols, prof, relationships, attribute_map)
            logger.info(f"LLM table doc {ti}/{len(tables_all)}: {full}, prompt_bytes={len(up)}")
            try:
                txt = llm.text_completion(SYS_TABLE, up)
                if txt:
                    table_desc = "(LLM) " + txt.strip()
            except Exception as e:
                logger.warning(f"LLM table doc failed for {full}: {e}")

        if not table_desc:
            table_desc = nlg_table_desc(full, cols, prof, relationships, attribute_map)

        rows_out.append({"table": full, "description": table_desc, "columns": []})

        # --- Column docs ---
        for c in cols:
            dtype = (tdef.get("columns", {}).get(c, {}) or {}).get("data_type", "")
            col_desc = ""
            if llm_enabled:
                cp = format_col_llm_prompt(full, c, dtype, prof, attribute_map)
                try:
                    txt = llm.text_completion(SYS_COLUMN, cp)
                    if txt:
                        col_desc = "(LLM) " + txt.strip()
                except Exception as e:
                    # keep going; we'll fallback
                    logger.debug(f"LLM column doc failed for {full}.{c}: {e}")

            if not col_desc:
                col_desc = nlg_col_desc(full, c, dtype, prof, attribute_map)

            rows_out[-1]["columns"].append({
                "column": c,
                "description": col_desc,
                "type": dtype,
                "nullability": (tdef.get("columns", {}).get(c, {}) or {}).get("is_nullable", ""),
                "examples": (prof or {}).get("columns", {}).get(c, {}).get("examples", [])[:3]
            })

    # Persist JSONL (one table per line)
    Path(out_docs_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_docs_jsonl, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Wrote documentation JSONL: {out_docs_jsonl}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description="Generate schema documentation via LLM (per table/column).")
    ap.add_argument("-c", "--config", required=True, help="Path to config YAML")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    generate_docs(cfg)
