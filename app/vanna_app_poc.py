# app/vanna_app_poc.py
# PG-vector only (no VannaDefault / no Chroma)
# Adds:
#   - Batch CSV testing across all instances in manifest
#   - Robust prompt budgeting + DataFrame clamping for summaries
#   - Schema+concepts "cheat sheet" trained as a high-priority doc
#   - Robust SQL extraction + permissive execution gating
#   - NEW: InfoSchema "catalog grounding" docs to prevent generic table hallucination
#
# Postgres must have: CREATE EXTENSION IF NOT EXISTS vector;

import os
import argparse
import json
import yaml
import multiprocessing
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from collections import defaultdict
import csv
import re

from flask import Flask, request, jsonify

import pandas as pd

from vanna.pgvector import PG_VectorStore
from vanna.openai import OpenAI_Chat
from vanna.flask import VannaFlaskApp

DEFAULT_POC_CFG = Path("configs/default_config.yaml")
DEFAULT_BUNDLE_DIR = Path("artifacts/vanna_bundle")
DEFAULT_VANNA_RUNS = Path("artifacts/vanna_runs")  # only for manifest/logs


# ----------------- tiny IO helpers -----------------
def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    except Exception:
        return ""


def chunk_text(text: str, max_chars: int = 1800) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    parts = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        subs = block.split("\n# ")
        for i, s in enumerate(subs):
            s = s.strip()
            if not s:
                continue
            parts.append(s if i == 0 else "# " + s)

    chunks, buff, size = [], [], 0
    for para in parts:
        if size + len(para) + 2 > max_chars and buff:
            chunks.append("\n\n".join(buff))
            buff, size = [], 0
        buff.append(para)
        size += len(para) + 2

    if buff:
        chunks.append("\n\n".join(buff))
    return chunks


def chunk_csv(path: Path, rows_per_chunk: int = 250, include_header: bool = True) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return []
    header = lines[0] if include_header else None
    body = lines[1:] if include_header else lines
    out = []
    for i in range(0, len(body), rows_per_chunk):
        block = body[i:i + rows_per_chunk]
        out.append("\n".join(([header] + block) if header else block))
    return out


def chunk_json(path: Path, max_chars: int = 1800) -> List[str]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        txt = json.dumps(obj, indent=2)
    except Exception:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    return chunk_text(txt, max_chars=max_chars)


def chunk_jsonl(path: Path, max_chars: int = 1800) -> List[str]:
    if not path.exists():
        return []
    chunks = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            chunks.append(line)
        else:
            chunks.extend(chunk_text(line, max_chars=max_chars))
    return chunks


# ----------------- config + creds -----------------
def load_poc_cfg(cfg_path: Path) -> dict:
    if cfg_path.exists():
        try:
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    legacy = Path("config.yaml")
    if legacy.exists():
        try:
            return yaml.safe_load(legacy.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {}


def resolve_openai_from_poc(cfg: dict,
                           cli_key: Optional[str],
                           cli_base: Optional[str],
                           cli_model: Optional[str]) -> Tuple[str, Optional[str], str]:
    llm = (cfg.get("llm") or {})
    cfg_key = llm.get("api_key")
    cfg_base = llm.get("base_url")
    cfg_model = llm.get("model")

    env_key = os.getenv("OPENAI_API_KEY")
    env_base = os.getenv("OPENAI_BASE_URL")
    env_model = os.getenv("OPENAI_MODEL")

    key = cli_key or env_key or cfg_key
    base = cli_base or env_base or cfg_base
    model = cli_model or env_model or cfg_model or "gpt-4.1"

    if not key:
        raise RuntimeError("OpenAI API key not found (config llm.api_key or env OPENAI_API_KEY).")
    return key, base, model


# ----------- training artifact flags -----------#
def _resolve_train_flags(global_cfg: dict) -> dict:
    defaults = {
        "infoschema": True,
        "docs": True,
        "concepts": True,
        "synonyms": True,
        "schema": True,
        "relationships": True,
        "ddl": True,
        "profiles": True,
        "attr_map": True,
        "values": True,
    }
    overrides = global_cfg.get("train_artifacts") or {}
    flags = defaults.copy()
    for k, v in overrides.items():
        if k in flags:
            flags[k] = bool(v)
    return flags


def _filter_bundle_parts(parts: List[str], flags: dict) -> List[str]:
    out: List[str] = []
    for p in parts:
        key = p.lower()
        if flags.get(key, True):
            out.append(key)
    return out


# ----------- connection lookups -----------#
def get_conn_string(cfg: dict, kind: str, name: str) -> str:
    conn_map = (cfg.get("connections") or {}).get(kind) or {}
    s = conn_map.get(name)
    if not s:
        raise KeyError(f"Connection not found for connections.{kind}.{name} in config.")
    return s


# ----------- DF clamp helper -----------#
def _clamp_df(df: pd.DataFrame,
              max_rows: int,
              max_cols: int,
              max_chars_per_cell: int) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    df2 = df.copy()

    if max_rows and len(df2) > max_rows:
        df2 = df2.head(max_rows)
    if max_cols and df2.shape[1] > max_cols:
        keep_cols = list(df2.columns[:max_cols])
        df2 = df2[keep_cols]

    if max_chars_per_cell and max_chars_per_cell > 0:
        def _truncate(x):
            s = str(x)
            return s if len(s) <= max_chars_per_cell else (s[:max_chars_per_cell] + "…")
        df2 = df2.applymap(_truncate)

    return df2


# ----------- SQL extraction + permissive gating -----------#
def extract_sql_candidate(text: str) -> str:
    """
    Robustly extract the first SQL statement from an LLM response.

    Handles:
      - ```sql ... ``` fences
      - ``` ... ``` fences
      - leading prose + SQL
      - mixed content (returns from first SELECT/WITH until end)
    """
    if not isinstance(text, str):
        return ""

    s = text.strip()
    if not s:
        return ""

    # Normalize common fence forms
    # e.g. ```sql\nSELECT ...\n``` or ```\nSELECT...\n```
    fence = re.search(r"```(?:sql)?\s*(.*?)\s*```", s, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        s = fence.group(1).strip()

    # Remove stray backticks (single-line code formatting)
    s = s.strip("`").strip()

    # Find first SQL start
    m = re.search(r"(?is)\b(select|with)\b", s)
    if m:
        return s[m.start():].strip()

    # If nothing found, return stripped (caller will decide)
    return s


def _looks_like_sql(sql_text: str) -> bool:
    """
    PERMISSIVE: If we have a SELECT/WITH anywhere, treat as SQL.
    (This is closer to Nov-6 behavior and avoids blocking execution.)
    """
    if not isinstance(sql_text, str):
        return False
    s = sql_text.strip()
    if not s:
        return False
    low = s.lower()
    return ("select " in low) or low.startswith("select") or ("with " in low) or low.startswith("with")


# ----------- Vanna wrappers (with prompt + summary budgeting) -----------#
class CustomVanna(PG_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        config = config or {}
        self.max_input_bytes = int(config.get("max_input_bytes", 120_000))
        self.summary_max_rows = int(config.get("summary_max_rows", 200))
        self.summary_max_cols = int(config.get("summary_max_cols", 40))
        self.summary_max_chars_per_cell = int(config.get("summary_max_chars_per_cell", 500))
        if "temperature" in config:
            self.temperature = float(config["temperature"])
        PG_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

    def submit_prompt(self, message_log, **kwargs):
        msgs = []
        for m in (message_log or []):
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)

            per_msg_cap = max(4000, self.max_input_bytes // 2)
            if len(content) > per_msg_cap:
                content = content[:per_msg_cap] + "\n...[truncated]"
            msgs.append({"role": role, "content": content})

        def total_size(mm): return sum(len(x["content"]) for x in mm)

        head = msgs[:1]
        tail = msgs[1:]
        while (len(head) + len(tail) > 2) and (total_size(head + tail) > self.max_input_bytes):
            tail.pop(0)

        pruned = head + tail
        while total_size(pruned) > self.max_input_bytes and len(pruned) > 1:
            if len(pruned) == 2:
                pruned[0]["content"] = pruned[0]["content"][: self.max_input_bytes // 3] + "\n...[truncated]"
                break
            else:
                pruned.pop(1)

        return OpenAI_Chat.submit_prompt(self, pruned, **kwargs)

    def generate_summary(self, question: str, df=None, **kwargs):
        try:
            df = _clamp_df(
                df,
                max_rows=self.summary_max_rows,
                max_cols=self.summary_max_cols,
                max_chars_per_cell=self.summary_max_chars_per_cell,
            )
        except Exception:
            pass
        from vanna.base.base import Base
        return Base.generate_summary(self, question=question, df=df, **kwargs)


def init_vanna_pg(openai_key: str,
                  openai_model: str,
                  pg_conn_string: str,
                  max_input_bytes: Optional[int] = None,
                  temperature: Optional[float] = None,
                  summary_max_rows: Optional[int] = None,
                  summary_max_cols: Optional[int] = None,
                  summary_max_chars_per_cell: Optional[int] = None) -> CustomVanna:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["OPENAI_MODEL"] = openai_model
    cfg = {
        "api_key": openai_key,
        "openai_model": openai_model,
        "connection_string": pg_conn_string,
    }
    if max_input_bytes is not None:
        cfg["max_input_bytes"] = int(max_input_bytes)
    if temperature is not None:
        cfg["temperature"] = float(temperature)
    if summary_max_rows is not None:
        cfg["summary_max_rows"] = int(summary_max_rows)
    if summary_max_cols is not None:
        cfg["summary_max_cols"] = int(summary_max_cols)
    if summary_max_chars_per_cell is not None:
        cfg["summary_max_chars_per_cell"] = int(summary_max_chars_per_cell)
    return CustomVanna(config=cfg)


# ----------- training helpers -----------#
def _bundle_files(bundle_dir: Path) -> Dict[str, Path]:
    return {
        "ddl": bundle_dir / "ddl.sql",
        "docs": bundle_dir / "documentation.md",
        "profiles": bundle_dir / "profiles_summary.csv",
        "schema_json": bundle_dir / "schema.json",
        "relationships_json": bundle_dir / "relationships.json",
        "synonyms_json": bundle_dir / "synonyms.json",
        "attribute_map_json": bundle_dir / "attribute_map.json",
        "concept_catalog": bundle_dir / "concept_catalog.csv",
        "concept_alias": bundle_dir / "concept_alias.csv",
        "concept_attributes": bundle_dir / "concept_attributes.csv",
        "concept_rules": bundle_dir / "concept_rules.csv",
        "concept_layer_llm": bundle_dir / "concept_layer_llm.json",
        "value_aliases_llm": bundle_dir / "value_aliases_llm.json",
        "value_domains_llm": bundle_dir / "value_domains_llm.csv",
        "cheatsheet": bundle_dir / "schema_concepts_cheatsheet.txt",
    }


def _train_doc(vn: CustomVanna, path: Path, mode: str):
    if not path.exists():
        return
    if mode == "text":
        for c in chunk_text(_read_text(path), 1800):
            if c.strip():
                vn.train(documentation=c)
    elif mode == "csv":
        for c in chunk_csv(path, 250, True):
            if c.strip():
                vn.train(documentation=c)
    elif mode == "json":
        for c in chunk_json(path, 1800):
            if c.strip():
                vn.train(documentation=c)


# ----------- schema + concepts cheat sheet generator -----------#
def build_schema_concepts_cheatsheet(bundle_dir: Path) -> Path:
    schema_path = bundle_dir / "schema.json"
    attrs_path = bundle_dir / "concept_attributes.csv"
    cat_path = bundle_dir / "concept_catalog.csv"
    out_path = bundle_dir / "schema_concepts_cheatsheet.txt"

    if not schema_path.exists():
        out_path.write_text("", encoding="utf-8")
        return out_path

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception:
        schema = {}

    concept_name_by_id: Dict[str, str] = {}
    if cat_path.exists():
        try:
            with cat_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = (row.get("concept_id") or "").strip()
                    cname = (row.get("concept_name") or "").strip()
                    if cid and cname:
                        concept_name_by_id[cid] = cname
        except Exception:
            pass

    table_col_concepts: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    if attrs_path.exists():
        try:
            with attrs_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tbl = (row.get("table") or "").strip()
                    col = (row.get("column") or "").strip()
                    cid = (row.get("concept_id") or "").strip()
                    role = (row.get("role") or "").strip() or "attr"
                    if tbl and col and cid:
                        table_col_concepts[tbl][col].append((cid, role))
        except Exception:
            pass

    lines: List[str] = []
    lines.append("# SCHEMA / CONCEPT CHEAT SHEET (AUTO-GENERATED)")
    lines.append("# Dense mapping of physical names to concepts for SQL grounding.\n")

    tables = schema.get("tables") or {}
    for full_name, tdef in sorted(tables.items()):
        # full_name expected like reporting.Employee
        if "." in full_name:
            sch, tbl = full_name.split(".", 1)
        else:
            sch, tbl = "dbo", full_name

        physical_label = f"[{sch}].[{tbl}]"
        t_desc = (tdef.get("description") or "").strip()
        lines.append(f"TABLE {physical_label}  -- {t_desc or full_name}")

        cols = (tdef.get("columns") or {})
        for col_name, cmeta in sorted(cols.items()):
            data_type = (cmeta.get("data_type") or "").strip()
            nullable = str(cmeta.get("is_nullable") or "").strip()
            key = (cmeta.get("constraint_type") or "").strip()

            concept_info = table_col_concepts.get(full_name, {}).get(col_name, [])
            if concept_info:
                concept_strs = []
                for cid, role in concept_info:
                    cname = concept_name_by_id.get(cid, "")
                    concept_strs.append(f"{cid}({cname},role={role})" if cname else f"{cid}(role={role})")
                concept_block = "; ".join(concept_strs)
            else:
                concept_block = ""

            lines.append(
                f"  COLUMN [{col_name}] type={data_type or 'UNKNOWN'} nullable={nullable or 'UNKNOWN'} "
                f"key={key or ''} concepts={concept_block}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def train_from_bundle_select(vn: CustomVanna, bundle_dir: Path, parts: Set[str]):
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    f = _bundle_files(bundle_dir)

    # Keep cheat sheet (helps grounding) BUT only if schema.json exists
    cheat_path = build_schema_concepts_cheatsheet(bundle_dir)
    _train_doc(vn, cheat_path, "text")

    parts_lower = {p.lower() for p in parts}

    if "ddl" in parts_lower:
        _train_doc(vn, f["ddl"], "text")
    if "docs" in parts_lower:
        _train_doc(vn, f["docs"], "text")
    if "profiles" in parts_lower:
        _train_doc(vn, f["profiles"], "csv")
    if "schema" in parts_lower:
        _train_doc(vn, f["schema_json"], "json")
    if "relationships" in parts_lower:
        _train_doc(vn, f["relationships_json"], "json")
    if "synonyms" in parts_lower:
        _train_doc(vn, f["synonyms_json"], "json")
    if "attr_map" in parts_lower:
        _train_doc(vn, f["attribute_map_json"], "json")
    if "concepts" in parts_lower:
        for name in ["concept_catalog", "concept_alias", "concept_attributes", "concept_rules"]:
            _train_doc(vn, f[name], "csv")
    if "values" in parts_lower:
        _train_doc(vn, f["value_aliases_llm"], "json")
        _train_doc(vn, f["value_domains_llm"], "csv")

    if "docs" in parts_lower:
        jpath = Path("artifacts") / "schema_docs.jsonl"
        if jpath.exists():
            for c in chunk_jsonl(jpath, 1800):
                if c.strip():
                    vn.train(documentation=c)

    print(f"✅ Trained from bundle parts {sorted(parts_lower)} at {bundle_dir.resolve()}")


def train_from_bundle_chunked(vn: CustomVanna, bundle_dir: Path):
    all_parts = {
        "ddl", "docs", "profiles", "schema", "relationships",
        "concepts", "synonyms", "attr_map", "values"
    }
    train_from_bundle_select(vn, bundle_dir=bundle_dir, parts=all_parts)


# ----------- NEW: InfoSchema catalog grounding -----------#
def _train_infoschema_catalog_grounding(vn: CustomVanna, df_cols: pd.DataFrame, schema_filter: Optional[str] = None):
    """
    Adds compact documentation that explicitly lists schema-qualified table names and columns.
    This is the single most effective fix for the 'Employees/Benefits' hallucination pattern.
    """
    if df_cols is None or not isinstance(df_cols, pd.DataFrame) or df_cols.empty:
        return

    need = {"TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME"}
    if not need.issubset(set(df_cols.columns)):
        return

    # Optional schema filtering (defensive)
    df2 = df_cols.copy()
    if schema_filter:
        df2 = df2[df2["TABLE_SCHEMA"].astype(str).str.lower() == str(schema_filter).lower()]

    # Group and build compact docs
    grouped = df2.groupby(["TABLE_SCHEMA", "TABLE_NAME"])["COLUMN_NAME"].apply(list).reset_index()

    lines = []
    lines.append("# INFORMATION_SCHEMA CATALOG (AUTO-GENERATED)")
    lines.append("# Use ONLY these schema-qualified tables/columns.\n")

    for _, row in grouped.iterrows():
        sch = str(row["TABLE_SCHEMA"]).strip()
        tbl = str(row["TABLE_NAME"]).strip()
        cols = [str(c).strip() for c in (row["COLUMN_NAME"] or []) if str(c).strip()]
        cols = sorted(set(cols))

        # schema-qualified and bracketed forms
        fq_dot = f"{sch}.{tbl}"
        fq_br = f"[{sch}].[{tbl}]"

        lines.append(f"TABLE {fq_dot}  (also {fq_br})")
        if cols:
            # keep it compact
            lines.append("COLUMNS: " + ", ".join(cols[:120]))
            if len(cols) > 120:
                lines.append(f"... (+{len(cols) - 120} more columns)")
        lines.append("")

    doc = "\n".join(lines).strip()
    for c in chunk_text(doc, max_chars=1800):
        if c.strip():
            vn.train(documentation=c)

    print("✅ Trained InfoSchema catalog grounding docs.")


def train_from_info_schema_mssql(vn: CustomVanna, mssql_conn_string: str, schema_filter: Optional[str] = None):
    vn.connect_to_mssql(mssql_conn_string)
    ok = vn.run_sql("SELECT 1 AS ok")
    assert int(ok.iloc[0]["ok"]) == 1, "MSSQL connectivity failed"

    if schema_filter:
        df = vn.run_sql(f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{schema_filter}'")
    else:
        df = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

    plan = vn.get_training_plan_generic(df)
    vn.train(plan=plan)

    # NEW: Add catalog grounding docs
    _train_infoschema_catalog_grounding(vn, df_cols=df, schema_filter=schema_filter)

    print(f"✅ Trained from INFORMATION_SCHEMA (schema={schema_filter or 'ALL'})")


# ----------- instance orchestration -----------#
def _vn_from_cfg(cfg: dict, pg_store_key: str, openai_key: str, openai_model: str) -> CustomVanna:
    pg_conn = get_conn_string(cfg, "postgres", pg_store_key)
    llm_cfg = (cfg.get("llm") or {})
    return init_vanna_pg(
        openai_key, openai_model, pg_conn,
        max_input_bytes=llm_cfg.get("max_input_bytes"),
        temperature=llm_cfg.get("temperature"),
        summary_max_rows=llm_cfg.get("summary_max_rows"),
        summary_max_cols=llm_cfg.get("summary_max_cols"),
        summary_max_chars_per_cell=llm_cfg.get("summary_max_chars_per_cell"),
    )


def train_instance(instance_cfg: dict, global_cfg: dict, openai_key: str, openai_model: str) -> dict:
    method = instance_cfg.get("method", "bundle")
    bundle_dir = Path(
        instance_cfg.get("bundle_dir")
        or global_cfg.get("outputs", {}).get("vanna_bundle_dir")
        or DEFAULT_BUNDLE_DIR
    )

    pg_store_key = instance_cfg.get("pg_store") or (
        global_cfg.get("vanna", {}).get("sql", {}).get("pg_store_name")
    )
    if not pg_store_key:
        raise RuntimeError(f"Instance '{instance_cfg.get('id')}' requires 'pg_store' in config.")
    vn = _vn_from_cfg(global_cfg, pg_store_key, openai_key, openai_model)

    mssql_key = instance_cfg.get("mssql_db") or (
        global_cfg.get("vanna", {}).get("sql", {}).get("mssql_db_name")
    )
    schema_filter = instance_cfg.get("schema")
    train_flags = _resolve_train_flags(global_cfg)

    bundle_parts_for_rec: Optional[List[str]] = None

    if method == "bundle":
        all_parts = [
            "ddl", "docs", "profiles", "schema",
            "relationships", "concepts", "synonyms",
            "attr_map", "values",
        ]
        effective_parts = _filter_bundle_parts(all_parts, train_flags)
        print(f"[{instance_cfg['id']}] bundle effective_parts={effective_parts}")
        if effective_parts:
            train_from_bundle_select(vn, bundle_dir=bundle_dir, parts=set(effective_parts))
        bundle_parts_for_rec = effective_parts

    elif method == "bundle_select":
        requested_parts = [p.lower() for p in (instance_cfg.get("bundle_parts") or [])]
        if not requested_parts:
            raise ValueError("bundle_select requires 'bundle_parts': e.g. ['docs','concepts']")
        effective_parts = _filter_bundle_parts(requested_parts, train_flags)
        print(f"[{instance_cfg['id']}] bundle_select requested={requested_parts} effective={effective_parts}")
        if effective_parts:
            train_from_bundle_select(vn, bundle_dir=bundle_dir, parts=set(effective_parts))
        bundle_parts_for_rec = effective_parts

    elif method == "info_schema":
        if train_flags.get("infoschema", True):
            mssql_conn = get_conn_string(global_cfg, "mssql", mssql_key)
            train_from_info_schema_mssql(vn, mssql_conn_string=mssql_conn, schema_filter=schema_filter)
        else:
            print(f"[{instance_cfg['id']}] Skipping info_schema (train_artifacts.infoschema = false)")

    elif method == "bundle_plus_info":
        all_parts = [
            "ddl", "docs", "profiles", "schema",
            "relationships", "concepts", "synonyms",
            "attr_map", "values",
        ]
        effective_parts = _filter_bundle_parts(all_parts, train_flags)
        print(f"[{instance_cfg['id']}] bundle_plus_info effective_parts={effective_parts}")
        if effective_parts:
            train_from_bundle_select(vn, bundle_dir=bundle_dir, parts=set(effective_parts))
        bundle_parts_for_rec = effective_parts

        if train_flags.get("infoschema", True):
            mssql_conn = get_conn_string(global_cfg, "mssql", mssql_key)
            train_from_info_schema_mssql(vn, mssql_conn_string=mssql_conn, schema_filter=schema_filter)

    elif method == "bundle_select_plus_info":
        requested_parts = [p.lower() for p in (instance_cfg.get("bundle_parts") or [])]
        if not requested_parts:
            raise ValueError("bundle_select_plus_info requires 'bundle_parts': e.g. ['docs','synonyms']")
        effective_parts = _filter_bundle_parts(requested_parts, train_flags)
        print(f"[{instance_cfg['id']}] bundle_select_plus_info requested={requested_parts} effective={effective_parts}")
        if effective_parts:
            train_from_bundle_select(vn, bundle_dir=bundle_dir, parts=set(effective_parts))
        bundle_parts_for_rec = effective_parts

        if train_flags.get("infoschema", True):
            mssql_conn = get_conn_string(global_cfg, "mssql", mssql_key)
            train_from_info_schema_mssql(vn, mssql_conn_string=mssql_conn, schema_filter=schema_filter)

    else:
        raise ValueError(f"Unknown training method: {method}")

    # smoke
    try:
        q = "List full-time employees and exclude smokers"
        raw = vn.generate_sql(q)
        sql = extract_sql_candidate(raw)
        print(f"[{instance_cfg['id']}] Example SQL:\n{sql}\n")
    except Exception as e:
        print(f"[{instance_cfg['id']}] SQL generation note: {repr(e)}")

    rec = {
        "id": instance_cfg["id"],
        "method": method,
        "bundle_dir": str(bundle_dir),
        "pg_store": pg_store_key,
    }
    if "port" in instance_cfg:
        rec["port"] = int(instance_cfg["port"])
    if mssql_key:
        rec["mssql_db"] = mssql_key
    if schema_filter:
        rec["schema"] = schema_filter
    if bundle_parts_for_rec:
        rec["bundle_parts"] = sorted(bundle_parts_for_rec)

    return rec


def _build_vn_pg_from_rec(rec: dict, cfg: dict, openai_key: str, openai_model: str) -> CustomVanna:
    pg_store_key = rec.get("pg_store") or (cfg.get("vanna", {}).get("sql", {}).get("pg_store_name"))
    vn = _vn_from_cfg(cfg, pg_store_key, openai_key, openai_model)
    if rec.get("mssql_db"):
        vn.connect_to_mssql(get_conn_string(cfg, "mssql", rec["mssql_db"]))
    return vn


# ----------- serving -----------#
def _serve_instance(rec: dict, cfg: dict, openai_key: str, openai_model: str, host: str):
    pg_store_key = rec.get("pg_store") or (cfg.get("vanna", {}).get("sql", {}).get("pg_store_name"))
    if not pg_store_key:
        raise RuntimeError(f"Instance '{rec.get('id')}' missing 'pg_store' and no default in vanna.sql.pg_store_name.")
    vn = _vn_from_cfg(cfg, pg_store_key, openai_key, openai_model)

    if rec.get("mssql_db"):
        mssql_conn = get_conn_string(cfg, "mssql", rec["mssql_db"])
        vn.connect_to_mssql(mssql_conn)

    port = int(rec.get("port") or 0)
    if not port:
        raise RuntimeError(f"Instance '{rec['id']}' has no 'port' configured for split-serve")
    app = VannaFlaskApp(vn, allow_llm_to_see_data=True)
    print(f"[serve:{rec['id']}] UI → http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


def start_multi_flask_one_port(manifest_path: Path, cfg: dict, host: str, port: int):
    app = Flask(__name__)
    instances: Dict[str, CustomVanna] = {}
    key, _, model = resolve_openai_from_poc(cfg, None, None, None)
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"[manifest] Loaded: {manifest_path} (instances={len(man.get('instances', []))})")
    for rec in man.get("instances", []):
        instances[rec["id"]] = _build_vn_pg_from_rec(rec, cfg, key, model)

    @app.get("/instances")
    def list_instances():
        return jsonify({"instances": list(instances.keys())})

    @app.post("/ask")
    def ask_one():
        data = request.get_json(force=True)
        q = (data.get("question") or "").strip()
        inst = data.get("instance")
        if not q or inst not in instances:
            return jsonify({"error": "Missing question or invalid instance"}), 400
        vn = instances[inst]
        raw = vn.generate_sql(q)
        sql = extract_sql_candidate(raw)
        result = None
        exec_error = None
        if data.get("execute"):
            if _looks_like_sql(sql):
                try:
                    result = vn.run_sql(sql)
                except Exception as e:
                    exec_error = str(e)
                    result = {"execution_error": exec_error}
            else:
                exec_error = "LLM_NON_SQL_RESPONSE"
                result = {"execution_error": exec_error, "raw_text": raw}
        return jsonify({"instance": inst, "sql": sql, "raw_sql": raw, "result": result})

    print(f"[multi-serve(one-port)] http://{host}:{port} | instances={list(instances.keys())}")
    app.run(host=host, port=port, debug=False)


def start_multi_flask_split_ports(manifest_path: Path, cfg: dict, host: str, base_port: Optional[int] = None):
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = man.get("instances", [])
    print(f"[manifest] Loaded: {manifest_path} (instances={len(instances)})")

    yaml_instances = ((cfg.get("vanna") or {}).get("instances") or [])
    yaml_by_id = {i.get("id"): i for i in yaml_instances if i.get("id")}

    next_port = base_port if base_port else None
    procs: List[multiprocessing.Process] = []

    openai_key, _, openai_model = resolve_openai_from_poc(cfg, None, None, None)

    for rec in instances:
        port = rec.get("port")
        if not port:
            y = yaml_by_id.get(rec.get("id"))
            if y and y.get("port"):
                port = int(y["port"])
                rec["port"] = port
        if not port and next_port is not None:
            rec["port"] = next_port
            port = next_port
            next_port += 1
        if not port:
            raise RuntimeError(
                f"Manifest instance '{rec.get('id')}' missing 'port', and no base_port provided.\n"
                f"→ Add 'port' in YAML or pass --port to auto-assign."
            )

        p = multiprocessing.Process(
            target=_serve_instance,
            args=(rec, cfg, openai_key, openai_model, host),
            daemon=False
        )
        p.start()
        procs.append(p)
        print(f"[spawned] {rec['id']} on port {rec['port']}")

    for p in procs:
        p.join()


# ----------- Batch CSV across instances -----------#
def run_batch_questions(
    manifest_path: Path,
    cfg: dict,
    input_csv: Path,
    output_csv: Path,
    do_execute: bool = True,
):
    """
    Reads input_csv with a column 'question' and runs it across all instances in the manifest.
    Writes long-form CSV with:
      question,instance,method,bundle_parts,train_artifacts,
      sql,raw_sql,executed,execution_error,rowcount,sample_json
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Batch input CSV not found: {input_csv}")
    df_in = pd.read_csv(input_csv)
    if "question" not in df_in.columns:
        raise ValueError("Input CSV must have a 'question' column")

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = man.get("instances", [])
    if not instances:
        raise RuntimeError("No instances in manifest for batch run")

    key, _, model = resolve_openai_from_poc(cfg, None, None, None)

    train_artifacts_cfg = cfg.get("train_artifacts") or {}
    train_artifacts_str = json.dumps(train_artifacts_cfg, sort_keys=True)

    rows = []
    for rec in instances:
        inst_id = rec.get("id")
        method = rec.get("method", "")
        bundle_parts = rec.get("bundle_parts") or []
        bundle_parts_str = ",".join(sorted(map(str, bundle_parts))) if bundle_parts else ""

        vn = _build_vn_pg_from_rec(rec, cfg, key, model)

        for q in df_in["question"].astype(str).tolist():
            q2 = q.strip()
            if not q2:
                continue

            raw_sql = ""
            sql = ""
            executed = False
            rowcount = None
            sample_json = None
            exec_err = None

            try:
                raw_sql = vn.generate_sql(q2)
                sql = extract_sql_candidate(raw_sql)

                if do_execute and sql:
                    if _looks_like_sql(sql):
                        try:
                            res = vn.run_sql(sql)
                            if isinstance(res, pd.DataFrame):
                                executed = True
                                rowcount = int(len(res))
                                sample_json = res.head(5).to_json(orient="records")
                            else:
                                executed = True
                                sample_json = json.dumps(res)[:2000]
                        except Exception as ex:
                            exec_err = str(ex)
                    else:
                        exec_err = "LLM_NON_SQL_RESPONSE"

            except Exception as gen_ex:
                exec_err = f"SQL_GEN_ERROR: {gen_ex}"

            rows.append({
                "question": q2,
                "instance": inst_id,
                "method": method,
                "bundle_parts": bundle_parts_str,
                "train_artifacts": train_artifacts_str,
                "sql": sql,
                "raw_sql": raw_sql,
                "executed": executed,
                "execution_error": exec_err,
                "rowcount": rowcount,
                "sample_json": sample_json,
            })

    df_out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[batch] Wrote results → {output_csv}")


# ----------------- CLI -----------------#
def parse_args():
    p = argparse.ArgumentParser(description="Vanna (PG-only) trainer/runner.")
    p.add_argument("--train-bundle", action="store_true")
    p.add_argument("--train-info", action="store_true")
    p.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    p.add_argument("--mssql-db", default=None, help="Key under connections.mssql")
    p.add_argument("--schema", default=None)

    p.add_argument("--train-instances", action="store_true")
    p.add_argument("--manifest", default=str(DEFAULT_VANNA_RUNS / "manifest.json"))

    p.add_argument("--serve", action="store_true")
    p.add_argument("--serve-multi", action="store_true", help="Serve all instances behind one port")
    p.add_argument("--serve-multi-split", action="store_true",
                   help="Serve each instance on its own port (requires vanna.instances[*].port or --port as base)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8084, help="For single-port serve OR base port for split-serve")

    p.add_argument("--batch-csv", default=None, help="Path to input CSV with a 'question' column")
    p.add_argument("--batch-out", default="artifacts/vanna_runs/batch_results.csv", help="Path to write batch output CSV")
    p.add_argument("--no-exec", action="store_true", help="Do not execute generated SQL during batch run")

    p.add_argument("--openai-key", default=None)
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--openai-model", default=None)

    p.add_argument("-c", "--config", default=str(DEFAULT_POC_CFG))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_poc_cfg(Path(args.config))

    key, base, model = resolve_openai_from_poc(cfg, args.openai_key, args.openai_base_url, args.openai_model)
    if base:
        os.environ["OPENAI_BASE_URL"] = base

    if args.batch_csv:
        mp = Path(args.manifest)
        if not mp.exists():
            raise FileNotFoundError(f"Manifest not found: {mp}. Run with --train-instances first.")
        run_batch_questions(
            manifest_path=mp,
            cfg=cfg,
            input_csv=Path(args.batch_csv),
            output_csv=Path(args.batch_out),
            do_execute=(not args.no_exec),
        )
        return

    if args.train_bundle or args.train_info or args.serve:
        pg_default_key = (cfg.get("vanna", {}).get("sql", {}).get("pg_store_name"))
        if not pg_default_key:
            raise RuntimeError("Missing vanna.sql.pg_store_name in config for PG vector store.")
        llm_cfg = (cfg.get("llm") or {})
        vn = init_vanna_pg(
            key, model, get_conn_string(cfg, "postgres", pg_default_key),
            max_input_bytes=llm_cfg.get("max_input_bytes"),
            temperature=llm_cfg.get("temperature"),
            summary_max_rows=llm_cfg.get("summary_max_rows"),
            summary_max_cols=llm_cfg.get("summary_max_cols"),
            summary_max_chars_per_cell=llm_cfg.get("summary_max_chars_per_cell"),
        )

        bundle_dir = Path(args.bundle_dir)
        if args.train_bundle:
            train_from_bundle_chunked(vn, bundle_dir=bundle_dir)
        if args.train_info:
            mssql_key = args.mssql_db or (cfg.get("vanna", {}).get("sql", {}).get("mssql_db_name"))
            mssql_conn = get_conn_string(cfg, "mssql", mssql_key)
            train_from_info_schema_mssql(vn, mssql_conn_string=mssql_conn, schema_filter=args.schema)

        if args.serve:
            mssql_key = args.mssql_db or (cfg.get("vanna", {}).get("sql", {}).get("mssql_db_name"))
            mssql_conn = get_conn_string(cfg, "mssql", mssql_key)
            vn.connect_to_mssql(mssql_conn)
            app = VannaFlaskApp(vn, allow_llm_to_see_data=True)
            print(f"\n🚀 http://{args.host}:{args.port}")
            app.run(host=args.host, port=args.port, debug=True)
        else:
            print("\nDone (no server).")
        return

    if args.train_instances:
        instances = (cfg.get("vanna") or {}).get("instances") or []
        if not instances:
            raise RuntimeError("No vanna.instances configured in configs/default_config.yaml")
        manifest = {"instances": []}
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        for inst in instances:
            rec = train_instance(inst, cfg, key, model)
            manifest["instances"].append(rec)
        Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[vanna] Trained {len(manifest['instances'])} instance(s). Manifest -> {args.manifest}")

    if args.serve_multi:
        mp = Path(args.manifest)
        if not mp.exists():
            raise FileNotFoundError(f"Manifest not found: {mp}. Run with --train-instances first.")
        start_multi_flask_one_port(mp, cfg, args.host, args.port)

    if args.serve_multi_split:
        mp = Path(args.manifest)
        if not mp.exists():
            raise FileNotFoundError(f"Manifest not found: {mp}. Run with --train-instances first.")
        start_multi_flask_split_ports(mp, cfg, args.host, base_port=args.port)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
