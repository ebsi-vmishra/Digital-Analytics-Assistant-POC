# app/vanna_app_poc.py
# PG-vector only (no VannaDefault / no Chroma)
# Adds:
#   - Batch CSV testing across all instances in manifest
#   - Robust prompt budgeting + DataFrame clamping for summaries
#
# Postgres must have: CREATE EXTENSION IF NOT EXISTS vector;

import os
import argparse
import json
import yaml
import multiprocessing
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple

from flask import Flask, request, jsonify

# NEW: pandas for batch CSV and DF handling
import pandas as pd

# Vanna PG + OpenAI mixins
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
            chunks.append("\n\n".join(buff)); buff, size = [], 0
        buff.append(para); size += len(para) + 2
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
        block = body[i:i+rows_per_chunk]
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
    cfg_key   = llm.get("api_key")
    cfg_base  = llm.get("base_url")
    cfg_model = llm.get("model")

    env_key   = os.getenv("OPENAI_API_KEY")
    env_base  = os.getenv("OPENAI_BASE_URL")
    env_model = os.getenv("OPENAI_MODEL")

    key   = cli_key  or env_key  or cfg_key
    base  = cli_base or env_base or cfg_base
    model = cli_model or env_model or cfg_model or "gpt-4.1"

    if not key:
        raise RuntimeError("OpenAI API key not found (config llm.api_key or env OPENAI_API_KEY).")
    return key, base, model

# ----------- connection lookups -----------
def get_conn_string(cfg: dict, kind: str, name: str) -> str:
    conn_map = (cfg.get("connections") or {}).get(kind) or {}
    s = conn_map.get(name)
    if not s:
        raise KeyError(f"Connection not found for connections.{kind}.{name} in config.")
    return s

# ----------- DF clamp helper (prevents context blowups) -----------
def _clamp_df(df: pd.DataFrame,
              max_rows: int,
              max_cols: int,
              max_chars_per_cell: int) -> pd.DataFrame:
    if df is None:
        return df
    if not isinstance(df, pd.DataFrame):
        return df
    df2 = df.copy()

    # limit rows/cols
    if max_rows and len(df2) > max_rows:
        df2 = df2.head(max_rows)
    if max_cols and df2.shape[1] > max_cols:
        keep_cols = list(df2.columns[:max_cols])
        df2 = df2[keep_cols]

    # limit cell sizes (string truncation)
    if max_chars_per_cell and max_chars_per_cell > 0:
        def _truncate(x):
            s = str(x)
            return s if len(s) <= max_chars_per_cell else (s[:max_chars_per_cell] + "…")
        df2 = df2.applymap(_truncate)

    return df2

# ----------- Vanna wrappers (with prompt + summary budgeting) -----------
class CustomVanna(PG_VectorStore, OpenAI_Chat):
    """
    Adds prompt-size budgeting using:
      - max_input_bytes        (default 120k)
      - summary_max_rows       (default 200)
      - summary_max_cols       (default 40)
      - summary_max_chars_per_cell (default 500)
      - temperature            (optional)
    And clamps DataFrames in generate_summary() to stop context explosions.
    """
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
        # Trim per message and overall budget
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

    # NEW: clamp DF before summary prompt is created
    def generate_summary(self, question: str, df=None, **kwargs):
        try:
            df = _clamp_df(
                df,
                max_rows=self.summary_max_rows,
                max_cols=self.summary_max_cols,
                max_chars_per_cell=self.summary_max_chars_per_cell,
            )
        except Exception:
            # best effort; even if clamping fails, fall back to parent
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

# ----------- training helpers -----------
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
    }

def _train_doc(vn: CustomVanna, path: Path, mode: str):
    if mode == "text":
        for c in chunk_text(_read_text(path), 1800):
            if c.strip(): vn.train(documentation=c)
    elif mode == "csv":
        for c in chunk_csv(path, 250, True):
            if c.strip(): vn.train(documentation=c)
    elif mode == "json":
        for c in chunk_json(path, 1800):
            if c.strip(): vn.train(documentation=c)

def train_from_bundle_chunked(vn: CustomVanna, bundle_dir: Path):
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    f = _bundle_files(bundle_dir)
    _train_doc(vn, f["ddl"], "text")
    _train_doc(vn, f["docs"], "text")
    _train_doc(vn, f["profiles"], "csv")
    for name in ["schema_json", "relationships_json", "synonyms_json", "attribute_map_json"]:
        _train_doc(vn, f[name], "json")
    for name in ["concept_catalog", "concept_alias", "concept_attributes", "concept_rules"]:
        _train_doc(vn, f[name], "csv")
    for name in ["concept_layer_llm", "value_aliases_llm"]:
        _train_doc(vn, f[name], "json")
    _train_doc(vn, f["value_domains_llm"], "csv")
    jpath = Path("artifacts") / "schema_docs.jsonl"
    if jpath.exists():
        for c in chunk_jsonl(jpath, 1800):
            if c.strip(): vn.train(documentation=c)
    print(f"✅ Trained from bundle (chunked): {bundle_dir.resolve()}")

def train_from_bundle_select(vn: CustomVanna, bundle_dir: Path, parts: Set[str]):
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    f = _bundle_files(bundle_dir)
    if "ddl" in parts:            _train_doc(vn, f["ddl"], "text")
    if "docs" in parts:           _train_doc(vn, f["docs"], "text")
    if "profiles" in parts:       _train_doc(vn, f["profiles"], "csv")
    if "schema" in parts:         _train_doc(vn, f["schema_json"], "json")
    if "relationships" in parts:  _train_doc(vn, f["relationships_json"], "json")
    if "synonyms" in parts:       _train_doc(vn, f["synonyms_json"], "json")
    if "attr_map" in parts:       _train_doc(vn, f["attribute_map_json"], "json")
    if "concepts" in parts:
        for name in ["concept_catalog", "concept_alias", "concept_attributes", "concept_rules"]:
            _train_doc(vn, f[name], "csv")
    if "values" in parts:
        _train_doc(vn, f["value_aliases_llm"], "json")
        _train_doc(vn, f["value_domains_llm"], "csv")
    print(f"✅ Trained from bundle parts {sorted(parts)} at {bundle_dir.resolve()}")

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
    print(f"✅ Trained from INFORMATION_SCHEMA (schema={schema_filter or 'ALL'})")

# ----------- instance orchestration -----------
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
    bundle_dir = Path(instance_cfg.get("bundle_dir") or global_cfg.get("outputs", {}).get("vanna_bundle_dir") or DEFAULT_BUNDLE_DIR)

    pg_store_key = instance_cfg.get("pg_store") or (global_cfg.get("vanna", {}).get("sql", {}).get("pg_store_name"))
    if not pg_store_key:
        raise RuntimeError(f"Instance '{instance_cfg.get('id')}' requires 'pg_store' in config.")
    vn = _vn_from_cfg(global_cfg, pg_store_key, openai_key, openai_model)

    mssql_key = instance_cfg.get("mssql_db") or (global_cfg.get("vanna", {}).get("sql", {}).get("mssql_db_name"))
    schema_filter = instance_cfg.get("schema")

    if method == "bundle":
        train_from_bundle_chunked(vn, bundle_dir=bundle_dir)
    elif method == "bundle_select":
        parts = set(map(str.lower, instance_cfg.get("bundle_parts") or []))
        if not parts:
            raise ValueError("bundle_select requires 'bundle_parts': e.g. ['docs','concepts']")
        train_from_bundle_select(vn, bundle_dir=bundle_dir, parts=parts)
    elif method == "info_schema":
        mssql_conn = get_conn_string(global_cfg, "mssql", mssql_key)
        train_from_info_schema_mssql(vn, mssql_conn_string=mssql_conn, schema_filter=schema_filter)
    elif method == "bundle_plus_info":
        train_from_bundle_chunked(vn, bundle_dir=bundle_dir)
        mssql_conn = get_conn_string(global_cfg, "mssql", mssql_key)
        train_from_info_schema_mssql(vn, mssql_conn_string=mssql_conn, schema_filter=schema_filter)
    elif method == "bundle_select_plus_info":
        parts = set(map(str.lower, instance_cfg.get("bundle_parts") or []))
        if not parts:
            raise ValueError("bundle_select_plus_info requires 'bundle_parts': e.g. ['docs','synonyms']")
        train_from_bundle_select(vn, bundle_dir=bundle_dir, parts=parts)
        mssql_conn = get_conn_string(global_cfg, "mssql", mssql_key)
        train_from_info_schema_mssql(vn, mssql_conn_string=mssql_conn, schema_filter=schema_filter)
    elif method == "docs_only":
        train_from_bundle_select(vn, bundle_dir=bundle_dir, parts={"docs","concepts"})
    else:
        raise ValueError(f"Unknown training method: {method}")

    try:
        q = "List full-time employees and exclude smokers"
        sql = vn.generate_sql(q)
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
    return rec

def _build_vn_pg_from_rec(rec: dict, cfg: dict, openai_key: str, openai_model: str) -> CustomVanna:
    pg_store_key = rec.get("pg_store") or (cfg.get("vanna", {}).get("sql", {}).get("pg_store_name"))
    vn = _vn_from_cfg(cfg, pg_store_key, openai_key, openai_model)
    if rec.get("mssql_db"):
        vn.connect_to_mssql(get_conn_string(cfg, "mssql", rec["mssql_db"]))
    return vn

def _serve_instance(rec: dict, cfg: dict, openai_key: str, openai_model: str, host: str):
    pg_store_key = rec.get("pg_store") or (cfg.get("vanna", {}).get("sql", {}).get("pg_store_name"))
    if not pg_store_key:
        raise RuntimeError(f"Instance '{rec.get('id')}' missing 'pg_store' and no default in vanna.sql.pg_store_name.")
    vn = _vn_from_cfg(cfg, pg_store_key, openai_key, openai_model)
    if rec.get("mssql_db"):
        mssql_conn = get_conn_string(cfg, "mssql", rec["mssql_db"])
        try:
            vn.connect_to_mssql(mssql_conn)
        except Exception as e:
            print(f"[serve:{rec['id']}] WARNING: couldn't attach MSSQL runtime connection: {e}")

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
        sql = vn.generate_sql(q)
        result = None
        if data.get("execute"):
            try:
                result = vn.run_sql(sql)
            except Exception as e:
                result = {"execution_error": str(e)}
        return jsonify({"instance": inst, "sql": sql, "result": result})

    @app.post("/compare")
    def compare():
        data = request.get_json(force=True)
        q = (data.get("question") or "").strip()
        selected = data.get("instances") or list(instances.keys())
        execq = bool(data.get("execute", False))
        answers = []
        for inst in selected:
            if inst not in instances:
                answers.append({"instance": inst, "error": "unknown instance"})
                continue
            vn = instances[inst]
            sql = vn.generate_sql(q)
            res = None
            if execq:
                try:
                    res = vn.run_sql(sql)
                except Exception as e:
                    res = {"execution_error": str(e)}
            answers.append({"instance": inst, "sql": sql, "result": res})
        return jsonify({"question": q, "answers": answers})

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

# ----------- Batch CSV across instances -----------
def run_batch_questions(manifest_path: Path, cfg: dict, input_csv: Path, output_csv: Path, do_execute: bool = True):
    """
    Reads input_csv with a column 'question' and runs it across all instances in the manifest.
    Writes a long-form CSV with columns:
      question,instance,sql,executed,execution_error,rowcount,sample_json
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

    rows = []
    for rec in instances:
        inst_id = rec.get("id")
        vn = _build_vn_pg_from_rec(rec, cfg, key, model)
        # try attach MSSQL for execution
        if rec.get("mssql_db"):
            try:
                vn.connect_to_mssql(get_conn_string(cfg, "mssql", rec["mssql_db"]))
            except Exception as e:
                print(f"[batch:{inst_id}] WARNING: MSSQL attach failed: {e}")

        for q in df_in["question"].astype(str).tolist():
            q2 = q.strip()
            if not q2:
                continue
            sql = ""
            executed = False
            rowcount = None
            sample_json = None
            exec_err = None
            try:
                sql = vn.generate_sql(q2)
                if do_execute and sql:
                    try:
                        res = vn.run_sql(sql)
                        if isinstance(res, pd.DataFrame):
                            executed = True
                            rowcount = int(len(res))
                            # keep tiny preview for debugging
                            sample_json = res.head(5).to_json(orient="records")
                        else:
                            # If some engines return list/dict
                            executed = True
                            sample_json = json.dumps(res)[:2000]
                    except Exception as ex:
                        exec_err = str(ex)
            except Exception as gen_ex:
                exec_err = f"SQL_GEN_ERROR: {gen_ex}"

            rows.append({
                "question": q2,
                "instance": inst_id,
                "sql": sql,
                "executed": executed,
                "execution_error": exec_err,
                "rowcount": rowcount,
                "sample_json": sample_json,
            })

    df_out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[batch] Wrote results → {output_csv}")

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Vanna (PG-only) trainer/runner.")
    # single
    p.add_argument("--train-bundle", action="store_true")
    p.add_argument("--train-info", action="store_true")
    p.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    p.add_argument("--mssql-db", default=None, help="Key under connections.mssql")
    p.add_argument("--schema", default=None)
    # multi
    p.add_argument("--train-instances", action="store_true")
    p.add_argument("--manifest", default=str(DEFAULT_VANNA_RUNS / "manifest.json"))
    # serving
    p.add_argument("--serve", action="store_true")
    p.add_argument("--serve-multi", action="store_true", help="Serve all instances behind one port")
    p.add_argument("--serve-multi-split", action="store_true", help="Serve each instance on its own port (requires vanna.instances[*].port or --port as base)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8084, help="For single-port serve OR base port for split-serve")
    # batch testing
    p.add_argument("--batch-csv", default=None, help="Path to input CSV with a 'question' column")
    p.add_argument("--batch-out", default="artifacts/vanna_runs/batch_results.csv", help="Path to write batch output CSV")
    p.add_argument("--no-exec", action="store_true", help="Do not execute generated SQL during batch run")
    # OpenAI overrides (optional; config llm.api_key also works)
    p.add_argument("--openai-key", default=None)
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--openai-model", default=None)
    # config file
    p.add_argument("-c", "--config", default=str(DEFAULT_POC_CFG))
    return p.parse_args()

# ---------------- main ----------------
def main():
    args = parse_args()
    cfg = load_poc_cfg(Path(args.config))

    key, base, model = resolve_openai_from_poc(cfg, args.openai_key, args.openai_base_url, args.openai_model)
    if base:
        os.environ["OPENAI_BASE_URL"] = base

    # Batch first (so you can run it standalone)
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

    # Single run
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

    # Multi-instance via config
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
