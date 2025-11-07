# scripts/pipeline.py
import argparse
import yaml
from pathlib import Path

# Local stages (direct function calls)
from scripts.auto_docs_llm import generate_docs
from scripts.build_concepts_llm import build_concepts
from scripts.synonyms_llm import build_synonyms_llm
from scripts.ssms_ingest import run as ingest_run
from scripts.quick_analysis import run as profile_run
from utils.db import resolve_mssql_url


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _schema_include(cfg: dict) -> list[str]:
    inc = (cfg.get("ingest") or {}).get("schema_include") or []
    if isinstance(inc, str):
        inc = [inc]
    # keep non-empty strings only; preserve user’s casing (ssms_ingest handles case-insensitively)
    return [s for s in (str(x).strip() for x in inc) if s]


def run_pipeline(cfg_path: str):
    cfg = _load_cfg(cfg_path)

    # Surface table-limit for test runs (honored inside each LLM script)
    table_limit = int((cfg.get("llm") or {}).get("table_limit") or 0)
    print(f"[pipeline] LLM table limit: {'ALL' if table_limit <= 0 else table_limit}")

    # Resolve DB URL once
    db_url = resolve_mssql_url(cfg)

    # ---------------- Ingestion ----------------
    if (cfg.get("flow", {}).get("steps", {}) or {}).get("ingest_schema"):
        print("[pipeline] Running ingestion...")
        schema_filter = _schema_include(cfg)
        # Pass the filter down to ssms_ingest so only selected schemas are ingested
        ingest_run(
            db_url=db_url,
            out_schema_path=cfg["inputs"]["schema_json"],
            out_rels_path=cfg["inputs"]["relationships_json"],
            schema_filter=schema_filter,
        )

    # ---------------- Profiling ----------------
    if (cfg.get("flow", {}).get("steps", {}) or {}).get("profile_data"):
        print("[pipeline] Running profiling...")
        sample_rows = int(cfg.get("sample_rows_per_table", 500))
        # quick_analysis profiles ONLY tables present in schema.json, so no extra filter needed here
        profile_run(
            db_url=db_url,
            schema_json_path=cfg["inputs"]["schema_json"],
            profiles_dir=cfg["inputs"]["profiles_dir"],
            profiles_summary_csv=cfg["inputs"]["profiles_summary_csv"],
            sample_rows=sample_rows,
        )

    # ---------------- 1) Docs (LLM) ----------------
    try:
        if cfg.get("llm", {}).get("enabled") and (cfg.get("flow", {}).get("steps", {}) or {}).get("build_docs_llm"):
            print("[pipeline] Generating documentation (LLM)...")
            generate_docs(cfg)
    except Exception as e:
        print(f"[pipeline] Docs step failed: {e}")

    # ---------------- 2) Concepts (LLM) ----------------
    try:
        if cfg.get("llm", {}).get("enabled") and (cfg.get("flow", {}).get("steps", {}) or {}).get("build_concepts_llm"):
            print("[pipeline] Building concept layer (LLM)...")
            build_concepts(cfg)
    except Exception as e:
        print(f"[pipeline] Concepts step failed: {e}")

    # ---------------- 3) Synonyms ----------------
    try:
        mode = (cfg.get("flow") or {}).get("synonyms_mode", "llm")
        steps = (cfg.get("flow", {}).get("steps", {}) or {})
        if mode == "llm" and cfg.get("llm", {}).get("enabled") and steps.get("build_synonyms_llm"):
            print("[pipeline] Building synonyms via LLM (docs + concepts)...")
            build_synonyms_llm(cfg)
        elif mode == "heuristic":
            print("[pipeline] Using existing heuristic synonyms artifacts; no LLM synonyms step.")
        else:
            print(f"[pipeline] Unknown synonyms_mode '{mode}', skipping synonyms.")
    except Exception as e:
        print(f"[pipeline] Synonyms step failed: {e}")

    # ---------------- 4) Vanna bundle export ----------------
    try:
        if (cfg.get("flow", {}).get("steps", {}) or {}).get("export_vanna_bundle"):
            print("[pipeline] Exporting Vanna bundle...")
            from utils.bundle_export import export_vanna_bundle
            bundle_dir = export_vanna_bundle(cfg)
            print(f"[pipeline] Bundle ready at: {bundle_dir}")
    except Exception as e:
        print(f"[pipeline] Bundle export failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    run_pipeline(args.config)
