# scripts/pipeline.py
import argparse
from pathlib import Path
from typing import List

import yaml

from scripts.auto_docs_llm import generate_docs
from scripts.build_concepts_llm import build_concepts
from scripts.synonyms_llm import build_synonyms_llm
from scripts.ssms_ingest import run as ingest_run
from scripts.quick_analysis import run as profile_run
from utils.db import resolve_mssql_url


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _schema_include(cfg: dict) -> List[str]:
    """
    Return list of schemas to include for ingest, based on cfg.ingest.schema_include.
    Keeps non-empty strings only; preserves casing (ssms_ingest is case-insensitive).
    """
    inc = (cfg.get("ingest") or {}).get("schema_include") or []
    if isinstance(inc, str):
        inc = [inc]
    return [s for s in (str(x).strip() for x in inc) if s]


def run_pipeline(cfg_path: str) -> None:
    cfg = _load_cfg(cfg_path)

    llm_cfg = cfg.get("llm") or {}
    flow_cfg = cfg.get("flow") or {}
    steps = flow_cfg.get("steps") or {}
    synonyms_mode = flow_cfg.get("synonyms_mode", "llm")

    # Surface table-limit for test runs (honored inside LLM scripts themselves)
    table_limit = int(llm_cfg.get("table_limit") or 0)
    print(f"[pipeline] LLM table limit: {'ALL' if table_limit <= 0 else table_limit}")

    # Resolve DB URL once (used by ingest + profiling)
    db_url = resolve_mssql_url(cfg)

    # -------------------------------------------------------------------------
    # 1) Ingestion: build infoschema (schema.json + relationships.json)
    # -------------------------------------------------------------------------
    if steps.get("ingest"):
        print("[pipeline] Running ingestion (schema + relationships)...")
        schema_filter = _schema_include(cfg)
        ingest_run(
            db_url=db_url,
            out_schema_path=cfg["inputs"]["schema_json"],
            out_rels_path=cfg["inputs"]["relationships_json"],
            schema_filter=schema_filter,
        )
    else:
        print("[pipeline] Skipping ingestion (flow.steps.ingest = false)")

    # -------------------------------------------------------------------------
    # 2) Profiling: build profiling artifacts from schema.json
    # -------------------------------------------------------------------------
    if steps.get("profile"):
        print("[pipeline] Running profiling...")
        sample_rows = int(cfg.get("sample_rows_per_table", 500))
        profile_run(
            db_url=db_url,
            schema_json_path=cfg["inputs"]["schema_json"],
            profiles_dir=cfg["inputs"]["profiles_dir"],
            profiles_summary_csv=cfg["inputs"]["profiles_summary_csv"],
            sample_rows=sample_rows,
        )
    else:
        print("[pipeline] Skipping profiling (flow.steps.profile = false)")

    # -------------------------------------------------------------------------
    # 3) Docs (LLM): generate NL documentation chunks
    # -------------------------------------------------------------------------
    if steps.get("docs"):
        if llm_cfg.get("enabled"):
            try:
                print("[pipeline] Generating documentation (LLM)...")
                generate_docs(cfg)
            except Exception as e:
                print(f"[pipeline] Docs step failed: {e}")
        else:
            print("[pipeline] Skipping docs (LLM disabled)")
    else:
        print("[pipeline] Skipping docs (flow.steps.docs = false)")

    # -------------------------------------------------------------------------
    # 4) Concepts (LLM): build concept catalog/aliases/attributes/rules CSVs
    # -------------------------------------------------------------------------
    if steps.get("concepts"):
        if llm_cfg.get("enabled"):
            try:
                print("[pipeline] Building concept layer (LLM)...")
                build_concepts(cfg)
            except Exception as e:
                print(f"[pipeline] Concepts step failed: {e}")
        else:
            print("[pipeline] Skipping concepts (LLM disabled)")
    else:
        print("[pipeline] Skipping concepts (flow.steps.concepts = false)")

    # -------------------------------------------------------------------------
    # 5) Synonyms: build synonyms + attribute_map (LLM or heuristic)
    # -------------------------------------------------------------------------
    if steps.get("synonyms"):
        try:
            if synonyms_mode == "llm":
                if llm_cfg.get("enabled"):
                    print("[pipeline] Building synonyms via LLM (docs + concepts)...")
                    build_synonyms_llm(cfg)
                else:
                    print("[pipeline] Skipping synonyms LLM (LLM disabled)")
            elif synonyms_mode == "heuristic":
                print("[pipeline] Using existing heuristic synonyms artifacts; no LLM synonyms step.")
            else:
                print(f"[pipeline] Unknown synonyms_mode '{synonyms_mode}', skipping synonyms.")
        except Exception as e:
            print(f"[pipeline] Synonyms step failed: {e}")
    else:
        print("[pipeline] Skipping synonyms (flow.steps.synonyms = false)")

    # -------------------------------------------------------------------------
    # 6) DDL export: export table DDLs for selected schemas
    # -------------------------------------------------------------------------
    if steps.get("export_ddl"):
        try:
            print("[pipeline] Exporting DDLs...")
            from scripts.export_ddl import export as export_ddl
            export_ddl(cfg_path)
        except Exception as e:
            print(f"[pipeline] DDL export failed: {e}")
    else:
        print("[pipeline] Skipping DDL export (flow.steps.export_ddl = false)")

    # -------------------------------------------------------------------------
    # 7) Concept-layer compilation: YAML rules/policies + artifacts → prompts/rules
    # -------------------------------------------------------------------------
    if steps.get("compile_concepts"):
        try:
            print("[pipeline] Compiling concept-layer prompts and rules...")
            from scripts.compile_concepts import compile_concepts
            compile_concepts(cfg_path)
        except FileNotFoundError as e:
            print(f"[pipeline] Skipping compile_concepts (missing inputs): {e}")
        except Exception as e:
            print(f"[pipeline] compile_concepts failed: {e}")
    else:
        print("[pipeline] Skipping compile_concepts (flow.steps.compile_concepts = false)")

    # -------------------------------------------------------------------------
    # 8) Emit combined system prompt: concepts_prompt + summaries + alias preview
    # -------------------------------------------------------------------------
    if steps.get("emit_system_prompt"):
        try:
            print("[pipeline] Emitting combined system prompt...")
            from scripts.emit_system_prompt import emit as emit_system_prompt
            emit_system_prompt(cfg_path)
        except FileNotFoundError as e:
            print(f"[pipeline] Skipping emit_system_prompt (missing prompts/rules): {e}")
        except Exception as e:
            print(f"[pipeline] emit_system_prompt failed: {e}")
    else:
        print("[pipeline] Skipping emit_system_prompt (flow.steps.emit_system_prompt = false)")

    # -------------------------------------------------------------------------
    # 9) Vanna bundle export: pack artifacts into artifacts/vanna_bundle
    # -------------------------------------------------------------------------
    if steps.get("export_bundle"):
        try:
            print("[pipeline] Exporting Vanna bundle...")
            from utils.bundle_export import export_vanna_bundle
            bundle_dir = export_vanna_bundle(cfg)
            print(f"[pipeline] Bundle ready at: {bundle_dir}")
        except Exception as e:
            print(f"[pipeline] Bundle export failed: {e}")
    else:
        print("[pipeline] Skipping bundle export (flow.steps.export_bundle = false)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    run_pipeline(args.config)
