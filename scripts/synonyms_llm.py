# scripts/synonyms_llm.py
# End-to-end LLM-based synonyms builder with heuristic fallback, rich logging,
# and optional table limiting via limits.llm_tables_max.

import os
import json
import yaml
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from libs.log_utils import setup_logger

import pandas as pd

# --- project root import shim ---
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------

from libs.llm_client import LLMClient
from libs.table_utils import pick_tables  # for limits.llm_tables_max


# -----------------------
# Basic IO helpers
# -----------------------
def read_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(obj, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def read_jsonl(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # tolerate stray lines
                pass
    return rows




# -----------------------
# Heuristic fallback
# -----------------------
def _heuristic_synonyms_from_concepts(attrs_df: pd.DataFrame, alias_rows: List[Dict]) -> Tuple[Dict[str, List[str]], List[Dict]]:
    """
    Build minimal synonyms and attribute_map from concept attributes + concept aliases when
    the LLM returns nothing.

    FIX: concept alias column is 'alias_text' (not 'alias').
    """
    # concept_id -> list[alias]
    by_concept: Dict[str, List[str]] = {}
    for a in alias_rows or []:
        cid = str(a.get("concept_id") or "").strip()
        al  = str(a.get("alias_text") or "").strip()  # <-- FIXED
        if cid and al:
            by_concept.setdefault(cid, []).append(al)

    synonyms: Dict[str, List[str]] = {}
    attr_map: List[Dict] = []

    for _, r in attrs_df.iterrows():
        cid = str(r.get("concept_id") or "").strip()
        t   = str(r.get("table") or "").strip()
        c   = str(r.get("column") or "").strip()
        if not (cid and t and c):
            continue

        target = f"{t}.{c}"
        crude_aliases = set()

        # Column name variants
        col_pretty = c.replace("_", " ").replace("-", " ").strip()
        crude_aliases.update({c, col_pretty, col_pretty.lower(), col_pretty.title()})

        # Concept aliases (if any)
        for al in by_concept.get(cid, []):
            crude_aliases.add(al)
            crude_aliases.add(al.lower())
            crude_aliases.add(al.title())

        # Finalize
        ls = synonyms.setdefault(target, [])
        for al in sorted(crude_aliases):
            if al and al not in ls:
                ls.append(al)
                attr_map.append({
                    "alias": al,
                    "target": target,
                    "confidence": 0.6 if al.lower() == col_pretty.lower() else 0.5,
                    "source": "concepts-heuristic"
                })

    return synonyms, attr_map


# -----------------------
# Prompt builder
# -----------------------
def _select_tables_from_attrs(attrs_df: pd.DataFrame) -> List[str]:
    tset = set()
    for _, r in attrs_df.iterrows():
        t = str(r.get("table") or "").strip()
        if t:
            tset.add(t)
    return sorted(tset)

def _build_prompt(
    schema: Dict,
    docs_by_table: Dict[str, Dict],
    concept_catalog: List[Dict],
    concept_aliases: List[Dict],
    attrs_subset: pd.DataFrame,
    max_docs_tables: int = 60,
) -> str:
    """
    Construct a compact, high-signal prompt for this batch:
    - Minimal SCHEMA subset (only tables referenced by attrs_subset)
    - Minimal DOCS subset (those tables) with column descriptions
    - FULL concept catalog + aliases (high value)
    - The attributes (schema.table.column) we need synonyms for in this batch

    Includes the literal word 'json' so OpenAI json_object mode is allowed.
    """
    tables_needed = _select_tables_from_attrs(attrs_subset)[:max_docs_tables]

    # schema subset
    schema_subset = {"tables": {}}
    all_tables = (schema or {}).get("tables") or {}
    for t in tables_needed:
        if t in all_tables:
            schema_subset["tables"][t] = all_tables[t]

    # docs subset
    docs_subset = []
    for t in tables_needed:
        d = docs_by_table.get(t)
        if d:
            docs_subset.append(d)

    # attrs for this batch
    attrs_list = []
    for _, r in attrs_subset.iterrows():
        attrs_list.append({
            "concept_id": str(r.get("concept_id") or ""),
            "tenant_id":  str(r.get("tenant_id") or ""),
            "table":      str(r.get("table") or ""),
            "column":     str(r.get("column") or ""),
            "role":       str(r.get("role") or ""),
            "transform_sql": str(r.get("transform_sql") or ""),
        })

    return (
        "You are a careful data analyst and metadata architect. "
        "Use only the provided inputs. Produce a strict JSON object (no commentary) "
        "for synonyms and attribute_map. The word json appears here intentionally.\n\n"
        "== SCHEMA_SUBSET ==\n"
        + json.dumps(schema_subset, ensure_ascii=False)
        + "\n\n== DOCS_SUBSET ==\n"
        + json.dumps(docs_subset, ensure_ascii=False)
        + "\n\n== CONCEPT_CATALOG ==\n"
        + json.dumps(concept_catalog, ensure_ascii=False)
        + "\n\n== CONCEPT_ALIASES ==\n"
        + json.dumps(concept_aliases, ensure_ascii=False)
        + "\n\n== ATTRIBUTES_THIS_BATCH ==\n"
        + json.dumps(attrs_list, ensure_ascii=False)
        + "\n\nReturn a json object EXACTLY like:\n"
        "{\n"
        '  "synonyms": {"schema.table.column": ["alias1","alias2"]},\n'
        '  "attribute_map": [\n'
        '    {"alias":"...", "target":"schema.table.column", "confidence":0.0, "source":"concepts+docs"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Only map aliases to targets present in ATTRIBUTES_THIS_BATCH (schema.table.column).\n"
        "- When building aliases, consider concept aliases bound to the attribute's concept_id, "
        "  table/column names (and readable variants), and phrases in DOCS_SUBSET.\n"
        "- Confidence ∈ [0,1]. Use higher values when concept alias and docs align closely to the column name/meaning.\n"
        "- Do NOT invent new tables or columns. If unsure, omit.\n"
    )


# -----------------------
# Main builder
# -----------------------
def build_synonyms_llm(cfg: dict):
    logger = setup_logger("synonyms_llm", (cfg.get("llm",{}) or {}).get("log_dir","artifacts/logs"),(cfg.get("llm",{}) or {}).get("log_level","INFO"))

    # Inputs
    schema_path           = cfg["inputs"]["schema_json"]
    docs_path             = cfg["outputs"]["docs_jsonl"]
    concept_catalog_path  = cfg["outputs"]["concepts"]["catalog_csv"]
    concept_alias_path    = cfg["outputs"]["concepts"]["alias_csv"]
    concept_attrs_path    = cfg["outputs"]["concepts"]["attributes_csv"]

    # Outputs
    synonyms_out = cfg["outputs"]["synonyms_json"]
    attr_map_out = cfg["outputs"]["attribute_map_json"]

    # Batch size (configurable)
    batch_size = int(cfg.get("llm", {}).get("synonyms_batch_size", 200))

    # Load inputs
    schema    = read_json(schema_path) or {}
    docs_rows = read_jsonl(docs_path)
    cat_df    = pd.read_csv(concept_catalog_path) if Path(concept_catalog_path).exists() else pd.DataFrame()
    alias_df  = pd.read_csv(concept_alias_path) if Path(concept_alias_path).exists() else pd.DataFrame()
    attrs_df  = pd.read_csv(concept_attrs_path) if Path(concept_attrs_path).exists() else pd.DataFrame()

    # Optional table limit (keep consistent with docs/concepts)
    # If set, we will keep only attributes whose table is in the picked set.
    table_limit = int((cfg.get("limits") or {}).get("llm_tables_max") or 0)
    if table_limit > 0 and not attrs_df.empty and "table" in [c.lower() for c in attrs_df.columns]:
        selected_tables = set(pick_tables(schema, table_limit))
        # normalize case for the column name
        table_col = [c for c in attrs_df.columns if c.lower() == "table"][0]
        before = len(attrs_df)
        attrs_df = attrs_df[attrs_df[table_col].astype(str).isin(selected_tables)].copy()
        logger.info(f"Applied table limit: kept {len(attrs_df)}/{before} attributes across {len(selected_tables)} tables")

    if attrs_df.empty:
        logger.info("No attributes found; writing empty synonyms artifacts.")
        write_json({}, synonyms_out)
        write_json([], attr_map_out)
        return

    # Normalize expected column names
    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        need = {
            "concept_id":    pick("concept_id"),
            "tenant_id":     pick("tenant_id"),
            "table":         pick("table"),
            "column":        pick("column"),
            "role":          pick("role"),
            "transform_sql": pick("transform_sql"),
            # alias_df uses alias_text (we won't rename here, we handle it where needed)
        }
        ren = {}
        for want, have in need.items():
            if have and have != want:
                ren[have] = want
        return df.rename(columns=ren)

    attrs_df = _norm_cols(attrs_df)
    cat_df   = _norm_cols(cat_df) if not cat_df.empty else cat_df
    alias_df = alias_df  # keep as-is; remember to use 'alias_text' when reading

    # Build docs_by_table lookup (compact)
    docs_by_table: Dict[str, Dict] = {}
    for row in docs_rows:
        t = str(row.get("table") or "").strip()
        if t:
            keep = {
                "table": t,
                "description": row.get("description"),
                "columns": row.get("columns"),
                "relationships": row.get("relationships"),
            }
            docs_by_table[t] = keep

    # Concept lists for prompt
    cat_rows   = cat_df.to_dict(orient="records") if not cat_df.empty else []
    alias_rows = alias_df.to_dict(orient="records") if not alias_df.empty else []

    # LLM client
    llm = LLMClient(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        request_timeout_sec=cfg["llm"]["request_timeout_sec"],
        connect_timeout_sec=cfg["llm"].get("connect_timeout_sec", 30),
        read_timeout_sec=cfg["llm"].get("read_timeout_sec", 120),
        max_retries=cfg["llm"]["max_retries"],
        json_strict=cfg["llm"]["json_strict"],
        api_key=cfg["llm"].get("api_key"),  # reads from config or env
        verify_ssl=cfg["llm"].get("verify_ssl", True),
        ca_bundle_path=cfg["llm"].get("ca_bundle_path") or None,
        proxies=cfg["llm"].get("proxies") or None,
        base_url=cfg["llm"].get("base_url") or None,
        log_dir=cfg["llm"]["log_dir"],
        log_level=cfg["llm"]["log_level"],
        log_prompts=cfg["llm"].get("log_prompts", True),
        redact_values=cfg["llm"].get("redact_values") or [],
        max_input_bytes=cfg["llm"].get("max_input_bytes", 180000),
    )

    # Batching
    n = len(attrs_df)
    batches = math.ceil(n / batch_size)
    logger.info(f"Synonyms: {n} attributes, batch_size={batch_size}, batches={batches}")

    full_synonyms: Dict[str, List[str]] = {}
    full_attr_map: List[Dict] = []

    # Optional debug output per batch
    log_dir = Path(cfg.get("llm", {}).get("log_dir", "artifacts/logs")) / "synonyms_batches"
    log_dir.mkdir(parents=True, exist_ok=True)

    for bi in range(batches):
        lo = bi * batch_size
        hi = min((bi + 1) * batch_size, n)
        batch_df = attrs_df.iloc[lo:hi].copy()

        logger.info(f"Synonyms batch {bi+1}/{batches}: rows={len(batch_df)}")

        prompt = _build_prompt(
            schema=schema,
            docs_by_table=docs_by_table,
            concept_catalog=cat_rows,
            concept_aliases=alias_rows,
            attrs_subset=batch_df,
            max_docs_tables=int(cfg.get("llm", {}).get("max_docs_tables_per_batch", 60)),
        )

        try:
            data = llm.json_completion(
                system_prompt=cfg["llm"]["system_prompt"],
                user_prompt=prompt
            ) or {}
        except Exception as e:
            logger.error(f"Batch {bi+1} LLM error: {e}")
            data = {}

        # Minimal shape guard
        syn  = data.get("synonyms") if isinstance(data, dict) else None
        amap = data.get("attribute_map") if isinstance(data, dict) else None
        if not isinstance(syn, dict):
            syn = {}
        if not isinstance(amap, list):
            amap = []

        # Log small breadcrumbs
        syn_ct = sum(len(v or []) for v in syn.values())
        logger.info(
            f"Synonyms batch {bi+1}/{batches} -> "
            f"syn_targets={len(syn)}, syn_total_aliases={syn_ct}, attr_map_rows={len(amap)}"
        )
        # Persist raw batch output for debug
        (log_dir / f"batch_{bi+1:03d}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Merge into full results
        for tgt, aliases in syn.items():
            if not isinstance(aliases, list):
                continue
            bucket = full_synonyms.setdefault(tgt, [])
            for a in aliases:
                a = str(a).strip()
                if a and a not in bucket:
                    bucket.append(a)

        for row in amap:
            try:
                alias = str(row.get("alias") or "").strip()
                target = str(row.get("target") or "").strip()
                conf = float(row.get("confidence") or 0.0)
                source = str(row.get("source") or "concepts+docs").strip()
                if alias and target:
                    full_attr_map.append({
                        "alias": alias,
                        "target": target,
                        "confidence": max(0.0, min(1.0, conf)),
                        "source": source or "concepts+docs"
                    })
            except Exception:
                # ignore bad rows
                pass

    # Fallback if the LLM yielded nothing
    if not full_synonyms and not full_attr_map:
        logger.warning("LLM returned no synonyms/attr_map; building heuristic synonyms from concept aliases.")
        heur_syn, heur_map = _heuristic_synonyms_from_concepts(attrs_df, alias_rows)
        full_synonyms = heur_syn
        full_attr_map = heur_map

    # Write outputs
    write_json(full_synonyms, synonyms_out)
    write_json(full_attr_map, attr_map_out)
    logger.info(f"Wrote synonyms to {synonyms_out} and attribute_map to {attr_map_out}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build synonyms via LLM using docs + concepts")
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ensure required config defaults
    cfg.setdefault("llm", {})
    cfg["llm"].setdefault("log_dir", "artifacts/logs")
    cfg["llm"].setdefault("log_level", "INFO")
    cfg["llm"].setdefault("synonyms_batch_size", 200)
    cfg["llm"].setdefault("max_docs_tables_per_batch", 60)

    build_synonyms_llm(cfg)
