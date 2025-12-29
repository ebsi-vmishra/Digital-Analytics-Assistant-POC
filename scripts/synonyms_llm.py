# scripts/synonyms_llm.py
# End-to-end LLM-based synonyms builder with heuristic fallback, rich logging,
# and optional table limiting via limits.llm_tables_max.
#
# Updates:
# - Uses a dedicated synonyms system prompt (not the SQL system prompt).
# - Applies lexical constraints from compiled_rules.json to attribute_map.
# - Rebuilds synonyms.json from the filtered attribute_map so both stay aligned.

import os
import json
import yaml
import math
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from libs.log_utils import setup_logger

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
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


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

    NOTE: concept alias column is 'alias_text' (not 'alias').
    """
    # concept_id -> list[alias]
    by_concept: Dict[str, List[str]] = {}
    for a in alias_rows or []:
        cid = str(a.get("concept_id") or "").strip()
        al = str(a.get("alias_text") or "").strip()
        if cid and al:
            by_concept.setdefault(cid, []).append(al)

    synonyms: Dict[str, List[str]] = {}
    attr_map: List[Dict] = []

    for _, r in attrs_df.iterrows():
        cid = str(r.get("concept_id") or "").strip()
        t = str(r.get("table") or "").strip()
        c = str(r.get("column") or "").strip()
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
                    "source": "concepts-heuristic",
                })

    return synonyms, attr_map


# -----------------------
# Lexical constraint helpers
# -----------------------
def _build_banned_pairs(compiled_rules: Dict) -> List[Tuple[str, str]]:
    """
    Extract (from, to) pairs from compiled_rules.policies where:
      type: lexical_constraint
      action: ban_substitution
    All lowercased.
    """
    banned: List[Tuple[str, str]] = []
    for p in (compiled_rules.get("policies") or []):
        if (p.get("type") or "").lower() != "lexical_constraint":
            continue
        if (p.get("action") or "").lower() != "ban_substitution":
            continue
        fr = (p.get("from") or "").strip().lower()
        to = (p.get("to") or "").strip().lower()
        if fr and to:
            banned.append((fr, to))
    return banned


def _target_tokens(target: str) -> List[str]:
    """
    Crude tokenization for target like 'reporting.EmployeeCoverageFact.Employee_Id':
    splits on schema/table/col and grabs alphanumerics.
    """
    if not target:
        return []
    # Take table + column parts
    parts = target.split(".")
    tail = " ".join(parts[-2:]) if len(parts) >= 2 else target
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", tail)]


def _apply_lexical_constraints_to_attr_map(
    attr_map: List[Dict],
    banned_pairs: List[Tuple[str, str]],
    logger: logging.Logger,
) -> List[Dict]:
    """
    Remove attribute_map rows that would realize a banned lexical substitution.
    Example: ('dependent','employee') means:
      if alias == 'dependent' and target tokens contain 'employee' → drop.
    """
    if not banned_pairs or not attr_map:
        return attr_map

    kept: List[Dict] = []
    dropped = 0

    for row in attr_map:
        alias = (row.get("alias") or "").strip().lower()
        target = (row.get("target") or "").strip()
        tokens = _target_tokens(target)
        if not alias or not target:
            continue

        bad = False
        for fr, to in banned_pairs:
            if alias == fr and to in tokens:
                bad = True
                break

        if bad:
            dropped += 1
            continue
        kept.append(row)

    if dropped:
        logger.info(f"Lexical constraints: dropped {dropped} attribute_map rows due to banned substitutions")
    return kept


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
            "tenant_id": str(r.get("tenant_id") or ""),
            "table": str(r.get("table") or ""),
            "column": str(r.get("column") or ""),
            "role": str(r.get("role") or ""),
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
    logger = setup_logger(
        "synonyms_llm",
        (cfg.get("llm", {}) or {}).get("log_dir", "artifacts/logs"),
        (cfg.get("llm", {}) or {}).get("log_level", "INFO"),
    )

    # Inputs
    schema_path = cfg["inputs"]["schema_json"]
    docs_path = cfg["outputs"]["docs_jsonl"]
    concept_catalog_path = cfg["outputs"]["concepts"]["catalog_csv"]
    concept_alias_path = cfg["outputs"]["concepts"]["alias_csv"]
    concept_attrs_path = cfg["outputs"]["concepts"]["attributes_csv"]

    # Outputs
    synonyms_out = cfg["outputs"]["synonyms_json"]
    attr_map_out = cfg["outputs"]["attribute_map_json"]

    # Batch size (configurable)
    batch_size = int(cfg.get("llm", {}).get("synonyms_batch_size", 200))

    # Load inputs
    schema = read_json(schema_path) or {}
    docs_rows = read_jsonl(docs_path)
    cat_df = pd.read_csv(concept_catalog_path) if Path(concept_catalog_path).exists() else pd.DataFrame()
    alias_df = pd.read_csv(concept_alias_path) if Path(concept_alias_path).exists() else pd.DataFrame()
    attrs_df = pd.read_csv(concept_attrs_path) if Path(concept_attrs_path).exists() else pd.DataFrame()

    # Optional table limit (keep consistent with docs/concepts)
    table_limit = int((cfg.get("limits") or {}).get("llm_tables_max") or 0)
    if table_limit > 0 and not attrs_df.empty and "table" in [c.lower() for c in attrs_df.columns]:
        selected_tables = set(pick_tables(schema, table_limit))
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
            "concept_id": pick("concept_id"),
            "tenant_id": pick("tenant_id"),
            "table": pick("table"),
            "column": pick("column"),
            "role": pick("role"),
            "transform_sql": pick("transform_sql"),
        }
        ren = {}
        for want, have in need.items():
            if have and have != want:
                ren[have] = want
        return df.rename(columns=ren)

    attrs_df = _norm_cols(attrs_df)
    cat_df = _norm_cols(cat_df) if not cat_df.empty else cat_df
    alias_df = alias_df  # keep as-is; uses alias_text

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
    cat_rows = cat_df.to_dict(orient="records") if not cat_df.empty else []
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

    # Dedicated synonyms system prompt (do NOT reuse SQL system prompt)
    synonyms_system_prompt = (
        "You are a meticulous metadata synonyms generator. "
        "You ONLY return strict JSON valid for a synonyms and attribute_map object. "
        "Use only the provided schema, docs, concepts, and attributes. "
        "Never invent tables or columns."
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
                system_prompt=synonyms_system_prompt,
                user_prompt=prompt,
            ) or {}
        except Exception as e:
            logger.error(f"Batch {bi+1} LLM error: {e}")
            data = {}

        # Minimal shape guard
        syn = data.get("synonyms") if isinstance(data, dict) else None
        amap = data.get("attribute_map") if isinstance(data, dict) else None
        if not isinstance(syn, dict):
            syn = {}
        if not isinstance(amap, list):
            amap = []

        syn_ct = sum(len(v or []) for v in syn.values())
        logger.info(
            f"Synonyms batch {bi+1}/{batches} -> "
            f"syn_targets={len(syn)}, syn_total_aliases={syn_ct}, attr_map_rows={len(amap)}"
        )
        (log_dir / f"batch_{bi+1:03d}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
                        "source": source or "concepts+docs",
                    })
            except Exception:
                # ignore bad rows
                pass

    # Fallback if the LLM yielded nothing at all
    if not full_synonyms and not full_attr_map:
        logger.warning("LLM returned no synonyms/attr_map; building heuristic synonyms from concept aliases.")
        heur_syn, heur_map = _heuristic_synonyms_from_concepts(attrs_df, alias_rows)
        full_synonyms = heur_syn
        full_attr_map = heur_map

    # Apply lexical constraints from compiled_rules.json (if present)
    outputs_cfg = cfg.get("outputs") or {}
    concepts_dir = Path(outputs_cfg.get("concepts_out_dir", "artifacts/concepts"))
    compiled_rules_path = concepts_dir / "compiled_rules.json"
    compiled_rules = read_json(str(compiled_rules_path)) or {}
    banned_pairs = _build_banned_pairs(compiled_rules)
    if banned_pairs:
        full_attr_map = _apply_lexical_constraints_to_attr_map(full_attr_map, banned_pairs, logger)

    # Rebuild synonyms from the final attribute_map so they stay aligned
    rebuilt_synonyms: Dict[str, List[str]] = {}
    for row in full_attr_map:
        tgt = row.get("target")
        al = row.get("alias")
        if not tgt or not al:
            continue
        bucket = rebuilt_synonyms.setdefault(tgt, [])
        if al not in bucket:
            bucket.append(al)
    full_synonyms = rebuilt_synonyms

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
