# scripts/build_concepts_llm.py

# --- project root import shim ---
import sys, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------

from libs.llm_client import LLMClient
from libs.log_utils import setup_logger
from utils.io_utils import write_csv

# Prompt / concepts artifact locations
PROMPTS_DIR  = Path("artifacts/prompts")
CONCEPTS_DIR = Path("artifacts/concepts")

CAT_FIELDS   = ["concept_id","concept_name","description","grain","owner","status"]
ALIAS_FIELDS = ["concept_id","alias_text","locale","confidence","source","is_preferred"]
ATTR_FIELDS  = ["concept_id","tenant_id","table","column","role","transform_sql","effective_start","effective_end","notes"]
RULE_FIELDS  = ["rule_id","concept_id","tenant_id","rule_type","sql_template","parameters","notes"]

# ------------------------ small loaders ------------------------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_json_if_exists(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _load_docs_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def _docs_index_by_table(docs_rows: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for row in docs_rows:
        t = row.get("table")
        if t:
            out[t] = row
    return out

# ------------------------ helpers: batching & dedup ------------------------

def _dedup_dict_rows(rows: List[Dict], keys: List[str]) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        k = tuple((r.get(k) or "").strip() for k in keys)
        if k not in seen:
            out.append(r)
            seen.add(k)
    return out

def _split_tables(schema: Dict, batch_size: int) -> List[List[str]]:
    tables = sorted((schema.get("tables") or {}).keys())
    if not tables:
        return []
    return [tables[i:i+batch_size] for i in range(0, len(tables), batch_size)]

def _subset_schema(schema: Dict, tables: List[str]) -> Dict:
    return {"tables": {t: schema["tables"][t] for t in tables if t in schema.get("tables", {})}}

def _subset_docs(docs_by_table: Dict[str, Dict], tables: List[str]) -> List[Dict]:
    return [docs_by_table[t] for t in tables if t in docs_by_table]

def _clip(obj: Any, limit: int) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s[:limit]

# ------------------------ heuristic backfill ------------------------

_SMOKER_PAT = re.compile(r"\b(smok(e|er|ing)|tobacco)\b", re.IGNORECASE)
_EMPLOY_PAT = re.compile(r"\b(employment[_\s]?type|emp[_\s]?type|ft|pt|full[-\s]?time|part[-\s]?time)\b", re.IGNORECASE)

def _heuristic_concepts(schema_subset: Dict, docs_subset: List[Dict], tenant_id: str) -> Dict[str, List[Dict]]:
    cat, alias, attrs, rules = [], [], [], []

    def add_concept(cid: str, name: str, desc: str, grain: str="entity"):
        if not any(c.get("concept_id")==cid for c in cat):
            cat.append({"concept_id": cid, "concept_name": name, "description": desc, "grain": grain, "owner": "", "status": "draft"})

    def add_alias(cid: str, text: str, conf: float=0.7, preferred: bool=False, src="heuristic"):
        alias.append({"concept_id": cid, "alias_text": text, "locale": "", "confidence": conf, "source": src, "is_preferred": preferred})

    def add_attr(cid: str, table: str, column: str, role="attr", tx="", notes=""):
        attrs.append({
            "concept_id": cid, "tenant_id": tenant_id, "table": table, "column": column,
            "role": role, "transform_sql": tx, "effective_start": "", "effective_end": "", "notes": notes
        })

    for d in docs_subset:
        t = d.get("table","")
        cols = d.get("columns") or []
        t_text = (d.get("description") or "") + " " + t
        if _SMOKER_PAT.search(t_text):
            add_concept("C_SMOKER", "Tobacco/Smoker Indicator", "Indicates whether a member uses tobacco.", "member")
            add_alias("C_SMOKER", "Smoker", preferred=True)
            add_alias("C_SMOKER", "TobaccoUser")
        if _EMPLOY_PAT.search(t_text):
            add_concept("C_EMPTYPE", "Employment Type", "Full-Time/Part-Time style classification.", "member")
            add_alias("C_EMPTYPE", "EmploymentType", preferred=True)
            add_alias("C_EMPTYPE", "EmpType")

        for c in cols:
            colname = c.get("column","")
            c_text = (c.get("description") or "") + " " + colname
            if _SMOKER_PAT.search(c_text):
                add_concept("C_SMOKER", "Tobacco/Smoker Indicator", "Indicates whether a member uses tobacco.", "member")
                add_attr(
                    "C_SMOKER", t, colname, role="attr",
                    tx="CASE WHEN {col} IN ('Y','1','TRUE') THEN 1 WHEN {col} IN ('N','0','FALSE') THEN 0 ELSE NULL END",
                    notes="Normalize Y/N or 1/0/TRUE/FALSE to 1/0."
                )
            if _EMPLOY_PAT.search(c_text):
                add_concept("C_EMPTYPE", "Employment Type", "Full-Time/Part-Time style classification.", "member")
                add_attr(
                    "C_EMPTYPE", t, colname, role="attr",
                    tx=("CASE WHEN {col} IN ('FT','FULL TIME','FULL-TIME') THEN 'Full-Time' "
                        "WHEN {col} IN ('PT','PART TIME','PART-TIME') THEN 'Part-Time' ELSE {col} END"),
                    notes="Normalize FT/PT tokens to canonical labels."
                )

    return {
        "concept_catalog": cat,
        "concept_alias": alias,
        "concept_attributes": attrs,
        "concept_rules": rules
    }

# ------------------------ prompt construction ------------------------

def _build_prompt(
    tenant_id: str,
    schema_subset: Dict,
    relationships: Dict,
    docs_subset: List[Dict],
    alias_preview: List[str] = None,
    syn_preview: List[str] = None,
) -> str:
    alias_block = ""
    syn_block = ""
    if alias_preview:
        alias_block = "\n**Aliases (sanitized attribute_map)**\n" + "\n".join(f"- {x}" for x in alias_preview[:40])
    if syn_preview:
        syn_block = "\n**Synonyms (sanitized)**\n" + "\n".join(f"- {x}" for x in syn_preview[:30])

    return f"""
You are a meticulous data analyst creating a Concept Layer for tenant "{tenant_id}".
Use the inputs and produce tight, consistent concepts. Honor lexical bans (e.g., do NOT substitute 'dependent' for 'employee').
Be conservative; use only tables/columns present in SCHEMA and DOCUMENTATION.

CRITICAL NAMING RULES
- Use table names EXACTLY as they appear in SCHEMA (for example, "reporting.Employee"), not generic plurals like "Employees".
- The "table" field in concept_attributes MUST match one of the SCHEMA keys (e.g., "reporting.Employee", "reporting.Benefit", etc.).
- Do NOT invent new table names; if you are unsure, omit the mapping instead of guessing.

INPUTS
- SCHEMA (subset for batch tables):
{_clip(schema_subset, 90000)}

- RELATIONSHIPS (full):
{_clip(relationships, 45000)}

- DOCUMENTATION (subset for batch tables):
{_clip(docs_subset, 90000)}

- PREVIEW HINTS:
{alias_block}
{syn_block}

TASK
1) concept_catalog: id, name, description, grain (employee/dependent/member/enrollment/etc), owner (optional), status="draft".
2) concept_alias: alias_text per concept (include technical variants), confidence (0-1), source "docs+profiles" (or similar), locale "", is_preferred true for one alias.
3) concept_attributes: map ACTUAL columns to concepts with: tenant_id "{tenant_id}", table (MUST be an exact schema key like "reporting.Employee"), column, role ("key"|"attr"|"metric"), optional transform_sql for normalization (FT/PT → 'Full-Time'/'Part-Time'; Y/N or 0/1 → TRUE/FALSE).
4) concept_rules: optional derivations; include rule_type ("derivation"|"quality"|"security"), sql_template (parameterized), parameters (JSON).
5) Do not invent tables/columns. If you cannot confidently map a concept to any table/column, leave concept_attributes empty for that concept.

Return a json object with these arrays (keys required on each row):
{{
  "concept_catalog": [
    {{"concept_id":"","concept_name":"","description":"","grain":"","owner":"","status":""}}
  ],
  "concept_alias": [
    {{"concept_id":"","alias_text":"","locale":"","confidence":0.0,"source":"docs+profiles","is_preferred":false}}
  ],
  "concept_attributes": [
    {{"concept_id":"","tenant_id":"","table":"schema.table","column":"","role":"","transform_sql":"","effective_start":"","effective_end":"","notes":""}}
  ],
  "concept_rules": [
    {{"rule_id":"","concept_id":"","tenant_id":"","rule_type":"","sql_template":"","parameters":"","notes":""}}
  ]
}}
Only valid json. No extra commentary.
"""

# ------------------------ normalization & constraints ------------------------

def _slugify_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()

def _build_table_normalizer(schema: Dict) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build helpers to validate/normalize table names:

    - exact_map:   exact lowercased key -> full key (e.g. "reporting.employee")
    - base_map:    base name lowercased ("employee") -> [full keys ...]
    """
    tables = (schema.get("tables") or {}).keys()
    exact_map: Dict[str, str] = {}
    base_map: Dict[str, List[str]] = {}
    for full in tables:
        low = full.lower()
        exact_map[low] = full
        base = full.split(".")[-1].lower()
        base_map.setdefault(base, []).append(full)
    return exact_map, base_map

def _normalize_table_name(raw_table: str, exact_map: Dict[str, str], base_map: Dict[str, List[str]]) -> str:
    """
    Normalize LLM-produced table name to an existing schema key.

    Strategy:
      1) Exact case-insensitive match on full key.
      2) Match by base name (last segment). If exactly one candidate, use it.
      3) Otherwise, return "" to signal "unresolvable".
    """
    t = (raw_table or "").strip()
    if not t:
        return ""
    low = t.lower()

    # 1) Exact full-key match
    if low in exact_map:
        return exact_map[low]

    # 2) base name match
    base = t.split(".")[-1].lower()
    candidates = base_map.get(base) or []
    if len(candidates) == 1:
        return candidates[0]

    # 3) Unclear → drop
    return ""

def _normalize_rows(data: Dict[str, Any], tenant_id: str, schema: Dict) -> Dict[str, List[Dict]]:
    out = {
        "concept_catalog": [],
        "concept_alias": [],
        "concept_attributes": [],
        "concept_rules": [],
    }

    exact_map, base_map = _build_table_normalizer(schema)

    # Catalog
    for r in (data.get("concept_catalog") or []):
        concept_id   = r.get("concept_id") or _slugify_id(r.get("concept_name") or r.get("concept") or "")
        concept_name = r.get("concept_name") or r.get("concept") or ""
        out["concept_catalog"].append({
            "concept_id":   concept_id,
            "concept_name": concept_name,
            "description":  r.get("description",""),
            "grain":        r.get("grain",""),
            "owner":        r.get("owner",""),
            "status":       r.get("status","draft") or "draft",
        })

    # Alias
    for r in (data.get("concept_alias") or []):
        out["concept_alias"].append({
            "concept_id":   r.get("concept_id") or _slugify_id(r.get("concept") or ""),
            "alias_text":   r.get("alias_text") or r.get("alias") or "",
            "locale":       r.get("locale",""),
            "confidence":   float(r.get("confidence", 0.0) or 0.0),
            "source":       (r.get("source") or "llm"),
            "is_preferred": bool(r.get("is_preferred", False)),
        })

    # Attributes (with schema validation)
    for r in (data.get("concept_attributes") or []):
        raw_col = (r.get("column") or r.get("col") or r.get("column_name") or "").strip()
        raw_tbl = (r.get("table")  or r.get("tbl") or r.get("table_name")  or "").strip()
        if not raw_tbl or not raw_col:
            continue

        norm_tbl = _normalize_table_name(raw_tbl, exact_map, base_map)
        if not norm_tbl:
            # Drop mappings to tables that do not exist in schema
            continue

        role = (r.get("role") or "attr").strip() or "attr"
        out["concept_attributes"].append({
            "concept_id":      r.get("concept_id") or _slugify_id(r.get("concept") or ""),
            "tenant_id":       r.get("tenant_id") or tenant_id,
            "table":           norm_tbl,
            "column":          raw_col,
            "role":            role,
            "transform_sql":   r.get("transform_sql",""),
            "effective_start": r.get("effective_start",""),
            "effective_end":   r.get("effective_end",""),
            "notes":           r.get("notes",""),
        })

    # Rules
    for r in (data.get("concept_rules") or []):
        out["concept_rules"].append({
            "rule_id":      r.get("rule_id",""),
            "concept_id":   r.get("concept_id") or _slugify_id(r.get("concept") or ""),
            "tenant_id":    r.get("tenant_id") or tenant_id,
            "rule_type":    r.get("rule_type",""),
            "sql_template": r.get("sql_template",""),
            "parameters":   r.get("parameters",""),
            "notes":        r.get("notes",""),
        })

    return out

def _apply_lexical_constraints(
    cat: List[Dict],
    alias: List[Dict],
    attrs: List[Dict],
    compiled_rules: Dict
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Enforce lexical bans from concepts/policies (compiled_rules.json).

    Example policy:
      - type: lexical_constraint
        action: ban_substitution
        from: "dependent"
        to:   "employee"

    We remove any alias rows that would realize a banned mapping.
    Additionally, we emit a warning if removals affect core business terms
    (employee, benefit, coverage, dependent, member) so you can inspect YAML.
    """
    banned_pairs = set()
    for p in (compiled_rules.get("policies") or []):
        if (p.get("type") or "").lower() == "lexical_constraint" and (p.get("action") or "").lower() == "ban_substitution":
            fr = (p.get("from") or "").strip().lower()
            to = (p.get("to") or "").strip().lower()
            if fr and to:
                banned_pairs.add((fr, to))

    if not banned_pairs:
        return cat, alias, attrs

    # concept_id -> concept_name
    cname_by_id = {c.get("concept_id", ""): (c.get("concept_name", "") or "") for c in cat}
    core_terms = {"employee", "employees", "benefit", "benefits", "coverage", "coverages", "dependent", "dependents", "member", "members"}

    removed_core = []

    def violates(row: Dict) -> bool:
        a = (row.get("alias_text") or "").strip().lower()
        cid = row.get("concept_id") or ""
        cn = (cname_by_id.get(cid, "") or "").strip().lower()
        if (a, cn) in banned_pairs:
            # Track if this is touching a core lexeme for debug
            if any(t in a or t in cn for t in core_terms):
                removed_core.append({"alias": a, "concept_name": cn})
            return True
        return False

    alias_kept = [r for r in alias if not violates(r)]

    if removed_core:
        # Keep behavior (aliases still removed), but emit a loud breadcrumb.
        print(
            "[concepts_llm] NOTE: lexical constraints removed aliases for core terms: "
            + ", ".join(f"{r['alias']}→{r['concept_name']}" for r in removed_core)
        )

    return cat, alias_kept, attrs

def _inject_rules_from_compiled(cat: List[Dict], rules: List[Dict], tenant_id: str, compiled_rules: Dict) -> List[Dict]:
    """
    Map YAML derivations/equivalences into concept_rules rows.
    - derivations: rule_type='derivation' with sql_template from logic_sql/logic_text
    - equivalences: rule_type='equivalence' with parameters carrying relation string
    """
    by_name = { (c.get("concept_name") or "").strip().lower() : c["concept_id"] for c in cat }
    out = list(rules)

    # Derivations
    for d in (compiled_rules.get("derivations") or []):
        name = (d.get("name") or d.get("id") or "derivation").strip()
        applies = d.get("applies_to") or {}
        derived_nm = (applies.get("derived_concept") or name).strip()
        cid = by_name.get(derived_nm.lower()) or _slugify_id(derived_nm)
        sql = d.get("logic_sql") or d.get("logic_text") or ""
        params = {}
        out.append({
            "rule_id":      _slugify_id(f"RULE_{name}"),
            "concept_id":   cid,
            "tenant_id":    tenant_id,
            "rule_type":    "derivation",
            "sql_template": sql,
            "parameters":   json.dumps(params),
            "notes":        (d.get("description") or ""),
        })

    # Equivalences / Composition
    for e in (compiled_rules.get("equivalences") or []):
        name = (e.get("name") or e.get("id") or "equivalence").strip()
        relation = (e.get("relation") or "").strip()  # e.g., "member = employee ∪ dependent"
        member_id = by_name.get("member") or _slugify_id("member")
        out.append({
            "rule_id":      _slugify_id(f"RULE_{name}"),
            "concept_id":   member_id,
            "tenant_id":    tenant_id,
            "rule_type":    "equivalence",
            "sql_template": "",
            "parameters":   json.dumps({"relation": relation}),
            "notes":        (e.get("description") or ""),
        })

    return out

# ------------------------ main entrypoint ------------------------

def build_concepts(cfg: Dict) -> Tuple[str, str, str, str]:
    """
    Returns (catalog_csv, alias_csv, attributes_csv, rules_csv)
    """
    logger = setup_logger("concepts_llm", cfg["llm"]["log_dir"], cfg["llm"]["log_level"])

    llm = LLMClient(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        request_timeout_sec=cfg["llm"]["request_timeout_sec"],
        connect_timeout_sec=cfg["llm"].get("connect_timeout_sec", 30),
        read_timeout_sec=cfg["llm"].get("read_timeout_sec", 180),
        max_retries=cfg["llm"]["max_retries"],
        json_strict=cfg["llm"]["json_strict"],
        api_key=cfg["llm"].get("api_key"),
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

    # Inputs
    schema_path = cfg["inputs"]["schema_json"]
    relationships_path = cfg["inputs"]["relationships_json"]
    docs_jsonl_path = cfg["outputs"]["docs_jsonl"]

    if not Path(schema_path).exists():
        raise FileNotFoundError(f"Schema JSON not found: {schema_path}")
    if not Path(relationships_path).exists():
        raise FileNotFoundError(f"Relationships JSON not found: {relationships_path}")
    if not Path(docs_jsonl_path).exists():
        raise FileNotFoundError(f"Documentation JSONL not found: {docs_jsonl_path} (Run docs step first)")

    schema = _load_json(schema_path)
    relationships = _load_json(relationships_path)
    docs_rows = _load_docs_jsonl(docs_jsonl_path)
    docs_by_table = _docs_index_by_table(docs_rows)

    tenant_id = cfg.get("tenant_id", "")

    # Outputs
    out_catalog   = cfg["outputs"]["concepts"]["catalog_csv"]
    out_alias     = cfg["outputs"]["concepts"]["alias_csv"]
    out_attrs     = cfg["outputs"]["concepts"]["attributes_csv"]
    out_rules     = cfg["outputs"]["concepts"]["rules_csv"]
    Path(out_catalog).parent.mkdir(parents=True, exist_ok=True)

    # New: load compiled rules + sanitized previews
    compiled_rules     = _read_json_if_exists(CONCEPTS_DIR / "compiled_rules.json", {})
    sanitized_amap     = _read_json_if_exists(PROMPTS_DIR / "sanitized_attribute_map.json", [])
    sanitized_syn      = _read_json_if_exists(PROMPTS_DIR / "sanitized_synonyms.json", {})

    # Build preview strings (short hints only)
    alias_preview = [
        f"{r.get('alias')} → {r.get('target')} (conf={r.get('confidence','')})"
        for r in sanitized_amap
        if isinstance(r, dict) and r.get("alias") and r.get("target")
    ]
    syn_preview = []
    if isinstance(sanitized_syn, dict):
        for k, lst in sanitized_syn.items():
            if isinstance(lst, list) and lst:
                syn_preview.append(f"{k}: {', '.join(map(str, lst))}")

    # Batching (+ optional table limit)
    bsz = int(cfg["llm"].get("concepts_batch_size", 10))
    tlimit = int((cfg.get("limits") or {}).get("llm_tables_max", 0) or 0)
    all_tables = sorted((schema.get("tables") or {}).keys())
    if tlimit > 0:
        all_tables = all_tables[:tlimit]
        schema = {"tables": {t: schema["tables"][t] for t in all_tables}}
    table_batches = [all_tables[i:i+bsz] for i in range(0, len(all_tables), bsz)]

    logger.info(f"Concepts: {sum(len(b) for b in table_batches)} tables, batch_size={bsz}, batches={len(table_batches)}")

    all_cat: List[Dict]   = []
    all_alias: List[Dict] = []
    all_attr: List[Dict]  = []
    all_rules: List[Dict] = []

    # System prompt: prefer emitted system_prompt.txt, else fallback to cfg["llm"]["system_prompt"]
    system_prompt_path = PROMPTS_DIR / "system_prompt.txt"
    if system_prompt_path.exists():
        sys_prompt_text = system_prompt_path.read_text(encoding="utf-8")
    else:
        sys_prompt_text = cfg["llm"]["system_prompt"]

    for bi, tables in enumerate(table_batches, start=1):
        logger.info(f"Concepts batch {bi}/{len(table_batches)}: tables={len(tables)}")
        schema_subset = _subset_schema(schema, tables)
        docs_subset   = _subset_docs(docs_by_table, tables)

        if not docs_subset:
            logger.warning(f"Concepts batch {bi}: No docs for selected tables; skipping.")
            continue

        prompt = _build_prompt(
            tenant_id,
            schema_subset,
            relationships,
            docs_subset,
            alias_preview=alias_preview,
            syn_preview=syn_preview,
        )

        data = {}
        try:
            data = llm.json_completion(system_prompt=sys_prompt_text, user_prompt=prompt) or {}
        except Exception as e:
            logger.error(f"Concepts batch {bi} LLM error: {e}")

        got_any = any(len(data.get(k) or []) > 0 for k in ("concept_catalog","concept_alias","concept_attributes","concept_rules"))
        if not got_any:
            logger.warning(f"Concepts batch {bi} returned empty; seeding heuristic backfill for this batch.")
            data = _heuristic_concepts(schema_subset, docs_subset, tenant_id)

        data_norm = _normalize_rows(data, tenant_id, schema)

        # Merge
        all_cat.extend(data_norm.get("concept_catalog", []))
        all_alias.extend(data_norm.get("concept_alias", []))
        all_attr.extend(data_norm.get("concept_attributes", []))
        all_rules.extend(data_norm.get("concept_rules", []))

    # Normalize & de-duplicate (pre-constraints)
    def _norm_rows(rows: List[Dict], fields: List[str]) -> List[Dict]:
        out = []
        for r in rows:
            rr = {k: ("" if r.get(k) is None else r.get(k)) for k in fields}
            out.append(rr)
        return out

    all_cat   = _norm_rows(all_cat,   CAT_FIELDS)
    all_alias = _norm_rows(all_alias, ALIAS_FIELDS)
    all_attr  = _norm_rows(all_attr,  ATTR_FIELDS)
    all_rules = _norm_rows(all_rules, RULE_FIELDS)

    # Apply lexical constraints from compiled rules
    all_cat, all_alias, all_attr = _apply_lexical_constraints(all_cat, all_alias, all_attr, compiled_rules)

    # Inject rule rows from derivations/equivalences in compiled rules
    all_rules = _inject_rules_from_compiled(all_cat, all_rules, tenant_id, compiled_rules)

    # De-duplicate after injections
    all_cat   = _dedup_dict_rows(all_cat,   ["concept_id", "concept_name"])
    all_alias = _dedup_dict_rows(all_alias, ["concept_id", "alias_text"])
    all_attr  = _dedup_dict_rows(all_attr,  ["concept_id", "tenant_id", "table", "column", "role"])
    all_rules = _dedup_dict_rows(all_rules, ["rule_id", "concept_id", "tenant_id"])

    # Write CSV artifacts
    write_csv(all_cat,   out_catalog, CAT_FIELDS)
    write_csv(all_alias, out_alias,   ALIAS_FIELDS)
    write_csv(all_attr,  out_attrs,   ATTR_FIELDS)
    write_csv(all_rules, out_rules,   RULE_FIELDS)

    logger.info(f"Wrote concept_catalog:   {out_catalog} (rows={len(all_cat)})")
    logger.info(f"Wrote concept_alias:     {out_alias} (rows={len(all_alias)})")
    logger.info(f"Wrote concept_attributes:{out_attrs} (rows={len(all_attr)})")
    logger.info(f"Wrote concept_rules:     {out_rules} (rows={len(all_rules)})")

    if not (len(all_cat)+len(all_alias)+len(all_attr)+len(all_rules)):
        raise RuntimeError("Concepts step produced 0 rows; check logs/prompts.")

    return (out_catalog, out_alias, out_attrs, out_rules)

# Optional direct run
if __name__ == "__main__":
    import yaml
    cfg_path = sys.argv[1] if len(sys.argv) >= 2 else "configs/default_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    build_concepts(cfg)
