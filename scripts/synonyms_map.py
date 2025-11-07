# synonyms_map.py
import json, re, math, glob, argparse
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("artifacts")
SCHEMA_FILE = Path("artifacts/schema.json")
REL_FILE = Path("artifacts/relationships.json")
PROFILES_DIR = Path("artifacts/profiles")

# -------------------------
# Tokenization & variants
# -------------------------
WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")

def tokenize_name(name: str):
    # strip schema for table tokens but keep full for column keys elsewhere
    s = name.split(".")[-1]
    s = re.sub(r"[_\\-]+", " ", s)
    return [p.lower() for p in WORD_RE.findall(s)]

def plural_forms(token: str):
    forms = {token}
    if token.endswith("y") and len(token) > 2:
        forms.add(token[:-1] + "ies")
    if not token.endswith("s"):
        forms.add(token + "s")
    if token.endswith("s"):
        forms.add(token.rstrip("s"))
    return forms

def variants_from_tokens(tokens):
    forms = set()
    base = " ".join(tokens).strip()
    if base:
        forms.add(base)
        forms.update(plural_forms(base))
    # individual tokens and their plurals
    for t in tokens:
        if t:
            forms.add(t)
            forms.update(plural_forms(t))
    # compacted forms
    if tokens:
        forms.add("_".join(tokens))
        forms.add("".join(tokens))
    # keep non-empty only
    return {f for f in forms if f}

def acronym(tokens):
    letters = [t[0] for t in tokens if t and t[0].isalnum()]
    return "".join(letters) if letters else ""

def type_hint_aliases(sql_type: str):
    t = (sql_type or "").lower()
    out = set()
    if not t:
        return out
    if "date" in t or "time" in t:
        out.update({"date","datetime","timestamp"})
    if "char" in t or "text" in t or "string" in t or "nchar" in t or "nvarchar" in t or "varchar" in t:
        out.update({"text","string"})
    if "int" in t or "decimal" in t or "numeric" in t or "float" in t or "money" in t:
        out.update({"number","numeric","amount","value"})
    if "uniqueidentifier" in t:
        out.update({"uuid","guid","id"})
    return out

# -------------------------
# Relationship-driven expansions
# -------------------------
def fk_expansions(relationships):
    """
    From FK: schemaA.TableA.colX -> schemaB.TableB.colY
    Generate table/table-column aliases: 'tableb', 'tableb id', etc. for the FK side.
    """
    fk_map = defaultdict(set)
    for fk in (relationships.get("foreign_keys") or []):
        fk_table = fk.get("fk_table")
        pk_table = fk.get("pk_table")
        fk_col = fk.get("fk_column")

        if not fk_table or not pk_table:
            continue

        # expand child table by parent table name
        fk_map[f"{fk_table}"].add(pk_table)

        # Also link the specific columns (child.col -> parent.table alias)
        if fk_col:
            fk_map[f"{fk_table}.{fk_col}"].add(pk_table)
    return fk_map  # mapping: object -> {related_table_names}

# -------------------------
# Profiles (for categorical-only filter)
# -------------------------
def load_profiles_map():
    out = {}
    for p in glob.glob(str(PROFILES_DIR / "*.json")):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            if "table" in d:
                out[d["table"]] = d
        except Exception:
            pass
    return out

def is_enum_column(profiles_map, full_table: str, col: str) -> bool:
    pm = profiles_map.get(full_table) or {}
    cmeta = (pm.get("columns") or {}).get(col, {})
    return bool(cmeta.get("enum_candidate"))

# -------------------------
# Scoring
# -------------------------
def score_alias(alias: str, target: str, source: str, tokens_target):
    """
    Score combines:
    - exact match
    - token overlap
    - acronym match
    - source boost (fk/type/tokens)
    """
    a = alias.lower().strip()
    t = target.lower().strip()

    if not a or not t:
        return 0.0

    if a == t or a == t.split(".")[-1]:
        base = 0.98
    else:
        # token overlap
        alias_tokens = set(tokenize_name(a))
        overlap = len(alias_tokens.intersection(tokens_target))
        base = 0.50 + 0.10 * min(overlap, 5)  # up to +0.5

    # acronym boost if alias equals acronym of target tokens
    acro = acronym(list(tokens_target))
    if acro and a == acro:
        base = max(base, 0.80)

    # source boosts
    if source == "exact":
        base = max(base, 0.98)
    elif source == "acronym":
        base = max(base, 0.82)
    elif source == "fk":
        base = max(base, 0.75)
    elif source == "type":
        base = max(base, 0.72)
    else:  # token/variant
        base = max(base, 0.60)

    return round(min(base, 0.99), 2)

# -------------------------
# Build synonyms
# -------------------------
def build_synonyms(schema, relationships, profiles_map=None, categorical_only=False):
    """
    Returns dict: { target_object -> { "aliases": [ ... ] } }
    target_object is either 'schema.Table' (table) or 'schema.Table.Column' (column)
    """
    syn = defaultdict(lambda: {"aliases": set()})
    fk_map = fk_expansions(relationships)

    # 1) Table & column token/variant/acronym aliases
    for full, tdef in (schema.get("tables") or {}).items():
        # table tokens
        t_tokens = tokenize_name(full)
        t_acro = acronym(t_tokens)
        for v in variants_from_tokens(t_tokens):
            syn[full]["aliases"].add(v)
        if t_acro:
            syn[full]["aliases"].add(t_acro)

        for c, meta in (tdef.get("columns") or {}).items():
            # if filtering to categorical only, skip non-enum columns
            if categorical_only and not is_enum_column(profiles_map or {}, full, c):
                continue

            key = f"{full}.{c}"
            c_tokens = tokenize_name(c)
            c_acro = acronym(c_tokens)
            for v in variants_from_tokens(c_tokens):
                syn[key]["aliases"].add(v)
            if c_acro:
                syn[key]["aliases"].add(c_acro)

            # type-based hints
            for hint in type_hint_aliases(meta.get("data_type")):
                syn[key]["aliases"].add(hint)

            # canonical ID expansion
            cl = c.lower()
            if cl in ("id","guid","uuid") or cl.endswith("_id"):
                syn[key]["aliases"].update({"id","identifier","guid","uuid"})

    # 2) FK-based expansions (child gets parent table name as alias; child.col too)
    for obj, related_tables in fk_map.items():
        # if categorical_only and obj is a column, ensure it's enum
        if categorical_only and "." in obj:
            maybe_table, maybe_col = obj.rsplit(".", 1)
            if not is_enum_column(profiles_map or {}, maybe_table, maybe_col):
                continue

        for rt in related_tables:
            rt_tokens = tokenize_name(rt)
            syn[obj]["aliases"].update(variants_from_tokens(rt_tokens))
            ac = acronym(rt_tokens)
            if ac: syn[obj]["aliases"].add(ac)

    # freeze to sorted lists
    return {k: {"aliases": sorted(list(v["aliases"]))} for k,v in syn.items()}

# -------------------------
# Build attribute map
# -------------------------
def build_attribute_map(schema, synonyms):
    """
    Builds list of {alias, target, confidence, source}
    - target is 'schema.Table' or 'schema.Table.Column'
    - alias is lowercased
    - source in {exact, acronym, type, tokens, fk?} (fk handled via score/overlap)
    Deduped to highest-confidence per (alias, target)
    """
    # Precompute target tokens for overlap scoring
    tgt_tokens = {}
    for target in synonyms.keys():
        parts = target.split(".")
        tokens = []
        if len(parts) >= 2:
            tokens += tokenize_name(parts[-2])   # table name tokens
        tokens += tokenize_name(parts[-1])       # column or table tokens
        tgt_tokens[target] = set(tokens)

    # Collect raw rows
    rows = []
    for target, payload in synonyms.items():
        tokens_target = tgt_tokens.get(target, set())
        # Determine the base acronym for 'acronym' source check
        target_acro = acronym(list(tokens_target)) if tokens_target else ""

        for alias in payload["aliases"]:
            a = alias.strip().lower()
            if not a:
                continue
            # infer a simple 'source' label
            if a == target.lower() or a == target.split(".")[-1].lower():
                source = "exact"
            elif target_acro and a == target_acro.lower():
                source = "acronym"
            elif a in {"uuid","guid","identifier","date","datetime","timestamp",
                       "number","numeric","amount","value","text","string"}:
                source = "type"
            else:
                source = "tokens"

            conf = score_alias(a, target, source, tokens_target)
            rows.append({
                "alias": a,
                "target": target,
                "confidence": conf,
                "source": source
            })

    # Deduplicate: keep highest confidence per (alias, target)
    best = {}
    for r in rows:
        key = (r["alias"], r["target"])
        prev = best.get(key)
        if (prev is None) or (r["confidence"] > prev["confidence"]):
            best[key] = r

    # Sort for readability
    out_rows = list(best.values())
    out_rows.sort(key=lambda r: (-r["confidence"], r["alias"], r["target"]))
    return out_rows

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categorical-only", action="store_true",
                        help="Only emit aliases/attribute_map for columns detected as categorical/enum in profiles.")
    args = parser.parse_args()

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Missing {SCHEMA_FILE}. Run: python -m src.ssms_ingest")

    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    relationships = json.loads(REL_FILE.read_text(encoding="utf-8")) if REL_FILE.exists() else {"foreign_keys": []}

    profiles_map = load_profiles_map() if args.categorical_only else {}
    synonyms = build_synonyms(schema, relationships, profiles_map, categorical_only=args.categorical_only)
    attribute_map = build_attribute_map(schema, synonyms)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "synonyms.json").write_text(json.dumps(synonyms, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "attribute_map.json").write_text(json.dumps(attribute_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # quick summary
    tgt_ct = len(synonyms)
    alias_ct = sum(len(v["aliases"]) for v in synonyms.values())
    print(f"Wrote synonyms for {tgt_ct} targets with {alias_ct} total aliases → artifacts/synonyms.json")
    print(f"Wrote {len(attribute_map)} alias→target rows → artifacts/attribute_map.json")
    if args.categorical_only:
        print("Mode: categorical-only (columns filtered by enum_candidate from profiles)")

if __name__ == "__main__":
    main()
