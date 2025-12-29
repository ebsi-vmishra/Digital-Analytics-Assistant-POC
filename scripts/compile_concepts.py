# scripts/compile_concepts.py
"""
Compile concept-layer artifacts into:
  - artifacts/concepts/compiled_rules.json
  - artifacts/prompts/concepts_prompt.md
  - artifacts/prompts/sanitized_attribute_map.json
  - artifacts/prompts/sanitized_synonyms.json

Goals:
  - Keep lexical bans (e.g., do NOT map 'dependent' → 'employee')
  - Avoid over-sanitizing core alias/synonym mappings
  - Keep output shapes compatible with utils/prompt_loader.build_system_prompt()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml


# -------------------- small IO helpers --------------------


def _read_yaml(path: Path):
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# -------------------- compile rules/policies --------------------


def _collect_yaml_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for ext in (".yml", ".yaml"):
        files.extend(sorted(root.rglob(f"*{ext}")))
    return files


def _normalize_policy(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize lexical constraint policies so prompt_loader can display them sensibly.
    We keep the fields as-is and only ensure 'type' is present.
    """
    out = dict(p)
    ptype = (out.get("type") or "").strip().lower()
    if not ptype:
        out["type"] = "policy"
    return out


def _compile_rules_and_policies(rules_dir: Path, policies_dir: Path) -> Dict[str, Any]:
    """
    Produce a compact JSON object:
      {
        "derivations": [...],
        "equivalences": [...],
        "policies": [...]
      }

    We are intentionally loose about YAML structure so this doesn't break if files differ.
    """
    compiled = {
        "derivations": [],
        "equivalences": [],
        "policies": [],
    }

    # Helper to classify a rule-ish dict
    def _classify_rule(obj: Dict[str, Any]) -> str:
        t = (obj.get("type") or "").lower()
        if "derivation" in t:
            return "derivations"
        if "equivalence" in t or "composition" in t:
            return "equivalences"
        # lexical constraints and other policies go to "policies"
        return "policies"

    # Process rules
    for p in _collect_yaml_files(rules_dir):
        data = _read_yaml(p)
        if data is None:
            continue
        if isinstance(data, dict):
            # Direct dict might contain typed rule
            bucket = _classify_rule(data)
            compiled[bucket].append(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    bucket = _classify_rule(item)
                    compiled[bucket].append(item)

    # Process policies
    for p in _collect_yaml_files(policies_dir):
        data = _read_yaml(p)
        if data is None:
            continue
        if isinstance(data, dict):
            compiled["policies"].append(_normalize_policy(data))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    compiled["policies"].append(_normalize_policy(item))

    return compiled


# -------------------- sanitization helpers --------------------


def _extract_banned_pairs(compiled: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    From compiled['policies'], extract lexical BAN_SUBSTITUTION style constraints:
      - type: lexical_constraint
      - from: "dependent"
      - to: "employee"

    Returns list of (from_lower, to_lower).
    """
    banned: List[Tuple[str, str]] = []
    for p in compiled.get("policies") or []:
        ptype = (p.get("type") or "").lower()
        if ptype != "lexical_constraint":
            continue
        fr = (p.get("from") or "").strip()
        to = (p.get("to") or "").strip()
        if fr and to:
            banned.append((fr.lower(), to.lower()))
    return banned


CORE_TERMS = {
    "employee",
    "employees",
    "benefit",
    "benefits",
    "coverage",
    "dependent",
    "dependents",
    "member",
    "members",
}


def _sanitize_attribute_map(raw_map: List[Dict[str, Any]], banned_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Input shape: list of objects, ideally containing:
      - alias
      - target
      - confidence (optional)

    We:
      - Drop only mappings that exactly match a banned lexical substitution pair.
      - Keep everything else, especially schema-qualified targets and core business terms.
    """
    banned_set = set(banned_pairs)
    sanitized: List[Dict[str, Any]] = []

    for rec in raw_map or []:
        if not isinstance(rec, dict):
            continue
        alias = (rec.get("alias") or rec.get("term") or "").strip()
        target = (rec.get("target") or rec.get("field") or "").strip()
        if not alias or not target:
            continue

        alias_l = alias.lower()
        target_l = target.lower()

        # Skip only direct banned substitution (e.g., dependent → employee)
        if (alias_l, target_l) in banned_set:
            continue

        # Keep confidence if present and numeric; otherwise leave as-is
        conf = rec.get("confidence")
        new_rec = {
            "alias": alias,
            "target": target,
        }
        if conf is not None:
            new_rec["confidence"] = conf

        # Optionally mark "protected" mappings (for debugging / introspection only)
        if any(t in alias_l for t in CORE_TERMS) or any(t in target_l for t in CORE_TERMS) or "." in target:
            new_rec["protected"] = True

        sanitized.append(new_rec)

    return sanitized


def _sanitize_synonyms(raw_syns: Dict[str, Any], banned_pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Input shape: dict canonical_term -> [synonym1, synonym2, ...]

    We:
      - Drop synonyms that exactly realize a banned lexical substitution pair.
        (e.g., canonical='employee', synonym='dependent' with policy dependent→employee)
      - Keep all other synonyms.
    """
    banned_set = set(banned_pairs)
    out: Dict[str, List[str]] = {}

    for canonical, vals in (raw_syns or {}).items():
        if not isinstance(vals, list):
            continue
        canon = (canonical or "").strip()
        if not canon:
            continue
        canon_l = canon.lower()
        cleaned: List[str] = []
        for syn in vals:
            s = (str(syn) or "").strip()
            if not s:
                continue
            s_l = s.lower()
            # Drop only exact banned pair directions
            if (s_l, canon_l) in banned_set or (canon_l, s_l) in banned_set:
                continue
            cleaned.append(s)
        if cleaned:
            out[canon] = cleaned

    return out


# -------------------- concepts prompt builder --------------------


def _build_concepts_prompt(compiled: Dict[str, Any]) -> str:
    """
    Build a human/LLM-readable markdown snippet summarizing core concept rules.
    This is what goes into artifacts/prompts/concepts_prompt.md.
    """
    lines: List[str] = []

    lines.append("# Concept Layer Guidance")
    lines.append("")
    lines.append(
        "You map business questions to the reporting schema using the concept catalog, "
        "derivations, equivalences, and lexical policies. Prefer documented mappings "
        "over naive string matching."
    )
    lines.append("")

    deriv = compiled.get("derivations") or []
    equiv = compiled.get("equivalences") or []
    policies = compiled.get("policies") or []

    if deriv:
        lines.append("## Derivations (Compact Summary)")
        for d in deriv[:40]:
            nm = (d.get("name") or d.get("id") or "unnamed_derivation").strip()
            desc = (d.get("description") or "").strip()
            applies = d.get("applies_to") or {}
            lines.append(f"- **{nm}**: {desc} | applies_to={json.dumps(applies)}")
        if len(deriv) > 40:
            lines.append(f"- ... (+{len(deriv) - 40} more derivations)")
        lines.append("")

    if equiv:
        lines.append("## Equivalences / Compositions (Compact Summary)")
        for e in equiv[:30]:
            nm = (e.get("name") or e.get("id") or "unnamed_equivalence").strip()
            desc = (e.get("description") or "").strip()
            relation = (e.get("relation") or "").strip()
            lines.append(f"- **{nm}**: {desc} | relation={relation}")
        if len(equiv) > 30:
            lines.append(f"- ... (+{len(equiv) - 30} more equivalences)")
        lines.append("")

    if policies:
        lines.append("## Lexical / Safety Policies (Compact Summary)")
        for p in policies[:50]:
            ptype = (p.get("type") or "").strip()
            if ptype.lower() == "lexical_constraint":
                fr = (p.get("from") or "").strip()
                to = (p.get("to") or "").strip()
                lines.append(f"- BAN_SUBSTITUTION: from='{fr}' to='{to}'")
            else:
                name = (p.get("name") or p.get("id") or "policy").strip()
                desc = (p.get("description") or "").strip()
                lines.append(f"- {name}: {desc}")
        if len(policies) > 50:
            lines.append(f"- ... (+{len(policies) - 50} more policies)")
        lines.append("")

    if not (deriv or equiv or policies):
        lines.append("_No explicit derivations, equivalences, or policies were found in the concept layer._")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


# -------------------- main compile entrypoint --------------------


def compile_concepts(cfg_path: str) -> Dict[str, str]:
    """
    Main entrypoint called from pipeline.py.

    Steps:
      1) Load config and resolve directories.
      2) Compile YAML rules + policies → artifacts/concepts/compiled_rules.json
      3) Load attribute_map + synonyms (LLM or heuristic) and sanitize:
         → artifacts/prompts/sanitized_attribute_map.json
         → artifacts/prompts/sanitized_synonyms.json
      4) Build concepts_prompt.md in prompts_dir.
    Returns a dict of paths written.
    """
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}

    base_dir = Path(cfg.get("base_dir") or ".").resolve()
    concepts_cfg = cfg.get("concepts") or {}
    outputs = cfg.get("outputs") or {}
    inputs = cfg.get("inputs") or {}

    # Directories
    rules_dir = base_dir / concepts_cfg.get("rules_dir", "concepts/rules")
    policies_dir = base_dir / concepts_cfg.get("policies_dir", "concepts/policies")
    concepts_out_dir = _ensure_dir(base_dir / outputs.get("concepts_out_dir", "artifacts/concepts"))
    prompts_dir = _ensure_dir(base_dir / outputs.get("prompts_dir", "artifacts/prompts"))

    # 1) Compile rules + policies
    compiled = _compile_rules_and_policies(rules_dir, policies_dir)

    compiled_path = concepts_out_dir / "compiled_rules.json"
    compiled_path.write_text(json.dumps(compiled, indent=2), encoding="utf-8")
    print(f"[compile_concepts] Wrote compiled_rules.json → {compiled_path}")

    # Extract banned lexical pairs (for sanitization)
    banned_pairs = _extract_banned_pairs(compiled)
    if banned_pairs:
        print(f"[compile_concepts] Lexical banned pairs: {banned_pairs}")

    # 2) Load attribute_map + synonyms (prefer LLM outputs; fall back to heuristic inputs)
    # Attribute map
    attr_llm_path = base_dir / outputs.get("attribute_map_json", "artifacts/attribute_map.json")
    attr_heuristic_path = base_dir / inputs.get("heuristic_attribute_map_json", "artifacts/attribute_map.json")
    raw_attr_map = _read_json(attr_llm_path, None)
    if raw_attr_map is None:
        raw_attr_map = _read_json(attr_heuristic_path, [])
    if not isinstance(raw_attr_map, list):
        raw_attr_map = []

    # Synonyms
    syn_llm_path = base_dir / outputs.get("synonyms_json", "artifacts/synonyms.json")
    syn_heuristic_path = base_dir / inputs.get("heuristic_synonyms_json", "artifacts/synonyms.json")
    raw_syns = _read_json(syn_llm_path, None)
    if raw_syns is None:
        raw_syns = _read_json(syn_heuristic_path, {})
    if not isinstance(raw_syns, dict):
        raw_syns = {}

    # 3) Sanitize attribute_map + synonyms with lexical bans
    sanitized_attr = _sanitize_attribute_map(raw_attr_map, banned_pairs)
    sanitized_syns = _sanitize_synonyms(raw_syns, banned_pairs)

    sanitized_attr_path = prompts_dir / "sanitized_attribute_map.json"
    sanitized_syns_path = prompts_dir / "sanitized_synonyms.json"

    sanitized_attr_path.write_text(json.dumps(sanitized_attr, indent=2), encoding="utf-8")
    sanitized_syns_path.write_text(json.dumps(sanitized_syns, indent=2), encoding="utf-8")

    print(f"[compile_concepts] Wrote sanitized_attribute_map.json → {sanitized_attr_path}")
    print(f"[compile_concepts] Wrote sanitized_synonyms.json → {sanitized_syns_path}")

    # 4) Build concepts_prompt.md for system prompt
    concepts_prompt = _build_concepts_prompt(compiled)
    concepts_prompt_path = prompts_dir / "concepts_prompt.md"
    concepts_prompt_path.write_text(concepts_prompt, encoding="utf-8")
    print(f"[compile_concepts] Wrote concepts_prompt.md → {concepts_prompt_path}")

    return {
        "compiled_rules": str(compiled_path),
        "sanitized_attribute_map": str(sanitized_attr_path),
        "sanitized_synonyms": str(sanitized_syns_path),
        "concepts_prompt": str(concepts_prompt_path),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compile concept-layer artifacts (rules, policies, alias/synonym sanitization).")
    ap.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML config (e.g., configs/default_config.yaml)",
    )
    args = ap.parse_args()
    compile_concepts(args.config)
