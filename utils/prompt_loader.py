# utils/prompt_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _truncate_preview(items: List[str], max_items: int = 40) -> List[str]:
    if len(items) <= max_items:
        return items
    return items[:max_items] + [f"... (+{len(items) - max_items} more)"]


def _ascii_clean(text: str) -> str:
    """
    Normalize any non-ASCII symbols in summaries so we don't leak things like
    '∪' or other special glyphs into the final system prompt.

    We do a conservative normalization:
      - Best-effort keep ASCII; drop non-ASCII characters.
      - Callers should keep domain words (union, intersection, etc.) ASCII already.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.encode("ascii", "ignore").decode("ascii")


def _strip_conflicting_sections(concepts_md: str) -> str:
    """
    Remove sections from concepts_prompt.md that may conflict with the
    programmatically constructed lexical/safety rules and output contract.

    Specifically we strip headings like:
      - '## Lexical & Safety Constraints'
      - '## Global Guardrails'
      - '## Output Format'
      - '## Final Guardrails'
    and everything under them until the next '## ' heading (or EOF).
    """
    if not concepts_md.strip():
        return concepts_md

    lines = concepts_md.splitlines()
    cleaned: List[str] = []

    banned_headings = [
        "## Lexical & Safety Constraints",
        "## Global Guardrails",
        "## Output Format",
        "## Final Guardrails",
    ]

    skip = False
    for line in lines:
        stripped = line.strip()

        # Start skipping when we hit one of the banned headings
        if any(stripped.startswith(h) for h in banned_headings):
            skip = True
            continue

        # If we see a *new* top-level heading while skipping, stop skipping
        if skip and stripped.startswith("## ") and not any(
            stripped.startswith(h) for h in banned_headings
        ):
            skip = False

        if not skip:
            cleaned.append(line)

    return "\n".join(cleaned).strip()


def _summarize_derivations(compiled: Dict[str, Any]) -> List[str]:
    deriv = compiled.get("derivations") or []
    if not deriv:
        return []

    out: List[str] = ["### Compact Summary: Derivations"]
    for d in deriv:
        nm = d.get("name") or d.get("id") or "unnamed_derivation"
        desc = _ascii_clean((d.get("description") or "").strip())
        ap = d.get("applies_to") or {}
        out.append(f"- {nm}: {desc} | applies_to={_ascii_clean(json.dumps(ap, sort_keys=True))}")
    return out


def _summarize_equivalences(compiled: Dict[str, Any]) -> List[str]:
    equiv = compiled.get("equivalences") or []
    if not equiv:
        return []

    out: List[str] = ["", "### Compact Summary: Equivalences / Compositions"]
    for e in equiv:
        nm = e.get("name") or e.get("id") or "unnamed_equivalence"
        desc = _ascii_clean((e.get("description") or "").strip())
        relation = _ascii_clean(e.get("relation") or "")
        concepts = e.get("concepts") or e.get("concept_ids") or []
        concepts_str = ", ".join(map(str, concepts)) if concepts else ""
        out.append(
            f"- {nm}: {desc}"
            + (f" | relation={relation}" if relation else "")
            + (f" | concepts=[{concepts_str}]" if concepts_str else "")
        )
    return out


def _extract_lexical_pairs(
    compiled: Dict[str, Any],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Returns (banned_pairs, allowed_pairs) based on compiled['policies'].

    We treat an entry as a lexical constraint if:
      (p['type'] or '').lower() == 'lexical_constraint'

    Within that:
      - If 'allow_substitution' is explicitly truthy -> allowed_pairs.
      - Else if 'ban_substitution' is explicitly truthy -> banned_pairs.
      - Else default to banned_pairs (these are usually 'no_X_to_Y' style).
    """
    banned: Set[Tuple[str, str]] = set()
    allowed: Set[Tuple[str, str]] = set()

    policies = compiled.get("policies") or []
    for p in policies:
        if (p.get("type") or "").lower() != "lexical_constraint":
            continue

        src = (p.get("from") or p.get("source") or "").strip()
        dst = (p.get("to") or p.get("target") or "").strip()
        if not src or not dst:
            continue

        # Normalize to lower-case for stability in the prompt
        src_l = src.lower()
        dst_l = dst.lower()

        allow_flag = str(p.get("allow_substitution", "")).lower()
        ban_flag = str(p.get("ban_substitution", "")).lower()
        policy = str(p.get("policy", "")).lower()

        allow = allow_flag in ("true", "yes", "1")
        ban = ban_flag in ("true", "yes", "1") or policy in ("ban", "block", "forbid")

        if allow and not ban:
            allowed.add((src_l, dst_l))
        else:
            # Default is to BAN, given our policies are mostly 'no_X_to_Y' semantics.
            banned.add((src_l, dst_l))

    return banned, allowed


def _summarize_lexical_constraints(compiled: Dict[str, Any]) -> List[str]:
    """
    Build a clean, directional + mutual view of lexical constraints.

    This is where we enforce the "vice versa" behavior:
      - If (a -> b) and (b -> a) are both banned, we declare them MUTUALLY not interchangeable.
      - If only one direction appears, we call it a one-way ban.
      - Allowed substitutions are listed separately for clarity.

    Examples we care about (given your rules):
      - dependent ↔ employee  => MUTUAL BAN (not interchangeable)
      - plan ↔ benefit        => MUTUAL BAN (not interchangeable)
      - member, participant, subscriber, employee, dependent => allowed equivalence
        (but usually expressed via equivalence rules, not lexical constraints).
    """
    banned_pairs, allowed_pairs = _extract_lexical_pairs(compiled)
    if not banned_pairs and not allowed_pairs:
        return []

    out: List[str] = ["", "### Compact Summary: Lexical / Safety Constraints"]

    # Mutually banned pairs (A->B and B->A)
    seen_mutual: Set[frozenset] = set()
    mutual_lines: List[str] = []
    one_way_lines: List[str] = []

    for src, dst in banned_pairs:
        if (dst, src) in banned_pairs:
            key = frozenset({src, dst})
            if key in seen_mutual:
                continue
            seen_mutual.add(key)
            a, b = sorted(list(key))
            mutual_lines.append(
                f"- MUTUAL BAN: do NOT treat '{a}' and '{b}' as interchangeable in any direction "
                f"(never substitute one for the other)."
            )
        else:
            one_way_lines.append(
                f"- BAN_SUBSTITUTION (one-way): never substitute '{src}' when the user clearly means '{dst}'."
            )

    if mutual_lines:
        out.append("**Mutually non-interchangeable terms (vice-versa bans):**")
        out.extend(mutual_lines)

    if one_way_lines:
        if mutual_lines:
            out.append("")
        out.append("**One-way lexical bans:**")
        out.extend(one_way_lines)

    if allowed_pairs:
        out.append("")
        out.append("**Explicitly allowed substitutions (directional synonyms):**")
        for src, dst in sorted(allowed_pairs):
            out.append(
                f"- ALLOW_SUBSTITUTION: you may treat '{src}' as equivalent to '{dst}' "
                f"when interpreting user questions, unless other filters contradict this."
            )

    # Final explicit guidance for the big cases we care about.
    out.append("")
    out.append("**Key non-interchangeable pairs (reinforced):**")
    out.append("- 'dependent' vs 'employee': never treat them as the same; they are different roles.")
    out.append("- 'plan' vs 'benefit': never treat them as the same; plans group one or more benefits.")
    out.append(
        "- When asked about 'member' or 'participant', consider both employees and dependents "
        "only when the question is explicitly about all covered individuals (not just employees)."
    )

    return out


def _build_alias_synonym_preview(
    attribute_map: List[Dict[str, Any]],
    synonyms: Dict[str, Any],
    max_alias_rows: int = 40,
    max_syn_rows: int = 30,
) -> str:
    alias_lines: List[str] = []
    for r in attribute_map:
        alias = r.get("alias")
        target = r.get("target")
        conf = r.get("confidence", "")
        if alias and target:
            alias_str = _ascii_clean(str(alias))
            target_str = _ascii_clean(str(target))
            alias_lines.append(f"{alias_str}  →  {target_str} (conf={conf})")
    alias_lines = _truncate_preview(alias_lines, max_alias_rows)

    syn_lines: List[str] = []
    if isinstance(synonyms, dict):
        for k, lst in synonyms.items():
            if isinstance(lst, list) and lst:
                key = _ascii_clean(str(k))
                vals = ", ".join(_ascii_clean(str(v)) for v in lst)
                syn_lines.append(f"{key}: {vals}")
    syn_lines = _truncate_preview(syn_lines, max_syn_rows)

    if not alias_lines and not syn_lines:
        return ""

    parts: List[str] = [
        "### Read-only Alias / Synonym Preview",
        "_Use these hints cautiously; lexical/safety constraints ALWAYS win over aliases/synonyms._",
        "",
        "**Aliases (attribute_map)**:",
    ]
    if alias_lines:
        parts.extend([f"- {line}" for line in alias_lines])
    else:
        parts.append("- (none)")

    parts.append("")
    parts.append("**Synonyms**:")
    if syn_lines:
        parts.extend([f"- {line}" for line in syn_lines])
    else:
        parts.append("- (none)")
    parts.append("")

    return "\n".join(parts)


def build_system_prompt(
    prompts_dir: Path,
    concepts_dir: Path,
    include_alias_preview: bool = True,
) -> str:
    """
    Builds a single LLM-ready system prompt by combining:
      - artifacts/prompts/concepts_prompt.md        (after stripping conflicting sections)
      - artifacts/concepts/compiled_rules.json      (compact derivations/equivalences/policies)
      - artifacts/prompts/sanitized_attribute_map.json (preview only)
      - artifacts/prompts/sanitized_synonyms.json      (preview only)

    This function also:
      - Avoids duplicate/conflicting lexical guardrails from older prompts.
      - Enforces vice-versa (mutual) bans where applicable.
      - Normalizes summaries to ASCII-only to avoid special glyphs like '∪'.
      - Appends a strict SQL-only output contract at the end.
    """

    # Core docs
    raw_concepts_md = _read_text(prompts_dir / "concepts_prompt.md")
    concepts_md = _strip_conflicting_sections(raw_concepts_md)
    if not concepts_md.strip():
        concepts_md = "# Concept Layer Guidance\n\n(No explicit concepts_prompt.md found.)\n"

    compiled = _read_json(concepts_dir / "compiled_rules.json", {})
    amap = _read_json(prompts_dir / "sanitized_attribute_map.json", [])
    syns = _read_json(prompts_dir / "sanitized_synonyms.json", {})

    # Compact summaries from compiled rules
    summary_lines: List[str] = []
    summary_lines.extend(_summarize_derivations(compiled))
    summary_lines.extend(_summarize_equivalences(compiled))
    summary_lines.extend(_summarize_lexical_constraints(compiled))

    # Optional alias/synonym preview
    alias_preview = ""
    if include_alias_preview:
        alias_preview = _build_alias_synonym_preview(amap, syns)

    # Assemble final system prompt; order matters:
    #  1) High-level concept guidance (from curated markdown, cleaned).
    #  2) Machine-compiled summaries: derivations, equivalences, lexical policies.
    #  3) Read-only alias/synonym preview.
    #  4) Final, unambiguous guardrails + SQL-only output contract.
    parts: List[str] = []

    # 1) Concept guidance
    parts.append(concepts_md.strip())

    # 2) Summaries
    if summary_lines:
        parts.append("")
        parts.append("\n".join(summary_lines).strip())

    # 3) Alias/synonym preview
    if alias_preview:
        parts.append("")
        parts.append(alias_preview.strip())

    # 4) Final guardrails + SQL contract
    final_guardrails: List[str] = [
        "",
        "### Final Guardrails (Authoritative)",
        "- Never violate lexical bans from the Lexical / Safety Constraints summary above.",
        "- When a pair is marked as MUTUAL BAN, they are *not interchangeable in any direction*.",
        "- When a substitution is explicitly allowed, you may treat the two terms as synonyms, "
        "but still respect table/column semantics from the schema and documentation.",
        "- Prefer concept derivations and compiled rules over raw string matching.",
        "- Always ground your reasoning in the provided reporting schema tables and columns; "
        "never invent new table or column names.",
        "",
        "### Output Format (STRICT, SQL-only)",
        "- You are a SQL generator for the Empyrean reporting schema.",
        "- Always respond with a single T-SQL SELECT statement and nothing else.",
        "- Do NOT include markdown fences, comments, explanations, or prose in your response.",
        "- Never refuse to write SQL because you \"cannot see the data\"; always produce the "
        "best-effort valid query based on the schema and documentation.",
        "- Always use existing tables and columns only, preferring fully qualified names like "
        "reporting.Employee, reporting.Benefit, reporting.EmployeeCoverageFact, etc.",
        "- If a question is ambiguous, choose the safest, most conservative interpretation that "
        "still returns a meaningful result rather than hallucinating new entities.",
        "",
    ]
    parts.append("\n".join(final_guardrails).strip())

    final_prompt = "\n".join([p for p in parts if p is not None and str(p).strip() != ""]).strip() + "\n"
    return final_prompt
