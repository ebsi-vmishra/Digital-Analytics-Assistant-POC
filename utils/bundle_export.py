import shutil, os, json
from pathlib import Path

def export_vanna_bundle(cfg):
    outdir = Path(cfg["outputs"]["vanna_bundle_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # copy canonical artifacts (assuming DDL and documentation.md are optional)
    to_copy = [
        cfg["inputs"]["schema_json"],
        cfg["inputs"]["relationships_json"],
        cfg["inputs"]["profiles_summary_csv"],
        cfg["outputs"]["synonyms_json"],
        cfg["outputs"]["attribute_map_json"],
        cfg["outputs"]["docs_jsonl"],
        cfg["outputs"]["concepts"]["catalog_csv"],
        cfg["outputs"]["concepts"]["alias_csv"],
        cfg["outputs"]["concepts"]["attributes_csv"],
        cfg["outputs"]["concepts"]["rules_csv"],
    ]

    for p in to_copy:
        if p and os.path.exists(p):
            shutil.copy2(p, outdir / os.path.basename(p))

    # Optional: generate documentation.md from docs_jsonl (simple rollup)
    docs_md = outdir / "documentation.md"
    if os.path.exists(cfg["outputs"]["docs_jsonl"]):
        with open(cfg["outputs"]["docs_jsonl"], "r", encoding="utf-8") as fsrc, open(docs_md, "w", encoding="utf-8") as fdst:
            fdst.write("# Documentation\n\n")
            for line in fsrc:
                obj = json.loads(line)
                fdst.write(f"## Table: {obj.get('table')}\n\n")
                if obj.get("description"):
                    fdst.write(obj["description"] + "\n\n")
                for col in obj.get("columns", []):
                    fdst.write(f"- **{col.get('column')}** ({col.get('type')}) — {col.get('description')}\n")
                fdst.write("\n")
    return str(outdir)
