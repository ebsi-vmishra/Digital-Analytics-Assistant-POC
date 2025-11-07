import os, io, json, yaml, zipfile, subprocess, pandas as pd, streamlit as st
from pathlib import Path

st.set_page_config(page_title="Semantic Layer POC – Orchestrator", layout="wide")

DEFAULT_CFG_PATH = "configs/default_config.yaml"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def run_pipeline(cfg_path):
    # Import and run directly to keep logs in-app
    from scripts.pipeline import run_pipeline
    run_pipeline(cfg_path)

def layer_file_section(title, paths):
    st.subheader(title)
    cols = st.columns(min(3, len(paths) or 1))
    for i, p in enumerate(paths):
        with cols[i % len(cols)]:
            if os.path.exists(p):
                st.code(p, language="text")
                if p.endswith(".csv"):
                    df = pd.read_csv(p)
                    st.dataframe(df)
                elif p.endswith(".json") or p.endswith(".jsonl"):
                    # show first items
                    try:
                        if p.endswith(".json"):
                            st.json(json.load(open(p, "r", encoding="utf-8")))
                        else:
                            lines = [json.loads(l) for _, l in zip(range(5), open(p, "r", encoding="utf-8")) if l.strip()]
                            st.json(lines)
                    except Exception as e:
                        st.warning(f"Unable to preview {os.path.basename(p)}: {e}")
            else:
                st.text(f"Missing: {p}")

def make_bundle_zip(dir_path: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dir_path):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, dir_path))
    mem.seek(0)
    return mem.read()

st.title("Semantic Layer POC – End-to-End Orchestrator")

with st.sidebar:
    st.header("Configuration")
    cfg_source = st.radio("Config Source", ["Use default YAML", "Upload YAML"])
    if cfg_source == "Upload YAML":
        uploaded = st.file_uploader("Upload YAML", type=["yaml", "yml"])
        if uploaded:
            cfg = yaml.safe_load(uploaded)
            save_yaml(cfg, DEFAULT_CFG_PATH)
            st.success("Uploaded config applied.")
    cfg = load_yaml(DEFAULT_CFG_PATH)

    st.text_input("Tenant ID", value=cfg.get("tenant_id", ""), key="tenant_id")
    st.selectbox("Synonyms Mode", ["llm", "heuristic"], index=(0 if cfg["flow"].get("synonyms_mode","llm")=="llm" else 1), key="syn_mode")
    st.toggle("LLM Enabled", value=cfg["llm"]["enabled"], key="llm_enabled")

    # Database selection
    st.subheader("Database")
    st.text_input("DSN (optional)", value=cfg["database"].get("dsn",""), key="dsn")
    st.text_input("ODBC Conn String (optional)", value=cfg["database"].get("odbc_connection_string",""), key="odbc")

    if st.button("Save Config"):
        cfg["tenant_id"] = st.session_state["tenant_id"]
        cfg["flow"]["synonyms_mode"] = st.session_state["syn_mode"]
        cfg["llm"]["enabled"] = st.session_state["llm_enabled"]
        cfg["database"]["dsn"] = st.session_state["dsn"]
        cfg["database"]["odbc_connection_string"] = st.session_state["odbc"]
        save_yaml(cfg, DEFAULT_CFG_PATH)
        st.success("Saved.")

st.markdown("### Run Process Flow")
col_run, col_dl = st.columns([1,1])

with col_run:
    if st.button("Run Pipeline"):
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                run_pipeline(DEFAULT_CFG_PATH)
                status.update(label="Pipeline complete.", state="complete")
            except Exception as e:
                status.update(label="Pipeline failed.", state="error")
                st.exception(e)

with col_dl:
    if cfg["flow"]["steps"].get("export_vanna_bundle"):
        bundle_dir = cfg["outputs"]["vanna_bundle_dir"]
        if os.path.isdir(bundle_dir):
            blob = make_bundle_zip(bundle_dir)
            st.download_button("Download Vanna Training Bundle", data=blob, file_name="vanna_bundle.zip", mime="application/zip")

st.divider()

# Deep dives per layer + why it matters (training help)
st.header("Explore Layers & Why They Matter")
exp = st.expander("Help: Layer Importance")
exp.markdown("""
- **Schema & Profiles (Deterministic):** Exact structure and real distributions; foundation for all enrichment.
- **Documentation (LLM):** Business-readable context; improves searchability and NL→SQL generation.
- **Concepts (LLM):** Normalizes cross-client drift (e.g., Smoker vs TobaccoUser); creates stable business truth.
- **Synonyms (LLM via Docs+Concepts):** Rich aliasing with provenance/confidence; fuels robust NL→SQL and Vanna recall.
- **Vanna Bundle:** Portable training artifacts (docs, profiles, synonyms, concepts) for LLM-based SQL generation.
""")

st.subheader("Artifacts")

# Files from config
layer_file_section("Documentation (schema_docs.jsonl)", [cfg["outputs"]["docs_jsonl"]])
layer_file_section("Concept Layer", [
    cfg["outputs"]["concepts"]["catalog_csv"],
    cfg["outputs"]["concepts"]["alias_csv"],
    cfg["outputs"]["concepts"]["attributes_csv"],
    cfg["outputs"]["concepts"]["rules_csv"],
])
layer_file_section("Synonyms", [
    cfg["outputs"]["synonyms_json"],
    cfg["outputs"]["attribute_map_json"],
])
if cfg["flow"]["steps"].get("export_vanna_bundle"):
    layer_file_section("Vanna Bundle Directory", [cfg["outputs"]["vanna_bundle_dir"]])
