# 🧠 Semantic Layer POC – How to Run

This guide explains how to set up, configure, and execute the **Semantic Layer Proof of Concept (POC)** end-to-end — from raw SQL schema ingestion to generation of a **Vanna-compatible training bundle**.

---

## 📋 1. Prerequisites

### ✅ System Requirements
- Python **3.10+**  
- Access to a **SQL Server database** (with valid credentials)  
- Internet access (for OpenAI API calls)  
- Corporate CA certificate if behind a proxy (optional)  

### ✅ Python Dependencies
Inside the project folder:
```bash
python -m venv .venv
.venv\Scripts\activate   # (Windows)
# or
source .venv/bin/activate  # (Mac/Linux)

pip install -r requirements.txt
```

**requirements.txt (minimal example):**
```
openai>=1.12.0
httpx>=0.26.0
sqlalchemy>=2.0.20
pandas>=2.2.0
pyodbc
pyyaml
python-docx
```

---

## ⚙️ 2. Configuration

### File: `configs/default_config.yaml`

Edit this YAML to point to your database and API key.

```yaml
run_id: "demo_run"
output_dir: "outputs"
tenant_id: "ReportingDevCo"

database:
  dsn: "ReportingDevCo"
  odbc_connection_string: "Driver={ODBC Driver 17 for SQL Server};Server=YOURSERVER;Database=YOURDB;UID=YOURUSER;PWD=YOURPASS;Trusted_Connection=no;"

llm:
  enabled: true
  model: "gpt-4.1"
  api_key: "sk-xxxx..."                # required
  temperature: 0.1
  request_timeout_sec: 60
  read_timeout_sec: 180
  connect_timeout_sec: 30
  max_retries: 4
  json_strict: true
  verify_ssl: true
  ca_bundle_path: "C:/certs/corp-root.cer"   # optional
  proxies: {}                                # leave blank if no proxy
  log_dir: "artifacts/logs"
  log_level: "INFO"

flow:
  steps:
    ingest_schema: true
    profile_data: true
    build_docs_llm: true
    build_concepts_llm: true
    build_synonyms_llm: true
    export_vanna_bundle: true
  synonyms_mode: "llm"

inputs:
  schema_json: "artifacts/schema.json"
  relationships_json: "artifacts/relationships.json"
  profiles_dir: "artifacts/profiles"
  profiles_summary_csv: "artifacts/profiles_summary.csv"
  heuristic_synonyms_json: "artifacts/synonyms.json"
  heuristic_attribute_map_json: "artifacts/attribute_map.json"

outputs:
  docs_jsonl: "artifacts/schema_docs.jsonl"
  concepts:
    catalog_csv: "artifacts/concepts/concept_catalog.csv"
    alias_csv: "artifacts/concepts/concept_alias.csv"
    attributes_csv: "artifacts/concepts/concept_attributes.csv"
    rules_csv: "artifacts/concepts/concept_rules.csv"
  synonyms_json: "artifacts/synonyms.json"
  attribute_map_json: "artifacts/attribute_map.json"
  vanna_bundle_dir: "artifacts/vanna_bundle"
```

> 💡 **Important:**  
> - Keep the API key secure — do not commit it to GitHub.  
> - You can also set it as an environment variable:  
>   `setx OPENAI_API_KEY "sk-xxxx"`

---

## 🗄️ 3. Project Structure

Ensure the following folder layout:

```
semantic_layer_poc_v2/
│
├── configs/
│   └── default_config.yaml
│
├── scripts/
│   ├── ssms_ingest.py
│   ├── quick_analysis.py
│   ├── auto_docs_llm.py
│   ├── build_concepts_llm.py
│   ├── synonyms_llm.py
│   └── pipeline.py
│
├── libs/
│   ├── llm_client.py
│   └── log_utils.py
│
├── utils/
│   ├── io_utils.py
│   └── bundle_export.py
│
└── artifacts/
    └── (generated automatically)
```

---

## ▶️ 4. How to Run the POC

### Option 1 — Full Pipeline (Recommended)
Run everything end-to-end:
```bash
python -m scripts.pipeline -c configs/default_config.yaml
```

### Option 2 — Run Layers Individually
If you only want to run specific layers:

```bash
python scripts/ssms_ingest.py
python scripts/quick_analysis.py
python scripts/auto_docs_llm.py -c configs/default_config.yaml
python scripts/build_concepts_llm.py -c configs/default_config.yaml
python scripts/synonyms_llm.py -c configs/default_config.yaml
```

---

## 🧩 5. Outputs Overview

After successful execution, you’ll have:

```
artifacts/
├── schema.json
├── relationships.json
├── profiles/
│   ├── schema.table.json
│   └── ...
├── profiles_summary.csv
├── schema_docs.jsonl
├── concepts/
│   ├── concept_catalog.csv
│   ├── concept_alias.csv
│   ├── concept_attributes.csv
│   └── concept_rules.csv
├── synonyms.json
├── attribute_map.json
└── vanna_bundle/
    ├── schema.json
    ├── relationships.json
    ├── documentation.md
    ├── profiles_summary.csv
    ├── synonyms.json
    ├── attribute_map.json
    ├── concept_catalog.csv
    ├── concept_alias.csv
    ├── concept_attributes.csv
    └── concept_rules.csv
```

---

## ⚡ 6. Common Issues & Fixes

| Problem | Likely Cause | Fix |
|----------|---------------|-----|
| SSL error (`CERTIFICATE_VERIFY_FAILED`) | Behind corporate proxy | Set `llm.ca_bundle_path` to your `.cer` or `.pem` file |
| Proxy errors (`Unknown scheme`) | Empty strings in YAML | Use `proxies: {}` instead of `http: ""` |
| Empty outputs | Timeout or rate limit | Increase `read_timeout_sec` to 180+, rerun |
| Empty llm outputs | Wrong API key not passed | Ensure `api_key=cfg["llm"].get("api_key")` in `_llm.py` |
| Slow runs | Large schema | Reduce batch sizes (`docs_batch_size`, `concepts_batch_size`, `synonyms_batch_size`) |

---

## ✅ 7. Verification Checklist

| Verify | Expected Result |
|---------|----------------|
| `artifacts/schema.json` exists | ✅ schema extracted |
| `artifacts/profiles_summary.csv` has rows | ✅ profiling done |
| `artifacts/schema_docs.jsonl` not empty | ✅ documentation generated |
| `artifacts/concepts/concept_catalog.csv` has rows | ✅ concepts built |
| `artifacts/synonyms.json` not empty | ✅ synonyms mapped |
| `artifacts/vanna_bundle/` exists | ✅ bundle ready for Vanna |

---

## 🎯 8. Key Takeaways

- Configure the YAML before running.  
- Each layer feeds the next — don’t skip ingestion/profiling.  
- Vanna uses the exported bundle for **semantic retrieval** and **NL→SQL generation**.  
- Logs for every LLM call are available under `artifacts/logs/`.
