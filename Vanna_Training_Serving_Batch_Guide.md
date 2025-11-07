# Vanna Training, Serving & Batch Testing Guide

This document describes how to **train**, **serve**, and **batch-test** multiple **Vanna AI** instances using the pre-built Semantic Layer bundle.

---

## 1. What You Have

- A project folder containing:
  - `app/vanna_app_poc.py` → main orchestration script  
  - `configs/default_config.yaml` → configuration for all runs  
  - `artifacts/vanna_bundle/` → Semantic Layer export bundle  
  - `artifacts/vanna_runs/` → runtime outputs (manifest, test results)
- OpenAI API key is already defined in your config (`llm.api_key`).
- Dependencies installed via:
  ```bash
  pip install "vanna[pgvector]==0.7.9" "psycopg[binary]>=3.1" "pgvector>=0.2.5" "SQLAlchemy>=2.0" flask pandas langchain-huggingface
  ```
- PostgreSQL should have:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

---

## 2. Understanding the Three Vanna Instances

| Instance ID | Method | Data Used | Schema | Vector Store | Port |
|--------------|---------|-----------|---------|---------------|-------|
| **A_info_only** | `info_schema` | MSSQL Information Schema only | `reporting` | `connections.postgres.my_postgres2` | **8084** |
| **B_info_plus_bundle_whole** | `bundle_plus_info` | Full bundle + Info schema | `reporting` | `connections.postgres.my_postgres3` | **8085** |
| **C_info_plus_selected** | `bundle_select_plus_info` | Selected bundle parts (`docs`, `concepts`) + Info schema | `reporting` | `connections.postgres.my_postgres` | **8086** |

### Quick Summary
- **A_info_only** → trains on DB structure (no bundle).  
- **B_info_plus_bundle_whole** → combines schema + all bundle artifacts.  
- **C_info_plus_selected** → combines schema + selective bundle parts (e.g., `docs` and `concepts`).

---

## 3. Training Commands

### (a) Train All Instances
```bash
python app/vanna_app_poc.py --train-instances -c configs/default_config.yaml
```

**Produces:**
```
artifacts/vanna_runs/manifest.json
```

---

### (b) Train a Single Instance Manually

**Bundle only:**
```bash
python app/vanna_app_poc.py --train-bundle --bundle-dir artifacts/vanna_bundle -c configs/default_config.yaml
```

**Info schema only:**
```bash
python app/vanna_app_poc.py --train-info --mssql-db reporting_devco --schema reporting -c configs/default_config.yaml
```

**Combined:**
```bash
python app/vanna_app_poc.py --train-bundle --train-info --mssql-db reporting_devco --schema reporting -c configs/default_config.yaml
```

---

## 4. Serving Trained Instances

### (a) Multi-Instance — Split Ports (Recommended)
```bash
python app/vanna_app_poc.py --serve-multi-split --manifest artifacts/vanna_runs/manifest.json -c configs/default_config.yaml
```

Example output:
```
[spawned] A_info_only on port 8084
[spawned] B_info_plus_bundle_whole on port 8085
[spawned] C_info_plus_selected on port 8086
```

Access:
- `http://localhost:8084`
- `http://localhost:8085`
- `http://localhost:8086`

---

### (b) Multi-Instance — One Port (Gateway Mode)
```bash
python app/vanna_app_poc.py --serve-multi --manifest artifacts/vanna_runs/manifest.json --port 8090 -c configs/default_config.yaml
```

Accessible via unified API:
```
http://localhost:8090
```

---

### (c) Single Instance Serve
```bash
python app/vanna_app_poc.py --serve --port 8087 -c configs/default_config.yaml
```

---

## 5. API Examples

**List Instances**
```bash
curl http://localhost:8090/instances
```

**Ask a Question**
```bash
curl -X POST http://localhost:8090/ask -H "Content-Type: application/json"   -d '{"instance":"B_info_plus_bundle_whole","question":"List full-time employees and exclude smokers"}'
```

**Compare Answers**
```bash
curl -X POST http://localhost:8090/compare -H "Content-Type: application/json"   -d '{"instances":["A_info_only","B_info_plus_bundle_whole","C_info_plus_selected"],"question":"Top 10 plans by enrollment"}'
```

---

## 6. Batch Testing Mode (New Feature)

### Prepare Input CSV
Create `artifacts/vanna_runs/questions.csv`:

```csv
question
List all full-time employees
Show all smoker members
Top 10 plans by enrollment
```

### Run Batch Testing
```bash
python app/vanna_app_poc.py   --batch-csv artifacts/vanna_runs/questions.csv   --batch-out artifacts/vanna_runs/batch_results.csv   --manifest artifacts/vanna_runs/manifest.json   -c configs/default_config.yaml
```

**Output File:** `batch_results.csv`

| Column | Description |
|---------|-------------|
| `question` | Question asked |
| `instance` | Instance ID (A/B/C) |
| `sql` | SQL generated |
| `executed` | True/False |
| `execution_error` | Error (if failed) |
| `rowcount` | Rows returned |
| `sample_json` | JSON preview of result |

---

## 7. Preventing Context Overflow (Token Control)

This build includes **prompt budgeting** and **automatic truncation** to prevent  
`context_length_exceeded` errors.

### Config Example
```yaml
llm:
  model: gpt-4.1
  api_key: "sk-xxxx"
  max_input_bytes: 120000   # caps overall LLM input
  summary_max_rows: 200     # row limit for data summarization
  summary_max_cols: 40      # column limit
  summary_max_chars_per_cell: 400
  temperature: 0.2
```

✅ Vanna now prunes oversized context dynamically during calls — no more 400 errors.

---

## 8. Troubleshooting

| Issue | Fix |
|--------|------|
| **Context length exceeded** | Increase `max_input_bytes` or reduce data returned. |
| **Manifest not found** | Re-run `--train-instances`. |
| **Port conflict** | Adjust `port:` in config. |
| **SQL failed to execute** | Check DB connection under `connections.mssql`. |
| **Vector store connection error** | Verify `connections.postgres.*` URLs and extensions. |
| **OpenAI auth error** | Ensure valid API key in `llm.api_key`. |

---

## 9. Quick Command Summary

| Task | Command |
|------|----------|
| Train all instances | `python app/vanna_app_poc.py --train-instances -c configs/default_config.yaml` |
| Serve all (split ports) | `python app/vanna_app_poc.py --serve-multi-split --manifest artifacts/vanna_runs/manifest.json -c configs/default_config.yaml` |
| Serve all (one port) | `python app/vanna_app_poc.py --serve-multi --manifest artifacts/vanna_runs/manifest.json --port 8090 -c configs/default_config.yaml` |
| Batch test questions | `python app/vanna_app_poc.py --batch-csv artifacts/vanna_runs/questions.csv --batch-out artifacts/vanna_runs/batch_results.csv --manifest artifacts/vanna_runs/manifest.json -c configs/default_config.yaml` |
| Single instance serve | `python app/vanna_app_poc.py --serve --port 8087 -c configs/default_config.yaml` |

---

## 10. Notes

- Each instance has its own **Postgres vector store** and port.
- The bundle path is fixed by `outputs.vanna_bundle_dir` in config.
- Info-schema uses the schema set in `schema:` within YAML.
- All credentials are pulled from config (`llm.api_key`, `connections.*`).
- For Windows, multi-serve uses safe `multiprocessing.freeze_support()`.

---

**End of Guide**  
🧠 *Includes batch testing, context budgeting, and multi-instance serving.*
