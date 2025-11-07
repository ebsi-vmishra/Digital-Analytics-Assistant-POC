Vanna Training, Serving & Batch Testing Guide

This guide walks you (the teammate) through training and serving Vanna using the pre-created bundle from the Semantic Layer POC, and how to batch-test questions across the three instances.

Everything on the Semantic Layer side (ingestion, profiling, docs, concepts, synonyms, bundle creation) is already done. You only need to run the Vanna training, serving, and testing steps.

1) What You Have

Project folder containing:

app/vanna_app_poc.py (main training/serving/testing script)

configs/default_config.yaml (configuration file)

artifacts/vanna_bundle/ (the complete training bundle)

artifacts/vanna_runs/ (manifest and batch outputs will be created here)

Dependencies installed:

vanna[pgvector]==0.7.9, flask, psycopg[binary]>=3.1, pgvector>=0.2.5, SQLAlchemy>=2.0, langchain-huggingface, pandas

OpenAI API key in llm.api_key (or via env OPENAI_API_KEY)

Postgres note: your DB must have CREATE EXTENSION IF NOT EXISTS vector;

2) The 3 Training Instances
Instance ID	Method	Data Sources Used	Schema	Vector Store	Port
A_info_only	info_schema	MSSQL INFORMATION_SCHEMA.COLUMNS only	reporting	connections.postgres.my_postgres2	8084
B_info_plus_bundle_whole	bundle_plus_info	Full bundle + Info schema	reporting	connections.postgres.my_postgres3	8085
C_info_plus_selected	bundle_select_plus_info	Selected bundle parts (docs, concepts) + Info schema	reporting	connections.postgres.my_postgres	8086

Quick take

A_info_only → trains only on DB metadata (no bundle).

B_info_plus_bundle_whole → trains on the entire bundle + live schema.

C_info_plus_selected → trains only on targeted bundle parts (e.g., docs, concepts) + live schema.

3) Artifact → Training Tag Map

These are the bundle files and what training “tag” they feed:

Bundle File	Path in artifacts/vanna_bundle/	Tag used by trainer
DDL	ddl.sql	ddl
Documentation	documentation.md	docs
Column Profiles	profiles_summary.csv	profiles
Logical Schema	schema.json	schema
Relationships	relationships.json	relationships
Synonyms	synonyms.json	synonyms
Attribute Map	attribute_map.json	attr_map
Concepts (catalog/alias/attributes/rules)	concept_*.csv	concepts
Value Aliases (LLM)	value_aliases_llm.json	values
Value Domains	value_domains_llm.csv	values

With method: bundle_select[_plus_info], use bundle_parts: [docs, concepts, synonyms, schema, relationships, ddl, profiles, attr_map, values].

4) Training Commands

Run from the project root.

(a) Train all 3 instances (recommended)
python app/vanna_app_poc.py --train-instances -c configs/default_config.yaml


This:

Trains all 3 instances defined in the YAML

Saves a manifest to artifacts/vanna_runs/manifest.json

Example:

[vanna] Trained 3 instance(s). Manifest -> artifacts/vanna_runs/manifest.json

(b) Train a single instance manually (optional)

Bundle only

python app/vanna_app_poc.py --train-bundle --bundle-dir artifacts/vanna_bundle -c configs/default_config.yaml


Info schema only

python app/vanna_app_poc.py --train-info --mssql-db reporting_devco --schema reporting -c configs/default_config.yaml


Both

python app/vanna_app_poc.py --train-bundle --train-info --schema reporting --mssql-db reporting_devco -c configs/default_config.yaml

5) Serving the Trained Instances
(a) Separate ports (full Vanna UI per instance) — recommended
python app/vanna_app_poc.py --serve-multi-split \
  --manifest artifacts/vanna_runs/manifest.json \
  -c configs/default_config.yaml


You’ll see:

[spawned] A_info_only on port 8084
[spawned] B_info_plus_bundle_whole on port 8085
[spawned] C_info_plus_selected on port 8086


Open:

http://localhost:8084 → A_info_only

http://localhost:8085 → B_info_plus_bundle_whole

http://localhost:8086 → C_info_plus_selected

Tip: If ports are missing in the manifest, pass --port 8084 to auto-assign a base 8084...

(b) Single port (API multiplexer)
python app/vanna_app_poc.py --serve-multi \
  --manifest artifacts/vanna_runs/manifest.json \
  --port 8090 \
  -c configs/default_config.yaml


Endpoints:

GET /instances

POST /ask

POST /compare

(c) Single instance UI
python app/vanna_app_poc.py --serve --port 8087 -c configs/default_config.yaml

6) API Usage (single-port mode)

List instances

curl http://localhost:8090/instances


Ask a question

curl -X POST http://localhost:8090/ask -H "Content-Type: application/json" \
  -d '{"instance":"B_info_plus_bundle_whole","question":"List full-time employees and exclude smokers","execute":true}'


Compare across instances

curl -X POST http://localhost:8090/compare -H "Content-Type: application/json" \
  -d '{"instances":["A_info_only","B_info_plus_bundle_whole","C_info_plus_selected"],"question":"Top 10 plans by enrollment","execute":true}'


Split-ports example (instance B @ 8085)

curl -X POST http://localhost:8085/ask -H "Content-Type: application/json" \
  -d '{"question":"Show all smoker members","execute":true}'

7) Batch Testing From CSV (new)

Prepare an input CSV (e.g., artifacts/vanna_runs/questions.csv):

question
List full-time employees and exclude smokers
Top 10 plans by enrollment
Show all smoker members


Run batch across all instances in the manifest:

python app/vanna_app_poc.py \
  --batch-csv artifacts/vanna_runs/questions.csv \
  --batch-out artifacts/vanna_runs/batch_results.csv \
  --manifest artifacts/vanna_runs/manifest.json \
  -c configs/default_config.yaml


To generate SQL only (no DB execution), add --no-exec.

Output CSV columns

question

instance

sql

executed (True/False)

execution_error (string if any)

rowcount (int when executed and DataFrame)

sample_json (first few rows as lightweight JSON preview)

8) Optional Runtime Overrides
python app/vanna_app_poc.py --train-instances --openai-model gpt-4o-mini -c configs/default_config.yaml
python app/vanna_app_poc.py --serve-multi --openai-base-url https://yourproxy/v1 --manifest artifacts/vanna_runs/manifest.json -c configs/default_config.yaml

9) Context-Size Safeguards (fixed)

In configs/default_config.yaml under llm:

llm:
  api_key: "YOUR_KEY"
  base_url: null          # or your proxy
  model: gpt-4.1
  max_input_bytes: 180000               # trims/prunes message history before LLM calls
  summary_max_rows: 200                 # clamps DF size before generating summaries
  summary_max_cols: 40
  summary_max_chars_per_cell: 500
  temperature: 0.2


How it works

The script prunes/truncates large prompts to keep them below max_input_bytes.

When summarizing SQL results, it clamps the DataFrame to avoid “small question → huge context” errors.

If you still hit a rare context error:

Raise max_input_bytes (e.g., to 220000), and/or

Tighten summary_max_* limits if result sets are very large.

10) Troubleshooting
Issue	Fix
Manifest not found	Run --train-instances before serving.
MSSQL connection errors	Verify connections.mssql.<key> in YAML and network/ODBC driver.
Vector store connection errors	Check Postgres DSN values in connections.postgres.* and ensure pgvector extension.
OpenAI auth error	Confirm llm.api_key (or env OPENAI_API_KEY) is valid.
Context length exceeded	The safeguards are on—raise llm.max_input_bytes or reduce summary_max_* if needed.
Port already in use	Change each instance’s port in YAML or pass a different --port base for split-serve.
11) Summary of Key Commands
Purpose	Command
Train all instances	python app/vanna_app_poc.py --train-instances -c configs/default_config.yaml
Serve all instances (split ports)	python app/vanna_app_poc.py --serve-multi-split --manifest artifacts/vanna_runs/manifest.json -c configs/default_config.yaml
Serve all instances (single port)	python app/vanna_app_poc.py --serve-multi --manifest artifacts/vanna_runs/manifest.json --port 8090 -c configs/default_config.yaml
Batch test from CSV	python app/vanna_app_poc.py --batch-csv artifacts/vanna_runs/questions.csv --batch-out artifacts/vanna_runs/batch_results.csv --manifest artifacts/vanna_runs/manifest.json -c configs/default_config.yaml
Ask question (split-ports example)	curl -X POST http://localhost:8085/ask -H "Content-Type: application/json" -d '{"question":"List full-time employees","execute":true}'
12) Notes

Each instance is isolated (separate PG vector stores).

The bundle path defaults to artifacts/vanna_bundle/ (override with --bundle-dir for single-instance training).

Info-schema training respects the schema: filter in YAML.

All credentials default from YAML (llm.api_key, connections.*) but can be overridden via CLI flags.

On Windows, multi-process serving is already safeguarded with multiprocessing.freeze_support().

End of Guide