import os, json, csv
from pathlib import Path
from typing import Any, Dict, List

def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def write_json(data: Any, path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_jsonl(rows: List[Dict], path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(rows: List[Dict[str, Any]], path: str, fieldnames: List[str]):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
