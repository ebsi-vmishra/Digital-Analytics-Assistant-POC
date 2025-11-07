# libs/path_utils.py
from pathlib import Path

def run_root(cfg: dict) -> Path:
    out_base = (cfg.get("output_dir") or "outputs")
    rid = (cfg.get("run_id") or "run")
    root = Path(out_base) / rid
    root.mkdir(parents=True, exist_ok=True)
    return root

def run_path(cfg: dict, *parts: str) -> Path:
    p = run_root(cfg).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
