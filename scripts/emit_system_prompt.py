# scripts/emit_system_prompt.py
import argparse
from pathlib import Path
import yaml

# Works when run as module or script
try:
    from utils.prompt_loader import build_system_prompt
except Exception:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.prompt_loader import build_system_prompt


def emit(cfg_path: str) -> str:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) if cfg_path else {}
    base = Path(cfg.get("base_dir") or ".").resolve()
    outputs = cfg.get("outputs") or {}

    prompts_dir = base / outputs.get("prompts_dir", "artifacts/prompts")
    concepts_dir = base / outputs.get("concepts_out_dir", "artifacts/concepts")
    out_path = prompts_dir / "system_prompt.txt"

    prompts_dir.mkdir(parents=True, exist_ok=True)
    sp = build_system_prompt(prompts_dir, concepts_dir, include_alias_preview=True)
    out_path.write_text(sp, encoding="utf-8")

    print(f"[emit_system_prompt] Wrote combined system prompt → {out_path}")
    return str(out_path)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=False, help="Path to YAML config")
    args = ap.parse_args()
    emit(args.config or "")
