# libs/table_utils.py
from typing import List, Dict, Optional

def pick_tables(schema_json: Dict, limit: Optional[int]) -> List[str]:
    all_tables = sorted((schema_json.get("tables") or {}).keys())
    if not limit or limit <= 0 or limit >= len(all_tables):
        return all_tables
    return all_tables[: int(limit)]
