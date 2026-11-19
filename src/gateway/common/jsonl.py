import json
from pathlib import Path
from typing import Any

def save_jsonl(records: list[Any], file_path: str | Path) -> None:
    """
    Saves a list of dictionaries or Pydantic models to a JSONL file.
    Automatically creates the parent directories if they do not exist.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            # If it's a Pydantic model, use model_dump_json()
            if hasattr(record, "model_dump_json"):
                f.write(record.model_dump_json() + "\n")
            # Otherwise, use standard json serialization
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_jsonl(file_path: str | Path) -> list[dict]:
    """
    Reads a JSONL file into a list of dictionaries.
    """
    path = Path(file_path)
    records = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                
    return records