import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_jsonl(
    records: Iterable[Any],
    file_path: str | Path,
    schema: type[BaseModel] | None = None,
) -> None:
    """Atomically saves an iterable of dicts or Pydantic models to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                if schema is not None:
                    if isinstance(record, dict):
                        record = schema.model_validate(record)
                    elif not isinstance(record, schema):
                        raise TypeError(f"Expected {schema.__name__}, got {type(record).__name__}")

                if isinstance(record, BaseModel):
                    f.write(record.model_dump_json() + "\n")
                else:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()
            os.fsync(f.fileno())

        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@overload
def load_jsonl(file_path: str | Path, schema: type[T]) -> list[T]: ...


@overload
def load_jsonl(file_path: str | Path, schema: None = None) -> list[dict[str, Any]]: ...


def load_jsonl(
    file_path: str | Path,
    schema: type[T] | None = None,
) -> list[T] | list[dict[str, Any]]:
    """Loads a JSONL file into validated Pydantic models or plain dicts."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                if schema is not None:
                    records.append(schema.model_validate_json(line))
                else:
                    records.append(json.loads(line))
            except Exception as err:
                err.add_note(f"Failed parsing JSONL at {path}:{line_num}")
                raise

    return records
