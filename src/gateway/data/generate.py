import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from gateway.common.jsonl import save_jsonl
from gateway.common.schema import PIIRecord

load_dotenv()

SYSTEM_PROMPT = """
You are a synthetic data generator for an Italian enterprise.
Generate exactly 20 realistic IT helpdesk tickets in Italian.
They must contain realistic Italian PII (Names, Phone Numbers, Email Addresses, Credit Cards).

You MUST output ONLY valid JSON in this exact format:
{
  "records": [
    {
      "id": "syn-1",
      "original_prompt": "Mi chiamo Lorenzo Bianchi e ho perso la password della mia email lorenzo.b@azienda.it.",
      "entities": [
        {"text": "Lorenzo Bianchi", "label": "PERSON"},
        {"text": "lorenzo.b@azienda.it", "label": "EMAIL"}
      ]
    }
  ]
}
"""


def parse_llm_batch(raw_json_string: str, start_index: int = 1) -> list[PIIRecord]:
    """Parses raw LLM JSON output and keeps only valid PIIRecords with exact entity substrings."""
    raw_dict = json.loads(raw_json_string)
    raw_list = raw_dict.get("records", [])
    if not isinstance(raw_list, list):
        return []

    validated: list[PIIRecord] = []
    for item in raw_list:
        if isinstance(item, str):
            item = json.loads(item)
        if not isinstance(item, dict):
            continue

        item["id"] = f"syn-{start_index + len(validated)}"
        record = PIIRecord.model_validate(item)

        if all(ent.text in record.original_prompt for ent in record.entities):
            validated.append(record)

    return validated


def generate_batch(client: OpenAI, start_index: int = 1) -> list[PIIRecord]:
    """Requests one batch of 20 synthetic Italian records from local Ollama."""
    response = client.chat.completions.create(
        model="llama3",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    raw_json_string = response.choices[0].message.content or ""
    try:
        return parse_llm_batch(raw_json_string, start_index=start_index)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as err:
        print(f"Skipping malformed LLM batch: {err}")
        return []


def build_dataset(
    target_size: int = 2000,
    output_path: str | Path = "data/raw/synthetic_italian.jsonl",
    max_consecutive_failures: int = 10,
) -> list[PIIRecord]:
    """Generates target_size validated records and saves them atomically as JSONL."""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "local-mode"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    )

    all_records: list[PIIRecord] = []
    failures = 0

    print(f"Generating {target_size} records...")
    while len(all_records) < target_size:
        batch = generate_batch(client, start_index=len(all_records) + 1)
        if not batch:
            failures += 1
            if failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"Aborted after {failures} consecutive failed batches from Ollama."
                )
            continue

        failures = 0
        all_records.extend(batch)
        print(f"Progress: {len(all_records)} / {target_size}")

    all_records = all_records[:target_size]
    save_jsonl(all_records, output_path, schema=PIIRecord)
    print(f"Saved {len(all_records)} records to {output_path}")
    return all_records


if __name__ == "__main__":
    build_dataset()
