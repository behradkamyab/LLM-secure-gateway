import json
import os
from openai import OpenAI
from pydantic import ValidationError
from dotenv import load_dotenv

# Import our strict data contracts
from gateway.common.schema import SyntheticDataRecord, PIIEntity

# Load the local endpoint from .env
load_dotenv()
CLIENT = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "local-mode"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
)

SYSTEM_PROMPT = """
You are a synthetic data generator for an Italian enterprise. 
Generate exactly 20 realistic IT helpdesk tickets in Italian. 
They must contain realistic Italian PII (Names, Phone Numbers, Email Addresses, Organization names).

You MUST output ONLY valid JSON in this exact format:
{
  "records": [
    {
      "id": "unique-id-1",
      "original_prompt": "Mi chiamo Lorenzo Bianchi e ho perso la password della mia email lorenzo.b@azienda.it.",
      "entities": [
        {"text": "Lorenzo Bianchi", "label": "PERSON"},
        {"text": "lorenzo.b@azienda.it", "label": "EMAIL"}
      ]
    }
  ]
}
"""

def generate_batch() -> list[SyntheticDataRecord]:
    """Generates a batch of 20 records using local Llama-3."""
    response = CLIENT.chat.completions.create(
        model="llama3", 
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        response_format={"type": "json_object"}, 
        temperature=0.7
    )
    
    raw_json_string = response.choices[0].message.content
    validated_records = []
    
    try:
        # Parse the string into a standard Python dictionary
        raw_dict = json.loads(raw_json_string)
        
        # Safely extract the list from the "records" key
        raw_list = raw_dict.get("records", [])
        
        #  Force each item through our strict Pydantic contract
        for item in raw_list:
            # Catch LLM double-encoding hallucinations where the dict is wrapped in a string
            if isinstance(item, str):
                item = json.loads(item)
                
            record = SyntheticDataRecord(**item)
            validated_records.append(record)
            
    except Exception as e:
        # Catch absolutely any error (TypeError, JSONDecodeError, ValidationError)
        print(f"\n⚠️ LLM hallucinated bad format. Dropping batch. Error: {e}")
        print(f"--- RAW LLM OUTPUT FOR DEBUGGING ---\n{raw_json_string}\n-----")
        
    return validated_records

def build_dataset(target_size: int = 2000, output_path: str = "data/raw/synthetic_italian.jsonl"):
    """Loops until we have the required number of perfect records."""
    all_records = []    
    
    # Ensure the target directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating {target_size} records...")
    while len(all_records) < target_size:
        batch = generate_batch()
        if not batch:
            continue  # Skip to the next iteration if the batch failed
            
        all_records.extend(batch)
        print(f"Progress: {len(all_records)} / {target_size}")
        
    # Trim excess if the last batch pushed us slightly over 2000
    all_records = all_records[:target_size]
    
    # Save safely to our git-ignored data folder
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(record.model_dump_json() + "\n")
            
    print(f"✅ Successfully saved {target_size} records to {output_path}")

if __name__ == "__main__":
    build_dataset(target_size=2000, output_path="data/raw/synthetic_italian.jsonl")