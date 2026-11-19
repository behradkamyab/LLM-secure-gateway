import json
from pathlib import Path
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from gateway.infer.engine import GatewayEngine
from gateway.common.jsonl import load_jsonl

def align_to_subword_bio(text: str, entities: list[str], tokenizer) -> list[str]:
    """
    Production BIO alignment using Hugging Face subword tokenization and character offsets.
    Prevents tokenization mismatches caused by punctuation or subword splits.
    """
    # Tokenize the text to get character start/end coordinates for every subword
    inputs = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = inputs["offset_mapping"]
    
    tags = ["O"] * len(offsets)
    
    # Find the exact character coordinates of our ground-truth strings
    entity_spans = []
    for ent in entities:
        start_char = text.find(ent)
        if start_char != -1:
            end_char = start_char + len(ent)
            entity_spans.append((start_char, end_char))
            
    # Map the subword tokens to the character spans
    for ent_start, ent_end in entity_spans:
        entity_started = False
        
        for idx, (tok_start, tok_end) in enumerate(offsets):
            # If the token's characters fall entirely inside the entity's character span
            if tok_start >= ent_start and tok_end <= ent_end and tok_start != tok_end:
                if not entity_started:
                    tags[idx] = "B-PII"
                    entity_started = True
                else:
                    tags[idx] = "I-PII"
                    
    return tags

def evaluate_zeroshot():
    print("Loading Gateway Engine (Zero-Shot Baseline)...")
    engine = GatewayEngine(config_path="configs/e0_zeroshot.yaml")
    tokenizer = engine.redactor.tokenizer
    
    print("Loading test dataset...")
    # Use standard json.load for an array, not our jsonl line-by-line reader
    with open("data/splits/test.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    
    y_true = []
    y_pred = []
    
    print(f"Evaluating {len(records)} records against {engine.config['model']['ner_engine']['name']}...")
    
    for record in records:
        # Use 'text' instead of 'original_prompt'
        text = record["text"]
        
        # 'true_entities' is already a list of strings, so we don't need a list comprehension
        true_ents = record["true_entities"] 
        
        # Predict using the redactor
        pred_ents = engine.redactor.extract_entities(text)
        
        # Align BOTH ground-truth strings and predicted strings to the DeBERTa subwords
        y_true.append(align_to_subword_bio(text, true_ents, tokenizer))
        y_pred.append(align_to_subword_bio(text, pred_ents, tokenizer))
        
    print("\n--------- Zero-Shot Baseline Results ----")
    report = classification_report(y_true, y_pred, mode="strict", scheme=IOB2)
    print(report)

if __name__ == "__main__":
    evaluate_zeroshot()