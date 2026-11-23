from pathlib import Path

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from gateway.common.jsonl import load_jsonl
from gateway.common.schema import PIIRecord
from gateway.infer.engine import GatewayEngine


def align_to_subword_bio(text: str, entities: list[str], tokenizer) -> list[str]:
    """
    Production BIO alignment using Hugging Face subword tokenization and character offsets.
    Prevents tokenization mismatches caused by punctuation or subword splits.
    """
    inputs = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = inputs["offset_mapping"]

    tags = ["O"] * len(offsets)

    entity_spans = []
    for ent in entities:
        if not ent:
            continue
        start_char = text.find(ent)
        if start_char != -1:
            end_char = start_char + len(ent)
            entity_spans.append((start_char, end_char))

    for ent_start, ent_end in entity_spans:
        entity_started = False

        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start >= ent_start and tok_end <= ent_end and tok_start != tok_end:
                if not entity_started:
                    tags[idx] = "B-PII"
                    entity_started = True
                else:
                    tags[idx] = "I-PII"

    return tags


def evaluate_zeroshot(config_path: str | Path = "configs/e0_zeroshot.yaml") -> str:
    print("Loading Gateway Engine (Zero-Shot Baseline)...")
    engine = GatewayEngine(config_path=str(config_path))
    tokenizer = engine.redactor.tokenizer

    test_path = engine.config["data"]["test_path"]
    print(f"Loading validated test dataset from {test_path}...")
    records = load_jsonl(test_path, schema=PIIRecord)

    y_true = []
    y_pred = []

    ner_name = engine.config["model"]["ner_engine"]["name"]
    print(f"Evaluating {len(records)} records against {ner_name}...")

    for record in records:
        text = record.original_prompt
        true_ents = [ent.text for ent in record.entities]
        pred_ents = engine.redactor.extract_entities(text)

        y_true.append(align_to_subword_bio(text, true_ents, tokenizer))
        y_pred.append(align_to_subword_bio(text, pred_ents, tokenizer))

    print("\n--------- Zero-Shot Baseline Results ----")
    report = classification_report(y_true, y_pred, mode="strict", scheme=IOB2)
    print(report)
    return str(report)


if __name__ == "__main__":
    evaluate_zeroshot()
