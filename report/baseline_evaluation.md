# Zero-Shot Baseline Evaluation

**Date:** 2026-11-23  
**Phase:** Issue #6 / Issue #7 (Pre-Fine-Tuning Baseline on Italian JSONL Split)  
**Task:** Token Classification (NER) for PII Redaction  
**Model:** `yonigo/deberta-v3-base-pii-en` (English-tuned)  
**Dataset:** `data/processed/test.jsonl` (100 Italian IT Helpdesk records, validated with `PIIRecord`)  

## Summary
I ran the zero-shot baseline evaluation on the 100-record Italian helpdesk test split (`data/processed/test.jsonl`). Every record uses full Italian sentence structure and is validated against the `PIIRecord` schema through `load_jsonl()`.

The English-tuned DeBERTa-v3 model reaches 0.68 precision, 0.66 recall, and 0.67 F1 under strict IOB2 span matching. It still misses 34% of the ground-truth PII entities in the test set.

## Quantitative Metrics (Strict Boundary IOB2)

| Entity | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **PII** | 0.68 | 0.66 | 0.67 | 134 |
| | | | | |
| **Micro Avg** | 0.68 | 0.66 | 0.67 | 134 |
| **Macro Avg** | 0.68 | 0.66 | 0.67 | 134 |
| **Weighted Avg** | 0.68 | 0.66 | 0.67 | 134 |

*(Previous score on the legacy mixed-language `data/splits/test.json` array was Precision `0.66`, Recall `0.37`, F1 `0.48` on the same 134 entities.)*

## Error Breakdown
1. **Missed PII (Recall 0.66):** The model misses 34% of target entities (45 out of 134). Part of this drop happens because `PIIRedactor.extract_entities()` currently runs `_apply_regex()` first and masks all 22 email addresses to `[REDACTED]` before DeBERTa runs. The remaining misses come from Italian multi-word names and numbers.
2. **False Positives (Precision 0.68):** 32% of predicted spans are wrong. The model tags non-PII capitalized words such as hardware names (`Acer Nitro`) and Italian company names (`Faugno-Fittipaldi SPA`) in the false-positive trap rows.
3. **Subword Alignment:** Both ground-truth spans and predicted spans are aligned to DeBERTa character offsets (`return_offsets_mapping=True`) and scored with `seqeval` (`mode="strict"`, `scheme=IOB2`). Any partial name match counts as a miss.
