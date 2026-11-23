# Zero-Shot Baseline Evaluation

**Date:** 2026-11-23  
**Phase:** Issue #6 / Issue #7 (Pre-Fine-Tuning Hybrid Baseline on Italian JSONL Split)  
**Task:** Hybrid Regex + Token Classification (NER) for PII Redaction  
**Model:** `yonigo/deberta-v3-base-pii-en` (English-tuned) + Email Regex  
**Dataset:** `data/processed/test.jsonl` (100 Italian IT Helpdesk records, validated with `PIIRecord`)  

## Summary
I evaluated the zero-shot gateway baseline on the 100-record Italian helpdesk test split (`data/processed/test.jsonl`). Every record uses full Italian sentence structure and is loaded via `load_jsonl()` with `PIIRecord` validation.

After fixing `PIIRedactor.extract_entities()` so it returns regex-matched email addresses alongside DeBERTa-v3 NER spans, the hybrid zero-shot baseline reaches **0.73 precision, 0.81 recall, and 0.77 F1** under strict IOB2 span matching.

## Quantitative Metrics (Strict Boundary IOB2)

| Pipeline Stage | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Hybrid (Email Regex + DeBERTa-v3 NER)** | **0.73** | **0.81** | **0.77** | **134** |
| Pure DeBERTa-v3 (before email eval fix) | 0.68 | 0.66 | 0.67 | 134 |
| Legacy `test.json` (English templates, pre-cleanup) | 0.66 | 0.37 | 0.48 | 134 |

## Error Breakdown
1. **Email Regex Eval Fix (+15% Recall):** In the earlier evaluation code, `_apply_regex()` replaced all 22 email addresses (`16.4%` of the 134 entities) with `[REDACTED]` before DeBERTa ran, and never added those emails to the predicted entity list. I also fixed the email regex so it does not consume a trailing sentence period (`.`). Returning regex-matched emails alongside NER spans raised recall from `0.66` to `0.81`.
2. **Remaining Missed PII (Recall 0.81, 19% missed):** Even with deterministic email matching, the English-tuned DeBERTa-v3 model still misses 19% of target entities (25 out of 134), mainly multi-word Italian names and localized card numbers.
3. **False Positives (Precision 0.73, 27% false alarms):** 27% of predicted spans are false positives. The model still tags non-PII capitalized hardware names (`Acer Nitro`) and Italian company names (`Faugno-Fittipaldi SPA`) in the 13 false-positive trap rows.
4. **Subword Alignment:** Both ground-truth spans and predicted spans are mapped to DeBERTa character offsets (`return_offsets_mapping=True`) and scored with `seqeval` (`mode="strict"`, `scheme=IOB2`).
