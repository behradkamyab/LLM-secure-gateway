# Zero-Shot Baseline Evaluation

**Date:** 2026-11-18  
**Phase:** Issue #6 (Pre-Fine-Tuning Baseline)  
**Task:** Token Classification (NER) for PII Redaction  
**Model:** `yonigo/deberta-v3-base-pii-en` (English-tuned)  
**Dataset:** `data/splits/test.jsonl` (100 Italian IT Helpdesk records)  

## Executive Summary
This evaluation establishes the baseline performance of our semantic router and PII redaction gateway before any domain-specific fine-tuning. The base DeBERTa-v3 model was evaluated on a zero-shot basis against synthetic Italian enterprise data.

As hypothesized in the architecture design, the English-tuned model exhibits catastrophic recall failure (37%) when exposed to Italian naming conventions, syntax, and corporate ID formats. This validates the business requirement for a LoRA fine-tuning phase to prevent data leakage at the edge.

## Quantitative Metrics (Strict Boundary BIO)

| Entity | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **PII** | 0.66 | 0.37 | 0.48 | 134 |
| | | | | |
| **Micro Avg** | 0.66 | 0.37 | 0.48 | 134 |
| **Macro Avg** | 0.66 | 0.37 | 0.48 | 134 |
| **Weighted Avg** | 0.66 | 0.37 | 0.48 | 134 |

## Engineering Analysis
1. **Critical Data Leakage (Recall 0.37):** The model missed 63% of the target PII. It consistently fails to recognize Italian surnames, multi-word corporate names, and localized formatting.
2. **False Positives (Precision 0.66):** While the model is conservative, 34% of its redactions were incorrect, masking benign Italian vocabulary that structurally resembled English names or entities.
3. **Subword Alignment:** The evaluation utilized robust subword-to-character BIO alignment to prevent tokenization artifacts from skewing the metrics, guaranteeing that the 0.48 F1 score is a true reflection of model capability, not a string-matching error.

## Next Steps
This baseline formally justifies the transition to **Phase 2: PEFT/LoRA Fine-Tuning**. By training low-rank adapters on our synthetic Italian dataset (`data/raw/synthetic_italian.jsonl`), our target is to elevate this 0.37 recall metric to >0.95, establishing a secure perimeter before requests are routed to external LLMs