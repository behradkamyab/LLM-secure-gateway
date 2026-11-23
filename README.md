# AI Security Gateway & Semantic Router

I built an async FastAPI microservice that sits in front of Enterprise RAG systems. It intercepts user prompts, redacts PII with a hybrid Regex and NER pipeline, and routes repeated queries to cached answers using FAISS vector search before calling an external LLM.
I evaluate zero-shot `DeBERTa-v3` (`yonigo/deberta-v3-base-pii-en`) and `all-MiniLM-L6-v2` against regex and keyword baselines on 100 synthetic Italian IT helpdesk records (134 PII spans). The initial zero-shot NER baseline scored 0.37 recall (0.48 F1). After cleaning the test split into full Italian JSONL records and fixing an email regex evaluation bug in Issue #7, the hybrid zero-shot baseline reaches 0.81 recall (0.77 F1), while still missing 19% of Italian names and card numbers. I am fine-tuning LoRA PEFT adapters on synthetic Italian data to push recall above 0.95 before ONNX export.

| Step | What                                                          | Where                                         |
| :--- | :------------------------------------------------------------ | :-------------------------------------------- |
| 1    | Synthetic Italian data generation and JSONL/BIO preprocessing | `src/gateway/data/`, `notebooks/`         |
| 2    | Zero-shot baseline evaluation and ONNX routing (`E0`)       | `src/gateway/infer/`, `src/gateway/eval/` |
| 3    | LoRA PEFT fine-tuning (`E1+`)                               | `src/gateway/train/`                        |
| 4    | FastAPI microservice and startup hooks                        | `src/gateway/infer/main.py`                 |

## Project Roadmap & Status

- [X] **Phase 1: Infrastructure & Baseline Benchmarks (Completed)**
  - Async FastAPI service with `lifespan` model loading and typed Pydantic schemas.
  - Strict-boundary BIO baseline for zero-shot ONNX `DeBERTa-v3` vs Regex, and FAISS vs keyword routing.
  - Modular project layout with `uv`, `pytest`, and `pre-commit` guards.
- [ ] **Phase 2: Domain Adaptation via LoRA PEFT (Active)**
  - LLM-assisted synthetic Italian IT helpdesk generation (`data/raw/synthetic_italian.jsonl`).
  - Subword-to-character BIO alignment and tagging pipeline.
  - LoRA fine-tuning on DeBERTa to raise Italian PII recall above 0.95.
- [ ] **Phase 3: Inference Optimization (Planned)**
  - Merge LoRA adapters and export the fine-tuned model to ONNX.
  - Benchmark end-to-end latency and memory usage.

## Setup

Requires Python 3.11.

```bash
uv venv -p 3.11
uv pip install -e ".[dev]"            # Schema, config, and unit tests (CPU only)
uv pip install -e ".[dev,infer]"      # FastAPI, ONNX Runtime, and FAISS (API server)
uv pip install -e ".[dev,train,eval]" # PyTorch, Transformers, PEFT, and seqeval
pre-commit install                    # Runs ruff and blocks raw data or model weights
cp .env.example .env                  # Add local API / tracking keys (never commit .env)
```

## Folder Map

- `configs/`: One YAML config per experiment (`E0` zero-shot, `E1` LoRA).
- `data/`: Preprocessing scripts and split manifests (no raw JSON files in Git).
- `src/gateway/data/`: Synthetic data generation and JSONL/BIO formatting.
- `src/gateway/infer/`: FastAPI gateway, ONNX PII redactor, and FAISS router.
- `src/gateway/train/`: LoRA supervised fine-tuning loops.
- `src/gateway/eval/`: `seqeval` strict-BIO metrics and evaluation scripts.
- `src/gateway/common/`: Shared schemas, config loader, and logging.
- `outputs/`: Trained adapters and score files (gitignored).
- `report/`: Diagnostic evaluation reports.
- `tests/`: Unit and repository hygiene tests.

## Running the API Gateway

The FastAPI `lifespan` hook loads the ONNX models into RAM and builds the FAISS index at startup before serving requests.

```bash
uvicorn src.gateway.infer.main:app --reload
```

### API Usage

**Endpoint:** `POST /api/v1/gateway`
**Request:**

```json
{
  "prompts": [
    "My name is Behrad and I spilled coffee on my ThinkPad."
  ]
}
```

**Response:**

```json
{
  "results": [
    {
      "original_prompt": "My name is Behrad and I spilled coffee on my ThinkPad.",
      "redacted_prompt": "My name is [REDACTED] and I spilled coffee on my ThinkPad.",
      "routing": {
        "action": "CACHE_HIT",
        "score": 0.6389,
        "response": "What do I do if I spilled water on my laptop?"
      }
    }
  ]
}
```

## Architecture

Every prompt passes through four stages before reaching an external LLM:

1. **Input normalization:** Cleans whitespace and casing.
2. **Regex redaction:** Strips structured PII such as emails, phone numbers, and card numbers.
3. **NER redaction:** Runs ONNX `DeBERTa-v3` token classification for names and organizations, merging split subword spans.
4. **Semantic routing:** Embeds the redacted prompt with `all-MiniLM-L6-v2` and runs cosine search over a FAISS `IndexFlatIP` index to return cached FAQ answers.

## Evaluation Results (Phase 1 Baseline)

### Semantic Router (Hit Rate @ 1)

| Engine             | Strategy                                 | Hit Rate        |
| :----------------- | :--------------------------------------- | :-------------- |
| **Baseline** | Keyword Substrings                       | 0.0%            |
| **FAISS**    | Cosine Similarity (`all-MiniLM-L6-v2`) | **80.0%** |

### PII Redaction (Strict-Boundary BIO, 100 Italian IT Helpdesk Records / 134 PII Spans)

| Stage | Model / Strategy | Dataset Split | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Regex & Casing | `data/splits/test.json` | 0.02 | 0.07 | 0.03 |
| **Zero-Shot v1 (Initial Baseline)** | `yonigo/deberta-v3-base-pii-en` | `data/splits/test.json` (English templates + Italian names) | **0.66** | **0.37** | **0.48** |
| **Zero-Shot v2 (Italian JSONL Split)** | `yonigo/deberta-v3-base-pii-en` (Pure NER) | `data/processed/test.jsonl` (Full Italian sentences) | 0.68 | 0.66 | 0.67 |
| **Zero-Shot v3 (Current Hybrid Baseline)** | Email Regex + `yonigo/deberta-v3-base-pii-en` | `data/processed/test.jsonl` (Full Italian sentences) | **0.73** | **0.81** | **0.77** |

#### Why the Baseline Metrics Changed (Issue #7 Evaluation Cleanup)

If you came here from my resume, you saw the initial zero-shot baseline (**Precision `0.66`, Recall `0.37`, F1 `0.48`**). During the Issue #7 data-layer audit, I found and fixed two evaluation bottlenecks on the same 100 records (134 PII entities):

1. **Zero-Shot v1 -> v2 (`0.37` -> `0.66` Recall):** The original `data/splits/test.json` file placed Italian names inside 5 English sentence templates (such as `"Please contact Antonietta Gotti..."`). On those mixed-language sentences, DeBERTa often tagged only the surname (`"Gotti"`) and missed the Italian first name. Because `seqeval` runs in strict IOB2 mode, partial name matches count as both a miss and a false positive. When I converted the 100 records into full Italian sentences in `data/processed/test.jsonl` (`"Per favore contatta Antonietta Gotti..."`), DeBERTa predicted the full two-word name spans, raising pure-NER recall to `0.66` (`F1 0.67`).
2. **Zero-Shot v2 -> v3 (`0.66` -> `0.81` Recall):** In `PIIRedactor.extract_entities()`, `_apply_regex()` was replacing all 22 email addresses (`16.4%` of the 134 entities) with `[REDACTED]` before DeBERTa ran, and omitting those regex matches from the returned entity list. I updated `extract_entities()` to return both regex-matched emails and DeBERTa NER spans (and tightened the email regex so it does not consume trailing sentence periods). This raised hybrid zero-shot recall to `0.81` (`F1 0.77`).
3. **Remaining Gap for Phase 2 LoRA:** Even after fixing the email regex path, the English-tuned DeBERTa model still misses **19% of PII** (Italian names and card numbers) and produces **27% false positives** (`Precision 0.73`) on hardware names (`Acer Nitro`) and Italian company names (`Faugno-Fittipaldi SPA`). See `report/baseline_evaluation.md` for the full diagnostic breakdown.

## Data & Reproducibility Policy

- Raw data, weights, and secrets (`data/`, `outputs/**`, `*.safetensors`, `*.onnx`, `mlruns/`, `.env`) are gitignored and blocked by `pre-commit` hooks.
- Global seed `42` is fixed across Python, NumPy, and PyTorch.
