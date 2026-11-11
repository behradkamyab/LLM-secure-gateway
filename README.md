# AI Security Gateway & Semantic Router

An asynchronous, production-ready FastAPI microservice acting as a security and routing layer for Enterprise RAG systems. It intercepts user prompts, redacts PII using a hybrid Regex/NER approach, and uses Vector Search to route queries to cached FAQs or downstream LLMs.

## 📍 Project Roadmap & Status

This project follows an iterative MLOps lifecycle:

- [x] **Phase 1: Infrastructure & Baseline Benchmarks (Completed)**
  - Async FastAPI setup with `lifespan` model loading.
  - Regex vs. Zero-Shot DeBERTa-v3 token classification evaluation.
  - Keyword vs. FAISS (`all-MiniLM-L6-v2`) semantic router evaluation.
- [ ] **Phase 2: Domain Adaptation via LoRA PEFT (Active)**
  - BIO-tagging synthetic enterprise data.
  - Parameter-Efficient Fine-Tuning (PEFT) on DeBERTa to resolve Italian/Enterprise domain shift.
  - MLflow experiment tracking.
- [ ] **Phase 3: Production Inference Optimization (Planned)**
  - Merging LoRA adapters and compiling the finalized model to ONNX.
  - End-to-end latency and memory benchmarking.

## 📂 Repository Structure
```text
.
├── data/                  # Synthetic datasets (golden_dataset.json)
├── deberta_onnx/          # Compiled ONNX binary for PII token classification
├── docs/                  # Architecture diagrams, evaluation reports, and charts
├── minilm_onnx/           # Compiled ONNX binary for semantic embeddings
├── scripts/               # Task-based entrypoints (generate data, evaluate, charts)
├── src/                   # FastAPI gateway, redactor, and router microservices
├── tests/                 # Pytest suite for engine unit testing
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Quick Start & Script Workflow

The repository is divided into the evaluation pipeline and the live API gateway.

### 1. The Evaluation Pipeline
If you want to reproduce the baseline metrics or generate new synthetic data, run the modular scripts:

```bash
# 1. Generate the synthetic JSON dataset
python scripts/generate_golden_data.py

# 2. Run the benchmarking evaluation (NER & Router)
python scripts/evaluate.py

# 3. Generate Matplotlib performance charts
python scripts/generate_charts.py
```

### 2. The API Gateway
On startup, the API initializes the brain. Through the async `lifespan` handler, it safely downloads and loads the ONNX models into RAM, compiles the target FAISS dataset, and prepares your CPU-accelerated worker pool *before* allowing the gateway to receive its first network request.

```bash
uvicorn src.main:app --reload
```

## 📊 Evaluation Results

The project report and experiments compare legacy deterministic baselines against zero-shot AI models for enterprise routing and redaction.

### Semantic Router (Hit Rate @ 1)
| Engine | Strategy | Hit Rate |
| :--- | :--- | :--- |
| **Baseline** | Rigid Keyword Substrings | 0.0% |
| **FAISS AI** | Cosine Similarity (`all-MiniLM-L6-v2`) | **80.0%** |

### PII Redaction (Entity Extraction)
| Engine | Strategy | Recall | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Regex & Casing | 0.07 | 0.02 | 0.03 |
| **Zero-Shot AI** | `DeBERTa-v3` Token Classification | **0.54** | **0.62** | **0.57** |

*Note: The 54% Recall rate on the DeBERTa model identified a critical domain shift vulnerability regarding Italian naming conventions and enterprise structures. This is actively being resolved in Phase 2 via LoRA fine-tuning. See `docs/model_evaluation_report.md` for the full diagnostic breakdown.*

## 🏗 Architecture

<p align="center">
  <img src="docs/images/architecture.jpg" alt="AI Security Gateway Pipeline Architecture" width="100%">
</p>

This gateway executes a strict pipeline before any prompt touches a generative LLM:
1. **Input Normalization:** Sanitizes whitespace and structural casing.
2. **Deterministic Redaction (Regex):** Strips structured PII (Emails, Phone Numbers, Credit Cards).
3. **Probabilistic Redaction (NER):** Uses `DeBERTa-v3` token classification for unstructured PII (Names, Organizations). Implements interval-merging for subword fragmentation.
4. **Semantic Routing:** Embeds the sanitized prompt using `all-MiniLM-L6-v2` and searches a `FAISS` IndexFlatIP (Cosine Similarity) C++ struct to return cached answers in $O(N)$ time.

## 🔌 API Usage

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

## 🛠 Tech Stack
- **Web Framework:** FastAPI, Uvicorn, Pydantic
- **AI & NLP:** ONNX Runtime, PyTorch, Transformers, Sentence-Transformers
- **Vector Search:** FAISS (Exact Search / Inner Product)
- **Package Management:** `uv`, `pip`