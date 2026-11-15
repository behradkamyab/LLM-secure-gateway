# Contributing to AI Security Gateway

First off, thank you for contributing! This project follows a strict MLOps architecture designed for reproducibility, data governance, and modularity. 

To ensure a smooth collaboration, please follow the guidelines below regarding environment setup, branching, and data handling.

## 1. Local Development Setup

We use `uv` for lightning-fast, globally cached dependency management. 

### Step 1: Clone the repository
```bash
git clone [https://github.com/your-username/llm-secure-gateway-v2.git](https://github.com/your-username/llm-secure-gateway-v2.git)
cd llm-secure-gateway-v2
```

### Step 2: Sync the environment
Install all dependencies (including training, inference, and dev tools) without globally installing them on your machine:
```bash
uv sync --all-extras
```

### Step 3: Install the Git Hooks (MANDATORY)
We use `pre-commit` to automatically format Python code and actively block the committing of raw data or model weights.
```bash
pre-commit install
```

## 2. Branching Naming Conventions

Please do not push directly to `main`. Create a branch using the following prefixes:
* `feat/` - For new application features (e.g., `feat/add-onnx-quantization`)
* `fix/` - For bug fixes (e.g., `fix/regex-overlap-bug`)
* `exp/` - For new ML experiments and training runs (e.g., `exp/lora-italian-names`)
* `docs/` - For documentation updates

## 3. Strict MLOps & Data Rules

This repository enforces a strict decoupling of data, training, and inference.

* **Never Commit Data or Weights:** The `data/` and `outputs/` directories are heavily `.gitignore`d. If you try to commit a `.json` dataset, `.onnx` binary, or `.safetensors` weight, the pre-commit hook will block you.
* **The Unified Data Contract:** If you write a new data generation or preprocessing script, it MUST output JSON data that strictly validates against the Pydantic schemas defined in `src/gateway/common/schema.py`.
* **Immutable Experiments:** Never edit an existing experiment configuration (e.g., `configs/e0_zeroshot.yaml` or `configs/e1_lora.yaml`) if its results have already been logged or merged. If you want to try a new batch size or learning rate, create a new file (e.g., `configs/e2_lora_batch16.yaml`).

## 4. Submitting a Pull Request (PR)

When you are ready to open a PR, ensure you meet the following checklist:
- [ ] Code passes formatting (run `pre-commit run --all-files` manually if needed).
- [ ] You have not included any `.env` files, API keys, or raw data.
- [ ] If you added a new pipeline step, you updated the `pyproject.toml` CLI entrypoints if necessary.
- [ ] You have provided a brief summary of the changes and attached any relevant evaluation metrics if this PR changes model performance.