import argparse

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="AI Security Gateway MLOps CLI")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file (e.g., configs/e0_zeroshot.yaml)",
    )
    return parser.parse_args()


def prepare_main():
    """Triggered by the 'gateway-prepare' terminal command."""
    args = parse_args()
    print(f"Starting Data Pipeline using config: {args.config}")
    # We will import the BIO-tagger here soon:
    # from gateway.data.bio_tagger import run_pipeline
    # run_pipeline(args.config)


def train_main():
    """Triggered by the 'gateway-train' terminal command."""
    args = parse_args()
    print(f"Starting LoRA Fine-Tuning using config: {args.config}")
    # from gateway.train.sft import run_training
    # run_training(args.config)


def eval_main():
    """Triggered by the 'gateway-eval' terminal command."""
    args = parse_args()
    print(f" Starting Evaluation using config: {args.config}")
    # from gateway.eval.metrics import evaluate_model
    # evaluate_model(args.config)


def serve_main():
    """Starts the production FastAPI gateway."""
    print(" Booting up AI Security Gateway (FastAPI)...")
    uvicorn.run("gateway.infer.main:app", host="0.0.0.0", port=8000, reload=True)
