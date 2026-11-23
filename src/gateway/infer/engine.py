import json
from pathlib import Path

from gateway.common.config import load_config
from gateway.infer.redactor import PIIRedactor
from gateway.infer.router import SemanticRouter


class GatewayEngine:
    def __init__(self, config_path: str = "configs/e0_zeroshot.yaml"):
        self.config = load_config(config_path)

        intents_path = Path("data/router_intents.json")
        if intents_path.exists():
            with open(intents_path, "r", encoding="utf-8") as f:
                faqs = json.load(f)
        else:
            faqs = {}

        print("Initializing PyTorch Semantic Router...")
        self.router = SemanticRouter(
            faqs=faqs,
            threshold=self.config["model"]["router_engine"]["threshold"],
            model_id=self.config["model"]["router_engine"]["name"],
        )

        print("Initializing PyTorch PII Redactor...")
        self.redactor = PIIRedactor(model_id=self.config["model"]["ner_engine"]["name"])
        print("Gateway Engine Ready.")

    def process_request(self, text: str) -> dict:
        """Routes the text, and if it is headed to the LLM, redacts it first."""
        route_res = self.router.route(text)

        if route_res.destination == "LLM_PIPELINE":
            redact_res = self.redactor.redact(text)
            return {
                "route": route_res.model_dump(),
                "redacted_payload": redact_res.model_dump(),
            }

        return {
            "route": route_res.model_dump(),
            "redacted_payload": None,
        }
