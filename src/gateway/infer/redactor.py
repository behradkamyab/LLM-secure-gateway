import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from gateway.common.schema import RedactedResponse

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]*[a-zA-Z0-9]")


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PIIRedactor:
    def __init__(self, model_id: str):
        self.device = _select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def _find_emails(self, text: str) -> list[str]:
        return EMAIL_PATTERN.findall(text)

    def _apply_regex(self, text: str) -> str:
        return EMAIL_PATTERN.sub("[REDACTED]", text)

    def _detect_ner_intervals(self, cleaned_text: str) -> list[list[int]]:
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0]
        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            best = torch.argmax(outputs.logits, dim=2)[0]

        intervals: list[list[int]] = []
        for token_idx in range(len(best)):
            label = self.id2label[best[token_idx].item()]
            char_start, char_end = offsets[token_idx].tolist()

            if label != "O" and not (char_start == 0 and char_end == 0):
                intervals.append([char_start, char_end])

        intervals.sort(key=lambda x: x[0])
        merged: list[list[int]] = []
        for current in intervals:
            if not merged or merged[-1][1] < current[0]:
                merged.append(current)
            else:
                merged[-1][1] = max(merged[-1][1], current[1])

        return merged

    def redact(self, text: str) -> RedactedResponse:
        """Single-string inference for the web gateway."""
        regex_emails = self._find_emails(text)
        cleaned_text = self._apply_regex(text)
        merged = self._detect_ner_intervals(cleaned_text)

        redacted_text = cleaned_text
        for start, end in reversed(merged):
            redacted_text = redacted_text[:start] + " [REDACTED] " + redacted_text[end:]

        redacted_text = re.sub(r"\s+", " ", redacted_text).strip()

        return RedactedResponse(
            original_text=text,
            redacted_text=redacted_text,
            entities_found=len(merged) + len(regex_emails),
        )

    def extract_entities(self, text: str) -> list[str]:
        """Evaluation helper: returns both regex-matched emails and NER entity strings."""
        regex_emails = self._find_emails(text)
        cleaned_text = self._apply_regex(text)
        merged = self._detect_ner_intervals(cleaned_text)

        extracted: list[str] = []
        for start, end in merged:
            span = cleaned_text[start:end].strip()
            if span and "REDACTED" not in span:
                extracted.append(span)

        for email in regex_emails:
            if email not in extracted:
                extracted.append(email)

        return extracted
