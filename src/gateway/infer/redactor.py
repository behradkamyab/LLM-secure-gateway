import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from gateway.common.schema import RedactedResponse

class PIIRedactor:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.id2label = self.model.config.id2label
        
    def _apply_regex(self, text: str) -> str:
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        return re.sub(email_pattern, '[REDACTED]', text)

    def redact(self, text: str) -> RedactedResponse:
        """Single-string inference for the web gateway."""
        cleaned_text = self._apply_regex(text)
        
        inputs = self.tokenizer(
            cleaned_text, 
            return_tensors="pt", 
            return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = self.model(**inputs)
            best = torch.argmax(outputs.logits, dim=2)[0]

        intervals = []
        for token_idx in range(len(best)):
            label = self.id2label[best[token_idx].item()]
            char_start, char_end = offsets[token_idx].tolist()
            
            if label != "O" and not (char_start == 0 and char_end == 0):
                intervals.append([char_start, char_end])
        
        intervals.sort(key=lambda x: x[0])
        merged = []
        for current in intervals:
            if not merged or merged[-1][1] < current[0]:
                merged.append(current)
            else:
                merged[-1][1] = max(merged[-1][1], current[1])
                
        redacted_text = cleaned_text
        for start, end in reversed(merged):
            redacted_text = redacted_text[:start] + " [REDACTED] " + redacted_text[end:]
            
        # Strip extra spaces caused by replacement
        redacted_text = re.sub(r'\s+', ' ', redacted_text).strip()
            
        return RedactedResponse(
            original_text=text,
            redacted_text=redacted_text,
            entities_found=len(merged)
        )