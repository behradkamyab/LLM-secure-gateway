import re
from typing import List
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification

class PIIRedactor:
    def __init__(self, model_id: str = "yonigo/deberta-v3-base-pii-en"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.id2label = self.model.config.id2label
        
        
    def _apply_regex_redaction(self,texts: List[str]) -> List[str]:
        # Email regex example
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        text = re.sub(email_pattern, '[REDACTED]', texts)
        return text

    def extract_entities(self, text: str) -> List[str]:
        """
        Evaluation helper method. Returns the exact PII strings instead of redacting them.
        """
        # Tokenize single string
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")[0]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            best = torch.argmax(outputs.logits, dim=2)[0]

        # Extract the exact strings
        extracted = []
        intervals = []
        for token_idx in range(len(best)):
            label = self.id2label[best[token_idx].item()]
            char_start, char_end = offsets[token_idx].tolist()
            
            if label != "O" and not (char_start == 0 and char_end == 0):
                intervals.append([char_start, char_end])
        
        # Merge overlapping intervals
        intervals.sort(key=lambda x: x[0])
        merged = []
        for current in intervals:
            if not merged or merged[-1][1] < current[0]:
                merged.append(current)
            else:
                merged[-1][1] = max(merged[-1][1], current[1])
                
        # Slice original text to get the actual flagged words
        for start, end in merged:
            extracted.append(text[start:end].strip())
            
        return extracted
    
    def redact_batch(self, texts: List[str]) -> List[str]:
        # Preprocess the entire batch
        cleaned_texts = []
        for text in texts:
            text = re.sub(r'\s+', ' ', text).strip()
            if not text.endswith(('.', '?', '!')):
                text += '.'
            cleaned_texts.append(self._apply_regex_redaction(text))

        #  Tokenize with padding enabled for the batch matrix
        inputs = self.tokenizer(
            cleaned_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")

        # Single batched forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            best = torch.argmax(logits, dim=2)

        # Extract and slice for each sentence in the batch
        redacted_batch = []
        for batch_idx in range(best.shape[0]):
            intervals = []
            
            for token_idx in range(best.shape[1]):
                predicted_id = best[batch_idx, token_idx].item()
                label = self.id2label[predicted_id]
                char_start, char_end = offsets[batch_idx, token_idx].tolist()
                
                if label != "O" and not (char_start == 0 and char_end == 0):
                    intervals.append([char_start, char_end])
            
            # Sort and Merge
            intervals.sort(key=lambda x: x[0])
            merged = []
            for current in intervals:
                if not merged or merged[-1][1] < current[0]:
                    merged.append(current)
                else:
                    merged[-1][1] = max(merged[-1][1], current[1])
            
            # Reverse Slice
            redacted_text = cleaned_texts[batch_idx]
            for start, end in reversed(merged):
                redacted_text = redacted_text[:start] + " [REDACTED]" + redacted_text[end:]
                
            redacted_batch.append(redacted_text)

        return redacted_batch