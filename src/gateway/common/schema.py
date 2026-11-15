from pydantic import BaseModel, Field
from typing import List


class PIIEntity(BaseModel):
    """A specific slice of text that represents private data."""
    text: str
    label: str  # e.g., "PERSON", "ORG", "CREDIT_CARD"

class SyntheticDataRecord(BaseModel):
    """The JSON contract for the unstructured generated text."""
    id: str
    original_prompt: str
    entities: List[PIIEntity] = Field(default_factory=list)



class TokenizedNERRecord(BaseModel):
    """The strict mathematical array contract required by PyTorch/Hugging Face."""
    id: str
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]  # Integers matching B-PII, I-PII, O, and -100 for ignored subwords