from pydantic import BaseModel, ConfigDict, Field, model_validator


class PIIEntity(BaseModel):
    """A slice of text that holds private data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1)
    label: str = Field(default="PII", min_length=1)


class PIIRecord(BaseModel):
    """Shared format for both synthetic training rows and test split rows."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., min_length=1)
    original_prompt: str = Field(..., min_length=1)
    entities: list[PIIEntity] = Field(default_factory=list)


class TokenizedNERRecord(BaseModel):
    """Token IDs, attention mask, and BIO label IDs for PyTorch and Hugging Face."""

    id: str = Field(..., min_length=1)
    input_ids: list[int] = Field(..., min_length=1)
    attention_mask: list[int] = Field(..., min_length=1)
    labels: list[int] = Field(..., min_length=1)

    @model_validator(mode="after")
    def check_array_lengths_match(self) -> "TokenizedNERRecord":
        n_ids = len(self.input_ids)
        if len(self.attention_mask) != n_ids or len(self.labels) != n_ids:
            raise ValueError("input_ids, attention_mask, and labels must have the same length.")
        return self


class RedactionRequest(BaseModel):
    """Incoming web request payload."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The raw user prompt to redact.",
    )


class RedactedResponse(BaseModel):
    """Outgoing redacted web response payload."""

    original_text: str
    redacted_text: str
    entities_found: int = Field(..., ge=0)


class RouteResponse(BaseModel):
    """Semantic router decision and similarity score."""

    model_config = ConfigDict(str_strip_whitespace=True)

    destination: str = Field(..., min_length=1)
    confidence_score: float
