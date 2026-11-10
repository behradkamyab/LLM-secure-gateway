import pytest
from src.redactor.engine import PIIRedactor

@pytest.fixture(scope="module")
def redactor():
    """
    Loads the PyTorch engine once for the entire test session.
    Mimics the FastAPI lifespan behavior.
    """
    return PIIRedactor()

def test_deterministic_regex_email(redactor):
    """
    Tests the standard Regex layer.
    """
    prompts = ["Please send the architectural specs to behrad.kamyab@studenti.polito.it immediately."]
    result = redactor.redact_batch(prompts)[0]
    
    assert "behrad.kamyab@studenti.polito.it" not in result
    assert "[REDACTED]" in result

def test_probabilistic_ner_name(redactor):
    """
    Tests the DeBERTa Transformer's ability to identify unstructured names.
    """
    prompts = ["My name is Behrad and I spilled coffee on my ThinkPad."]
    result = redactor.redact_batch(prompts)[0]
    
    assert "Behrad" not in result
    assert "My name is [REDACTED]" in result

def test_probabilistic_ner_organization(redactor):
    """
    Tests the Transformer's ability to catch corporate entities.
    """
    prompts = ["I used to manage Linux servers at Pardazesh Tasvir Kaman."]
    result = redactor.redact_batch(prompts)[0]
    
    assert "Pardazesh Tasvir Kaman" not in result
    assert "[REDACTED]" in result

def test_batch_matrix_processing(redactor):
    """
    Validates that the 3D matrix math processes multiple strings 
    simultaneously without cross-contamination.
    """
    prompts = [
        "My name is Behrad and I spilled coffee on my ThinkPad.",
        "Please send the architectural specs to behrad.kamyab@studenti.polito.it immediately.",
        "The server is running Ubuntu 24.04." 
    ]
    
    results = redactor.redact_batch(prompts)
    
    assert len(results) == 3
    
    # Verifies NER batching
    assert "Behrad" not in results[0]
    
    # Verifies Regex batching
    assert "behrad.kamyab@studenti.polito.it" not in results[1]
    
    # Verifies clean text passes through untouched
    assert results[2] == "The server is running Ubuntu 24.04."