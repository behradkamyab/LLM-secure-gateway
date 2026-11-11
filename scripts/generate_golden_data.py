import json
from pathlib import Path
import random
from faker import Faker


fake = Faker(['en_US', 'it_IT'])

def generate_ner_data(num_samples=100):
    templates = [
        ("Please contact {name} at {email}.", ["name", "email"]),
        ("Send the Acer Nitro invoice to {name}.", ["name"]),
        ("My name is {name} and my card is {cc}.", ["name", "cc"]),
        ("Call {company} support.", ["company"]), # False positive trap
        ("Is {name} available for the RAG architecture meeting?", ["name"])
    ]
    
    ner_data = []
    for _ in range(num_samples):
        template, entities_to_gen = random.choice(templates)
        
        # Generate fake data
        name = fake.name()
        email = fake.email()
        cc = fake.credit_card_number()
        company = fake.company()
        # Fill template
        text = template.format(name=name, email=email, cc=cc, company=company)
        
        # Track true entities
        true_entities = []
        if "name" in entities_to_gen: true_entities.append(name)
        if "email" in entities_to_gen: true_entities.append(email)
        if "cc" in entities_to_gen: true_entities.append(cc)
        
        ner_data.append({
            "text": text,
            "true_entities": true_entities,
            "type": "synthetic" if true_entities else "false_positive_trap"
        })
    return ner_data

def generate_router_data(num_samples=50):
    # The queries are designed to match the faqs in main.py
    router_cases = [
        {
            "query": "I spilled coffee on my ThinkPad, what do I do?", 
            "expected_faq_id": "What do I do if I spilled water on my laptop?"
        },
        {
            "query": "dropped water on laptop", 
            "expected_faq_id": "What do I do if I spilled water on my laptop?"
        },
        {
            "query": "forgot my active directory login", 
            "expected_faq_id": "How do I reset my Windows password?"
        },
        {
            "query": "cant get into my windows pc", 
            "expected_faq_id": "How do I reset my Windows password?"
        },
        {
            "query": "where are my paystubs for this month?",
            "expected_faq_id": "Where can I view my pay stubs?"
        },
        {
            "query": "my keyboard is missing keys need a new one",
            "expected_faq_id": "What is the process for replacing a broken keyboard?"
        },
        {
            "query": "got a weird spam email asking for my password",
            "expected_faq_id": "How do I report a phishing email?"
        },
        {
            "query": "lost my office keycard this morning",
            "expected_faq_id": "What do I do if I lost my access fob?"
        },
        {
            "query": "how many vacation days do i have left",
            "expected_faq_id": "How many PTO days do I have left?"
        },
        {
            "query": "need to get on the corporate virtual private network from home",
            "expected_faq_id": "How do I connect to the corporate VPN?"
        }
    ]
    return random.choices(router_cases, k=num_samples)

if __name__ == "__main__":
    dataset = {
        "ner_evaluation": generate_ner_data(100),
        "router_evaluation": generate_router_data(50)
    }
    
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / "golden_dataset.json"
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated 150 test cases at:\n{file_path}")