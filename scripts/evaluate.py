import json
import re
from pathlib import Path
import sys
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.redactor.engine import PIIRedactor
from src.router.engine import SemanticRouter

def load_dataset():
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir.parent / "data" / "golden_dataset.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def baseline_regex_ner(text):
    found = []
    found.extend(re.findall(r'[\w\.-]+@[\w\.-]+', text))
    found.extend(re.findall(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b', text))
    caps = re.findall(r'\b[A-Z][a-z]+\b', text)
    found.extend(caps)
    return list(set(found))

def baseline_keyword_router(query):
    query = query.lower()
    if "pricing" in query or "cost" in query:
        return "faq_pricing"
    elif "password" in query or "reset" in query:
        return "faq_password_reset"
    elif "spilled coffee" in query or "water" in query:
        return "faq_water_damage"
    return "unknown"


def evaluate_ner(dataset, engine_type="baseline", redactor_engine=None):
    y_true_all = []
    y_pred_all = []
    
    
    # Track how many errors we've printed so we don't flood the terminal
    errors_printed = 0
    
    for item in dataset["ner_evaluation"]:
        text = item["text"]
        true_entities = set(item["true_entities"])
        
        if engine_type == "baseline":
            pred_entities = set(baseline_regex_ner(text))
        elif engine_type == "onnx_ai" and redactor_engine:
            
            pred_entities = set(redactor_engine.extract_entities(text))
        else:
            pred_entities = set()
            
        all_possible_entities = true_entities.union(pred_entities)
        
        # ==========================================
        # DIAGNOSTIC ERROR LOGGING
        # ==========================================
        missed = true_entities - pred_entities
        wrong = pred_entities - true_entities
        
        if engine_type == "onnx_ai" and (missed or wrong) and errors_printed < 5:
            print(f"\n[AI ERROR] Sentence: {text}")
            if missed:
                print(f"  ❌ MISSED (False Negative): {missed}")
            if wrong:
                print(f"  ⚠️ WRONG (False Positive): {wrong}")
            errors_printed += 1
        # ==========================================
        
        for entity in all_possible_entities:
            y_true_all.append(1 if entity in true_entities else 0)
            y_pred_all.append(1 if entity in pred_entities else 0)

    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    
    print(f"--- NER Evaluation: {engine_type.upper()} ---")
    print(f"Recall:    {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1-Score:  {f1:.2f}\n")

def evaluate_router(dataset, engine_type="baseline", router_engine=None):
    hits = 0
    total = len(dataset["router_evaluation"])
    
    for item in dataset["router_evaluation"]:
        query = item["query"]
        expected_faq = item["expected_faq_id"]
        
        if engine_type == "baseline":
            pred_faq = baseline_keyword_router(query)
        elif engine_type == "faiss_ai" and router_engine:
            # Call your actual route method
            decision = router_engine.route(query)
            
            # Extract the response if it was a cache hit
            if decision["action"] == "CACHE_HIT":
                pred_faq = decision["response"]
            else:
                pred_faq = "unknown"
        else:
            pred_faq = "unknown"
            
        if pred_faq == expected_faq:
            hits += 1
            
    hit_rate = (hits / total) * 100
    print(f"--- Router Evaluation: {engine_type.upper()} ---")
    print(f"Hit Rate @ 1: {hit_rate:.1f}%\n")

if __name__ == "__main__":
    data = load_dataset()
    
    project_root = Path(__file__).resolve().parent.parent
    deberta_path = str(project_root / "deberta_onnx")
    minilm_path = str(project_root / "minilm_onnx")
    faqs = [
    # IT & Hardware
    "How do I reset my Windows password?",
    "How do I connect to the corporate VPN?",
    "Why is my Wi-Fi disconnecting in the conference room?",
    "How do I request a second monitor?",
    "How do I install approved software on my laptop?",
    "What is the process for replacing a broken keyboard?",
    "How do I set up my work email on my personal phone?",
    "Why is my account locked out?",
    "How do I map a network drive?",
    "Where do I download the latest antivirus update?",
    "How do I clear my browser cache?",
    "How do I access the shared IT drive?",
    "Why is my laptop battery draining so fast?",
    "How do I request admin rights for my machine?",
    "What do I do if I spilled water on my laptop?",
    "How do I connect to the wireless printer?",
    "Why is my Zoom audio not working?",
    "How do I update my operating system?",
    "How do I transfer files to my new computer?",
    "Who do I contact for server downtime?",
    
    # HR & Benefits
    "How many PTO days do I have left?",
    "How do I request paid time off?",
    "What are the official company holidays this year?",
    "How do I enroll in the company 401k?",
    "What is the company match for retirement?",
    "How do I add a dependent to my health insurance?",
    "What is the maternity leave policy?",
    "What is the paternity leave policy?",
    "How do I report a grievance with a coworker?",
    "Where can I find the employee handbook?",
    "How do I apply for internal job postings?",
    "What is the policy on remote work?",
    "How do I request a desk assessment for ergonomics?",
    "What is the process for submitting a performance review?",
    "How do I access my employee assistance program (EAP)?",
    "What is the dress code policy?",
    "How do I get reimbursement for a training course?",
    "Can I cash out my unused PTO?",
    "What are the core working hours?",
    "How do I update my emergency contact information?",
    
    # Payroll & Billing
    "Where can I view my pay stubs?",
    "How do I change my direct deposit account?",
    "When do W2 forms get mailed out?",
    "How do I submit an expense report?",
    "What is the mileage reimbursement rate?",
    "How do I apply for a corporate credit card?",
    "What do I do if my corporate card is declined?",
    "What is the deadline for submitting monthly expenses?",
    "How do I change my tax withholding allowances?",
    "Why is my paycheck lower than expected?",
    "How do I get a copy of last year's tax documents?",
    "What expenses are eligible for the wellness stipend?",
    "How do I bill a client for travel time?",
    "What is the approval workflow for hardware purchases?",
    "How do I dispute a payroll discrepancy?",
    "When are annual bonuses paid out?",
    "How do I track billable hours?",
    "What is the per diem rate for international travel?",
    "How do I report a lost corporate card?",
    "Can I use the corporate card for personal emergencies?",
    
    # Security & Compliance
    "How do I report a phishing email?",
    "What is the policy for sharing passwords?",
    "How do I get a replacement ID badge?",
    "What is the procedure for registering a visitor?",
    "Can I bring my family into the office?",
    "What is the clean desk policy?",
    "How do I securely send a document to an external vendor?",
    "What do I do if I lost my access fob?",
    "How do I report a physical security breach?",
    "Where is the emergency assembly point?",
    "How do I lock my computer when walking away?",
    "Can I use public Wi-Fi on my work laptop?",
    "What is the data retention policy for emails?",
    "How do I classify a document as confidential?",
    "Who is the Data Protection Officer?",
    "What should I do if my laptop is stolen?",
    "How often do I need to complete security compliance training?",
    "Are we allowed to use USB flash drives?",
    "How do I request a building access audit?",
    "What is the protocol for active shooter situations?",
    
    # Office & Facilities
    "Where do I park my car at the main office?",
    "How do I reserve a conference room?",
    "Who do I contact if the restroom needs cleaning?",
    "Is there a lost and found?",
    "How do I operate the coffee machine in the breakroom?",
    "What time does the building front door lock?",
    "How do I request a standing desk?",
    "Where is the first aid kit located?",
    "What is the cafeteria menu for this week?",
    "Can I receive personal packages at the office?",
    "How do I adjust the thermostat in my area?",
    "What are the rules for storing food in the communal fridge?",
    "How do I request maintenance for a broken chair?",
    "Where is the designated smoking area?",
    "Are pets allowed in the office?",
    "How do I book the corporate shuttle?",
    "Who is responsible for watering the office plants?",
    "What are the gym access hours?",
    "How do I register my bicycle for the bike room?",
    "Where do I drop off outgoing corporate mail?"
]
    
    
    # Initialize your actual ONNX and FAISS engines
    print("Loading ONNX and FAISS models into memory...\n")
    # Update these paths and class names to match your repository
    redactor = PIIRedactor()
    router = SemanticRouter(faqs=faqs , model_id=minilm_path , threshold=0.0)
    
    print("Executing Benchmarks...\n")
    
    # Run NER Comparisons
    evaluate_ner(data, engine_type="baseline")
    evaluate_ner(data, engine_type="onnx_ai", redactor_engine=redactor)
    
    # Run Router Comparisons
    evaluate_router(data, engine_type="baseline")
    evaluate_router(data, engine_type="faiss_ai", router_engine=router)