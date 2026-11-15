from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from src.redactor.engine import PIIRedactor
from src.router.engine import SemanticRouter
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

# Global dictionary to hold AI engines in memory
ml_engines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    LIFESPAN: This function runs ONCE when the server starts and ONCE when it shuts down.
    It is used to load the AI engines into memory and free them when the server stops.
    """
    print("Booting up Security Gateway...")
    
    print("Loading PIIRedactor (DeBERTa-v3)...")
    ml_engines["redactor"] = PIIRedactor() 
    
    print("Loading SemanticRouter (MiniLM + FAISS)...")
    ml_engines["router"] = SemanticRouter(faqs=faqs) 
    
    print("✅ Gateway is live and accepting traffic.")
    
    yield # Hands control over to FastAPI to start accepting HTTP requests
    
    print("🛑 Shutting down Gateway and clearing VRAM/RAM...")
    ml_engines.clear()

# Initialize the FastAPI app with the lifespan manager
app = FastAPI(title="Enterprise AI Security Gateway", lifespan=lifespan)

# Strictly forces the incoming JSON to match this shape
class GatewayRequest(BaseModel):
    prompts: List[str]

@app.post("/api/v1/gateway")
def process_prompts(request: GatewayRequest):
    """
    This endpoint accepts a list of strings, redacts PII, and routes them to the appropriate response.
    
    """
    
    # Pass the entire list of strings into the batch redactor
    redacted_batch = ml_engines["redactor"].redact_batch(request.prompts)
    
    # Route the newly redacted strings
    results = []
    
    # Loop through the original and redacted prompts in parallel
    for original_text, redacted_text in zip(request.prompts, redacted_batch):
        
        # Check FAISS index
        routing_decision = ml_engines["router"].route(redacted_text)
        
        # Package the result
        results.append({
            "original_prompt": original_text,
            "redacted_prompt": redacted_text,
            "routing": routing_decision
        })
        
    return {"results": results}